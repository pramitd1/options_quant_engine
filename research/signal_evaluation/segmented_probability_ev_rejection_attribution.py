"""Attribution for rejected segmented-probability EV shadow candidates.

This module is research-only. It explains why a shadow-calibrated probability
candidate failed EV/risk review, writes advisory artifacts, and never changes
runtime configuration, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR,
    EV_SHADOW_REJECTED,
    SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME,
    SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME,
    _top_probability_mask,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_REGIME_FIELDS,
    DEFAULT_RETURN_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_ev_rejection_attribution"
)
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR / SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR / SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME
)

SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_JSON_FILENAME = (
    "latest_segmented_probability_ev_rejection_attribution.json"
)
SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_MARKDOWN_FILENAME = (
    "latest_segmented_probability_ev_rejection_attribution.md"
)
SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_CANDIDATES_FILENAME = (
    "latest_segmented_probability_ev_rejection_candidates.csv"
)
SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_SHADOW_ONLY_CANDIDATES_FILENAME = (
    "latest_segmented_probability_ev_rejection_shadow_only_candidates.csv"
)
SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_REGIMES_FILENAME = (
    "latest_segmented_probability_ev_rejection_regimes.csv"
)
SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_POLICIES_FILENAME = (
    "latest_segmented_probability_ev_rejection_policies.csv"
)

EV_REJECTION_ATTRIBUTION_ACTIONABLE = "EV_REJECTION_ATTRIBUTION_ACTIONABLE"
EV_REJECTION_ATTRIBUTION_WATCH = "EV_REJECTION_ATTRIBUTION_WATCH"
EV_REJECTION_ATTRIBUTION_NOT_REJECTED = "EV_REJECTION_ATTRIBUTION_NOT_REJECTED"
EV_REJECTION_ATTRIBUTION_INSUFFICIENT_DATA = "EV_REJECTION_ATTRIBUTION_INSUFFICIENT_DATA"

UNROUTED_IDENTITY_KEY = "unrouted_identity"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index)), errors="coerce")


def _first_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns or group.empty:
        return None
    values = group[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_routes(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _summary_for_frame(
    frame: pd.DataFrame,
    *,
    return_field: str = DEFAULT_RETURN_FIELD,
    max_spread_pct: float = 5.0,
) -> dict[str, Any]:
    count = int(len(frame))
    if count <= 0:
        return {
            "sample_count": 0,
            "hit_rate": None,
            "avg_return_bps": None,
            "avg_risk_adjusted_return_bps": None,
            "avg_liquidity_adjusted_return_bps": None,
        }

    returns = _numeric(frame, "_return_bps")
    if returns.notna().sum() == 0 and return_field in frame.columns:
        returns = _numeric(frame, return_field)
    risk_adjusted = _numeric(frame, "_risk_adjusted_return_bps")
    if risk_adjusted.notna().sum() == 0:
        risk_adjusted = returns
    liquidity_adjusted = _numeric(frame, "_liquidity_adjusted_return_bps")
    if liquidity_adjusted.notna().sum() == 0:
        liquidity_adjusted = risk_adjusted
    labels = _numeric(frame, "label")
    mfe = _numeric(frame, "mfe_60m_bps")
    mae = _numeric(frame, "mae_60m_bps").abs()
    spread = _numeric(frame, "selected_option_ba_spread_pct")
    volume = _numeric(frame, "selected_option_volume")
    open_interest = _numeric(frame, "selected_option_open_interest")
    high_spread_rate = None
    if spread.notna().any():
        high_spread_rate = float((spread > float(max_spread_pct)).mean())
    positive_returns = returns.loc[returns > 0]
    negative_returns = returns.loc[returns < 0]
    win_loss_ratio = None
    if not positive_returns.empty and not negative_returns.empty:
        win_loss_ratio = float(positive_returns.mean()) / max(abs(float(negative_returns.mean())), 1e-9)

    return {
        "sample_count": count,
        "hit_rate": _round_or_none(labels.mean(), 6),
        "avg_return_bps": _round_or_none(returns.mean(), 6),
        "median_return_bps": _round_or_none(returns.median(), 6),
        "avg_risk_adjusted_return_bps": _round_or_none(risk_adjusted.mean(), 6),
        "avg_liquidity_adjusted_return_bps": _round_or_none(liquidity_adjusted.mean(), 6),
        "total_risk_adjusted_return_bps": _round_or_none(risk_adjusted.sum(), 6),
        "avg_mfe_60m_bps": _round_or_none(mfe.mean(), 6),
        "avg_abs_mae_60m_bps": _round_or_none(mae.mean(), 6),
        "p95_abs_mae_60m_bps": _round_or_none(mae.quantile(0.95), 6),
        "win_loss_return_ratio": _round_or_none(win_loss_ratio, 6),
        "avg_spread_pct": _round_or_none(spread.mean(), 6),
        "high_spread_rate": _round_or_none(high_spread_rate, 6),
        "avg_selected_option_volume": _round_or_none(volume.mean(), 6),
        "avg_selected_option_open_interest": _round_or_none(open_interest.mean(), 6),
    }


def _delta(value: Any, baseline: Any) -> float | None:
    left = _safe_float(value)
    right = _safe_float(baseline)
    if left is None or right is None:
        return None
    return left - right


def _policy_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("policy_results", []) or []
    return [row for row in rows if isinstance(row, dict)]


def _policy_row(report: dict[str, Any], route_policy: str) -> dict[str, Any]:
    for row in _policy_rows(report):
        if str(row.get("route_policy")) == str(route_policy):
            return row
    return {}


def _select_route_policy(
    report: dict[str, Any],
    routes: pd.DataFrame,
    *,
    route_policy: str | None = None,
) -> str:
    if route_policy:
        return str(route_policy)
    selected = (report.get("selection_summary", {}) or {}).get("recommended_routing_policy")
    if selected:
        return str(selected)
    if "route_policy" in routes.columns:
        values = routes["route_policy"].dropna().astype(str).unique()
        if len(values) > 0:
            return str(values[0])
    return "unknown"


def _routes_for_policy(routes: pd.DataFrame, route_policy: str) -> pd.DataFrame:
    if routes.empty or "route_policy" not in routes.columns or route_policy == "unknown":
        return routes.copy()
    selected = routes.loc[routes["route_policy"].astype(str) == str(route_policy)].copy()
    return selected if not selected.empty else routes.copy()


def _candidate_status(
    *,
    candidate_key: str,
    attribution_basis: str,
    sample_count: int,
    avg_risk_adjusted_return_bps: float | None,
    share_of_bucket: float,
    min_candidate_sample: int,
    negative_route_keys: set[str],
) -> str:
    if sample_count < int(min_candidate_sample):
        return "LOW_SAMPLE_WATCH"
    if avg_risk_adjusted_return_bps is not None and avg_risk_adjusted_return_bps < 0:
        if candidate_key in negative_route_keys:
            return "PRUNE_CANDIDATE_ROUTE"
        if attribution_basis == "shadow_only_top_bucket":
            return "PROMOTED_ROWS_DAMAGE_REVIEW"
        return "TOP_BUCKET_DAMAGE_REVIEW"
    if share_of_bucket >= 0.20:
        return "KEEP_CANDIDATE_ROUTE_UNDER_REVIEW"
    return "CANDIDATE_ROUTE_WATCH"


def _candidate_attribution(
    frame: pd.DataFrame,
    *,
    attribution_basis: str,
    min_candidate_sample: int,
    return_field: str,
    max_spread_pct: float,
    negative_route_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if frame.empty or "assigned_candidate_key" not in frame.columns:
        return []
    total_count = max(int(len(frame)), 1)
    negative_keys = set(negative_route_keys or set())
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby("assigned_candidate_key", dropna=False):
        candidate_key = str(key)
        summary = _summary_for_frame(group, return_field=return_field, max_spread_pct=max_spread_pct)
        sample_count = int(summary.get("sample_count") or 0)
        share = float(sample_count) / float(total_count)
        avg_risk = _safe_float(summary.get("avg_risk_adjusted_return_bps"))
        status = _candidate_status(
            candidate_key=candidate_key,
            attribution_basis=attribution_basis,
            sample_count=sample_count,
            avg_risk_adjusted_return_bps=avg_risk,
            share_of_bucket=share,
            min_candidate_sample=min_candidate_sample,
            negative_route_keys=negative_keys,
        )
        if candidate_key == UNROUTED_IDENTITY_KEY and status == "PRUNE_CANDIDATE_ROUTE":
            status = "IDENTITY_ROUTE_DAMAGE"
        row = {
            "attribution_basis": attribution_basis,
            "candidate_key": candidate_key,
            "attribution_status": status,
            "share_of_bucket": _round_or_none(share, 6),
            "assigned_candidate_type": _first_value(group, "assigned_candidate_type"),
            "assigned_segment_field": _first_value(group, "assigned_segment_field"),
            "assigned_segment_value": _first_value(group, "assigned_segment_value"),
            "assigned_calibrator": _first_value(group, "assigned_calibrator"),
            "matched_candidate_count_avg": _round_or_none(_numeric(group, "matched_candidate_count").mean(), 6),
        }
        row.update(summary)
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            item.get("attribution_status") not in {"PRUNE_CANDIDATE_ROUTE", "IDENTITY_ROUTE_DAMAGE"},
            item.get("attribution_status") != "PROMOTED_ROWS_DAMAGE_REVIEW",
            item.get("attribution_status") != "TOP_BUCKET_DAMAGE_REVIEW",
            float(item.get("avg_risk_adjusted_return_bps") or 1e9),
            -float(item.get("share_of_bucket") or 0.0),
        ),
    )


def _negative_candidate_routes_from_report(report: dict[str, Any], route_policy: str) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("candidate_route_results", []) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("route_policy")) != str(route_policy):
            continue
        if str(row.get("ev_route_status")) != "EV_ROUTE_NEGATIVE":
            continue
        rows.append(
            {
                "route_policy": row.get("route_policy"),
                "candidate_key": row.get("candidate_key"),
                "ev_route_status": row.get("ev_route_status"),
                "sample_count": row.get("sample_count"),
                "avg_risk_adjusted_return_bps": row.get("avg_risk_adjusted_return_bps"),
                "hit_rate": row.get("hit_rate"),
                "assigned_candidate_type": row.get("assigned_candidate_type"),
                "assigned_segment_field": row.get("assigned_segment_field"),
                "assigned_segment_value": row.get("assigned_segment_value"),
                "assigned_calibrator": row.get("assigned_calibrator"),
            }
        )
    return sorted(rows, key=lambda item: float(item.get("avg_risk_adjusted_return_bps") or 1e9))


def _regime_attribution(
    frame: pd.DataFrame,
    *,
    attribution_basis: str,
    regime_fields: tuple[str, ...],
    min_regime_sample: int,
    return_field: str,
    max_spread_pct: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    total_count = max(int(len(frame)), 1)
    for field in regime_fields:
        if field not in frame.columns:
            continue
        for value, group in frame.groupby(field, dropna=False):
            if len(group) < int(min_regime_sample):
                continue
            summary = _summary_for_frame(group, return_field=return_field, max_spread_pct=max_spread_pct)
            avg_risk = _safe_float(summary.get("avg_risk_adjusted_return_bps"))
            status = "REGIME_DAMAGE" if avg_risk is not None and avg_risk < 0 else "REGIME_SUPPORTIVE"
            row = {
                "attribution_basis": attribution_basis,
                "regime_field": field,
                "regime_value": str(value),
                "attribution_status": status,
                "share_of_bucket": _round_or_none(float(len(group)) / float(total_count), 6),
            }
            row.update(summary)
            rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            item.get("attribution_status") != "REGIME_DAMAGE",
            float(item.get("avg_risk_adjusted_return_bps") or 1e9),
            -float(item.get("share_of_bucket") or 0.0),
        ),
    )


def _policy_comparison(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _policy_rows(report):
        rows.append(
            {
                "route_policy": row.get("route_policy"),
                "ev_shadow_status": row.get("ev_shadow_status"),
                "status_reason": row.get("status_reason"),
                "policy_score": row.get("policy_score"),
                "shadow_vs_raw_top_risk_adjusted_return_delta_bps": row.get(
                    "shadow_vs_raw_top_risk_adjusted_return_delta_bps"
                ),
                "shadow_vs_raw_top_hit_rate_delta": row.get("shadow_vs_raw_top_hit_rate_delta"),
                "shadow_top_vs_bottom_risk_adjusted_return_spread_bps": row.get(
                    "shadow_top_vs_bottom_risk_adjusted_return_spread_bps"
                ),
                "liquidity_status": row.get("liquidity_status"),
            }
        )
    return rows


def _best_policy_by_score(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not policy_rows:
        return None
    return sorted(
        policy_rows,
        key=lambda row: (
            float(row.get("policy_score") or -1e9),
            float(row.get("shadow_vs_raw_top_risk_adjusted_return_delta_bps") or -1e9),
        ),
        reverse=True,
    )[0]


def _recommended_actions(
    *,
    status: str,
    attribution_reasons: list[str],
    negative_route_candidates: list[dict[str, Any]],
    candidate_attribution: list[dict[str, Any]],
    regime_attribution: list[dict[str, Any]],
) -> list[str]:
    if status == EV_REJECTION_ATTRIBUTION_INSUFFICIENT_DATA:
        return [
            "Collect more quality-approved labeled route rows before diagnosing EV rejection.",
            "Keep the candidate bundle in research-only shadow review.",
        ]
    if status == EV_REJECTION_ATTRIBUTION_NOT_REJECTED:
        return [
            "No EV rejection was attributed for the selected route policy.",
            "Keep monitoring forward-shadow and EV/risk artifacts before any manual adoption review.",
        ]

    actions = [
        "Do not advance this segmented-probability candidate bundle while EV rejection attribution remains actionable.",
    ]
    negative_keys = [
        str(row.get("candidate_key"))
        for row in negative_route_candidates
        if row.get("candidate_key") is not None
    ]
    prune_keys = [
        str(row.get("candidate_key"))
        for row in candidate_attribution
        if row.get("attribution_status") in {"PRUNE_CANDIDATE_ROUTE", "IDENTITY_ROUTE_DAMAGE"}
    ]
    combined = list(dict.fromkeys(negative_keys + prune_keys))
    if combined:
        actions.append(
            "Prune or quarantine the damaging candidate route(s) in the next research bundle: "
            + ", ".join(f"`{key}`" for key in combined[:8])
            + "."
        )
    if "shadow_only_rows_negative_ev" in attribution_reasons:
        actions.append(
            "Separate calibration from ranking: keep calibrated probabilities advisory until the shadow top bucket "
            "beats the raw top bucket on risk-adjusted payoff."
        )
    if "shadow_top_replaced_raw_top_bucket" in attribution_reasons:
        actions.append(
            "Add a ranking-preservation guardrail so calibration cannot promote weaker rows into the top signal bucket."
        )
    damaged_regimes = [
        f"{row.get('regime_field')}={row.get('regime_value')}"
        for row in regime_attribution
        if row.get("attribution_status") == "REGIME_DAMAGE"
    ]
    if damaged_regimes:
        actions.append(
            "Review regime-specific gating for the damaged pocket(s): "
            + ", ".join(f"`{item}`" for item in damaged_regimes[:6])
            + "."
        )
    actions.append("Rerun EV shadow evaluation and attribution after the next candidate bundle is generated.")
    return actions


def build_segmented_probability_ev_rejection_attribution_report(
    ev_shadow_report: dict[str, Any],
    routes: pd.DataFrame,
    *,
    ev_shadow_report_path: str | Path | None = None,
    ev_shadow_routes_path: str | Path | None = None,
    route_policy: str | None = None,
    return_field: str = DEFAULT_RETURN_FIELD,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    top_fraction: float | None = None,
    min_bucket_sample: int = 25,
    min_candidate_sample: int = 30,
    min_regime_sample: int = 10,
    max_spread_pct: float = 5.0,
) -> dict[str, Any]:
    """Build a research-only attribution report for EV shadow rejection."""
    raw_routes = routes.copy() if isinstance(routes, pd.DataFrame) else pd.DataFrame()
    analysis_policy = _select_route_policy(ev_shadow_report, raw_routes, route_policy=route_policy)
    policy_routes = _routes_for_policy(raw_routes, analysis_policy).reset_index(drop=True)
    selected_policy_result = _policy_row(ev_shadow_report, analysis_policy)
    fraction = float(top_fraction if top_fraction is not None else ev_shadow_report.get("top_fraction") or 0.25)
    ev_shadow_status = str(ev_shadow_report.get("ev_shadow_status") or "unknown")

    raw_top_mask = _top_probability_mask(policy_routes.get("raw_probability", pd.Series(dtype=float)), top_fraction=fraction)
    shadow_top_mask = _top_probability_mask(
        policy_routes.get("shadow_probability", pd.Series(dtype=float)),
        top_fraction=fraction,
    )
    shadow_bottom_mask = _top_probability_mask(
        -pd.to_numeric(policy_routes.get("shadow_probability", pd.Series(dtype=float)), errors="coerce"),
        top_fraction=fraction,
    )
    overlap_mask = raw_top_mask & shadow_top_mask
    raw_only_mask = raw_top_mask & ~shadow_top_mask
    shadow_only_mask = shadow_top_mask & ~raw_top_mask

    raw_top = policy_routes.loc[raw_top_mask].copy()
    shadow_top = policy_routes.loc[shadow_top_mask].copy()
    shadow_bottom = policy_routes.loc[shadow_bottom_mask].copy()
    raw_only_top = policy_routes.loc[raw_only_mask].copy()
    shadow_only_top = policy_routes.loc[shadow_only_mask].copy()
    overlap_top = policy_routes.loc[overlap_mask].copy()

    raw_top_summary = _summary_for_frame(raw_top, return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_top_summary = _summary_for_frame(shadow_top, return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_bottom_summary = _summary_for_frame(
        shadow_bottom,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    raw_only_summary = _summary_for_frame(raw_only_top, return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_only_summary = _summary_for_frame(
        shadow_only_top,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    overlap_summary = _summary_for_frame(overlap_top, return_field=return_field, max_spread_pct=max_spread_pct)

    risk_delta = _delta(
        shadow_top_summary.get("avg_risk_adjusted_return_bps"),
        raw_top_summary.get("avg_risk_adjusted_return_bps"),
    )
    hit_delta = _delta(shadow_top_summary.get("hit_rate"), raw_top_summary.get("hit_rate"))
    liquidity_delta = _delta(
        shadow_top_summary.get("avg_liquidity_adjusted_return_bps"),
        raw_top_summary.get("avg_liquidity_adjusted_return_bps"),
    )
    ranking_spread = _delta(
        shadow_top_summary.get("avg_risk_adjusted_return_bps"),
        shadow_bottom_summary.get("avg_risk_adjusted_return_bps"),
    )
    raw_top_count = int(raw_top_summary.get("sample_count") or 0)
    shadow_top_count = int(shadow_top_summary.get("sample_count") or 0)
    overlap_count = int(overlap_mask.sum())
    overlap_rate = float(overlap_count) / float(max(shadow_top_count, 1))
    negative_route_candidates = _negative_candidate_routes_from_report(ev_shadow_report, analysis_policy)
    negative_route_keys = {
        str(row.get("candidate_key"))
        for row in negative_route_candidates
        if row.get("candidate_key") is not None
    }

    candidate_attribution = _candidate_attribution(
        shadow_top,
        attribution_basis="shadow_top_bucket",
        min_candidate_sample=min_candidate_sample,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
        negative_route_keys=negative_route_keys,
    )
    shadow_only_candidate_attribution = _candidate_attribution(
        shadow_only_top,
        attribution_basis="shadow_only_top_bucket",
        min_candidate_sample=min_candidate_sample,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
        negative_route_keys=negative_route_keys,
    )
    regime_attribution = _regime_attribution(
        shadow_top,
        attribution_basis="shadow_top_bucket",
        regime_fields=regime_fields,
        min_regime_sample=min_regime_sample,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    policy_comparison = _policy_comparison(ev_shadow_report)
    best_policy = _best_policy_by_score(policy_comparison)

    attribution_reasons: list[str] = []
    if ev_shadow_status == EV_SHADOW_REJECTED:
        attribution_reasons.append("ev_shadow_status_rejected")
    if raw_top_count < int(min_bucket_sample) or shadow_top_count < int(min_bucket_sample):
        attribution_reasons.append("top_bucket_sample_guardrail_failed")
    if risk_delta is not None and risk_delta < 0:
        attribution_reasons.append("shadow_top_worse_than_raw_top")
    if hit_delta is not None and hit_delta < 0:
        attribution_reasons.append("shadow_top_hit_rate_worse_than_raw_top")
    if liquidity_delta is not None and liquidity_delta < 0:
        attribution_reasons.append("shadow_top_liquidity_adjusted_payoff_worse_than_raw_top")
    if ranking_spread is not None and ranking_spread <= 0:
        attribution_reasons.append("shadow_ranking_inverted_vs_bottom_bucket")
    shadow_only_risk = _safe_float(shadow_only_summary.get("avg_risk_adjusted_return_bps"))
    if shadow_only_risk is not None and shadow_only_risk < 0:
        attribution_reasons.append("shadow_only_rows_negative_ev")
    raw_only_risk = _safe_float(raw_only_summary.get("avg_risk_adjusted_return_bps"))
    if raw_only_risk is not None and shadow_only_risk is not None and shadow_only_risk < raw_only_risk:
        attribution_reasons.append("shadow_selection_displaced_better_raw_top_rows")
    if overlap_rate < 0.50 and shadow_top_count >= int(min_bucket_sample):
        attribution_reasons.append("shadow_top_replaced_raw_top_bucket")
    if negative_route_candidates:
        attribution_reasons.append("candidate_routes_negative_overall")
    if any(row.get("attribution_status") == "REGIME_DAMAGE" for row in regime_attribution):
        attribution_reasons.append("regime_damage_detected")
    attribution_reasons = list(dict.fromkeys(attribution_reasons))

    insufficient = "top_bucket_sample_guardrail_failed" in attribution_reasons or policy_routes.empty
    actionable_reasons = [
        reason
        for reason in attribution_reasons
        if reason
        not in {
            "ev_shadow_status_rejected",
            "top_bucket_sample_guardrail_failed",
        }
    ]
    if insufficient:
        attribution_status = EV_REJECTION_ATTRIBUTION_INSUFFICIENT_DATA
    elif ev_shadow_status != EV_SHADOW_REJECTED and not actionable_reasons:
        attribution_status = EV_REJECTION_ATTRIBUTION_NOT_REJECTED
    elif actionable_reasons:
        attribution_status = EV_REJECTION_ATTRIBUTION_ACTIONABLE
    else:
        attribution_status = EV_REJECTION_ATTRIBUTION_WATCH

    routing_diagnostics = {
        "selected_policy_result": selected_policy_result,
        "best_policy_by_score": best_policy,
        "selected_policy_negative_candidate_route_count": int(len(negative_route_candidates)),
        "shadow_top_prune_candidate_count": int(
            sum(
                1
                for row in candidate_attribution
                if row.get("attribution_status") in {"PRUNE_CANDIDATE_ROUTE", "IDENTITY_ROUTE_DAMAGE"}
            )
        ),
        "shadow_only_candidate_count": int(len(shadow_only_candidate_attribution)),
        "shadow_top_raw_top_overlap_count": overlap_count,
        "shadow_top_raw_top_overlap_rate": _round_or_none(overlap_rate, 6),
        "likely_failure_mode": "NO_REJECTION_ATTRIBUTED",
    }
    if attribution_status == EV_REJECTION_ATTRIBUTION_INSUFFICIENT_DATA:
        routing_diagnostics["likely_failure_mode"] = "INSUFFICIENT_TOP_BUCKET_EVIDENCE"
    elif negative_route_candidates and shadow_only_risk is not None and shadow_only_risk < 0:
        routing_diagnostics["likely_failure_mode"] = "DAMAGING_CANDIDATE_ROUTE_PROMOTED_WEAK_ROWS"
    elif risk_delta is not None and risk_delta < 0 and "shadow_selection_displaced_better_raw_top_rows" in attribution_reasons:
        routing_diagnostics["likely_failure_mode"] = "CALIBRATION_DISPLACED_HIGH_EV_RAW_TOP_ROWS"
    elif risk_delta is not None and risk_delta < 0 and overlap_rate < 0.50:
        routing_diagnostics["likely_failure_mode"] = "CALIBRATED_RANKING_REORDERED_TOP_BUCKET_UNPROFITABLY"
    elif risk_delta is not None and risk_delta < 0:
        routing_diagnostics["likely_failure_mode"] = "CALIBRATED_TOP_BUCKET_UNDERPERFORMED_RAW_TOP_BUCKET"
    elif negative_route_candidates:
        routing_diagnostics["likely_failure_mode"] = "NEGATIVE_CANDIDATE_ROUTES_PRESENT"

    rejection_summary = {
        "raw_top_sample_count": raw_top_count,
        "shadow_top_sample_count": shadow_top_count,
        "shadow_bottom_sample_count": int(shadow_bottom_summary.get("sample_count") or 0),
        "overlap_sample_count": overlap_count,
        "shadow_top_raw_top_overlap_rate": _round_or_none(overlap_rate, 6),
        "top_bucket_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "top_bucket_hit_rate_delta": _round_or_none(hit_delta, 6),
        "top_bucket_liquidity_adjusted_return_delta_bps": _round_or_none(liquidity_delta, 6),
        "shadow_top_vs_bottom_risk_adjusted_return_spread_bps": _round_or_none(ranking_spread, 6),
        "raw_top": raw_top_summary,
        "shadow_top": shadow_top_summary,
        "shadow_bottom": shadow_bottom_summary,
        "raw_only_top": raw_only_summary,
        "shadow_only_top": shadow_only_summary,
        "overlap_top": overlap_summary,
    }

    report = {
        "report_type": "segmented_probability_ev_rejection_attribution",
        "generated_at": _utc_now(),
        "ev_shadow_report_path": str(ev_shadow_report_path) if ev_shadow_report_path is not None else None,
        "ev_shadow_routes_path": str(ev_shadow_routes_path) if ev_shadow_routes_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "ev_shadow_status": ev_shadow_status,
        "attribution_status": attribution_status,
        "attribution_reasons": attribution_reasons,
        "analysis_policy": analysis_policy,
        "top_fraction": fraction,
        "return_field": return_field,
        "route_decision_count": int(len(raw_routes)),
        "analysis_route_decision_count": int(len(policy_routes)),
        "min_bucket_sample": int(min_bucket_sample),
        "min_candidate_sample": int(min_candidate_sample),
        "min_regime_sample": int(min_regime_sample),
        "rejection_summary": rejection_summary,
        "candidate_attribution": candidate_attribution,
        "shadow_only_candidate_attribution": shadow_only_candidate_attribution,
        "negative_route_candidates": negative_route_candidates,
        "regime_attribution": regime_attribution,
        "policy_comparison": policy_comparison,
        "routing_diagnostics": routing_diagnostics,
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(
        status=attribution_status,
        attribution_reasons=attribution_reasons,
        negative_route_candidates=negative_route_candidates,
        candidate_attribution=candidate_attribution,
        regime_attribution=regime_attribution,
    )
    return _sanitize_value(report)


def render_segmented_probability_ev_rejection_attribution_markdown(report: dict[str, Any]) -> str:
    """Render EV rejection attribution as Markdown."""
    summary = report.get("rejection_summary", {}) or {}
    diagnostics = report.get("routing_diagnostics", {}) or {}
    lines = [
        "# Segmented Probability EV Rejection Attribution",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- EV shadow report: {report.get('ev_shadow_report_path') or 'inline'}",
        f"- EV shadow routes: {report.get('ev_shadow_routes_path') or 'inline'}",
        f"- EV shadow status: `{report.get('ev_shadow_status')}`",
        f"- Attribution status: `{report.get('attribution_status')}`",
        f"- Analysis policy: `{report.get('analysis_policy')}`",
        f"- Likely failure mode: `{diagnostics.get('likely_failure_mode')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Top Bucket Comparison",
        "",
        "| Bucket | N | Hit Rate | Avg Risk-Adj Return | Avg Liquidity-Adj Return | Avg Spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Raw top", "raw_top"),
        ("Shadow top", "shadow_top"),
        ("Shadow bottom", "shadow_bottom"),
        ("Raw-only top", "raw_only_top"),
        ("Shadow-only top", "shadow_only_top"),
        ("Overlap", "overlap_top"),
    ]:
        bucket = summary.get(key, {}) or {}
        lines.append(
            f"| {label} | {bucket.get('sample_count')} | {bucket.get('hit_rate')} | "
            f"{bucket.get('avg_risk_adjusted_return_bps')} | "
            f"{bucket.get('avg_liquidity_adjusted_return_bps')} | {bucket.get('avg_spread_pct')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Attribution",
            "",
            "| Candidate | Status | N | Share | Avg Risk-Adj Return | Hit Rate | Calibrator |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in (report.get("candidate_attribution", []) or [])[:20]:
        lines.append(
            f"| `{row.get('candidate_key')}` | `{row.get('attribution_status')}` | "
            f"{row.get('sample_count')} | {row.get('share_of_bucket')} | "
            f"{row.get('avg_risk_adjusted_return_bps')} | {row.get('hit_rate')} | "
            f"`{row.get('assigned_calibrator')}` |"
        )
    lines.extend(
        [
            "",
            "## Negative Overall Routes",
            "",
            "| Candidate | N | Avg Risk-Adj Return | Hit Rate | Segment |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in (report.get("negative_route_candidates", []) or [])[:20]:
        segment = f"{row.get('assigned_segment_field')}={row.get('assigned_segment_value')}"
        lines.append(
            f"| `{row.get('candidate_key')}` | {row.get('sample_count')} | "
            f"{row.get('avg_risk_adjusted_return_bps')} | {row.get('hit_rate')} | `{segment}` |"
        )
    lines.extend(
        [
            "",
            "## Damaged Regimes",
            "",
            "| Regime | Status | N | Share | Avg Risk-Adj Return | Hit Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (report.get("regime_attribution", []) or [])[:20]:
        if row.get("attribution_status") != "REGIME_DAMAGE":
            continue
        regime = f"{row.get('regime_field')}={row.get('regime_value')}"
        lines.append(
            f"| `{regime}` | `{row.get('attribution_status')}` | {row.get('sample_count')} | "
            f"{row.get('share_of_bucket')} | {row.get('avg_risk_adjusted_return_bps')} | "
            f"{row.get('hit_rate')} |"
        )
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("*Research-only artifact. Runtime config, parameter packs, data sources, and execution behavior are unchanged.*")
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "candidates_csv_path": output / f"{stem}_candidates.csv",
        "shadow_only_candidates_csv_path": output / f"{stem}_shadow_only_candidates.csv",
        "regimes_csv_path": output / f"{stem}_regimes.csv",
        "policies_csv_path": output / f"{stem}_policies.csv",
        "latest_json_path": output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_MARKDOWN_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_CANDIDATES_FILENAME,
        "latest_shadow_only_candidates_csv_path": (
            output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_SHADOW_ONLY_CANDIDATES_FILENAME
        ),
        "latest_regimes_csv_path": output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_REGIMES_FILENAME,
        "latest_policies_csv_path": output / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_POLICIES_FILENAME,
    }


def write_segmented_probability_ev_rejection_attribution_report(
    ev_shadow_report: dict[str, Any],
    routes: pd.DataFrame,
    *,
    ev_shadow_report_path: str | Path | None = None,
    ev_shadow_routes_path: str | Path | None = None,
    route_policy: str | None = None,
    return_field: str = DEFAULT_RETURN_FIELD,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    top_fraction: float | None = None,
    min_bucket_sample: int = 25,
    min_candidate_sample: int = 30,
    min_regime_sample: int = 10,
    max_spread_pct: float = 5.0,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write EV rejection attribution artifacts."""
    report = build_segmented_probability_ev_rejection_attribution_report(
        ev_shadow_report,
        routes,
        ev_shadow_report_path=ev_shadow_report_path,
        ev_shadow_routes_path=ev_shadow_routes_path,
        route_policy=route_policy,
        return_field=return_field,
        regime_fields=regime_fields,
        top_fraction=top_fraction,
        min_bucket_sample=min_bucket_sample,
        min_candidate_sample=min_candidate_sample,
        min_regime_sample=min_regime_sample,
        max_spread_pct=max_spread_pct,
    )
    assert_artifact_schema(report, "segmented_probability_ev_rejection_attribution")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_ev_rejection_attribution"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_ev_rejection_attribution_markdown(report)
    candidates = pd.DataFrame(report.get("candidate_attribution", []) or [])
    shadow_only_candidates = pd.DataFrame(report.get("shadow_only_candidate_attribution", []) or [])
    regimes = pd.DataFrame(report.get("regime_attribution", []) or [])
    policies = pd.DataFrame(report.get("policy_comparison", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    _atomic_write_csv(shadow_only_candidates, paths["shadow_only_candidates_csv_path"])
    _atomic_write_csv(regimes, paths["regimes_csv_path"])
    _atomic_write_csv(policies, paths["policies_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
        _atomic_write_csv(shadow_only_candidates, paths["latest_shadow_only_candidates_csv_path"])
        _atomic_write_csv(regimes, paths["latest_regimes_csv_path"])
        _atomic_write_csv(policies, paths["latest_policies_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_ev_rejection_attribution_report_from_paths(
    *,
    ev_shadow_report_path: str | Path | None = None,
    ev_shadow_routes_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load EV shadow artifacts and write rejection-attribution evidence."""
    report_path = Path(ev_shadow_report_path) if ev_shadow_report_path is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH
    routes_path = Path(ev_shadow_routes_path) if ev_shadow_routes_path is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH
    ev_shadow_report = _read_json(report_path)
    routes = _load_routes(routes_path)
    return write_segmented_probability_ev_rejection_attribution_report(
        ev_shadow_report,
        routes,
        ev_shadow_report_path=report_path,
        ev_shadow_routes_path=routes_path,
        **kwargs,
    )
