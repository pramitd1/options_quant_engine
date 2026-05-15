"""EV/risk-adjusted shadow evaluation for segmented probability candidates.

This module evaluates whether shadow-calibrated routing policies improve the
ranking of economically useful signals. It is research-only: it reads signal
evaluation data and candidate bundles, writes advisory artifacts, and never
changes runtime configuration, parameter packs, data sources, or execution
behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import (
    _clean_probability_and_label_frame,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_ROUTING_POLICIES,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    _evaluation_frame,
    _load_candidate_bundle,
    _route_decisions,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    DEFAULT_REGIME_FIELDS,
    DEFAULT_RETURN_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_ev_shadow_evaluation"
)

SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME = "latest_segmented_probability_ev_shadow_evaluation.json"
SEGMENTED_PROBABILITY_EV_SHADOW_MARKDOWN_FILENAME = "latest_segmented_probability_ev_shadow_evaluation.md"
SEGMENTED_PROBABILITY_EV_SHADOW_POLICIES_FILENAME = "latest_segmented_probability_ev_shadow_policies.csv"
SEGMENTED_PROBABILITY_EV_SHADOW_CANDIDATES_FILENAME = "latest_segmented_probability_ev_shadow_candidates.csv"
SEGMENTED_PROBABILITY_EV_SHADOW_REGIMES_FILENAME = "latest_segmented_probability_ev_shadow_regimes.csv"
SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME = "latest_segmented_probability_ev_shadow_routes.csv"

EV_SHADOW_PASS = "EV_SHADOW_EVALUATION_PASS"
EV_SHADOW_WATCH = "EV_SHADOW_EVALUATION_WATCH"
EV_SHADOW_REJECTED = "EV_SHADOW_EVALUATION_REJECTED"
EV_SHADOW_NEEDS_MORE_DATA = "EV_SHADOW_EVALUATION_NEEDS_MORE_DATA"
EV_SHADOW_NO_CANDIDATE_ROUTES = "EV_SHADOW_NO_CANDIDATE_ROUTES"

DEFAULT_PAYOFF_COLUMNS = (
    DEFAULT_RETURN_FIELD,
    "primary_outcome_return_bps",
    "mfe_60m_bps",
    "mae_60m_bps",
    "selected_option_ba_spread_pct",
    "selected_option_volume",
    "selected_option_open_interest",
    "selected_option_iv",
    "option_chain_is_valid",
    "option_chain_is_stale",
    "option_chain_validation_status",
    "market_data_provenance_status",
    "data_quality_score",
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _truthy_rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    tokens = values.dropna()
    if tokens.empty:
        return None
    if pd.api.types.is_bool_dtype(tokens):
        return float(tokens.astype(bool).mean())
    text = tokens.astype(str).str.strip().str.lower()
    truthy = text.isin({"true", "t", "yes", "y", "1", "1.0"})
    falsy = text.isin({"false", "f", "no", "n", "0", "0.0"})
    known = truthy | falsy
    if not bool(known.any()):
        numeric = pd.to_numeric(tokens, errors="coerce").dropna()
        if numeric.empty:
            return None
        return float((numeric != 0).mean())
    return float(truthy.loc[known].mean())


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index)), errors="coerce")


def _top_probability_mask(values: pd.Series, *, top_fraction: float) -> pd.Series:
    probabilities = pd.to_numeric(values, errors="coerce")
    mask = pd.Series(False, index=probabilities.index, dtype=bool)
    valid = probabilities.dropna()
    if valid.empty:
        return mask
    fraction = min(max(float(top_fraction), 0.01), 1.0)
    top_count = max(int(math.ceil(len(valid) * fraction)), 1)
    top_index = probabilities.loc[valid.index].rank(method="first", ascending=False).nsmallest(top_count).index
    mask.loc[top_index] = True
    return mask


def _enrich_route_decisions(
    evaluation: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    return_field: str,
    payoff_columns: tuple[str, ...],
    regime_fields: tuple[str, ...],
    downside_penalty_weight: float,
    spread_penalty_per_pct: float,
) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    evaluation_reset = evaluation.reset_index(drop=True)
    keep_columns = []
    for column in tuple(payoff_columns) + tuple(regime_fields):
        if column in evaluation_reset.columns and column not in keep_columns:
            keep_columns.append(column)
    extra = evaluation_reset.loc[:, keep_columns].copy() if keep_columns else pd.DataFrame(index=evaluation_reset.index)
    extra["_row_index"] = extra.index.astype(int)
    enriched = decisions.merge(extra, left_on="row_index", right_on="_row_index", how="left").drop(columns=["_row_index"])
    if return_field not in enriched.columns and "primary_outcome_return_bps" in enriched.columns:
        enriched[return_field] = enriched["primary_outcome_return_bps"]

    returns = _numeric(enriched, return_field)
    mae = _numeric(enriched, "mae_60m_bps").abs().fillna(0.0)
    spread = _numeric(enriched, "selected_option_ba_spread_pct").clip(lower=0.0)
    spread_penalty = spread.fillna(0.0) * float(spread_penalty_per_pct)
    enriched["_return_bps"] = returns
    enriched["_risk_adjusted_return_bps"] = returns - (mae * float(downside_penalty_weight))
    enriched["_liquidity_adjusted_return_bps"] = enriched["_risk_adjusted_return_bps"] - spread_penalty
    return enriched


def _summary_for_mask(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    return_field: str,
    max_spread_pct: float,
) -> dict[str, Any]:
    subset = frame.loc[mask.fillna(False)].copy()
    count = int(len(subset))
    if count <= 0:
        return {
            "sample_count": 0,
            "hit_rate": None,
            "avg_return_bps": None,
            "avg_risk_adjusted_return_bps": None,
            "avg_liquidity_adjusted_return_bps": None,
        }
    labels = pd.to_numeric(subset.get("label", pd.Series(index=subset.index)), errors="coerce")
    returns = _numeric(subset, "_return_bps")
    risk_adjusted = _numeric(subset, "_risk_adjusted_return_bps")
    liquidity_adjusted = _numeric(subset, "_liquidity_adjusted_return_bps")
    mfe = _numeric(subset, "mfe_60m_bps")
    mae = _numeric(subset, "mae_60m_bps").abs()
    spread = _numeric(subset, "selected_option_ba_spread_pct")
    volume = _numeric(subset, "selected_option_volume")
    open_interest = _numeric(subset, "selected_option_open_interest")
    positive_returns = returns.loc[returns > 0]
    negative_returns = returns.loc[returns < 0]
    win_loss_ratio = None
    if not positive_returns.empty and not negative_returns.empty:
        win_loss_ratio = float(positive_returns.mean()) / max(abs(float(negative_returns.mean())), 1e-9)
    high_spread_rate = None
    if spread.notna().any():
        high_spread_rate = float((spread > float(max_spread_pct)).mean())
    return {
        "sample_count": count,
        "hit_rate": _round_or_none(labels.mean(), 6),
        "avg_return_bps": _round_or_none(returns.mean(), 6),
        "median_return_bps": _round_or_none(returns.median(), 6),
        "avg_risk_adjusted_return_bps": _round_or_none(risk_adjusted.mean(), 6),
        "avg_liquidity_adjusted_return_bps": _round_or_none(liquidity_adjusted.mean(), 6),
        "total_return_bps": _round_or_none(returns.sum(), 6),
        "avg_mfe_60m_bps": _round_or_none(mfe.mean(), 6),
        "avg_abs_mae_60m_bps": _round_or_none(mae.mean(), 6),
        "p95_abs_mae_60m_bps": _round_or_none(mae.quantile(0.95), 6),
        "win_loss_return_ratio": _round_or_none(win_loss_ratio, 6),
        "avg_spread_pct": _round_or_none(spread.mean(), 6),
        "high_spread_rate": _round_or_none(high_spread_rate, 6),
        "avg_selected_option_volume": _round_or_none(volume.mean(), 6),
        "avg_selected_option_open_interest": _round_or_none(open_interest.mean(), 6),
        "option_chain_stale_rate": _round_or_none(
            _truthy_rate(subset.get("option_chain_is_stale", pd.Series(dtype=object))),
            6,
        ),
        "option_chain_valid_rate": _round_or_none(
            _truthy_rate(subset.get("option_chain_is_valid", pd.Series(dtype=object))),
            6,
        ),
        "return_field": return_field,
    }


def _delta(value: Any, baseline: Any) -> float | None:
    left = _safe_float(value)
    right = _safe_float(baseline)
    if left is None or right is None:
        return None
    return left - right


def _liquidity_status(summary: dict[str, Any], *, max_high_spread_rate: float) -> str:
    high_spread_rate = _safe_float(summary.get("high_spread_rate"))
    stale_rate = _safe_float(summary.get("option_chain_stale_rate"))
    valid_rate = _safe_float(summary.get("option_chain_valid_rate"))
    if high_spread_rate is not None and high_spread_rate > float(max_high_spread_rate):
        return "LIQUIDITY_WATCH"
    if stale_rate is not None and stale_rate > 0.0:
        return "STALE_CHAIN_WATCH"
    if valid_rate is not None and valid_rate < 0.95:
        return "CHAIN_VALIDITY_WATCH"
    return "OK"


def _policy_status(
    *,
    sample_count: int,
    assigned_count: int,
    raw_top: dict[str, Any],
    shadow_top: dict[str, Any],
    min_ev_sample: int,
    min_top_sample: int,
    min_risk_adjusted_improvement_bps: float,
    max_hit_rate_regression: float,
    max_liquidity_watch_rate: float,
) -> tuple[str, str]:
    if sample_count < int(min_ev_sample) or int(shadow_top.get("sample_count") or 0) < int(min_top_sample):
        return EV_SHADOW_NEEDS_MORE_DATA, "sample_size_guardrail_failed"
    if assigned_count <= 0:
        return EV_SHADOW_NO_CANDIDATE_ROUTES, "no_candidate_matched_evaluation_rows"
    risk_delta = _delta(
        shadow_top.get("avg_risk_adjusted_return_bps"),
        raw_top.get("avg_risk_adjusted_return_bps"),
    )
    hit_delta = _delta(shadow_top.get("hit_rate"), raw_top.get("hit_rate"))
    liquidity_watch = _safe_float(shadow_top.get("high_spread_rate"))
    if risk_delta is not None and risk_delta < -float(min_risk_adjusted_improvement_bps):
        return EV_SHADOW_REJECTED, "shadow_top_bucket_worsened_risk_adjusted_return"
    if hit_delta is not None and hit_delta < -float(max_hit_rate_regression):
        return EV_SHADOW_REJECTED, "shadow_top_bucket_worsened_hit_rate"
    if liquidity_watch is not None and liquidity_watch > float(max_liquidity_watch_rate):
        return EV_SHADOW_WATCH, "shadow_top_bucket_liquidity_watch"
    if risk_delta is not None and risk_delta >= float(min_risk_adjusted_improvement_bps):
        return EV_SHADOW_PASS, "shadow_top_bucket_improved_risk_adjusted_return"
    return EV_SHADOW_WATCH, "shadow_top_bucket_did_not_clear_ev_improvement_guardrail"


def _policy_result(
    decisions: pd.DataFrame,
    *,
    route_policy: str,
    return_field: str,
    top_fraction: float,
    min_ev_sample: int,
    min_top_sample: int,
    min_risk_adjusted_improvement_bps: float,
    max_hit_rate_regression: float,
    max_spread_pct: float,
    max_liquidity_watch_rate: float,
) -> dict[str, Any]:
    raw_top_mask = _top_probability_mask(decisions.get("raw_probability", pd.Series(dtype=float)), top_fraction=top_fraction)
    shadow_top_mask = _top_probability_mask(
        decisions.get("shadow_probability", pd.Series(dtype=float)),
        top_fraction=top_fraction,
    )
    shadow_bottom_mask = _top_probability_mask(
        -pd.to_numeric(decisions.get("shadow_probability", pd.Series(dtype=float)), errors="coerce"),
        top_fraction=top_fraction,
    )
    all_summary = _summary_for_mask(
        decisions,
        pd.Series(True, index=decisions.index),
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    raw_top = _summary_for_mask(decisions, raw_top_mask, return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_top = _summary_for_mask(decisions, shadow_top_mask, return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_bottom = _summary_for_mask(
        decisions,
        shadow_bottom_mask,
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    assigned_count = (
        int((decisions.get("assigned_candidate_key") != "unrouted_identity").sum())
        if not decisions.empty
        else 0
    )
    status, reason = _policy_status(
        sample_count=int(all_summary.get("sample_count") or 0),
        assigned_count=assigned_count,
        raw_top=raw_top,
        shadow_top=shadow_top,
        min_ev_sample=min_ev_sample,
        min_top_sample=min_top_sample,
        min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=max_hit_rate_regression,
        max_liquidity_watch_rate=max_liquidity_watch_rate,
    )
    risk_delta = _delta(
        shadow_top.get("avg_risk_adjusted_return_bps"),
        raw_top.get("avg_risk_adjusted_return_bps"),
    )
    liquidity_delta = _delta(
        shadow_top.get("avg_liquidity_adjusted_return_bps"),
        raw_top.get("avg_liquidity_adjusted_return_bps"),
    )
    hit_delta = _delta(shadow_top.get("hit_rate"), raw_top.get("hit_rate"))
    ranking_spread = _delta(
        shadow_top.get("avg_risk_adjusted_return_bps"),
        shadow_bottom.get("avg_risk_adjusted_return_bps"),
    )
    policy_score = None
    if risk_delta is not None and hit_delta is not None:
        policy_score = float(risk_delta) + (50.0 * float(hit_delta))
    return {
        "route_policy": route_policy,
        "ev_shadow_status": status,
        "status_reason": reason,
        "sample_count": int(all_summary.get("sample_count") or 0),
        "assigned_candidate_count": assigned_count,
        "top_fraction": float(top_fraction),
        "raw_top": raw_top,
        "shadow_top": shadow_top,
        "shadow_bottom": shadow_bottom,
        "shadow_vs_raw_top_hit_rate_delta": _round_or_none(hit_delta, 6),
        "shadow_vs_raw_top_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "shadow_vs_raw_top_liquidity_adjusted_return_delta_bps": _round_or_none(liquidity_delta, 6),
        "shadow_top_vs_bottom_risk_adjusted_return_spread_bps": _round_or_none(ranking_spread, 6),
        "liquidity_status": _liquidity_status(shadow_top, max_high_spread_rate=max_liquidity_watch_rate),
        "policy_score": _round_or_none(policy_score, 6),
        "all_rows": all_summary,
    }


def _candidate_route_results(
    decisions: pd.DataFrame,
    *,
    route_policy: str,
    return_field: str,
    min_candidate_sample: int,
    max_spread_pct: float,
) -> list[dict[str, Any]]:
    if decisions.empty or "assigned_candidate_key" not in decisions.columns:
        return []
    routed = decisions.loc[decisions["assigned_candidate_key"] != "unrouted_identity"].copy()
    if routed.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in routed.groupby("assigned_candidate_key", dropna=False):
        summary = _summary_for_mask(
            group,
            pd.Series(True, index=group.index),
            return_field=return_field,
            max_spread_pct=max_spread_pct,
        )
        sample_count = int(summary.get("sample_count") or 0)
        risk_return = _safe_float(summary.get("avg_risk_adjusted_return_bps"))
        if sample_count < int(min_candidate_sample):
            status = "INSUFFICIENT_EV_EVIDENCE"
        elif risk_return is not None and risk_return > 0:
            status = "EV_ROUTE_HELPFUL"
        elif risk_return is not None and risk_return < 0:
            status = "EV_ROUTE_NEGATIVE"
        else:
            status = "EV_ROUTE_WATCH"
        first = group.iloc[0]
        row = {
            "route_policy": route_policy,
            "candidate_key": str(key),
            "ev_route_status": status,
            "assigned_candidate_type": first.get("assigned_candidate_type"),
            "assigned_segment_field": first.get("assigned_segment_field"),
            "assigned_segment_value": first.get("assigned_segment_value"),
            "assigned_calibrator": first.get("assigned_calibrator"),
        }
        row.update(summary)
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            item.get("ev_route_status") != "EV_ROUTE_HELPFUL",
            -(float(item.get("avg_risk_adjusted_return_bps") or -1e9)),
        ),
    )


def _regime_payoff_results(
    decisions: pd.DataFrame,
    *,
    route_policy: str,
    regime_fields: tuple[str, ...],
    return_field: str,
    top_fraction: float,
    min_regime_sample: int,
    max_spread_pct: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return rows
    top_mask = _top_probability_mask(
        decisions.get("shadow_probability", pd.Series(dtype=float)),
        top_fraction=top_fraction,
    )
    top = decisions.loc[top_mask].copy()
    for regime_field in regime_fields:
        if regime_field not in top.columns:
            continue
        for regime_value, group in top.groupby(regime_field, dropna=False):
            if len(group) < int(min_regime_sample):
                continue
            summary = _summary_for_mask(
                group,
                pd.Series(True, index=group.index),
                return_field=return_field,
                max_spread_pct=max_spread_pct,
            )
            row = {
                "route_policy": route_policy,
                "regime_field": regime_field,
                "regime_value": str(regime_value),
            }
            row.update(summary)
            rows.append(row)
    return sorted(rows, key=lambda item: -(float(item.get("avg_risk_adjusted_return_bps") or -1e9)))


def _best_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not policy_rows:
        return None
    severity = {
        EV_SHADOW_PASS: 0,
        EV_SHADOW_WATCH: 1,
        EV_SHADOW_REJECTED: 2,
        EV_SHADOW_NO_CANDIDATE_ROUTES: 3,
        EV_SHADOW_NEEDS_MORE_DATA: 4,
    }
    return sorted(
        policy_rows,
        key=lambda row: (
            severity.get(str(row.get("ev_shadow_status")), 9),
            -(float(row.get("policy_score") or -1e9)),
            -(float(row.get("shadow_top_vs_bottom_risk_adjusted_return_spread_bps") or -1e9)),
        ),
    )[0]


def _overall_status(policy_rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("ev_shadow_status")) for row in policy_rows}
    if not statuses:
        return EV_SHADOW_NO_CANDIDATE_ROUTES
    if EV_SHADOW_PASS in statuses:
        return EV_SHADOW_PASS
    if EV_SHADOW_WATCH in statuses:
        return EV_SHADOW_WATCH
    if EV_SHADOW_NEEDS_MORE_DATA in statuses and len(statuses) == 1:
        return EV_SHADOW_NEEDS_MORE_DATA
    if EV_SHADOW_NO_CANDIDATE_ROUTES in statuses and len(statuses) == 1:
        return EV_SHADOW_NO_CANDIDATE_ROUTES
    return EV_SHADOW_REJECTED


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("ev_shadow_status")
    best = report.get("selection_summary", {}).get("recommended_routing_policy")
    if status == EV_SHADOW_PASS:
        return [
            f"Keep `{best}` in shadow review as the leading EV/risk routing policy.",
            "Compare this payoff evidence with Brier/ECE readiness before any manual calibration review.",
            "Do not change runtime probabilities, parameter packs, data sources, or execution behavior from this report.",
        ]
    if status == EV_SHADOW_NEEDS_MORE_DATA:
        return [
            "Collect more quality-approved labels before trusting EV/risk route rankings.",
            "Keep the candidate bundle in research-only shadow evaluation.",
        ]
    if status == EV_SHADOW_WATCH:
        return [
            "Keep the candidate bundle in watch mode until EV/risk improvement is clearer.",
            "Review top-bucket liquidity, spread, and regime rows before preferring a route policy.",
        ]
    return [
        "Do not advance the candidate bundle on payoff evidence; shadow ranking did not improve EV/risk results.",
        "Return to calibration/ranking diagnostics before another shadow cycle.",
    ]


def build_segmented_probability_ev_shadow_evaluation_report(
    frame: pd.DataFrame,
    *,
    candidate_bundle: dict[str, Any] | None = None,
    candidate_bundle_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    return_field: str = DEFAULT_RETURN_FIELD,
    train_fraction: float = 0.70,
    validation_mode: str = "auto",
    routing_policies: tuple[str, ...] = DEFAULT_ROUTING_POLICIES,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    top_fraction: float = 0.25,
    min_ev_sample: int = 100,
    min_top_sample: int = 25,
    min_candidate_sample: int = 30,
    min_regime_sample: int = 10,
    min_risk_adjusted_improvement_bps: float = 2.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    max_liquidity_watch_rate: float = 0.35,
) -> dict[str, Any]:
    """Build EV/risk-adjusted shadow evidence for candidate routing policies."""
    bundle = candidate_bundle if isinstance(candidate_bundle, dict) else _load_candidate_bundle(
        candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    )
    candidates = [candidate for candidate in bundle.get("candidates", []) or [] if isinstance(candidate, dict)]
    raw = frame if frame is not None else pd.DataFrame()
    working = _clean_probability_and_label_frame(
        raw,
        probability_field=probability_field,
        label_field=label_field,
    )
    evaluation, window = _evaluation_frame(
        working,
        bundle=bundle,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        min_shadow_sample=min_ev_sample,
    )

    policy_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    route_frames: list[pd.DataFrame] = []
    payoff_columns = tuple(dict.fromkeys(DEFAULT_PAYOFF_COLUMNS + (return_field,)))
    for policy in routing_policies:
        decisions = _route_decisions(evaluation, candidates, routing_policy=str(policy))
        enriched = _enrich_route_decisions(
            evaluation,
            decisions,
            return_field=return_field,
            payoff_columns=payoff_columns,
            regime_fields=regime_fields,
            downside_penalty_weight=downside_penalty_weight,
            spread_penalty_per_pct=spread_penalty_per_pct,
        )
        route_frames.append(enriched)
        policy_rows.append(
            _policy_result(
                enriched,
                route_policy=str(policy),
                return_field=return_field,
                top_fraction=top_fraction,
                min_ev_sample=min_ev_sample,
                min_top_sample=min_top_sample,
                min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
                max_hit_rate_regression=max_hit_rate_regression,
                max_spread_pct=max_spread_pct,
                max_liquidity_watch_rate=max_liquidity_watch_rate,
            )
        )
        candidate_rows.extend(
            _candidate_route_results(
                enriched,
                route_policy=str(policy),
                return_field=return_field,
                min_candidate_sample=min_candidate_sample,
                max_spread_pct=max_spread_pct,
            )
        )
        regime_rows.extend(
            _regime_payoff_results(
                enriched,
                route_policy=str(policy),
                regime_fields=regime_fields,
                return_field=return_field,
                top_fraction=top_fraction,
                min_regime_sample=min_regime_sample,
                max_spread_pct=max_spread_pct,
            )
        )

    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    best = _best_policy(policy_rows)
    report = {
        "report_type": "segmented_probability_ev_shadow_evaluation",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "candidate_bundle_path": str(candidate_bundle_path) if candidate_bundle_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "row_count": int(len(raw)),
        "quality_labeled_row_count": int(len(working)),
        "probability_field": probability_field,
        "label_field": label_field,
        "return_field": return_field,
        "candidate_count": int(len(candidates)),
        "train_fraction": float(train_fraction),
        "validation_window": window,
        "top_fraction": float(top_fraction),
        "ev_shadow_status": _overall_status(policy_rows),
        "selection_summary": {
            "recommended_routing_policy": best.get("route_policy") if best else None,
            "recommended_policy_status": best.get("ev_shadow_status") if best else None,
            "recommended_policy_score": best.get("policy_score") if best else None,
            "recommended_policy_risk_adjusted_return_delta_bps": (
                best.get("shadow_vs_raw_top_risk_adjusted_return_delta_bps") if best else None
            ),
            "recommended_policy_hit_rate_delta": best.get("shadow_vs_raw_top_hit_rate_delta") if best else None,
            "evaluated_routing_policy_count": int(len(policy_rows)),
            "route_policy_status_counts": {
                status: sum(1 for row in policy_rows if row.get("ev_shadow_status") == status)
                for status in sorted({str(row.get("ev_shadow_status")) for row in policy_rows})
            },
        },
        "policy_results": policy_rows,
        "candidate_route_results": candidate_rows,
        "regime_payoff_results": regime_rows,
        "route_decision_count": int(len(routes)),
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    sanitized = _sanitize_value(report)
    sanitized["_route_decisions_frame"] = routes
    return sanitized


def render_segmented_probability_ev_shadow_evaluation_markdown(report: dict[str, Any]) -> str:
    """Render EV/risk shadow evidence as Markdown."""
    selection = report.get("selection_summary", {}) or {}
    window = report.get("validation_window", {}) or {}
    lines = [
        "# Segmented Probability EV Shadow Evaluation",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Candidate bundle: {report.get('candidate_bundle_path') or 'inline'}",
        f"- Validation mode: `{window.get('validation_mode_used')}`",
        f"- Strict forward rows: {window.get('strict_forward_row_count')}",
        f"- Holdout replay rows: {window.get('holdout_replay_row_count')}",
        f"- Status: `{report.get('ev_shadow_status')}`",
        f"- Recommended routing policy: `{selection.get('recommended_routing_policy')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Routing Policies",
        "",
        "| Policy | Status | Top Risk-Adj Delta | Top Hit Delta | Ranking Spread | Liquidity |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("policy_results", []) or []:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('ev_shadow_status')}` | "
            f"{row.get('shadow_vs_raw_top_risk_adjusted_return_delta_bps')} | "
            f"{row.get('shadow_vs_raw_top_hit_rate_delta')} | "
            f"{row.get('shadow_top_vs_bottom_risk_adjusted_return_spread_bps')} | "
            f"`{row.get('liquidity_status')}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate Routes",
            "",
            "| Policy | Candidate | Status | N | Avg Risk-Adj Return | Hit Rate | Avg Spread |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (report.get("candidate_route_results", []) or [])[:20]:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('candidate_key')}` | "
            f"`{row.get('ev_route_status')}` | {row.get('sample_count')} | "
            f"{row.get('avg_risk_adjusted_return_bps')} | {row.get('hit_rate')} | "
            f"{row.get('avg_spread_pct')} |"
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
        "policies_csv_path": output / f"{stem}_policies.csv",
        "candidates_csv_path": output / f"{stem}_candidates.csv",
        "regimes_csv_path": output / f"{stem}_regimes.csv",
        "routes_csv_path": output / f"{stem}_routes.csv",
        "latest_json_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_MARKDOWN_FILENAME,
        "latest_policies_csv_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_POLICIES_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_CANDIDATES_FILENAME,
        "latest_regimes_csv_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_REGIMES_FILENAME,
        "latest_routes_csv_path": output / SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME,
    }


def write_segmented_probability_ev_shadow_evaluation_report(
    frame: pd.DataFrame,
    *,
    candidate_bundle: dict[str, Any] | None = None,
    candidate_bundle_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    return_field: str = DEFAULT_RETURN_FIELD,
    train_fraction: float = 0.70,
    validation_mode: str = "auto",
    routing_policies: tuple[str, ...] = DEFAULT_ROUTING_POLICIES,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    top_fraction: float = 0.25,
    min_ev_sample: int = 100,
    min_top_sample: int = 25,
    min_candidate_sample: int = 30,
    min_regime_sample: int = 10,
    min_risk_adjusted_improvement_bps: float = 2.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    max_liquidity_watch_rate: float = 0.35,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write EV/risk-adjusted shadow evaluation artifacts."""
    bundle_path = candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    report = build_segmented_probability_ev_shadow_evaluation_report(
        frame,
        candidate_bundle=candidate_bundle,
        candidate_bundle_path=bundle_path,
        dataset_path=dataset_path,
        probability_field=probability_field,
        label_field=label_field,
        return_field=return_field,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        routing_policies=routing_policies,
        regime_fields=regime_fields,
        top_fraction=top_fraction,
        min_ev_sample=min_ev_sample,
        min_top_sample=min_top_sample,
        min_candidate_sample=min_candidate_sample,
        min_regime_sample=min_regime_sample,
        min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=max_hit_rate_regression,
        downside_penalty_weight=downside_penalty_weight,
        spread_penalty_per_pct=spread_penalty_per_pct,
        max_spread_pct=max_spread_pct,
        max_liquidity_watch_rate=max_liquidity_watch_rate,
    )
    routes = report.pop("_route_decisions_frame", pd.DataFrame())
    assert_artifact_schema(report, "segmented_probability_ev_shadow_evaluation")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_ev_shadow_evaluation"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_ev_shadow_evaluation_markdown(report)
    policies = pd.json_normalize(report.get("policy_results", []) or [])
    candidates = pd.DataFrame(report.get("candidate_route_results", []) or [])
    regimes = pd.DataFrame(report.get("regime_payoff_results", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(policies, paths["policies_csv_path"])
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    _atomic_write_csv(regimes, paths["regimes_csv_path"])
    _atomic_write_csv(routes, paths["routes_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(policies, paths["latest_policies_csv_path"])
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
        _atomic_write_csv(regimes, paths["latest_regimes_csv_path"])
        _atomic_write_csv(routes, paths["latest_routes_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_ev_shadow_evaluation_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset and write EV/risk-adjusted shadow evidence."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_segmented_probability_ev_shadow_evaluation_report(frame, dataset_path=path, **kwargs)
