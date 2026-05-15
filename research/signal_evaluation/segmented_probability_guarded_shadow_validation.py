"""Guard-aware shadow validation for segmented-probability candidates.

This module validates a guarded segmented-probability candidate bundle using
the bundle's rank-preservation policy. It is research-only: it writes advisory
artifacts and never changes runtime configuration, parameter packs, data
sources, or execution behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import (
    _calibration_curve,
    _clean_probability_and_label_frame,
    _metrics,
)
from research.signal_evaluation.segmented_probability_ev_rejection_attribution import (
    _delta,
    _safe_float,
    _summary_for_frame,
)
from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    DEFAULT_PAYOFF_COLUMNS,
    _enrich_route_decisions,
    _top_probability_mask,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_ROUTING_POLICIES,
    _evaluation_frame,
    _load_candidate_bundle,
    _route_decisions,
)
from research.signal_evaluation.segmented_probability_guarded_candidate_bundle import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR,
    SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME,
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
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_guarded_shadow_validation"
)
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR
    / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME
)

SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation.json"
)
SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_MARKDOWN_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation.md"
)
SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_POLICIES_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation_policies.csv"
)
SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_CANDIDATES_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation_candidates.csv"
)
SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_ROUTES_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation_routes.csv"
)
SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_CURVE_FILENAME = (
    "latest_segmented_probability_guarded_shadow_validation_calibration_curve.csv"
)

GUARDED_SHADOW_VALIDATION_PASS = "GUARDED_SHADOW_VALIDATION_PASS"
GUARDED_SHADOW_VALIDATION_WATCH = "GUARDED_SHADOW_VALIDATION_WATCH"
GUARDED_SHADOW_VALIDATION_REJECTED = "GUARDED_SHADOW_VALIDATION_REJECTED"
GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA = "GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA"


def _guard_policy(bundle: dict[str, Any]) -> dict[str, Any]:
    policy = bundle.get("rank_preservation_policy")
    return policy if isinstance(policy, dict) else {}


def _negative_keys(bundle: dict[str, Any]) -> set[str]:
    return {str(key) for key in bundle.get("quarantined_candidate_keys", []) or [] if str(key).strip()}


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _apply_rank_preservation_guard(
    routes: pd.DataFrame,
    *,
    top_fraction: float,
    raw_rank_ceiling_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    guarded = routes.copy()
    if guarded.empty:
        guarded["guarded_probability"] = pd.Series(dtype=float)
        return guarded, {
            "raw_rank_guard_applied": False,
            "raw_rank_eligible_count": 0,
            "raw_rank_capped_count": 0,
        }
    raw = pd.to_numeric(guarded.get("raw_probability", pd.Series(index=guarded.index)), errors="coerce")
    shadow = pd.to_numeric(guarded.get("shadow_probability", pd.Series(index=guarded.index)), errors="coerce")
    guarded["guarded_probability"] = shadow
    valid = raw.dropna()
    if valid.empty:
        return guarded, {
            "raw_rank_guard_applied": False,
            "raw_rank_eligible_count": 0,
            "raw_rank_capped_count": 0,
        }
    fraction = min(max(float(top_fraction), 0.01), 1.0)
    multiplier = max(float(raw_rank_ceiling_multiplier), 1.0)
    eligible_count = max(int(len(valid) * min(fraction * multiplier, 1.0) + 0.999999), 1)
    eligible_index = raw.loc[valid.index].rank(method="first", ascending=False).nsmallest(eligible_count).index
    eligible_mask = pd.Series(False, index=guarded.index, dtype=bool)
    eligible_mask.loc[eligible_index] = True
    eligible_scores = shadow.loc[eligible_index].dropna()
    if eligible_scores.empty:
        guarded["raw_rank_guard_eligible"] = eligible_mask
        guarded["raw_rank_guard_capped"] = False
        return guarded, {
            "raw_rank_guard_applied": False,
            "raw_rank_eligible_count": int(eligible_count),
            "raw_rank_capped_count": 0,
        }
    cap = float(eligible_scores.min()) - 1e-9
    capped_mask = ~eligible_mask & (shadow > cap)
    guarded.loc[capped_mask, "guarded_probability"] = cap
    guarded["raw_rank_guard_eligible"] = eligible_mask
    guarded["raw_rank_guard_capped"] = capped_mask
    return guarded, {
        "raw_rank_guard_applied": True,
        "raw_rank_eligible_count": int(eligible_count),
        "raw_rank_capped_count": int(capped_mask.sum()),
        "raw_rank_ceiling_multiplier": float(raw_rank_ceiling_multiplier),
        "raw_rank_probability_cap": _round_or_none(cap, 10),
    }


def _calibration_result(
    routes: pd.DataFrame,
    *,
    route_policy: str,
    min_shadow_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    n_bins: int,
) -> dict[str, Any]:
    labels = pd.to_numeric(routes.get("label", pd.Series(dtype=float)), errors="coerce")
    raw = pd.to_numeric(routes.get("raw_probability", pd.Series(dtype=float)), errors="coerce")
    shadow = pd.to_numeric(routes.get("shadow_probability", pd.Series(dtype=float)), errors="coerce")
    guarded = pd.to_numeric(routes.get("guarded_probability", pd.Series(dtype=float)), errors="coerce")
    raw_metrics = _metrics(raw, labels, method="raw_probability", split_name=route_policy, n_bins=n_bins)
    shadow_metrics = _metrics(shadow, labels, method="shadow_probability", split_name=route_policy, n_bins=n_bins)
    guarded_metrics = _metrics(guarded, labels, method="guarded_probability", split_name=route_policy, n_bins=n_bins)
    raw_brier = raw_metrics.get("brier_score")
    shadow_brier = shadow_metrics.get("brier_score")
    raw_ece = raw_metrics.get("expected_calibration_error")
    shadow_ece = shadow_metrics.get("expected_calibration_error")
    improvement = None if raw_brier is None or shadow_brier is None else float(raw_brier) - float(shadow_brier)
    ece_change = None if raw_ece is None or shadow_ece is None else float(shadow_ece) - float(raw_ece)
    sample_count = int(shadow_metrics.get("sample_count") or 0)
    if sample_count < int(min_shadow_sample):
        status = GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA
        reason = "sample_size_guardrail_failed"
    elif improvement is not None and improvement < 0:
        status = GUARDED_SHADOW_VALIDATION_REJECTED
        reason = "shadow_calibration_worsened_brier"
    elif ece_change is not None and ece_change > float(max_ece_regression):
        status = GUARDED_SHADOW_VALIDATION_WATCH
        reason = "shadow_calibration_worsened_ece"
    elif improvement is not None and improvement >= float(min_brier_improvement):
        status = GUARDED_SHADOW_VALIDATION_PASS
        reason = "shadow_calibration_improved_brier_without_ece_regression"
    else:
        status = GUARDED_SHADOW_VALIDATION_WATCH
        reason = "shadow_calibration_did_not_clear_brier_improvement_guardrail"
    return {
        "calibration_status": status,
        "calibration_reason": reason,
        "sample_count": sample_count,
        "raw_brier_score": raw_metrics.get("brier_score"),
        "shadow_brier_score": shadow_metrics.get("brier_score"),
        "guarded_ranking_brier_score": guarded_metrics.get("brier_score"),
        "brier_improvement": _round_or_none(improvement, 8),
        "raw_expected_calibration_error": raw_metrics.get("expected_calibration_error"),
        "shadow_expected_calibration_error": shadow_metrics.get("expected_calibration_error"),
        "guarded_ranking_expected_calibration_error": guarded_metrics.get("expected_calibration_error"),
        "ece_change": _round_or_none(ece_change, 8),
        "actual_hit_rate": shadow_metrics.get("actual_hit_rate"),
    }


def _top_overlap_rate(left: pd.Series, right: pd.Series) -> float | None:
    right_count = int(right.sum())
    if right_count <= 0:
        return None
    return float((left & right).sum()) / float(right_count)


def _ev_result(
    routes: pd.DataFrame,
    *,
    route_policy: str,
    return_field: str,
    top_fraction: float,
    min_ev_sample: int,
    min_top_sample: int,
    min_risk_adjusted_improvement_bps: float,
    max_hit_rate_regression: float,
    max_spread_pct: float,
    quarantined_keys: set[str],
) -> dict[str, Any]:
    raw_top_mask = _top_probability_mask(routes.get("raw_probability", pd.Series(dtype=float)), top_fraction=top_fraction)
    shadow_top_mask = _top_probability_mask(
        routes.get("shadow_probability", pd.Series(dtype=float)),
        top_fraction=top_fraction,
    )
    guarded_top_mask = _top_probability_mask(
        routes.get("guarded_probability", pd.Series(dtype=float)),
        top_fraction=top_fraction,
    )
    guarded_bottom_mask = _top_probability_mask(
        -pd.to_numeric(routes.get("guarded_probability", pd.Series(dtype=float)), errors="coerce"),
        top_fraction=top_fraction,
    )
    raw_top = _summary_for_frame(routes.loc[raw_top_mask], return_field=return_field, max_spread_pct=max_spread_pct)
    shadow_top = _summary_for_frame(
        routes.loc[shadow_top_mask],
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    guarded_top_frame = routes.loc[guarded_top_mask].copy()
    guarded_top = _summary_for_frame(guarded_top_frame, return_field=return_field, max_spread_pct=max_spread_pct)
    guarded_bottom = _summary_for_frame(
        routes.loc[guarded_bottom_mask],
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    risk_delta = _delta(
        guarded_top.get("avg_risk_adjusted_return_bps"),
        raw_top.get("avg_risk_adjusted_return_bps"),
    )
    hit_delta = _delta(guarded_top.get("hit_rate"), raw_top.get("hit_rate"))
    liquidity_delta = _delta(
        guarded_top.get("avg_liquidity_adjusted_return_bps"),
        raw_top.get("avg_liquidity_adjusted_return_bps"),
    )
    ranking_spread = _delta(
        guarded_top.get("avg_risk_adjusted_return_bps"),
        guarded_bottom.get("avg_risk_adjusted_return_bps"),
    )
    assigned = guarded_top_frame.get("assigned_candidate_key", pd.Series(index=guarded_top_frame.index)).astype(str)
    negative_top_count = int(assigned.isin(quarantined_keys).sum()) if not guarded_top_frame.empty else 0
    top_count = int(guarded_top.get("sample_count") or 0)
    if int(len(routes)) < int(min_ev_sample) or top_count < int(min_top_sample):
        status = GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA
        reason = "ev_sample_size_guardrail_failed"
    elif negative_top_count > 0:
        status = GUARDED_SHADOW_VALIDATION_REJECTED
        reason = "quarantined_route_exposed_in_guarded_top_bucket"
    elif risk_delta is not None and risk_delta < -float(min_risk_adjusted_improvement_bps):
        status = GUARDED_SHADOW_VALIDATION_REJECTED
        reason = "guarded_top_bucket_worsened_risk_adjusted_return"
    elif hit_delta is not None and hit_delta < -float(max_hit_rate_regression):
        status = GUARDED_SHADOW_VALIDATION_REJECTED
        reason = "guarded_top_bucket_worsened_hit_rate"
    elif risk_delta is not None and risk_delta >= float(min_risk_adjusted_improvement_bps):
        status = GUARDED_SHADOW_VALIDATION_PASS
        reason = "guarded_top_bucket_preserved_or_improved_risk_adjusted_return"
    else:
        status = GUARDED_SHADOW_VALIDATION_WATCH
        reason = "guarded_top_bucket_needs_more_forward_confirmation"
    policy_score = None
    if risk_delta is not None and hit_delta is not None:
        policy_score = float(risk_delta) + (50.0 * float(hit_delta))
    return {
        "ev_status": status,
        "ev_reason": reason,
        "raw_top": raw_top,
        "shadow_top": shadow_top,
        "guarded_top": guarded_top,
        "guarded_bottom": guarded_bottom,
        "guarded_vs_raw_top_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "guarded_vs_raw_top_hit_rate_delta": _round_or_none(hit_delta, 6),
        "guarded_vs_raw_top_liquidity_adjusted_return_delta_bps": _round_or_none(liquidity_delta, 6),
        "guarded_top_vs_bottom_risk_adjusted_return_spread_bps": _round_or_none(ranking_spread, 6),
        "guarded_top_raw_top_overlap_rate": _round_or_none(_top_overlap_rate(raw_top_mask, guarded_top_mask), 6),
        "guarded_top_shadow_top_overlap_rate": _round_or_none(_top_overlap_rate(shadow_top_mask, guarded_top_mask), 6),
        "quarantined_route_top_count": negative_top_count,
        "quarantined_route_top_rate": _round_or_none(float(negative_top_count) / float(max(top_count, 1)), 6),
        "policy_score": _round_or_none(policy_score, 6),
    }


def _candidate_results(
    routes: pd.DataFrame,
    *,
    route_policy: str,
    top_fraction: float,
    return_field: str,
    max_spread_pct: float,
) -> list[dict[str, Any]]:
    if routes.empty or "assigned_candidate_key" not in routes.columns:
        return []
    top_mask = _top_probability_mask(routes.get("guarded_probability", pd.Series(dtype=float)), top_fraction=top_fraction)
    top = routes.loc[top_mask].copy()
    total = max(int(len(top)), 1)
    rows: list[dict[str, Any]] = []
    for key, group in top.groupby("assigned_candidate_key", dropna=False):
        summary = _summary_for_frame(group, return_field=return_field, max_spread_pct=max_spread_pct)
        rows.append(
            {
                "route_policy": route_policy,
                "candidate_key": str(key),
                "top_bucket_share": _round_or_none(float(len(group)) / float(total), 6),
                "sample_count": int(len(group)),
                "assigned_candidate_type": group["assigned_candidate_type"].dropna().iloc[0]
                if "assigned_candidate_type" in group.columns and not group["assigned_candidate_type"].dropna().empty
                else None,
                "assigned_calibrator": group["assigned_calibrator"].dropna().iloc[0]
                if "assigned_calibrator" in group.columns and not group["assigned_calibrator"].dropna().empty
                else None,
                "avg_risk_adjusted_return_bps": summary.get("avg_risk_adjusted_return_bps"),
                "hit_rate": summary.get("hit_rate"),
                "avg_spread_pct": summary.get("avg_spread_pct"),
            }
        )
    return sorted(rows, key=lambda item: (-float(item.get("top_bucket_share") or 0.0), str(item.get("candidate_key"))))


def _policy_status(calibration: dict[str, Any], ev: dict[str, Any]) -> tuple[str, str]:
    if ev.get("ev_status") == GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA:
        return GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA, str(ev.get("ev_reason"))
    if ev.get("ev_status") == GUARDED_SHADOW_VALIDATION_REJECTED:
        return GUARDED_SHADOW_VALIDATION_REJECTED, str(ev.get("ev_reason"))
    if calibration.get("calibration_status") == GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA:
        return GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA, str(calibration.get("calibration_reason"))
    if calibration.get("calibration_status") == GUARDED_SHADOW_VALIDATION_REJECTED:
        return GUARDED_SHADOW_VALIDATION_WATCH, "ev_passed_but_calibration_brier_regressed"
    if (
        ev.get("ev_status") == GUARDED_SHADOW_VALIDATION_PASS
        and calibration.get("calibration_status") == GUARDED_SHADOW_VALIDATION_PASS
    ):
        return GUARDED_SHADOW_VALIDATION_PASS, "guarded_ev_and_calibration_passed"
    if ev.get("ev_status") == GUARDED_SHADOW_VALIDATION_PASS:
        return GUARDED_SHADOW_VALIDATION_WATCH, "guarded_ev_passed_calibration_needs_review"
    return GUARDED_SHADOW_VALIDATION_WATCH, "guarded_validation_needs_review"


def _best_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not policy_rows:
        return None
    severity = {
        GUARDED_SHADOW_VALIDATION_PASS: 0,
        GUARDED_SHADOW_VALIDATION_WATCH: 1,
        GUARDED_SHADOW_VALIDATION_REJECTED: 2,
        GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA: 3,
    }
    return sorted(
        policy_rows,
        key=lambda row: (
            severity.get(str(row.get("guarded_policy_status")), 9),
            -float(row.get("policy_score") or -1e9),
            -float(row.get("brier_improvement") or -1e9),
        ),
    )[0]


def _overall_status(policy_rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("guarded_policy_status")) for row in policy_rows}
    if not statuses:
        return GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA
    if GUARDED_SHADOW_VALIDATION_PASS in statuses:
        return GUARDED_SHADOW_VALIDATION_PASS
    if GUARDED_SHADOW_VALIDATION_WATCH in statuses:
        return GUARDED_SHADOW_VALIDATION_WATCH
    if GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA in statuses and len(statuses) == 1:
        return GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA
    return GUARDED_SHADOW_VALIDATION_REJECTED


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("guarded_shadow_status")
    best = (report.get("selection_summary", {}) or {}).get("recommended_routing_policy")
    if status == GUARDED_SHADOW_VALIDATION_PASS:
        return [
            f"Keep `{best}` and the guarded candidate bundle in research-only forward monitoring.",
            "Run EV rejection attribution and readiness gating on this guard-aware validation before any manual review.",
            "Do not change runtime probabilities, parameter packs, data sources, or execution behavior from this report.",
        ]
    if status == GUARDED_SHADOW_VALIDATION_WATCH:
        return [
            "The guard-aware bundle improved selection safety but needs calibration or forward-label review.",
            "Keep runtime behavior unchanged and collect more guard-aware shadow evidence.",
        ]
    if status == GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA:
        return [
            "Collect more quality-approved labels before judging the guarded bundle.",
            "Keep the guarded bundle research-only.",
        ]
    return [
        "Do not advance the guarded bundle; guard-aware shadow validation failed.",
        "Return to guarded candidate generation or signal-quality diagnostics.",
    ]


def build_segmented_probability_guarded_shadow_validation_report(
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
    top_fraction: float | None = None,
    raw_rank_ceiling_multiplier: float | None = None,
    min_shadow_sample: int = 100,
    min_ev_sample: int = 100,
    min_top_sample: int = 25,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    min_risk_adjusted_improvement_bps: float = 0.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build guard-aware forward and EV shadow evidence for a guarded bundle."""
    bundle = candidate_bundle if isinstance(candidate_bundle, dict) else _load_candidate_bundle(
        candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH
    )
    candidates = [candidate for candidate in bundle.get("candidates", []) or [] if isinstance(candidate, dict)]
    policy = _guard_policy(bundle)
    fraction = float(top_fraction if top_fraction is not None else policy.get("top_fraction") or 0.25)
    multiplier = float(
        raw_rank_ceiling_multiplier
        if raw_rank_ceiling_multiplier is not None
        else policy.get("raw_rank_ceiling_multiplier") or 1.0
    )
    raw = frame if frame is not None else pd.DataFrame()
    working = _clean_probability_and_label_frame(raw, probability_field=probability_field, label_field=label_field)
    evaluation, window = _evaluation_frame(
        working,
        bundle=bundle,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        min_shadow_sample=min_shadow_sample,
    )
    payoff_columns = tuple(dict.fromkeys(DEFAULT_PAYOFF_COLUMNS + (return_field,)))
    quarantined_keys = _negative_keys(bundle)
    policy_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    route_frames: list[pd.DataFrame] = []
    curve_rows: list[dict[str, Any]] = []
    for route_policy in routing_policies:
        decisions = _route_decisions(evaluation, candidates, routing_policy=str(route_policy))
        enriched = _enrich_route_decisions(
            evaluation,
            decisions,
            return_field=return_field,
            payoff_columns=payoff_columns,
            regime_fields=regime_fields,
            downside_penalty_weight=downside_penalty_weight,
            spread_penalty_per_pct=spread_penalty_per_pct,
        )
        guarded, guard_metadata = _apply_rank_preservation_guard(
            enriched,
            top_fraction=fraction,
            raw_rank_ceiling_multiplier=multiplier,
        )
        calibration = _calibration_result(
            guarded,
            route_policy=str(route_policy),
            min_shadow_sample=min_shadow_sample,
            min_brier_improvement=min_brier_improvement,
            max_ece_regression=max_ece_regression,
            n_bins=n_bins,
        )
        ev = _ev_result(
            guarded,
            route_policy=str(route_policy),
            return_field=return_field,
            top_fraction=fraction,
            min_ev_sample=min_ev_sample,
            min_top_sample=min_top_sample,
            min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
            max_hit_rate_regression=max_hit_rate_regression,
            max_spread_pct=max_spread_pct,
            quarantined_keys=quarantined_keys,
        )
        status, reason = _policy_status(calibration, ev)
        row = {
            "route_policy": str(route_policy),
            "guarded_policy_status": status,
            "status_reason": reason,
        }
        row.update(calibration)
        row.update(ev)
        row.update(guard_metadata)
        policy_rows.append(row)
        candidate_rows.extend(
            _candidate_results(
                guarded,
                route_policy=str(route_policy),
                top_fraction=fraction,
                return_field=return_field,
                max_spread_pct=max_spread_pct,
            )
        )
        labels = pd.to_numeric(guarded.get("label", pd.Series(dtype=float)), errors="coerce")
        raw_probability = pd.to_numeric(guarded.get("raw_probability", pd.Series(dtype=float)), errors="coerce")
        shadow_probability = pd.to_numeric(guarded.get("shadow_probability", pd.Series(dtype=float)), errors="coerce")
        guarded_probability = pd.to_numeric(guarded.get("guarded_probability", pd.Series(dtype=float)), errors="coerce")
        curve_rows.extend(
            _calibration_curve(raw_probability, labels, method=f"{route_policy}:raw", split_name="guarded", n_bins=n_bins)
        )
        curve_rows.extend(
            _calibration_curve(
                shadow_probability,
                labels,
                method=f"{route_policy}:shadow",
                split_name="guarded",
                n_bins=n_bins,
            )
        )
        curve_rows.extend(
            _calibration_curve(
                guarded_probability,
                labels,
                method=f"{route_policy}:guarded_rank",
                split_name="guarded",
                n_bins=n_bins,
            )
        )
        route_frames.append(guarded)
    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    best = _best_policy(policy_rows)
    side_effects_clean = (
        bundle.get("runtime_config_changed") is False
        and bundle.get("parameter_pack_file_changed") is False
        and bundle.get("execution_behavior_changed") is False
    )
    bundle_research_only = _safe_bool(bundle.get("research_only"))
    bundle_approval_required = _safe_bool(bundle.get("approval_required_for_runtime_use"))
    report = {
        "report_type": "segmented_probability_guarded_shadow_validation",
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
        "quarantined_candidate_keys": sorted(quarantined_keys),
        "validation_window": window,
        "top_fraction": fraction,
        "raw_rank_ceiling_multiplier": multiplier,
        "rank_preservation_policy": policy,
        "rank_preservation_policy_present": bool(policy),
        "guarded_bundle_side_effect_flags_clean": side_effects_clean,
        "guarded_bundle_research_only": bool(bundle_research_only),
        "guarded_bundle_approval_required_for_runtime_use": bool(bundle_approval_required),
        "guarded_bundle_status": bundle.get("guarded_candidate_bundle_status") or bundle.get("calibration_status"),
        "guarded_shadow_status": _overall_status(policy_rows),
        "selection_summary": {
            "recommended_routing_policy": best.get("route_policy") if best else None,
            "recommended_policy_status": best.get("guarded_policy_status") if best else None,
            "recommended_policy_score": best.get("policy_score") if best else None,
            "recommended_policy_risk_delta_vs_raw_bps": (
                best.get("guarded_vs_raw_top_risk_adjusted_return_delta_bps") if best else None
            ),
            "recommended_policy_hit_delta_vs_raw": best.get("guarded_vs_raw_top_hit_rate_delta") if best else None,
            "evaluated_routing_policy_count": int(len(policy_rows)),
        },
        "policy_results": policy_rows,
        "candidate_route_results": candidate_rows,
        "calibration_curve": curve_rows,
        "route_decision_count": int(len(routes)),
        "recommended_next_actions": [],
    }
    if not side_effects_clean and report["guarded_shadow_status"] == GUARDED_SHADOW_VALIDATION_PASS:
        report["guarded_shadow_status"] = GUARDED_SHADOW_VALIDATION_WATCH
    report["recommended_next_actions"] = _recommended_actions(report)
    sanitized = _sanitize_value(report)
    sanitized["_route_decisions_frame"] = routes
    return sanitized


def render_segmented_probability_guarded_shadow_validation_markdown(report: dict[str, Any]) -> str:
    """Render guard-aware shadow validation as Markdown."""
    selection = report.get("selection_summary", {}) or {}
    window = report.get("validation_window", {}) or {}
    lines = [
        "# Segmented Probability Guard-Aware Shadow Validation",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Guarded candidate bundle: {report.get('candidate_bundle_path') or 'inline'}",
        f"- Validation mode: `{window.get('validation_mode_used')}`",
        f"- Status: `{report.get('guarded_shadow_status')}`",
        f"- Recommended routing policy: `{selection.get('recommended_routing_policy')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Policies",
        "",
        "| Policy | Status | EV Risk Delta | EV Hit Delta | Raw Overlap | Quarantine Top Rate | Brier Improvement | Guard Capped |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("policy_results", []) or []:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('guarded_policy_status')}` | "
            f"{row.get('guarded_vs_raw_top_risk_adjusted_return_delta_bps')} | "
            f"{row.get('guarded_vs_raw_top_hit_rate_delta')} | "
            f"{row.get('guarded_top_raw_top_overlap_rate')} | "
            f"{row.get('quarantined_route_top_rate')} | "
            f"{row.get('brier_improvement')} | {row.get('raw_rank_capped_count')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Exposure",
            "",
            "| Policy | Candidate | Top Share | N | Avg Risk-Adj Return | Hit Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (report.get("candidate_route_results", []) or [])[:30]:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('candidate_key')}` | "
            f"{row.get('top_bucket_share')} | {row.get('sample_count')} | "
            f"{row.get('avg_risk_adjusted_return_bps')} | {row.get('hit_rate')} |"
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
        "routes_csv_path": output / f"{stem}_routes.csv",
        "calibration_curve_csv_path": output / f"{stem}_calibration_curve.csv",
        "latest_json_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_MARKDOWN_FILENAME,
        "latest_policies_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_POLICIES_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_CANDIDATES_FILENAME,
        "latest_routes_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_ROUTES_FILENAME,
        "latest_calibration_curve_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_CURVE_FILENAME,
    }


def write_segmented_probability_guarded_shadow_validation_report(
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
    top_fraction: float | None = None,
    raw_rank_ceiling_multiplier: float | None = None,
    min_shadow_sample: int = 100,
    min_ev_sample: int = 100,
    min_top_sample: int = 25,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    min_risk_adjusted_improvement_bps: float = 0.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    n_bins: int = 10,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write guard-aware shadow validation artifacts."""
    bundle_path = candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH
    report = build_segmented_probability_guarded_shadow_validation_report(
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
        raw_rank_ceiling_multiplier=raw_rank_ceiling_multiplier,
        min_shadow_sample=min_shadow_sample,
        min_ev_sample=min_ev_sample,
        min_top_sample=min_top_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
        max_hit_rate_regression=max_hit_rate_regression,
        downside_penalty_weight=downside_penalty_weight,
        spread_penalty_per_pct=spread_penalty_per_pct,
        max_spread_pct=max_spread_pct,
        n_bins=n_bins,
    )
    routes = report.pop("_route_decisions_frame", pd.DataFrame())
    assert_artifact_schema(report, "segmented_probability_guarded_shadow_validation")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_guarded_shadow_validation"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_guarded_shadow_validation_markdown(report)
    policies = pd.json_normalize(report.get("policy_results", []) or [])
    candidates = pd.DataFrame(report.get("candidate_route_results", []) or [])
    curve = pd.DataFrame(report.get("calibration_curve", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(policies, paths["policies_csv_path"])
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    _atomic_write_csv(routes, paths["routes_csv_path"])
    _atomic_write_csv(curve, paths["calibration_curve_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(policies, paths["latest_policies_csv_path"])
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
        _atomic_write_csv(routes, paths["latest_routes_csv_path"])
        _atomic_write_csv(curve, paths["latest_calibration_curve_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_guarded_shadow_validation_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset and write guard-aware validation artifacts."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_segmented_probability_guarded_shadow_validation_report(frame, dataset_path=path, **kwargs)
