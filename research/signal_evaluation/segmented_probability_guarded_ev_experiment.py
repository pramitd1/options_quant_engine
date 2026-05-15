"""Guarded EV experiment for segmented-probability candidates.

This module tests whether EV-negative route quarantine and raw-rank
preservation would have prevented a rejected segmented-probability candidate
from degrading the top signal bucket. It is research-only: it writes advisory
artifacts and never changes runtime configuration, parameter packs, data
sources, or execution behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import _clean_probability_and_label_frame
from research.signal_evaluation.segmented_probability_ev_rejection_attribution import (
    DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR,
    EV_REJECTION_ATTRIBUTION_ACTIONABLE,
    SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_JSON_FILENAME,
    _delta,
    _read_json,
    _routes_for_policy,
    _safe_float,
    _summary_for_frame,
)
from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    DEFAULT_PAYOFF_COLUMNS,
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR,
    SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME,
    SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME,
    _enrich_route_decisions,
    _top_probability_mask,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    _candidate_key,
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
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_guarded_ev_experiment"
)
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR / SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR / SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_DIR
    / SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_JSON_FILENAME
)

SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_JSON_FILENAME = (
    "latest_segmented_probability_guarded_ev_experiment.json"
)
SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_MARKDOWN_FILENAME = (
    "latest_segmented_probability_guarded_ev_experiment.md"
)
SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_VARIANTS_FILENAME = (
    "latest_segmented_probability_guarded_ev_variants.csv"
)
SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_CANDIDATES_FILENAME = (
    "latest_segmented_probability_guarded_ev_candidates.csv"
)

GUARDED_EV_EXPERIMENT_PASS = "GUARDED_EV_EXPERIMENT_PASS"
GUARDED_EV_EXPERIMENT_WATCH = "GUARDED_EV_EXPERIMENT_WATCH"
GUARDED_EV_EXPERIMENT_REJECTED = "GUARDED_EV_EXPERIMENT_REJECTED"
GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA = "GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA"

VARIANT_BASELINE_SHADOW = "baseline_shadow"
VARIANT_QUARANTINE = "quarantine_negative_routes"
VARIANT_RANK_GUARD = "rank_preservation_guard"
VARIANT_QUARANTINE_PLUS_RANK_GUARD = "quarantine_plus_rank_guard"


def _load_routes(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _negative_route_keys(attribution_report: dict[str, Any]) -> list[str]:
    rows = attribution_report.get("negative_route_candidates", []) or []
    keys = [
        str(row.get("candidate_key"))
        for row in rows
        if isinstance(row, dict) and row.get("candidate_key")
    ]
    return list(dict.fromkeys(keys))


def _analysis_policy(ev_shadow_report: dict[str, Any], attribution_report: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    if attribution_report.get("analysis_policy"):
        return str(attribution_report.get("analysis_policy"))
    selected = (ev_shadow_report.get("selection_summary", {}) or {}).get("recommended_routing_policy")
    return str(selected or "regime_first")


def _candidate_bundle_candidates(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate for candidate in bundle.get("candidates", []) or [] if isinstance(candidate, dict)]


def _filter_candidate_bundle(bundle: dict[str, Any], quarantine_keys: set[str]) -> dict[str, Any]:
    filtered = dict(bundle or {})
    candidates = _candidate_bundle_candidates(bundle)
    kept = [candidate for candidate in candidates if _candidate_key(candidate) not in quarantine_keys]
    filtered["candidates"] = kept
    filtered["candidate_count"] = int(len(kept))
    filtered["guarded_ev_quarantined_candidate_keys"] = sorted(quarantine_keys)
    return filtered


def _existing_file(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def _load_bundle_for_report(ev_shadow_report: dict[str, Any], candidate_bundle_path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    path = _existing_file(candidate_bundle_path or ev_shadow_report.get("candidate_bundle_path"))
    if path is None:
        path = _existing_file(DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH)
    if path is None:
        return {"candidate_count": 0, "candidates": []}, None
    return _load_candidate_bundle(path), path


def _load_dataset_for_report(ev_shadow_report: dict[str, Any], dataset_path: str | Path | None) -> tuple[pd.DataFrame | None, Path | None]:
    path = _existing_file(dataset_path or ev_shadow_report.get("dataset_path"))
    if path is None:
        fallback = default_signal_quality_dataset_path()
        path = fallback if fallback.exists() else None
    if path is None:
        return None, None
    return pd.read_csv(path, low_memory=False), path


def _rebuild_routes_from_bundle(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    ev_shadow_report: dict[str, Any],
    route_policy: str,
    probability_field: str,
    label_field: str,
    return_field: str,
    train_fraction: float,
    validation_mode: str,
    regime_fields: tuple[str, ...],
    downside_penalty_weight: float,
    spread_penalty_per_pct: float,
    min_shadow_sample: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = _candidate_bundle_candidates(bundle)
    working = _clean_probability_and_label_frame(
        frame,
        probability_field=probability_field,
        label_field=label_field,
    )
    evaluation, window = _evaluation_frame(
        working,
        bundle=bundle,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        min_shadow_sample=min_shadow_sample,
    )
    decisions = _route_decisions(evaluation, candidates, routing_policy=route_policy)
    payoff_columns = tuple(dict.fromkeys(DEFAULT_PAYOFF_COLUMNS + (return_field,)))
    enriched = _enrich_route_decisions(
        evaluation,
        decisions,
        return_field=return_field,
        payoff_columns=payoff_columns,
        regime_fields=regime_fields,
        downside_penalty_weight=downside_penalty_weight,
        spread_penalty_per_pct=spread_penalty_per_pct,
    )
    return enriched, window


def _fallback_quarantine_routes(routes: pd.DataFrame, quarantine_keys: set[str]) -> pd.DataFrame:
    guarded = routes.copy()
    if guarded.empty or "assigned_candidate_key" not in guarded.columns:
        return guarded
    assigned = guarded["assigned_candidate_key"].astype(str)
    quarantine_mask = assigned.isin(quarantine_keys)
    guarded["quarantined_candidate_key"] = None
    guarded.loc[quarantine_mask, "quarantined_candidate_key"] = assigned.loc[quarantine_mask]
    guarded.loc[quarantine_mask, "shadow_probability"] = pd.to_numeric(
        guarded.loc[quarantine_mask, "raw_probability"],
        errors="coerce",
    )
    guarded.loc[quarantine_mask, "assigned_candidate_key"] = "quarantined_identity"
    guarded.loc[quarantine_mask, "assigned_candidate_type"] = "none"
    guarded.loc[quarantine_mask, "assigned_segment_field"] = None
    guarded.loc[quarantine_mask, "assigned_segment_value"] = None
    guarded.loc[quarantine_mask, "assigned_calibrator"] = "identity_after_quarantine"
    return guarded


def _apply_rank_preservation_guard(
    routes: pd.DataFrame,
    *,
    source_probability_column: str = "shadow_probability",
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
    source = pd.to_numeric(guarded.get(source_probability_column, pd.Series(index=guarded.index)), errors="coerce")
    guarded["guarded_probability"] = source
    valid = raw.dropna()
    if valid.empty:
        return guarded, {
            "raw_rank_guard_applied": False,
            "raw_rank_eligible_count": 0,
            "raw_rank_capped_count": 0,
        }
    fraction = min(max(float(top_fraction), 0.01), 1.0)
    multiplier = max(float(raw_rank_ceiling_multiplier), 1.0)
    eligible_count = max(int(math.ceil(len(valid) * min(fraction * multiplier, 1.0))), 1)
    eligible_index = raw.loc[valid.index].rank(method="first", ascending=False).nsmallest(eligible_count).index
    eligible_mask = pd.Series(False, index=guarded.index, dtype=bool)
    eligible_mask.loc[eligible_index] = True
    eligible_scores = source.loc[eligible_index].dropna()
    if eligible_scores.empty:
        return guarded, {
            "raw_rank_guard_applied": False,
            "raw_rank_eligible_count": int(eligible_count),
            "raw_rank_capped_count": 0,
        }
    cap = float(eligible_scores.min()) - 1e-9
    capped_mask = ~eligible_mask & (source > cap)
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


def _top_overlap_rate(left_mask: pd.Series, right_mask: pd.Series) -> float | None:
    right_count = int(right_mask.sum())
    if right_count <= 0:
        return None
    return float((left_mask & right_mask).sum()) / float(right_count)


def _variant_result(
    routes: pd.DataFrame,
    *,
    variant_name: str,
    score_column: str,
    top_fraction: float,
    return_field: str,
    min_top_sample: int,
    max_spread_pct: float,
    min_risk_adjusted_improvement_bps: float,
    max_hit_rate_regression: float,
    quarantine_keys: set[str],
    guard_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_top_mask = _top_probability_mask(routes.get("raw_probability", pd.Series(dtype=float)), top_fraction=top_fraction)
    baseline_top_mask = _top_probability_mask(
        routes.get("shadow_probability", pd.Series(dtype=float)),
        top_fraction=top_fraction,
    )
    variant_top_mask = _top_probability_mask(routes.get(score_column, pd.Series(dtype=float)), top_fraction=top_fraction)
    variant_bottom_mask = _top_probability_mask(
        -pd.to_numeric(routes.get(score_column, pd.Series(dtype=float)), errors="coerce"),
        top_fraction=top_fraction,
    )
    raw_top = _summary_for_frame(routes.loc[raw_top_mask], return_field=return_field, max_spread_pct=max_spread_pct)
    variant_top_frame = routes.loc[variant_top_mask].copy()
    variant_top = _summary_for_frame(variant_top_frame, return_field=return_field, max_spread_pct=max_spread_pct)
    variant_bottom = _summary_for_frame(
        routes.loc[variant_bottom_mask],
        return_field=return_field,
        max_spread_pct=max_spread_pct,
    )
    risk_delta = _delta(
        variant_top.get("avg_risk_adjusted_return_bps"),
        raw_top.get("avg_risk_adjusted_return_bps"),
    )
    hit_delta = _delta(variant_top.get("hit_rate"), raw_top.get("hit_rate"))
    liquidity_delta = _delta(
        variant_top.get("avg_liquidity_adjusted_return_bps"),
        raw_top.get("avg_liquidity_adjusted_return_bps"),
    )
    ranking_spread = _delta(
        variant_top.get("avg_risk_adjusted_return_bps"),
        variant_bottom.get("avg_risk_adjusted_return_bps"),
    )
    assigned = variant_top_frame.get("assigned_candidate_key", pd.Series(index=variant_top_frame.index)).astype(str)
    negative_top_count = int(assigned.isin(quarantine_keys).sum()) if not variant_top_frame.empty else 0
    top_count = int(variant_top.get("sample_count") or 0)
    negative_top_rate = float(negative_top_count) / float(max(top_count, 1))
    if top_count < int(min_top_sample):
        status = GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA
        reason = "top_bucket_sample_guardrail_failed"
    elif negative_top_count > 0:
        status = GUARDED_EV_EXPERIMENT_REJECTED
        reason = "ev_negative_route_still_exposed_in_top_bucket"
    elif risk_delta is not None and risk_delta < -float(min_risk_adjusted_improvement_bps):
        status = GUARDED_EV_EXPERIMENT_REJECTED
        reason = "guarded_top_bucket_worsened_risk_adjusted_return"
    elif hit_delta is not None and hit_delta < -float(max_hit_rate_regression):
        status = GUARDED_EV_EXPERIMENT_REJECTED
        reason = "guarded_top_bucket_worsened_hit_rate"
    elif risk_delta is not None and risk_delta >= float(min_risk_adjusted_improvement_bps):
        status = GUARDED_EV_EXPERIMENT_PASS
        reason = "guarded_top_bucket_preserved_or_improved_risk_adjusted_return"
    else:
        status = GUARDED_EV_EXPERIMENT_WATCH
        reason = "guarded_top_bucket_needs_more_forward_confirmation"
    policy_score = None
    if risk_delta is not None and hit_delta is not None:
        policy_score = float(risk_delta) + (50.0 * float(hit_delta))
    row = {
        "variant_name": variant_name,
        "variant_status": status,
        "status_reason": reason,
        "score_column": score_column,
        "sample_count": int(len(routes)),
        "top_fraction": float(top_fraction),
        "raw_top": raw_top,
        "variant_top": variant_top,
        "variant_bottom": variant_bottom,
        "variant_vs_raw_top_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "variant_vs_raw_top_hit_rate_delta": _round_or_none(hit_delta, 6),
        "variant_vs_raw_top_liquidity_adjusted_return_delta_bps": _round_or_none(liquidity_delta, 6),
        "variant_top_vs_bottom_risk_adjusted_return_spread_bps": _round_or_none(ranking_spread, 6),
        "variant_top_raw_top_overlap_rate": _round_or_none(_top_overlap_rate(raw_top_mask, variant_top_mask), 6),
        "variant_top_baseline_shadow_overlap_rate": _round_or_none(
            _top_overlap_rate(baseline_top_mask, variant_top_mask),
            6,
        ),
        "ev_negative_route_top_count": negative_top_count,
        "ev_negative_route_top_rate": _round_or_none(negative_top_rate, 6),
        "policy_score": _round_or_none(policy_score, 6),
    }
    row.update(guard_metadata or {})
    return row


def _candidate_exposure_rows(
    routes: pd.DataFrame,
    *,
    variant_name: str,
    score_column: str,
    top_fraction: float,
    return_field: str,
    max_spread_pct: float,
) -> list[dict[str, Any]]:
    if routes.empty or "assigned_candidate_key" not in routes.columns:
        return []
    top_mask = _top_probability_mask(routes.get(score_column, pd.Series(dtype=float)), top_fraction=top_fraction)
    top = routes.loc[top_mask].copy()
    if top.empty:
        return []
    total = max(int(len(top)), 1)
    rows: list[dict[str, Any]] = []
    for key, group in top.groupby("assigned_candidate_key", dropna=False):
        summary = _summary_for_frame(group, return_field=return_field, max_spread_pct=max_spread_pct)
        rows.append(
            {
                "variant_name": variant_name,
                "candidate_key": str(key),
                "top_bucket_share": _round_or_none(float(len(group)) / float(total), 6),
                "sample_count": int(len(group)),
                "assigned_candidate_type": group.get("assigned_candidate_type", pd.Series(dtype=object)).dropna().iloc[0]
                if "assigned_candidate_type" in group.columns and not group["assigned_candidate_type"].dropna().empty
                else None,
                "assigned_calibrator": group.get("assigned_calibrator", pd.Series(dtype=object)).dropna().iloc[0]
                if "assigned_calibrator" in group.columns and not group["assigned_calibrator"].dropna().empty
                else None,
                "avg_risk_adjusted_return_bps": summary.get("avg_risk_adjusted_return_bps"),
                "hit_rate": summary.get("hit_rate"),
                "avg_spread_pct": summary.get("avg_spread_pct"),
            }
        )
    return sorted(rows, key=lambda item: (-float(item.get("top_bucket_share") or 0.0), str(item.get("candidate_key"))))


def _best_guarded_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("variant_name") != VARIANT_BASELINE_SHADOW]
    if not candidates:
        return None
    severity = {
        GUARDED_EV_EXPERIMENT_PASS: 0,
        GUARDED_EV_EXPERIMENT_WATCH: 1,
        GUARDED_EV_EXPERIMENT_REJECTED: 2,
        GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA: 3,
    }
    return sorted(
        candidates,
        key=lambda row: (
            severity.get(str(row.get("variant_status")), 9),
            -float(row.get("policy_score") or -1e9),
            -float(row.get("variant_top_vs_bottom_risk_adjusted_return_spread_bps") or -1e9),
        ),
    )[0]


def _overall_status(best_variant: dict[str, Any] | None, baseline: dict[str, Any] | None) -> str:
    if best_variant is None:
        return GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA
    if best_variant.get("variant_status") == GUARDED_EV_EXPERIMENT_PASS:
        return GUARDED_EV_EXPERIMENT_PASS
    baseline_delta = _safe_float((baseline or {}).get("variant_vs_raw_top_risk_adjusted_return_delta_bps"))
    best_delta = _safe_float(best_variant.get("variant_vs_raw_top_risk_adjusted_return_delta_bps"))
    if best_delta is not None and baseline_delta is not None and best_delta > baseline_delta:
        return GUARDED_EV_EXPERIMENT_WATCH
    if best_variant.get("variant_status") == GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA:
        return GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA
    return GUARDED_EV_EXPERIMENT_REJECTED


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("guarded_ev_status")
    best = report.get("selection_summary", {}).get("recommended_guarded_variant")
    quarantine_keys = report.get("quarantined_candidate_keys", []) or []
    if status == GUARDED_EV_EXPERIMENT_PASS:
        actions = [
            f"Use `{best}` as the next research-only candidate-bundle design for forward shadow validation.",
            "Keep the runtime engine unchanged; this artifact is not an adoption approval.",
        ]
        if quarantine_keys:
            actions.append(
                "Exclude the EV-negative route(s) from the next research bundle: "
                + ", ".join(f"`{key}`" for key in quarantine_keys)
                + "."
            )
        actions.append("Require fresh forward-shadow and EV attribution evidence before any manual promotion review.")
        return actions
    if status == GUARDED_EV_EXPERIMENT_WATCH:
        return [
            f"Keep `{best}` in shadow research only; it improved the rejected baseline but did not clear every guardrail.",
            "Collect true post-candidate labels before considering a new candidate bundle.",
            "Do not change runtime probabilities, data sources, parameter packs, or execution behavior.",
        ]
    if status == GUARDED_EV_EXPERIMENT_NEEDS_MORE_DATA:
        return [
            "Collect more quality-approved labels before judging the guarded EV experiment.",
            "Keep the current candidate bundle blocked from promotion.",
        ]
    return [
        "Do not generate a replacement candidate from these guards alone; the guarded variants did not repair EV rejection.",
        "Return to signal-quality modeling, probability calibration, and feature/regime diagnostics.",
    ]


def build_segmented_probability_guarded_ev_experiment_report(
    ev_shadow_report: dict[str, Any],
    ev_rejection_attribution: dict[str, Any],
    routes: pd.DataFrame,
    *,
    dataset_frame: pd.DataFrame | None = None,
    candidate_bundle: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    candidate_bundle_path: str | Path | None = None,
    ev_shadow_report_path: str | Path | None = None,
    ev_rejection_attribution_path: str | Path | None = None,
    ev_shadow_routes_path: str | Path | None = None,
    route_policy: str | None = None,
    probability_field: str | None = None,
    label_field: str | None = None,
    return_field: str | None = None,
    train_fraction: float | None = None,
    validation_mode: str | None = None,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    top_fraction: float | None = None,
    min_shadow_sample: int = 100,
    min_top_sample: int = 25,
    min_risk_adjusted_improvement_bps: float = 0.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    raw_rank_ceiling_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build an advisory guarded EV experiment from rejected EV artifacts."""
    analysis_policy = _analysis_policy(ev_shadow_report, ev_rejection_attribution, route_policy)
    probability = str(probability_field or ev_shadow_report.get("probability_field") or DEFAULT_PROBABILITY_FIELD)
    label = str(label_field or ev_shadow_report.get("label_field") or DEFAULT_LABEL_FIELD)
    returns = str(return_field or ev_shadow_report.get("return_field") or DEFAULT_RETURN_FIELD)
    fraction = float(top_fraction if top_fraction is not None else ev_shadow_report.get("top_fraction") or 0.25)
    train = float(train_fraction if train_fraction is not None else ev_shadow_report.get("train_fraction") or 0.70)
    window = ev_shadow_report.get("validation_window", {}) or {}
    validation = str(validation_mode or window.get("validation_mode_used") or "holdout_replay")
    quarantine_keys = set(_negative_route_keys(ev_rejection_attribution))
    baseline_routes = _routes_for_policy(routes.copy() if isinstance(routes, pd.DataFrame) else pd.DataFrame(), analysis_policy)
    baseline_routes = baseline_routes.reset_index(drop=True)
    simulation_mode = "route_csv_fallback"
    rebuild_window: dict[str, Any] = {}
    bundle = candidate_bundle if isinstance(candidate_bundle, dict) else None
    dataset = dataset_frame.copy() if isinstance(dataset_frame, pd.DataFrame) else None
    bundle_path_resolved: Path | None = None
    dataset_path_resolved: Path | None = None
    if bundle is None:
        bundle, bundle_path_resolved = _load_bundle_for_report(ev_shadow_report, candidate_bundle_path)
    if dataset is None:
        dataset, dataset_path_resolved = _load_dataset_for_report(ev_shadow_report, dataset_path)
    if dataset is not None and bundle is not None and _candidate_bundle_candidates(bundle):
        simulation_mode = "dataset_candidate_bundle_replay"
        baseline_routes, rebuild_window = _rebuild_routes_from_bundle(
            dataset,
            bundle,
            ev_shadow_report=ev_shadow_report,
            route_policy=analysis_policy,
            probability_field=probability,
            label_field=label,
            return_field=returns,
            train_fraction=train,
            validation_mode=validation,
            regime_fields=regime_fields,
            downside_penalty_weight=downside_penalty_weight,
            spread_penalty_per_pct=spread_penalty_per_pct,
            min_shadow_sample=min_shadow_sample,
        )
        baseline_routes = baseline_routes.reset_index(drop=True)
        filtered_bundle = _filter_candidate_bundle(bundle, quarantine_keys)
        quarantine_routes, _ = _rebuild_routes_from_bundle(
            dataset,
            filtered_bundle,
            ev_shadow_report=ev_shadow_report,
            route_policy=analysis_policy,
            probability_field=probability,
            label_field=label,
            return_field=returns,
            train_fraction=train,
            validation_mode=validation,
            regime_fields=regime_fields,
            downside_penalty_weight=downside_penalty_weight,
            spread_penalty_per_pct=spread_penalty_per_pct,
            min_shadow_sample=min_shadow_sample,
        )
        quarantine_routes = quarantine_routes.reset_index(drop=True)
    else:
        quarantine_routes = _fallback_quarantine_routes(baseline_routes, quarantine_keys).reset_index(drop=True)

    baseline_routes["variant_probability"] = pd.to_numeric(baseline_routes.get("shadow_probability"), errors="coerce")
    quarantine_routes["variant_probability"] = pd.to_numeric(quarantine_routes.get("shadow_probability"), errors="coerce")
    rank_guard_routes, rank_guard_meta = _apply_rank_preservation_guard(
        baseline_routes,
        source_probability_column="shadow_probability",
        top_fraction=fraction,
        raw_rank_ceiling_multiplier=raw_rank_ceiling_multiplier,
    )
    quarantine_rank_guard_routes, quarantine_rank_guard_meta = _apply_rank_preservation_guard(
        quarantine_routes,
        source_probability_column="shadow_probability",
        top_fraction=fraction,
        raw_rank_ceiling_multiplier=raw_rank_ceiling_multiplier,
    )

    variants = [
        (
            VARIANT_BASELINE_SHADOW,
            baseline_routes,
            "variant_probability",
            {"raw_rank_guard_applied": False},
        ),
        (
            VARIANT_QUARANTINE,
            quarantine_routes,
            "variant_probability",
            {
                "raw_rank_guard_applied": False,
                "quarantine_applied": bool(quarantine_keys),
            },
        ),
        (
            VARIANT_RANK_GUARD,
            rank_guard_routes,
            "guarded_probability",
            rank_guard_meta,
        ),
        (
            VARIANT_QUARANTINE_PLUS_RANK_GUARD,
            quarantine_rank_guard_routes,
            "guarded_probability",
            {
                **quarantine_rank_guard_meta,
                "quarantine_applied": bool(quarantine_keys),
            },
        ),
    ]
    variant_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for name, frame, score_column, metadata in variants:
        variant_rows.append(
            _variant_result(
                frame,
                variant_name=name,
                score_column=score_column,
                top_fraction=fraction,
                return_field=returns,
                min_top_sample=min_top_sample,
                max_spread_pct=max_spread_pct,
                min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
                max_hit_rate_regression=max_hit_rate_regression,
                quarantine_keys=quarantine_keys,
                guard_metadata=metadata,
            )
        )
        candidate_rows.extend(
            _candidate_exposure_rows(
                frame,
                variant_name=name,
                score_column=score_column,
                top_fraction=fraction,
                return_field=returns,
                max_spread_pct=max_spread_pct,
            )
        )

    baseline_row = next((row for row in variant_rows if row.get("variant_name") == VARIANT_BASELINE_SHADOW), None)
    best = _best_guarded_variant(variant_rows)
    status = _overall_status(best, baseline_row)
    baseline_delta = _safe_float((baseline_row or {}).get("variant_vs_raw_top_risk_adjusted_return_delta_bps"))
    best_delta = _safe_float((best or {}).get("variant_vs_raw_top_risk_adjusted_return_delta_bps"))
    delta_vs_baseline = None if baseline_delta is None or best_delta is None else best_delta - baseline_delta
    attribution_status = str(ev_rejection_attribution.get("attribution_status") or "")
    report = {
        "report_type": "segmented_probability_guarded_ev_experiment",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path_resolved or dataset_path or ev_shadow_report.get("dataset_path") or ""),
        "candidate_bundle_path": str(
            bundle_path_resolved or candidate_bundle_path or ev_shadow_report.get("candidate_bundle_path") or ""
        ),
        "ev_shadow_report_path": str(ev_shadow_report_path) if ev_shadow_report_path is not None else None,
        "ev_rejection_attribution_path": (
            str(ev_rejection_attribution_path) if ev_rejection_attribution_path is not None else None
        ),
        "ev_shadow_routes_path": str(ev_shadow_routes_path) if ev_shadow_routes_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_ev_status": status,
        "analysis_policy": analysis_policy,
        "simulation_mode": simulation_mode,
        "ev_rejection_attribution_status": attribution_status,
        "ev_rejection_is_actionable": attribution_status == EV_REJECTION_ATTRIBUTION_ACTIONABLE,
        "probability_field": probability,
        "label_field": label,
        "return_field": returns,
        "top_fraction": fraction,
        "raw_rank_ceiling_multiplier": float(raw_rank_ceiling_multiplier),
        "quarantined_candidate_keys": sorted(quarantine_keys),
        "quarantined_candidate_count": int(len(quarantine_keys)),
        "candidate_count_before_quarantine": int(len(_candidate_bundle_candidates(bundle or {}))),
        "candidate_count_after_quarantine": int(
            len(_candidate_bundle_candidates(_filter_candidate_bundle(bundle or {}, quarantine_keys)))
        ),
        "route_decision_count": int(len(baseline_routes)),
        "validation_window": rebuild_window or window,
        "variant_results": variant_rows,
        "candidate_exposure": candidate_rows,
        "selection_summary": {
            "recommended_guarded_variant": best.get("variant_name") if best else None,
            "recommended_guarded_variant_status": best.get("variant_status") if best else None,
            "recommended_guarded_variant_score": best.get("policy_score") if best else None,
            "recommended_variant_risk_delta_vs_raw_bps": (
                best.get("variant_vs_raw_top_risk_adjusted_return_delta_bps") if best else None
            ),
            "recommended_variant_hit_delta_vs_raw": best.get("variant_vs_raw_top_hit_rate_delta") if best else None,
            "recommended_variant_risk_delta_improvement_vs_baseline_bps": _round_or_none(delta_vs_baseline, 6),
            "evaluated_variant_count": int(len(variant_rows)),
        },
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    return _sanitize_value(report)


def render_segmented_probability_guarded_ev_experiment_markdown(report: dict[str, Any]) -> str:
    """Render guarded EV experiment results as Markdown."""
    selection = report.get("selection_summary", {}) or {}
    lines = [
        "# Segmented Probability Guarded EV Experiment",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Status: `{report.get('guarded_ev_status')}`",
        f"- Analysis policy: `{report.get('analysis_policy')}`",
        f"- Simulation mode: `{report.get('simulation_mode')}`",
        f"- Recommended guarded variant: `{selection.get('recommended_guarded_variant')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Variants",
        "",
        "| Variant | Status | Risk Delta vs Raw | Hit Delta vs Raw | Raw Overlap | Baseline Overlap | Neg Route Top Rate | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("variant_results", []) or []:
        lines.append(
            f"| `{row.get('variant_name')}` | `{row.get('variant_status')}` | "
            f"{row.get('variant_vs_raw_top_risk_adjusted_return_delta_bps')} | "
            f"{row.get('variant_vs_raw_top_hit_rate_delta')} | "
            f"{row.get('variant_top_raw_top_overlap_rate')} | "
            f"{row.get('variant_top_baseline_shadow_overlap_rate')} | "
            f"{row.get('ev_negative_route_top_rate')} | {row.get('policy_score')} |"
        )
    lines.extend(["", "## Quarantined Routes", ""])
    for key in report.get("quarantined_candidate_keys", []) or ["None"]:
        lines.append(f"- `{key}`")
    lines.extend(
        [
            "",
            "## Candidate Exposure",
            "",
            "| Variant | Candidate | Top Share | N | Avg Risk-Adj Return | Hit Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in (report.get("candidate_exposure", []) or [])[:30]:
        lines.append(
            f"| `{row.get('variant_name')}` | `{row.get('candidate_key')}` | "
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
        "variants_csv_path": output / f"{stem}_variants.csv",
        "candidates_csv_path": output / f"{stem}_candidates.csv",
        "latest_json_path": output / SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_MARKDOWN_FILENAME,
        "latest_variants_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_VARIANTS_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_CANDIDATES_FILENAME,
    }


def write_segmented_probability_guarded_ev_experiment_report(
    ev_shadow_report: dict[str, Any],
    ev_rejection_attribution: dict[str, Any],
    routes: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write guarded EV experiment artifacts."""
    report = build_segmented_probability_guarded_ev_experiment_report(
        ev_shadow_report,
        ev_rejection_attribution,
        routes,
        **kwargs,
    )
    assert_artifact_schema(report, "segmented_probability_guarded_ev_experiment")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_EV_EXPERIMENT_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_guarded_ev_experiment"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_guarded_ev_experiment_markdown(report)
    variants = pd.json_normalize(report.get("variant_results", []) or [])
    candidates = pd.DataFrame(report.get("candidate_exposure", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(variants, paths["variants_csv_path"])
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(variants, paths["latest_variants_csv_path"])
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_guarded_ev_experiment_report_from_paths(
    *,
    ev_shadow_report_path: str | Path | None = None,
    ev_rejection_attribution_path: str | Path | None = None,
    ev_shadow_routes_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load rejected EV artifacts and write guarded EV experiment evidence."""
    shadow_path = Path(ev_shadow_report_path) if ev_shadow_report_path is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_REPORT_PATH
    attribution_path = (
        Path(ev_rejection_attribution_path)
        if ev_rejection_attribution_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_EV_REJECTION_ATTRIBUTION_PATH
    )
    routes_path = Path(ev_shadow_routes_path) if ev_shadow_routes_path is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_ROUTES_PATH
    ev_shadow_report = _read_json(shadow_path)
    ev_rejection_attribution = _read_json(attribution_path)
    routes = _load_routes(routes_path)
    return write_segmented_probability_guarded_ev_experiment_report(
        ev_shadow_report,
        ev_rejection_attribution,
        routes,
        ev_shadow_report_path=shadow_path,
        ev_rejection_attribution_path=attribution_path,
        ev_shadow_routes_path=routes_path,
        **kwargs,
    )
