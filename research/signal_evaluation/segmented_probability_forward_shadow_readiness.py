"""Readiness gate for segmented probability forward-shadow candidates.

This module reads the latest forward-shadow and accumulation artifacts and
emits a hard research/ops decision. It never changes runtime configuration,
parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import (
    assert_artifact_schema,
    validate_artifact_schema,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR,
    FORWARD_SHADOW_PASS,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_JSON_FILENAME,
)
from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR,
    EV_SHADOW_NEEDS_MORE_DATA,
    EV_SHADOW_NO_CANDIDATE_ROUTES,
    EV_SHADOW_REJECTED,
    SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME,
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (
    ACCUMULATION_TRUE_FORWARD_PASS,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
)
from research.signal_evaluation.segmented_probability_candidate_staleness import (
    ACTIVE_REVIEW,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR,
    EXPIRED,
    SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME,
    SUPERSEDED,
)
from research.signal_evaluation.segmented_probability_guarded_shadow_validation import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR,
    GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA,
    GUARDED_SHADOW_VALIDATION_PASS,
    GUARDED_SHADOW_VALIDATION_REJECTED,
    SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME,
)
from research.signal_evaluation.signal_quality_model_audit import (
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_forward_shadow_readiness"
)
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    / SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_REPORT_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR / SEGMENTED_PROBABILITY_FORWARD_SHADOW_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    / SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR
    / SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR
    / SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR
    / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME
)

SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_JSON_FILENAME = (
    "latest_segmented_probability_forward_shadow_readiness.json"
)
SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_MARKDOWN_FILENAME = (
    "latest_segmented_probability_forward_shadow_readiness.md"
)

FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW = "FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW"
FORWARD_SHADOW_READINESS_BLOCKED = "FORWARD_SHADOW_READINESS_BLOCKED"

ROUTE_REGRESSION_STATUSES = {
    "CANDIDATE_ROUTE_REGRESSED_BRIER",
    "CANDIDATE_ROUTE_REGRESSED_ECE",
}


def load_json_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_history_frame(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", ""}
    return value in {0, None}


def _candidate_age_days(candidate_generated_at: Any, *, as_of: Any = None) -> float | None:
    if not candidate_generated_at:
        return None
    candidate_ts = pd.to_datetime(candidate_generated_at, errors="coerce", utc=True)
    if pd.isna(candidate_ts):
        return None
    as_of_ts = pd.to_datetime(as_of, errors="coerce", utc=True) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(as_of_ts):
        as_of_ts = pd.Timestamp.now(tz="UTC")
    return max(float((as_of_ts - candidate_ts).total_seconds()) / 86400.0, 0.0)


def _policy_row(forward_shadow_report: dict[str, Any], policy: str | None) -> dict[str, Any]:
    if not policy:
        return {}
    for row in forward_shadow_report.get("routing_policy_results", []) or []:
        if str(row.get("route_policy")) == str(policy):
            return row if isinstance(row, dict) else {}
    return {}


def _candidate_route_regressions(forward_shadow_report: dict[str, Any]) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for row in forward_shadow_report.get("candidate_route_results", []) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("candidate_status")) in ROUTE_REGRESSION_STATUSES:
            regressions.append(row)
    return regressions


def _stable_policy_summary(
    history: pd.DataFrame,
    latest: dict[str, Any],
    *,
    recommended_policy: str | None,
    min_forward_sample: int,
    policy_lookback_runs: int,
    min_policy_stability_runs: int,
) -> dict[str, Any]:
    if history is None or history.empty:
        frame = pd.DataFrame([latest]) if latest else pd.DataFrame()
    else:
        frame = history.copy()
    if frame.empty or not recommended_policy:
        return {
            "routing_policy_stable": False,
            "stable_policy_run_count": 0,
            "conflicting_policy_count": 0,
            "policy_lookback_runs": int(policy_lookback_runs),
            "min_policy_stability_runs": int(min_policy_stability_runs),
        }

    lookback = frame.tail(max(int(policy_lookback_runs), 1)).copy()
    mode = lookback.get("validation_mode_used", pd.Series(dtype=str)).fillna("").astype(str)
    strict = pd.to_numeric(lookback.get("strict_forward_row_count", pd.Series(dtype=float)), errors="coerce")
    status = lookback.get("accumulation_status", pd.Series(dtype=str)).fillna("").astype(str)
    policy = lookback.get("recommended_routing_policy", pd.Series(dtype=str)).fillna("").astype(str)
    eligible = lookback.loc[
        (mode == "after_candidate_generated")
        & (strict >= int(min_forward_sample))
        & (status == ACCUMULATION_TRUE_FORWARD_PASS)
    ].copy()
    policy_values = eligible.get("recommended_routing_policy", pd.Series(dtype=str)).fillna("").astype(str)
    same_count = int((policy_values == str(recommended_policy)).sum())
    conflicting_count = int(((policy_values != str(recommended_policy)) & (policy_values != "")).sum())
    stable = same_count >= int(min_policy_stability_runs) and conflicting_count == 0
    return {
        "routing_policy_stable": bool(stable),
        "stable_policy_run_count": same_count,
        "conflicting_policy_count": conflicting_count,
        "eligible_true_forward_history_count": int(len(eligible)),
        "policy_lookback_runs": int(policy_lookback_runs),
        "min_policy_stability_runs": int(min_policy_stability_runs),
        "recommended_policy": recommended_policy,
    }


def _side_effects_absent(accumulation_latest: dict[str, Any], forward_shadow_report: dict[str, Any]) -> bool:
    fields = ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    return all(_is_false(accumulation_latest.get(field)) for field in fields) and all(
        _is_false(forward_shadow_report.get(field)) for field in fields
    )


def _candidate_staleness_gate(candidate_staleness_report: dict[str, Any]) -> dict[str, Any]:
    report = candidate_staleness_report if isinstance(candidate_staleness_report, dict) else {}
    schema = validate_artifact_schema(report, "segmented_probability_candidate_staleness")
    candidate = report.get("candidate_summary", {}) if isinstance(report.get("candidate_summary"), dict) else {}
    shift = (
        report.get("forward_label_population_shift", {})
        if isinstance(report.get("forward_label_population_shift"), dict)
        else {}
    )
    routing = (
        report.get("routing_policy_stability", {})
        if isinstance(report.get("routing_policy_stability"), dict)
        else {}
    )
    supersession = report.get("supersession", {}) if isinstance(report.get("supersession"), dict) else {}
    checked = report.get("checked_conditions", {}) if isinstance(report.get("checked_conditions"), dict) else {}
    status = str(report.get("staleness_status") or "")
    superseded = bool(supersession.get("superseded") or checked.get("candidate_bundle_superseded"))
    shifted = bool(shift.get("shifted_materially") or checked.get("forward_label_population_shifted"))
    routing_changed = bool(routing.get("routing_policy_changed")) or checked.get("routing_policy_stable") is False
    side_effects_absent = all(
        _is_false(report.get(field))
        for field in ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    )
    return {
        "candidate_staleness_schema_passed": schema.get("schema_status") == "PASS",
        "candidate_staleness_schema_validation": schema,
        "candidate_staleness_status": status,
        "candidate_staleness_status_active_review": status == ACTIVE_REVIEW,
        "candidate_staleness_reasons": report.get("staleness_reasons", []) or [],
        "candidate_staleness_generated_at": report.get("generated_at"),
        "candidate_staleness_candidate_generated_at": candidate.get("candidate_generated_at"),
        "candidate_staleness_candidate_age_days": candidate.get("candidate_age_days"),
        "candidate_staleness_candidate_count": _safe_int(candidate.get("candidate_count")),
        "candidate_bundle_not_expired": status != EXPIRED,
        "candidate_bundle_not_superseded": not superseded and status != SUPERSEDED,
        "candidate_forward_label_population_stable": not shifted,
        "candidate_staleness_routing_policy_stable": not routing_changed,
        "candidate_staleness_side_effects_absent": bool(side_effects_absent),
        "candidate_staleness_latest_routing_policy": routing.get("latest_recommended_routing_policy"),
        "candidate_staleness_post_candidate_label_count": _safe_int(
            shift.get("post_candidate_label_count")
        ),
        "candidate_staleness_shift_status": shift.get("shift_status"),
        "candidate_staleness_supersession_status": supersession.get("candidate_bundle_search_status"),
    }


def _ev_policy_row(ev_shadow_report: dict[str, Any], policy: str | None) -> dict[str, Any]:
    if not policy:
        return {}
    for row in ev_shadow_report.get("policy_results", []) or []:
        if isinstance(row, dict) and str(row.get("route_policy")) == str(policy):
            return row
    return {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _ev_shadow_gate(ev_shadow_report: dict[str, Any]) -> dict[str, Any]:
    report = ev_shadow_report if isinstance(ev_shadow_report, dict) else {}
    schema = validate_artifact_schema(report, "segmented_probability_ev_shadow_evaluation")
    selection = report.get("selection_summary", {}) if isinstance(report.get("selection_summary"), dict) else {}
    status = str(report.get("ev_shadow_status") or "")
    policy = selection.get("recommended_routing_policy")
    policy_row = _ev_policy_row(report, str(policy) if policy else None)
    risk_delta = _first_float(
        selection.get("recommended_policy_risk_adjusted_return_delta_bps"),
        policy_row.get("shadow_vs_raw_top_risk_adjusted_return_delta_bps"),
    )
    hit_delta = _first_float(
        selection.get("recommended_policy_hit_rate_delta"),
        policy_row.get("shadow_vs_raw_top_hit_rate_delta"),
    )
    liquidity_status = str(policy_row.get("liquidity_status") or "")
    candidate_rows = [
        row
        for row in report.get("candidate_route_results", []) or []
        if isinstance(row, dict) and (not policy or str(row.get("route_policy")) == str(policy))
    ]
    negative_candidate_rows = [
        row for row in candidate_rows if str(row.get("ev_route_status") or "") == "EV_ROUTE_NEGATIVE"
    ]
    side_effects_absent = all(
        _is_false(report.get(field))
        for field in ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    )
    return {
        "ev_shadow_schema_passed": schema.get("schema_status") == "PASS",
        "ev_shadow_schema_validation": schema,
        "ev_shadow_status": status,
        "ev_shadow_status_not_rejected": status != EV_SHADOW_REJECTED,
        "ev_shadow_has_sufficient_evidence": status not in {
            EV_SHADOW_NEEDS_MORE_DATA,
            EV_SHADOW_NO_CANDIDATE_ROUTES,
            "",
        },
        "ev_shadow_recommended_routing_policy": policy,
        "ev_shadow_recommended_policy_status": selection.get("recommended_policy_status"),
        "ev_shadow_recommended_policy_score": selection.get("recommended_policy_score"),
        "ev_shadow_policy_status_reason": policy_row.get("status_reason"),
        "ev_shadow_top_bucket_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "ev_shadow_top_bucket_hit_rate_delta": _round_or_none(hit_delta, 6),
        "ev_shadow_top_bucket_risk_adjusted_return_not_regressed": (
            risk_delta is not None and risk_delta >= 0.0
        ),
        "ev_shadow_top_bucket_hit_rate_not_regressed": hit_delta is not None and hit_delta >= 0.0,
        "ev_shadow_liquidity_status": liquidity_status or None,
        "ev_shadow_liquidity_status_ok": liquidity_status == "OK",
        "ev_shadow_key_candidate_negative_route_count": int(len(negative_candidate_rows)),
        "ev_shadow_key_candidate_negative_route_sample_count": int(
            sum(_safe_int(row.get("sample_count")) for row in negative_candidate_rows)
        ),
        "ev_shadow_key_candidate_routes_non_negative": len(negative_candidate_rows) == 0,
        "ev_shadow_side_effects_absent": bool(side_effects_absent),
    }


def _guarded_policy_row(guarded_shadow_report: dict[str, Any], policy: str | None) -> dict[str, Any]:
    if not policy:
        return {}
    for row in guarded_shadow_report.get("policy_results", []) or []:
        if isinstance(row, dict) and str(row.get("route_policy")) == str(policy):
            return row
    return {}


def _guarded_shadow_gate(guarded_shadow_report: dict[str, Any]) -> dict[str, Any]:
    report = guarded_shadow_report if isinstance(guarded_shadow_report, dict) else {}
    schema = validate_artifact_schema(report, "segmented_probability_guarded_shadow_validation")
    selection = report.get("selection_summary", {}) if isinstance(report.get("selection_summary"), dict) else {}
    window = report.get("validation_window", {}) if isinstance(report.get("validation_window"), dict) else {}
    status = str(report.get("guarded_shadow_status") or "")
    policy = selection.get("recommended_routing_policy")
    policy_row = _guarded_policy_row(report, str(policy) if policy else None)
    risk_delta = _first_float(
        selection.get("recommended_policy_risk_delta_vs_raw_bps"),
        policy_row.get("guarded_vs_raw_top_risk_adjusted_return_delta_bps"),
    )
    hit_delta = _first_float(
        selection.get("recommended_policy_hit_delta_vs_raw"),
        policy_row.get("guarded_vs_raw_top_hit_rate_delta"),
    )
    quarantined_top_count = _safe_int(policy_row.get("quarantined_route_top_count"))
    quarantined_top_rate = _safe_float(policy_row.get("quarantined_route_top_rate"))
    side_effects_absent = all(
        _is_false(report.get(field))
        for field in ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    )
    bundle_side_effects_absent = bool(report.get("guarded_bundle_side_effect_flags_clean"))
    return {
        "guarded_shadow_schema_passed": schema.get("schema_status") == "PASS",
        "guarded_shadow_schema_validation": schema,
        "guarded_shadow_status": status,
        "guarded_shadow_status_passed": status == GUARDED_SHADOW_VALIDATION_PASS,
        "guarded_shadow_has_sufficient_evidence": status not in {
            GUARDED_SHADOW_VALIDATION_NEEDS_MORE_DATA,
            "",
        },
        "guarded_shadow_status_not_rejected": status != GUARDED_SHADOW_VALIDATION_REJECTED,
        "guarded_shadow_recommended_routing_policy": policy,
        "guarded_shadow_recommended_policy_status": selection.get("recommended_policy_status"),
        "guarded_shadow_recommended_policy_score": selection.get("recommended_policy_score"),
        "guarded_shadow_policy_status_reason": policy_row.get("status_reason"),
        "guarded_shadow_validation_mode_used": window.get("validation_mode_used"),
        "guarded_shadow_strict_forward_row_count": _safe_int(window.get("strict_forward_row_count")),
        "guarded_shadow_holdout_replay_row_count": _safe_int(window.get("holdout_replay_row_count")),
        "guarded_shadow_top_bucket_risk_adjusted_return_delta_bps": _round_or_none(risk_delta, 6),
        "guarded_shadow_top_bucket_hit_rate_delta": _round_or_none(hit_delta, 6),
        "guarded_shadow_top_bucket_risk_adjusted_return_not_regressed": (
            risk_delta is not None and risk_delta >= 0.0
        ),
        "guarded_shadow_top_bucket_hit_rate_not_regressed": hit_delta is not None and hit_delta >= 0.0,
        "guarded_shadow_quarantined_route_top_count": quarantined_top_count,
        "guarded_shadow_quarantined_route_top_rate": _round_or_none(quarantined_top_rate, 6),
        "guarded_shadow_quarantined_route_top_exposure_zero": quarantined_top_count == 0,
        "guarded_shadow_rank_preservation_policy_present": bool(report.get("rank_preservation_policy_present")),
        "guarded_shadow_bundle_side_effects_absent": bool(bundle_side_effects_absent),
        "guarded_shadow_side_effects_absent": bool(side_effects_absent and bundle_side_effects_absent),
        "guarded_shadow_candidate_bundle_research_only": bool(report.get("guarded_bundle_research_only")),
        "guarded_shadow_candidate_bundle_approval_required": bool(
            report.get("guarded_bundle_approval_required_for_runtime_use")
        ),
        "guarded_shadow_candidate_count": _safe_int(report.get("candidate_count")),
    }


def _readiness_actions(readiness_status: str, reasons: list[str]) -> list[str]:
    if readiness_status == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW:
        return [
            "Open manual calibration-governance review for the recommended routing policy and candidate routes.",
            "Keep runtime probabilities unchanged until an explicit human-controlled adoption workflow approves a bundle.",
        ]
    actions: list[str] = []
    joined = " ".join(reasons)
    if "true_forward" in joined or "after_candidate_generated" in joined:
        actions.append("Continue running the forward-shadow accumulator until true post-candidate labels reach the gate.")
    if "sample" in joined:
        actions.append("Wait for more quality-approved post-candidate labels before manual review.")
    if "schema" in joined:
        actions.append("Fix artifact schema drift before relying on the readiness gate.")
    if "stale" in joined:
        actions.append("Regenerate the segmented calibration candidate bundle before further review.")
    if "expired" in joined:
        actions.append("Regenerate the candidate bundle before manual review; the current bundle is expired.")
    if "superseded" in joined:
        actions.append("Use the newer candidate bundle artifact before continuing manual review.")
    if "population_shifted" in joined:
        actions.append("Refresh calibration evidence because forward-label behavior shifted materially.")
    if "regression" in joined:
        actions.append("Do not advance the bundle while candidate route regressions are present.")
    if "ev_shadow_status_rejected" in joined or "ev_shadow_top_bucket" in joined:
        actions.append("Do not advance manual review until EV/risk shadow evidence no longer rejects the candidate.")
    if "ev_shadow_negative_candidate_routes" in joined:
        actions.append("Inspect and prune candidate routes with negative risk-adjusted payoff before another review.")
    if "ev_shadow_liquidity" in joined:
        actions.append("Review spread/liquidity quality before trusting EV/risk shadow evidence.")
    if "guarded_shadow" in joined:
        actions.append("Do not advance manual review until guard-aware shadow validation passes all guarded EV checks.")
    if "guarded_candidate_bundle" in joined:
        actions.append("Confirm the guarded candidate bundle remains research-only and approval-gated.")
    if not actions:
        actions.append("Keep the calibration bundle in shadow research; readiness requirements are not all satisfied.")
    actions.append("No runtime config, parameter pack, data source, or execution behavior should be changed by this gate.")
    return actions


def build_segmented_probability_forward_shadow_readiness_report(
    *,
    accumulation_dashboard: dict[str, Any],
    forward_shadow_report: dict[str, Any],
    candidate_staleness_report: dict[str, Any] | None = None,
    ev_shadow_report: dict[str, Any] | None = None,
    guarded_shadow_report: dict[str, Any] | None = None,
    history: pd.DataFrame | None = None,
    accumulation_dashboard_path: str | Path | None = None,
    forward_shadow_report_path: str | Path | None = None,
    candidate_staleness_path: str | Path | None = None,
    ev_shadow_path: str | Path | None = None,
    guarded_shadow_path: str | Path | None = None,
    history_path: str | Path | None = None,
    min_forward_sample: int = 100,
    min_policy_stability_runs: int = 1,
    policy_lookback_runs: int = 5,
    max_candidate_age_days: float = 14.0,
    allow_holdout_replay_guarded_validation: bool = False,
    as_of: Any = None,
) -> dict[str, Any]:
    """Build a hard readiness gate report for manual calibration review."""
    accumulation = accumulation_dashboard if isinstance(accumulation_dashboard, dict) else {}
    forward = forward_shadow_report if isinstance(forward_shadow_report, dict) else {}
    latest = accumulation.get("latest", {}) if isinstance(accumulation.get("latest"), dict) else {}
    selection = forward.get("selection_summary", {}) if isinstance(forward.get("selection_summary"), dict) else {}
    window = forward.get("validation_window", {}) if isinstance(forward.get("validation_window"), dict) else {}

    accumulation_schema = validate_artifact_schema(accumulation, "segmented_probability_forward_shadow_accumulation")
    forward_schema = validate_artifact_schema(forward, "segmented_probability_forward_shadow")
    staleness_gate = _candidate_staleness_gate(candidate_staleness_report or {})
    ev_gate = _ev_shadow_gate(ev_shadow_report or {})
    guarded_gate = _guarded_shadow_gate(guarded_shadow_report or {})

    recommended_policy = (
        guarded_gate.get("guarded_shadow_recommended_routing_policy")
        or selection.get("recommended_routing_policy")
        or latest.get("recommended_routing_policy")
    )
    recommended_policy_row = _policy_row(forward, str(recommended_policy) if recommended_policy else None)
    strict_forward = _safe_int(window.get("strict_forward_row_count", latest.get("strict_forward_row_count")))
    candidate_generated_at = window.get("candidate_generated_at") or latest.get("candidate_generated_at")
    age_days = _candidate_age_days(candidate_generated_at, as_of=as_of)
    regressions = _candidate_route_regressions(forward)
    policy_stability = _stable_policy_summary(
        history if history is not None else pd.DataFrame(),
        latest,
        recommended_policy=str(recommended_policy) if recommended_policy else None,
        min_forward_sample=min_forward_sample,
        policy_lookback_runs=policy_lookback_runs,
        min_policy_stability_runs=min_policy_stability_runs,
    )
    route_policy_regressions = _safe_int(recommended_policy_row.get("candidate_regression_count"))
    candidate_fresh = age_days is not None and age_days <= float(max_candidate_age_days)
    side_effects_absent = _side_effects_absent(latest, forward)
    validation_mode_used = str(window.get("validation_mode_used") or latest.get("validation_mode_used") or "")
    shadow_status = str(forward.get("shadow_validation_status") or latest.get("shadow_validation_status") or "")
    accumulation_status = str(latest.get("accumulation_status") or "")
    recommended_policy_status = str(
        guarded_gate.get("guarded_shadow_recommended_policy_status")
        or selection.get("recommended_policy_status")
        or latest.get("recommended_policy_status")
        or recommended_policy_row.get("shadow_status")
        or ""
    )
    guarded_holdout_sample = _safe_int(guarded_gate.get("guarded_shadow_holdout_replay_row_count"))
    guarded_mode = str(guarded_gate.get("guarded_shadow_validation_mode_used") or "")
    guarded_holdout_allowed = bool(
        allow_holdout_replay_guarded_validation
        and guarded_mode == "holdout_replay"
        and guarded_gate.get("guarded_shadow_status_passed")
    )

    conditions = {
        "accumulation_schema_passed": accumulation_schema.get("schema_status") == "PASS",
        "forward_shadow_schema_passed": forward_schema.get("schema_status") == "PASS",
        "validation_mode_used": validation_mode_used,
        "validation_mode_is_true_forward": validation_mode_used == "after_candidate_generated",
        "allow_holdout_replay_guarded_validation": bool(allow_holdout_replay_guarded_validation),
        "validation_mode_requirement_met": (
            validation_mode_used == "after_candidate_generated" or bool(guarded_holdout_allowed)
        ),
        "strict_forward_row_count": int(strict_forward),
        "min_forward_sample": int(min_forward_sample),
        "sufficient_forward_sample": strict_forward >= int(min_forward_sample),
        "sufficient_forward_or_guarded_holdout_sample": (
            strict_forward >= int(min_forward_sample)
            or bool(guarded_holdout_allowed and guarded_holdout_sample >= int(min_forward_sample))
        ),
        "accumulation_status": accumulation_status,
        "accumulation_status_passed": accumulation_status == ACCUMULATION_TRUE_FORWARD_PASS,
        "accumulation_or_guarded_holdout_passed": (
            accumulation_status == ACCUMULATION_TRUE_FORWARD_PASS or bool(guarded_holdout_allowed)
        ),
        "shadow_validation_status": shadow_status,
        "shadow_validation_passed": shadow_status == FORWARD_SHADOW_PASS,
        "shadow_or_guarded_validation_passed": (
            shadow_status == FORWARD_SHADOW_PASS or bool(guarded_gate.get("guarded_shadow_status_passed"))
        ),
        "recommended_routing_policy": recommended_policy,
        "recommended_policy_status": recommended_policy_status,
        "recommended_policy_passed": recommended_policy_status == FORWARD_SHADOW_PASS,
        "recommended_policy_or_guarded_policy_passed": (
            recommended_policy_status == FORWARD_SHADOW_PASS
            or bool(guarded_gate.get("guarded_shadow_status_passed"))
        ),
        "routing_policy_stable": bool(policy_stability.get("routing_policy_stable")),
        "routing_policy_requirement_met": bool(policy_stability.get("routing_policy_stable"))
        or bool(guarded_holdout_allowed),
        "policy_stability": policy_stability,
        "candidate_route_regression_count": int(len(regressions)),
        "recommended_policy_candidate_regression_count": int(route_policy_regressions),
        "candidate_routes_clean": len(regressions) == 0 and route_policy_regressions == 0,
        "side_effects_absent": bool(side_effects_absent),
        "candidate_count": _safe_int(forward.get("candidate_count", latest.get("candidate_count"))),
        "candidate_count_positive": _safe_int(forward.get("candidate_count", latest.get("candidate_count"))) > 0,
        "candidate_generated_at": candidate_generated_at,
        "candidate_age_days": _round_or_none(age_days, 6),
        "max_candidate_age_days": float(max_candidate_age_days),
        "candidate_bundle_fresh": bool(candidate_fresh),
        "accumulation_schema_validation": accumulation_schema,
        "forward_shadow_schema_validation": forward_schema,
    }
    conditions.update(staleness_gate)
    conditions.update(ev_gate)
    conditions.update(guarded_gate)

    readiness_reasons: list[str] = []
    checks = {
        "accumulation_schema_passed": "accumulation_schema_failed",
        "forward_shadow_schema_passed": "forward_shadow_schema_failed",
        "candidate_staleness_schema_passed": "candidate_staleness_schema_failed",
        "guarded_shadow_schema_passed": "guarded_shadow_schema_failed",
        "validation_mode_requirement_met": "validation_mode_not_after_candidate_generated",
        "sufficient_forward_or_guarded_holdout_sample": "insufficient_true_forward_sample",
        "accumulation_or_guarded_holdout_passed": "accumulation_status_not_true_forward_pass",
        "shadow_or_guarded_validation_passed": "forward_shadow_status_not_passed",
        "recommended_policy_or_guarded_policy_passed": "recommended_policy_status_not_passed",
        "routing_policy_requirement_met": "recommended_routing_policy_not_stable",
        "candidate_routes_clean": "candidate_route_regression_detected",
        "side_effects_absent": "runtime_or_parameter_side_effect_detected",
        "candidate_count_positive": "no_candidate_routes_available",
        "candidate_bundle_fresh": "candidate_bundle_stale_or_missing_timestamp",
        "candidate_staleness_status_active_review": "candidate_staleness_status_not_active_review",
        "candidate_bundle_not_expired": "candidate_bundle_expired",
        "candidate_bundle_not_superseded": "candidate_bundle_superseded",
        "candidate_forward_label_population_stable": "candidate_forward_label_population_shifted",
        "candidate_staleness_routing_policy_stable": "candidate_staleness_routing_policy_unstable",
        "candidate_staleness_side_effects_absent": "candidate_staleness_side_effect_detected",
        "guarded_shadow_status_passed": "guarded_shadow_status_not_passed",
        "guarded_shadow_has_sufficient_evidence": "guarded_shadow_insufficient_evidence",
        "guarded_shadow_status_not_rejected": "guarded_shadow_status_rejected",
        "guarded_shadow_side_effects_absent": "guarded_shadow_side_effect_detected",
        "guarded_shadow_rank_preservation_policy_present": "guarded_shadow_rank_policy_missing",
        "guarded_shadow_quarantined_route_top_exposure_zero": "guarded_shadow_quarantined_route_top_exposure",
        "guarded_shadow_top_bucket_risk_adjusted_return_not_regressed": (
            "guarded_shadow_top_bucket_risk_adjusted_return_regressed"
        ),
        "guarded_shadow_top_bucket_hit_rate_not_regressed": "guarded_shadow_top_bucket_hit_rate_regressed",
        "guarded_shadow_candidate_bundle_research_only": "guarded_candidate_bundle_not_research_only",
        "guarded_shadow_candidate_bundle_approval_required": "guarded_candidate_bundle_not_approval_gated",
    }
    if not conditions.get("guarded_shadow_schema_passed"):
        checks.update(
            {
                "ev_shadow_schema_passed": "ev_shadow_schema_failed",
                "ev_shadow_status_not_rejected": "ev_shadow_status_rejected",
                "ev_shadow_has_sufficient_evidence": "ev_shadow_insufficient_evidence",
                "ev_shadow_top_bucket_risk_adjusted_return_not_regressed": (
                    "ev_shadow_top_bucket_risk_adjusted_return_regressed"
                ),
                "ev_shadow_top_bucket_hit_rate_not_regressed": "ev_shadow_top_bucket_hit_rate_regressed",
                "ev_shadow_liquidity_status_ok": "ev_shadow_liquidity_watch",
                "ev_shadow_key_candidate_routes_non_negative": "ev_shadow_negative_candidate_routes_detected",
                "ev_shadow_side_effects_absent": "ev_shadow_side_effect_detected",
            }
        )
    for condition, reason in checks.items():
        if not conditions.get(condition):
            readiness_reasons.append(reason)

    readiness_status = (
        FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW
        if not readiness_reasons
        else FORWARD_SHADOW_READINESS_BLOCKED
    )
    report = {
        "report_type": "segmented_probability_forward_shadow_readiness",
        "generated_at": _utc_now(),
        "accumulation_dashboard_path": str(accumulation_dashboard_path) if accumulation_dashboard_path is not None else None,
        "forward_shadow_report_path": str(forward_shadow_report_path) if forward_shadow_report_path is not None else None,
        "candidate_staleness_path": str(candidate_staleness_path) if candidate_staleness_path is not None else None,
        "ev_shadow_path": str(ev_shadow_path) if ev_shadow_path is not None else None,
        "guarded_shadow_path": str(guarded_shadow_path) if guarded_shadow_path is not None else None,
        "history_path": str(history_path) if history_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "readiness_status": readiness_status,
        "readiness_reasons": readiness_reasons,
        "checked_conditions": conditions,
        "recommended_next_actions": _readiness_actions(readiness_status, readiness_reasons),
    }
    return _sanitize_value(report)


def render_segmented_probability_forward_shadow_readiness_markdown(report: dict[str, Any]) -> str:
    """Render forward-shadow readiness gate output as Markdown."""
    checks = report.get("checked_conditions", {}) or {}
    policy = checks.get("recommended_routing_policy")
    lines = [
        "# Segmented Probability Forward Shadow Readiness",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Readiness status: **{report.get('readiness_status')}**",
        f"- Recommended routing policy: `{policy}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Gate Checks",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Accumulation schema passed | {checks.get('accumulation_schema_passed')} |",
        f"| Forward-shadow schema passed | {checks.get('forward_shadow_schema_passed')} |",
        f"| Validation mode | `{checks.get('validation_mode_used')}` |",
        f"| True forward rows | {checks.get('strict_forward_row_count')} / {checks.get('min_forward_sample')} |",
        f"| Accumulation status | `{checks.get('accumulation_status')}` |",
        f"| Shadow validation status | `{checks.get('shadow_validation_status')}` |",
        f"| Recommended policy status | `{checks.get('recommended_policy_status')}` |",
        f"| Routing policy stable | {checks.get('routing_policy_stable')} |",
        f"| Candidate route regressions | {checks.get('candidate_route_regression_count')} |",
        f"| Candidate age days | {checks.get('candidate_age_days')} / {checks.get('max_candidate_age_days')} |",
        f"| Candidate staleness status | `{checks.get('candidate_staleness_status')}` |",
        f"| Candidate staleness active | {checks.get('candidate_staleness_status_active_review')} |",
        f"| Candidate superseded | {not checks.get('candidate_bundle_not_superseded')} |",
        f"| Forward-label population stable | {checks.get('candidate_forward_label_population_stable')} |",
        f"| Staleness routing policy stable | {checks.get('candidate_staleness_routing_policy_stable')} |",
        f"| EV shadow status | `{checks.get('ev_shadow_status')}` |",
        f"| EV top risk-adjusted delta | {checks.get('ev_shadow_top_bucket_risk_adjusted_return_delta_bps')} |",
        f"| EV top hit-rate delta | {checks.get('ev_shadow_top_bucket_hit_rate_delta')} |",
        f"| EV liquidity status | `{checks.get('ev_shadow_liquidity_status')}` |",
        f"| EV negative candidate routes | {checks.get('ev_shadow_key_candidate_negative_route_count')} |",
        f"| Guarded shadow status | `{checks.get('guarded_shadow_status')}` |",
        f"| Guarded validation mode | `{checks.get('guarded_shadow_validation_mode_used')}` |",
        f"| Guarded EV risk delta | {checks.get('guarded_shadow_top_bucket_risk_adjusted_return_delta_bps')} |",
        f"| Guarded EV hit-rate delta | {checks.get('guarded_shadow_top_bucket_hit_rate_delta')} |",
        f"| Guarded quarantined-route top exposure | {checks.get('guarded_shadow_quarantined_route_top_count')} |",
        f"| Guarded rank policy present | {checks.get('guarded_shadow_rank_preservation_policy_present')} |",
        f"| Guarded bundle research-only | {checks.get('guarded_shadow_candidate_bundle_research_only')} |",
        f"| Guarded bundle approval-gated | {checks.get('guarded_shadow_candidate_bundle_approval_required')} |",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("readiness_reasons", []) or ["ready"]:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("*Research-only artifact. It does not alter runtime config, parameter packs, data sources, or execution behavior.*")
    return "\n".join(lines)


def write_segmented_probability_forward_shadow_readiness_report(
    *,
    accumulation_dashboard_path: str | Path | None = None,
    forward_shadow_report_path: str | Path | None = None,
    candidate_staleness_path: str | Path | None = None,
    ev_shadow_path: str | Path | None = None,
    guarded_shadow_path: str | Path | None = None,
    history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_forward_sample: int = 100,
    min_policy_stability_runs: int = 1,
    policy_lookback_runs: int = 5,
    max_candidate_age_days: float = 14.0,
    allow_holdout_replay_guarded_validation: bool = False,
    as_of: Any = None,
) -> dict[str, Any]:
    """Read latest artifacts, build readiness gate report, and write JSON/Markdown."""
    accumulation_path = (
        Path(accumulation_dashboard_path)
        if accumulation_dashboard_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_PATH
    )
    shadow_path = (
        Path(forward_shadow_report_path)
        if forward_shadow_report_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_REPORT_PATH
    )
    stale_path = (
        Path(candidate_staleness_path)
        if candidate_staleness_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_PATH
    )
    ev_path = (
        Path(ev_shadow_path)
        if ev_shadow_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_PATH
    )
    guarded_path = (
        Path(guarded_shadow_path)
        if guarded_shadow_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_PATH
    )
    hist_path = (
        Path(history_path)
        if history_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH
    )
    accumulation = load_json_file(accumulation_path)
    forward = load_json_file(shadow_path)
    staleness = load_json_file(stale_path)
    ev_shadow = load_json_file(ev_path)
    guarded_shadow = load_json_file(guarded_path)
    history = load_history_frame(hist_path)
    report = build_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard=accumulation,
        forward_shadow_report=forward,
        candidate_staleness_report=staleness,
        ev_shadow_report=ev_shadow,
        guarded_shadow_report=guarded_shadow,
        history=history,
        accumulation_dashboard_path=accumulation_path,
        forward_shadow_report_path=shadow_path,
        candidate_staleness_path=stale_path,
        ev_shadow_path=ev_path,
        guarded_shadow_path=guarded_path,
        history_path=hist_path,
        min_forward_sample=min_forward_sample,
        min_policy_stability_runs=min_policy_stability_runs,
        policy_lookback_runs=policy_lookback_runs,
        max_candidate_age_days=max_candidate_age_days,
        allow_holdout_replay_guarded_validation=allow_holdout_replay_guarded_validation,
        as_of=as_of,
    )
    assert_artifact_schema(report, "segmented_probability_forward_shadow_readiness")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_JSON_FILENAME
    markdown_path = output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_segmented_probability_forward_shadow_readiness_markdown(report))
    return {
        "readiness_json_path": str(json_path),
        "readiness_markdown_path": str(markdown_path),
        "readiness_report": report,
    }
