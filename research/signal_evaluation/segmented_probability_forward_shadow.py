"""Forward-shadow validation for segmented probability calibration candidates.

The validator applies a candidate calibration bundle to quality-approved signal
rows using explicit routing policies. It compares raw versus shadow-calibrated
probabilities and writes advisory artifacts only; it never changes runtime
configuration, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import (
    DEFAULT_PROBABILITY_FIELD,
    _calibration_curve,
    _clean_probability_and_label_frame,
    _clip_probability,
    _metrics,
    _split_train_holdout,
)
from research.signal_evaluation.segmented_probability_calibration_experiment import (
    DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR,
    SEGMENTED_PROBABILITY_CALIBRATION_CANDIDATE_FILENAME,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_LABEL_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_forward_shadow"
)
DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_CALIBRATION_EXPERIMENT_DIR
    / SEGMENTED_PROBABILITY_CALIBRATION_CANDIDATE_FILENAME
)

SEGMENTED_PROBABILITY_FORWARD_SHADOW_JSON_FILENAME = "latest_segmented_probability_forward_shadow.json"
SEGMENTED_PROBABILITY_FORWARD_SHADOW_MARKDOWN_FILENAME = "latest_segmented_probability_forward_shadow.md"
SEGMENTED_PROBABILITY_FORWARD_SHADOW_POLICIES_FILENAME = "latest_segmented_probability_forward_shadow_policies.csv"
SEGMENTED_PROBABILITY_FORWARD_SHADOW_CANDIDATES_FILENAME = "latest_segmented_probability_forward_shadow_candidates.csv"
SEGMENTED_PROBABILITY_FORWARD_SHADOW_ROUTES_FILENAME = "latest_segmented_probability_forward_shadow_routes.csv"

DEFAULT_ROUTING_POLICIES = ("candidate_priority", "regime_first", "recency_first")

FORWARD_SHADOW_PASS = "FORWARD_SHADOW_VALIDATION_PASS"
SHADOW_REPLAY_PASS = "SHADOW_REPLAY_VALIDATION_PASS_FORWARD_DATA_PENDING"
FORWARD_SHADOW_WATCH = "FORWARD_SHADOW_VALIDATION_WATCH"
FORWARD_SHADOW_REJECTED = "FORWARD_SHADOW_VALIDATION_REJECTED"
NEEDS_MORE_FORWARD_DATA = "NEEDS_MORE_FORWARD_SHADOW_DATA"
NO_CANDIDATE_ROUTES = "NO_CANDIDATE_ROUTES"


def _load_candidate_bundle(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path)
    if not candidate_path.exists():
        return {
            "artifact_type": "segmented_probability_calibration_candidate_bundle",
            "generated_at": None,
            "candidate_count": 0,
            "candidates": [],
            "load_error": f"Candidate bundle not found: {candidate_path}",
        }
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def _safe_segment_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "UNKNOWN"
    except Exception:
        pass
    token = str(value).strip()
    return token if token else "UNKNOWN"


def _candidate_key(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate.get('candidate_type', 'unknown')}:"
        f"{candidate.get('segment_field', 'unknown')}={candidate.get('segment_value', 'unknown')}"
    )


def _candidate_matches(row: pd.Series, candidate: dict[str, Any]) -> bool:
    candidate_type = str(candidate.get("candidate_type") or "")
    if candidate_type == "recency_window":
        return True
    if candidate_type != "regime_segment":
        return False
    field = str(candidate.get("segment_field") or "")
    if not field or field not in row.index:
        return False
    return _safe_segment_value(row.get(field)) == str(candidate.get("segment_value") or "")


def _candidate_route_rank(candidate: dict[str, Any], *, routing_policy: str) -> tuple[int, int, str]:
    priority = int(candidate.get("candidate_priority") or 9999)
    candidate_type = str(candidate.get("candidate_type") or "")
    if routing_policy == "regime_first":
        type_rank = 0 if candidate_type == "regime_segment" else 1
    elif routing_policy == "recency_first":
        type_rank = 0 if candidate_type == "recency_window" else 1
    else:
        type_rank = 0
    return (type_rank, priority, _candidate_key(candidate))


def _select_candidate_for_row(
    row: pd.Series,
    candidates: list[dict[str, Any]],
    *,
    routing_policy: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    matches = [candidate for candidate in candidates if _candidate_matches(row, candidate)]
    match_keys = [_candidate_key(candidate) for candidate in matches]
    if not matches:
        return None, match_keys
    selected = sorted(matches, key=lambda item: _candidate_route_rank(item, routing_policy=routing_policy))[0]
    return selected, match_keys


def _interpolate_mapping(raw_probability: float, mapping: dict[str, Any]) -> float:
    if not mapping:
        return raw_probability
    points = sorted((float(key), float(value)) for key, value in mapping.items())
    score = float(raw_probability) * 100.0
    if score <= points[0][0]:
        return points[0][1]
    if score >= points[-1][0]:
        return points[-1][1]
    for idx in range(len(points) - 1):
        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]
        if x0 <= score <= x1:
            weight = (score - x0) / max(x1 - x0, 1.0)
            return y0 + weight * (y1 - y0)
    return raw_probability


def _apply_candidate_probability(raw_probability: float, candidate: dict[str, Any] | None) -> float:
    raw = float(_clip_probability(pd.Series([raw_probability])).iloc[0])
    if not candidate:
        return raw
    state = candidate.get("state") if isinstance(candidate.get("state"), dict) else {}
    method = str(state.get("method") or candidate.get("selected_calibrator") or "")
    if method == "linear_shrink":
        alpha = float(state.get("alpha", 1.0))
        base_rate = float(state.get("base_rate", raw))
        return float(_clip_probability(pd.Series([(alpha * raw) + ((1.0 - alpha) * base_rate)])).iloc[0])
    if method == "temperature_score":
        temperature = max(float(state.get("temperature", 1.0)), 0.1)
        scaled_score = 50.0 + ((raw * 100.0) - 50.0) / temperature
        return float(_clip_probability(pd.Series([scaled_score / 100.0])).iloc[0])
    if method == "isotonic_score":
        mapping = state.get("calibration_mapping") if isinstance(state.get("calibration_mapping"), dict) else {}
        return float(_clip_probability(pd.Series([_interpolate_mapping(raw, mapping)])).iloc[0])
    return raw


def _candidate_generated_at(bundle: dict[str, Any]) -> pd.Timestamp | None:
    raw = bundle.get("generated_at")
    if not raw:
        return None
    timestamp = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def _evaluation_frame(
    working: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    train_fraction: float,
    validation_mode: str,
    min_shadow_sample: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, holdout = _split_train_holdout(working, train_fraction=train_fraction)
    generated_at = _candidate_generated_at(bundle)
    future = working.iloc[0:0].copy()
    if generated_at is not None and "signal_timestamp" in working.columns:
        timestamps = pd.to_datetime(working["signal_timestamp"], errors="coerce", utc=True)
        future = working.loc[timestamps > generated_at].copy()

    requested = str(validation_mode or "auto").strip().lower()
    if requested not in {"auto", "after_candidate_generated", "holdout_replay"}:
        requested = "auto"

    if requested == "after_candidate_generated":
        used = "after_candidate_generated"
        selected = future
        fallback_reason = None
    elif requested == "holdout_replay":
        used = "holdout_replay"
        selected = holdout
        fallback_reason = None
    elif len(future) >= int(min_shadow_sample):
        used = "after_candidate_generated"
        selected = future
        fallback_reason = None
    else:
        used = "holdout_replay"
        selected = holdout
        fallback_reason = "insufficient_rows_after_candidate_generated"

    return selected.reset_index(drop=True), {
        "validation_mode_requested": requested,
        "validation_mode_used": used,
        "fallback_reason": fallback_reason,
        "candidate_generated_at": generated_at.isoformat() if generated_at is not None else None,
        "strict_forward_row_count": int(len(future)),
        "holdout_replay_row_count": int(len(holdout)),
        "train_count": int(len(train)),
    }


def _route_decisions(
    evaluation: pd.DataFrame,
    candidates: list[dict[str, Any]],
    *,
    routing_policy: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx, row in evaluation.reset_index(drop=True).iterrows():
        selected, matched_keys = _select_candidate_for_row(row, candidates, routing_policy=routing_policy)
        raw_probability = float(row.get("_probability"))
        shadow_probability = _apply_candidate_probability(raw_probability, selected)
        selected_key = _candidate_key(selected) if selected is not None else "unrouted_identity"
        records.append(
            {
                "route_policy": routing_policy,
                "row_index": int(idx),
                "signal_id": row.get("signal_id"),
                "signal_timestamp": row.get("signal_timestamp"),
                "direction": row.get("direction"),
                "raw_probability": raw_probability,
                "shadow_probability": shadow_probability,
                "label": float(row.get("_label")),
                "assigned_candidate_key": selected_key,
                "assigned_candidate_type": selected.get("candidate_type") if selected is not None else "none",
                "assigned_segment_field": selected.get("segment_field") if selected is not None else None,
                "assigned_segment_value": selected.get("segment_value") if selected is not None else None,
                "assigned_calibrator": selected.get("selected_calibrator") if selected is not None else "identity",
                "matched_candidate_count": int(len(matched_keys)),
                "matched_candidate_keys": "|".join(matched_keys),
            }
        )
    return pd.DataFrame(records)


def _metric_pair(
    decisions: pd.DataFrame,
    *,
    route_policy: str,
    min_shadow_sample: int,
    min_brier_improvement: float,
    max_ece_regression: float,
    n_bins: int,
    validation_mode_used: str,
    candidate_regression_count: int,
) -> dict[str, Any]:
    labels = pd.to_numeric(decisions.get("label", pd.Series(dtype=float)), errors="coerce")
    raw = pd.to_numeric(decisions.get("raw_probability", pd.Series(dtype=float)), errors="coerce")
    shadow = pd.to_numeric(decisions.get("shadow_probability", pd.Series(dtype=float)), errors="coerce")
    raw_metrics = _metrics(raw, labels, method="raw_probability", split_name=route_policy, n_bins=n_bins)
    shadow_metrics = _metrics(shadow, labels, method="shadow_probability", split_name=route_policy, n_bins=n_bins)
    sample_count = int(shadow_metrics.get("sample_count") or 0)
    assigned_count = int((decisions.get("assigned_candidate_key") != "unrouted_identity").sum()) if not decisions.empty else 0
    raw_brier = raw_metrics.get("brier_score")
    shadow_brier = shadow_metrics.get("brier_score")
    raw_ece = raw_metrics.get("expected_calibration_error")
    shadow_ece = shadow_metrics.get("expected_calibration_error")
    brier_improvement = None if raw_brier is None or shadow_brier is None else float(raw_brier) - float(shadow_brier)
    ece_change = None if raw_ece is None or shadow_ece is None else float(shadow_ece) - float(raw_ece)
    multiple_match_count = int((decisions.get("matched_candidate_count", pd.Series(dtype=int)) > 1).sum()) if not decisions.empty else 0

    if sample_count < int(min_shadow_sample):
        status = NEEDS_MORE_FORWARD_DATA
        reason = "sample_size_guardrail_failed"
    elif assigned_count <= 0:
        status = NO_CANDIDATE_ROUTES
        reason = "no_candidate_matched_evaluation_rows"
    elif brier_improvement is not None and brier_improvement < 0:
        status = FORWARD_SHADOW_REJECTED
        reason = "shadow_calibration_worsened_brier"
    elif ece_change is not None and ece_change > float(max_ece_regression):
        status = FORWARD_SHADOW_WATCH
        reason = "shadow_calibration_worsened_ece"
    elif candidate_regression_count > 0:
        status = FORWARD_SHADOW_WATCH
        reason = "one_or_more_candidate_routes_regressed"
    elif brier_improvement is not None and brier_improvement >= float(min_brier_improvement):
        status = SHADOW_REPLAY_PASS if validation_mode_used == "holdout_replay" else FORWARD_SHADOW_PASS
        reason = "shadow_calibration_improved_brier_without_ece_regression"
    else:
        status = FORWARD_SHADOW_WATCH
        reason = "shadow_calibration_did_not_clear_brier_improvement_guardrail"

    return {
        "route_policy": route_policy,
        "shadow_status": status,
        "status_reason": reason,
        "sample_count": sample_count,
        "assigned_candidate_count": assigned_count,
        "unrouted_count": max(sample_count - assigned_count, 0),
        "multiple_match_count": multiple_match_count,
        "multiple_match_rate": _round_or_none(multiple_match_count / max(sample_count, 1), 6),
        "raw_brier_score": raw_metrics.get("brier_score"),
        "shadow_brier_score": shadow_metrics.get("brier_score"),
        "brier_improvement": _round_or_none(brier_improvement, 8),
        "raw_expected_calibration_error": raw_metrics.get("expected_calibration_error"),
        "shadow_expected_calibration_error": shadow_metrics.get("expected_calibration_error"),
        "ece_change": _round_or_none(ece_change, 8),
        "raw_mean_predicted_probability": raw_metrics.get("mean_predicted_probability"),
        "shadow_mean_predicted_probability": shadow_metrics.get("mean_predicted_probability"),
        "actual_hit_rate": shadow_metrics.get("actual_hit_rate"),
        "raw_calibration_gap": raw_metrics.get("calibration_gap"),
        "shadow_calibration_gap": shadow_metrics.get("calibration_gap"),
        "candidate_regression_count": int(candidate_regression_count),
    }


def _candidate_route_metrics(
    decisions: pd.DataFrame,
    *,
    route_policy: str,
    min_candidate_sample: int,
    max_candidate_brier_regression: float,
    max_candidate_ece_regression: float,
    n_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return rows
    routed = decisions.loc[decisions["assigned_candidate_key"] != "unrouted_identity"].copy()
    if routed.empty:
        return rows
    for key, group in routed.groupby("assigned_candidate_key", dropna=False):
        labels = pd.to_numeric(group["label"], errors="coerce")
        raw = pd.to_numeric(group["raw_probability"], errors="coerce")
        shadow = pd.to_numeric(group["shadow_probability"], errors="coerce")
        raw_metrics = _metrics(raw, labels, method="raw_probability", split_name=route_policy, n_bins=n_bins)
        shadow_metrics = _metrics(shadow, labels, method="shadow_probability", split_name=route_policy, n_bins=n_bins)
        raw_brier = raw_metrics.get("brier_score")
        shadow_brier = shadow_metrics.get("brier_score")
        raw_ece = raw_metrics.get("expected_calibration_error")
        shadow_ece = shadow_metrics.get("expected_calibration_error")
        improvement = None if raw_brier is None or shadow_brier is None else float(raw_brier) - float(shadow_brier)
        ece_change = None if raw_ece is None or shadow_ece is None else float(shadow_ece) - float(raw_ece)
        sample_count = int(shadow_metrics.get("sample_count") or 0)
        if sample_count < int(min_candidate_sample):
            status = "INSUFFICIENT_CANDIDATE_SHADOW_EVIDENCE"
        elif improvement is not None and improvement < -float(max_candidate_brier_regression):
            status = "CANDIDATE_ROUTE_REGRESSED_BRIER"
        elif ece_change is not None and ece_change > float(max_candidate_ece_regression):
            status = "CANDIDATE_ROUTE_REGRESSED_ECE"
        elif improvement is not None and improvement > 0:
            status = "CANDIDATE_ROUTE_IMPROVED"
        else:
            status = "CANDIDATE_ROUTE_WATCH"
        first = group.iloc[0]
        rows.append(
            {
                "route_policy": route_policy,
                "candidate_key": str(key),
                "candidate_status": status,
                "sample_count": sample_count,
                "assigned_candidate_type": first.get("assigned_candidate_type"),
                "assigned_segment_field": first.get("assigned_segment_field"),
                "assigned_segment_value": first.get("assigned_segment_value"),
                "assigned_calibrator": first.get("assigned_calibrator"),
                "raw_brier_score": raw_metrics.get("brier_score"),
                "shadow_brier_score": shadow_metrics.get("brier_score"),
                "brier_improvement": _round_or_none(improvement, 8),
                "raw_expected_calibration_error": raw_metrics.get("expected_calibration_error"),
                "shadow_expected_calibration_error": shadow_metrics.get("expected_calibration_error"),
                "ece_change": _round_or_none(ece_change, 8),
                "actual_hit_rate": shadow_metrics.get("actual_hit_rate"),
                "shadow_mean_predicted_probability": shadow_metrics.get("mean_predicted_probability"),
                "shadow_calibration_gap": shadow_metrics.get("calibration_gap"),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item.get("candidate_status") != "CANDIDATE_ROUTE_IMPROVED",
            -(float(item.get("brier_improvement") or -1e9)),
        ),
    )


def _best_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    acceptable = [
        row
        for row in policy_rows
        if row.get("shadow_status") in {FORWARD_SHADOW_PASS, SHADOW_REPLAY_PASS, FORWARD_SHADOW_WATCH}
    ]
    if not acceptable:
        return None
    return sorted(
        acceptable,
        key=lambda item: (
            item.get("shadow_status") not in {FORWARD_SHADOW_PASS, SHADOW_REPLAY_PASS},
            -(float(item.get("brier_improvement") or -1e9)),
            float(item.get("shadow_expected_calibration_error") or 1e9),
        ),
    )[0]


def _overall_status(policy_rows: list[dict[str, Any]], *, validation_mode_used: str) -> str:
    if not policy_rows:
        return NO_CANDIDATE_ROUTES
    statuses = {str(row.get("shadow_status")) for row in policy_rows}
    if NEEDS_MORE_FORWARD_DATA in statuses and len(statuses) == 1:
        return NEEDS_MORE_FORWARD_DATA
    if validation_mode_used == "holdout_replay" and SHADOW_REPLAY_PASS in statuses:
        return SHADOW_REPLAY_PASS
    if FORWARD_SHADOW_PASS in statuses:
        return FORWARD_SHADOW_PASS
    if FORWARD_SHADOW_WATCH in statuses:
        return FORWARD_SHADOW_WATCH
    if NO_CANDIDATE_ROUTES in statuses and len(statuses) == 1:
        return NO_CANDIDATE_ROUTES
    return FORWARD_SHADOW_REJECTED


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("shadow_validation_status")
    validation_mode = report.get("validation_window", {}).get("validation_mode_used")
    best = report.get("selection_summary", {}).get("recommended_routing_policy")
    if status == SHADOW_REPLAY_PASS:
        return [
            f"Keep `{best}` in shadow review, but require true rows after candidate generation before manual adoption.",
            "Use the route-decision artifact to review overlapping matches between recency and regime candidates.",
            "Do not change runtime probabilities, parameter packs, data sources, or execution behavior from this replay.",
        ]
    if status == FORWARD_SHADOW_PASS:
        return [
            f"Escalate `{best}` to manual calibration-governance review after confirming route-level regressions are absent.",
            "Continue monitoring future labeled rows before any human-approved runtime adoption.",
        ]
    if status == NEEDS_MORE_FORWARD_DATA:
        return [
            "Collect more quality-approved forward rows after candidate generation before judging the bundle.",
            "Keep current runtime probabilities unchanged while the shadow evidence accumulates.",
        ]
    if status == FORWARD_SHADOW_WATCH:
        return [
            "Keep the candidate bundle in watch mode because at least one route policy or candidate slice failed a guardrail.",
            "Prefer the best non-regressing routing policy only for continued shadow research, not runtime adoption.",
        ]
    return [
        "Do not promote the candidate bundle; shadow validation did not improve the replay evidence.",
        "Return to probability-generation diagnostics or narrower segment candidates before another shadow cycle.",
    ]


def build_segmented_probability_forward_shadow_report(
    frame: pd.DataFrame,
    *,
    candidate_bundle: dict[str, Any] | None = None,
    candidate_bundle_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    validation_mode: str = "auto",
    routing_policies: tuple[str, ...] = DEFAULT_ROUTING_POLICIES,
    min_shadow_sample: int = 100,
    min_candidate_sample: int = 50,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_brier_regression: float = 0.002,
    max_candidate_ece_regression: float = 0.02,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build forward-shadow validation evidence for a segmented calibration bundle."""
    bundle = candidate_bundle if isinstance(candidate_bundle, dict) else _load_candidate_bundle(
        candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    )
    candidates = [candidate for candidate in bundle.get("candidates", []) or [] if isinstance(candidate, dict)]
    raw = frame if frame is not None else pd.DataFrame()
    working = _clean_probability_and_label_frame(raw, probability_field=probability_field, label_field=label_field)
    evaluation, window = _evaluation_frame(
        working,
        bundle=bundle,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        min_shadow_sample=min_shadow_sample,
    )

    policy_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    route_decision_frames: list[pd.DataFrame] = []
    curve_rows: list[dict[str, Any]] = []
    for policy in routing_policies:
        decisions = _route_decisions(evaluation, candidates, routing_policy=str(policy))
        route_decision_frames.append(decisions)
        candidate_metrics = _candidate_route_metrics(
            decisions,
            route_policy=str(policy),
            min_candidate_sample=min_candidate_sample,
            max_candidate_brier_regression=max_candidate_brier_regression,
            max_candidate_ece_regression=max_candidate_ece_regression,
            n_bins=n_bins,
        )
        candidate_regression_count = sum(
            row.get("candidate_status") in {"CANDIDATE_ROUTE_REGRESSED_BRIER", "CANDIDATE_ROUTE_REGRESSED_ECE"}
            for row in candidate_metrics
        )
        policy_rows.append(
            _metric_pair(
                decisions,
                route_policy=str(policy),
                min_shadow_sample=min_shadow_sample,
                min_brier_improvement=min_brier_improvement,
                max_ece_regression=max_ece_regression,
                n_bins=n_bins,
                validation_mode_used=str(window.get("validation_mode_used")),
                candidate_regression_count=int(candidate_regression_count),
            )
        )
        candidate_rows.extend(candidate_metrics)
        labels = pd.to_numeric(decisions.get("label", pd.Series(dtype=float)), errors="coerce")
        raw_prob = pd.to_numeric(decisions.get("raw_probability", pd.Series(dtype=float)), errors="coerce")
        shadow_prob = pd.to_numeric(decisions.get("shadow_probability", pd.Series(dtype=float)), errors="coerce")
        curve_rows.extend(
            _calibration_curve(raw_prob, labels, method=f"{policy}:raw", split_name="shadow", n_bins=n_bins)
        )
        curve_rows.extend(
            _calibration_curve(shadow_prob, labels, method=f"{policy}:shadow", split_name="shadow", n_bins=n_bins)
        )

    route_decisions = pd.concat(route_decision_frames, ignore_index=True) if route_decision_frames else pd.DataFrame()
    best = _best_policy(policy_rows)
    report = {
        "report_type": "segmented_probability_forward_shadow",
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
        "candidate_count": int(len(candidates)),
        "train_fraction": float(train_fraction),
        "validation_window": window,
        "shadow_validation_status": _overall_status(policy_rows, validation_mode_used=str(window.get("validation_mode_used"))),
        "selection_summary": {
            "recommended_routing_policy": best.get("route_policy") if best else None,
            "recommended_policy_status": best.get("shadow_status") if best else None,
            "recommended_policy_brier_improvement": best.get("brier_improvement") if best else None,
            "recommended_policy_ece_change": best.get("ece_change") if best else None,
            "evaluated_routing_policy_count": int(len(policy_rows)),
            "route_policy_status_counts": {
                status: sum(1 for row in policy_rows if row.get("shadow_status") == status)
                for status in sorted({str(row.get("shadow_status")) for row in policy_rows})
            },
        },
        "routing_policy_results": policy_rows,
        "candidate_route_results": candidate_rows,
        "calibration_curve": curve_rows,
        "route_decision_count": int(len(route_decisions)),
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    sanitized = _sanitize_value(report)
    sanitized["_route_decisions_frame"] = route_decisions
    return sanitized


def render_segmented_probability_forward_shadow_markdown(report: dict[str, Any]) -> str:
    """Render forward-shadow validation output as Markdown."""
    selection = report.get("selection_summary", {}) or {}
    window = report.get("validation_window", {}) or {}
    lines = [
        "# Segmented Probability Forward Shadow",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Candidate bundle: {report.get('candidate_bundle_path') or 'inline'}",
        f"- Rows: {report.get('row_count')}",
        f"- Quality-labeled rows: {report.get('quality_labeled_row_count')}",
        f"- Validation mode: `{window.get('validation_mode_used')}`",
        f"- Strict forward rows: {window.get('strict_forward_row_count')}",
        f"- Holdout replay rows: {window.get('holdout_replay_row_count')}",
        f"- Status: `{report.get('shadow_validation_status')}`",
        f"- Recommended routing policy: `{selection.get('recommended_routing_policy')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Routing Policies",
        "",
        "| Policy | Status | Assigned | Multi-Match | Brier Improvement | ECE Change | Shadow Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("routing_policy_results", []) or []:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('shadow_status')}` | {row.get('assigned_candidate_count')} | "
            f"{row.get('multiple_match_count')} | {row.get('brier_improvement')} | "
            f"{row.get('ece_change')} | {row.get('shadow_calibration_gap')} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Routes",
            "",
            "| Policy | Candidate | Status | N | Brier Improvement | ECE Change |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in (report.get("candidate_route_results", []) or [])[:20]:
        lines.append(
            f"| `{row.get('route_policy')}` | `{row.get('candidate_key')}` | `{row.get('candidate_status')}` | "
            f"{row.get('sample_count')} | {row.get('brier_improvement')} | {row.get('ece_change')} |"
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
        "latest_json_path": output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_JSON_FILENAME,
        "latest_markdown_path": output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_MARKDOWN_FILENAME,
        "latest_policies_csv_path": output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_POLICIES_FILENAME,
        "latest_candidates_csv_path": output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_CANDIDATES_FILENAME,
        "latest_routes_csv_path": output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_ROUTES_FILENAME,
    }


def write_segmented_probability_forward_shadow_report(
    frame: pd.DataFrame,
    *,
    candidate_bundle: dict[str, Any] | None = None,
    candidate_bundle_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    train_fraction: float = 0.70,
    validation_mode: str = "auto",
    routing_policies: tuple[str, ...] = DEFAULT_ROUTING_POLICIES,
    min_shadow_sample: int = 100,
    min_candidate_sample: int = 50,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_brier_regression: float = 0.002,
    max_candidate_ece_regression: float = 0.02,
    n_bins: int = 10,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write segmented probability forward-shadow artifacts."""
    bundle_path = candidate_bundle_path or DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    report = build_segmented_probability_forward_shadow_report(
        frame,
        candidate_bundle=candidate_bundle,
        candidate_bundle_path=bundle_path,
        dataset_path=dataset_path,
        probability_field=probability_field,
        label_field=label_field,
        train_fraction=train_fraction,
        validation_mode=validation_mode,
        routing_policies=routing_policies,
        min_shadow_sample=min_shadow_sample,
        min_candidate_sample=min_candidate_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_brier_regression=max_candidate_brier_regression,
        max_candidate_ece_regression=max_candidate_ece_regression,
        n_bins=n_bins,
    )
    routes = report.pop("_route_decisions_frame", pd.DataFrame())
    assert_artifact_schema(report, "segmented_probability_forward_shadow")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "segmented_probability_forward_shadow"
    paths = _artifact_paths(output, stem)
    markdown = render_segmented_probability_forward_shadow_markdown(report)
    policies = pd.DataFrame(report.get("routing_policy_results", []) or [])
    candidates = pd.DataFrame(report.get("candidate_route_results", []) or [])

    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(policies, paths["policies_csv_path"])
    _atomic_write_csv(candidates, paths["candidates_csv_path"])
    _atomic_write_csv(routes, paths["routes_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(policies, paths["latest_policies_csv_path"])
        _atomic_write_csv(candidates, paths["latest_candidates_csv_path"])
        _atomic_write_csv(routes, paths["latest_routes_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def write_segmented_probability_forward_shadow_report_from_path(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset and write segmented probability forward-shadow artifacts."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return write_segmented_probability_forward_shadow_report(frame, dataset_path=path, **kwargs)
