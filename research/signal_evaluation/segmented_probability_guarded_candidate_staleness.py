"""Guarded-bundle staleness governance for segmented probability candidates.

This research-only gate answers whether the guarded candidate bundle itself is
fresh enough for shadow review. It is intentionally separate from the original
candidate staleness report, because a guarded bundle can be newly generated
while the source candidate's historical routing policy has already drifted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import _clean_probability_and_label_frame
from research.signal_evaluation.segmented_probability_candidate_staleness import (
    _candidate_age_days,
    _load_history_frame,
    _load_json_file,
    _parse_timestamp,
    _population_shift_summary,
    _read_dataset,
    _safe_int,
    _timestamp_series,
)
from research.signal_evaluation.segmented_probability_guarded_candidate_bundle import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR,
    GUARDED_CANDIDATE_BUNDLE_READY,
    SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_LABEL_FIELD,
    DEFAULT_PROBABILITY_FIELD,
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR = (
    PROJECT_ROOT
    / "research"
    / "signal_evaluation"
    / "reports"
    / "segmented_probability_guarded_candidate_staleness"
)
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_DIR
    / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_JSON_FILENAME
)
DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_SOAK_HISTORY_PATH = (
    PROJECT_ROOT
    / "research"
    / "signal_evaluation"
    / "reports"
    / "segmented_probability_shadow_soak"
    / "segmented_probability_shadow_soak_history.csv"
)

SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_JSON_FILENAME = (
    "latest_segmented_probability_guarded_candidate_staleness.json"
)
SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_MARKDOWN_FILENAME = (
    "latest_segmented_probability_guarded_candidate_staleness.md"
)

GUARDED_ACTIVE_REVIEW = "GUARDED_ACTIVE_REVIEW"
GUARDED_ACCUMULATING_FORWARD_LABELS = "GUARDED_ACCUMULATING_FORWARD_LABELS"
GUARDED_STALE_WATCH = "GUARDED_STALE_WATCH"
GUARDED_EXPIRED = "GUARDED_EXPIRED"
GUARDED_SUPERSEDED = "GUARDED_SUPERSEDED"
GUARDED_BLOCKED = "GUARDED_BLOCKED"

GUARDED_STALENESS_NON_BLOCKING = {
    GUARDED_ACTIVE_REVIEW,
    GUARDED_ACCUMULATING_FORWARD_LABELS,
}


def _guarded_generated_at(bundle: dict[str, Any]) -> str | None:
    generated_at = bundle.get("generated_at")
    if generated_at:
        return str(generated_at)
    for candidate in bundle.get("candidates", []) or []:
        if isinstance(candidate, dict) and candidate.get("generated_at"):
            return str(candidate.get("generated_at"))
    return None


def _dataset_modified_at(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return pd.Timestamp.fromtimestamp(file_path.stat().st_mtime, tz="UTC").isoformat()
    except Exception:
        return None


def _guarded_dataset_currency_summary(
    raw: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    dataset_path: str | Path | None,
    guarded_ts: pd.Timestamp | None,
    max_new_rows_before_stale: int,
    max_new_labeled_rows_before_stale: int,
) -> dict[str, Any]:
    raw_timestamps = _timestamp_series(raw)
    labeled_timestamps = _timestamp_series(labeled)
    raw_after = 0
    labeled_after = 0
    latest_after = None
    if guarded_ts is not None:
        raw_after_mask = raw_timestamps > guarded_ts
        labeled_after_mask = labeled_timestamps > guarded_ts
        raw_after = int(raw_after_mask.fillna(False).sum())
        labeled_after = int(labeled_after_mask.fillna(False).sum())
        post = raw_timestamps.loc[raw_after_mask.fillna(False)].dropna()
        latest_after = post.max().isoformat() if not post.empty else None

    valid = raw_timestamps.dropna()
    earliest = valid.min().isoformat() if not valid.empty else None
    latest = valid.max().isoformat() if not valid.empty else None
    latest_ts = _parse_timestamp(latest)
    generated_from_latest = (
        bool(latest_ts <= guarded_ts) if latest_ts is not None and guarded_ts is not None else False
    )
    material_new_data = (
        raw_after >= int(max_new_rows_before_stale)
        or labeled_after >= int(max_new_labeled_rows_before_stale)
    )
    dataset_mtime = _dataset_modified_at(dataset_path)
    mtime_ts = _parse_timestamp(dataset_mtime)
    modified_after = bool(mtime_ts > guarded_ts) if mtime_ts is not None and guarded_ts is not None else False
    if guarded_ts is None:
        status = "GUARDED_TIMESTAMP_MISSING"
    elif raw_after <= 0:
        status = "CURRENT_AT_GUARDED_GENERATION"
    elif material_new_data:
        status = "MATERIAL_NEW_POST_GUARDED_DATA"
    else:
        status = "POST_GUARDED_ROWS_ACCUMULATING"

    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "row_count": int(len(raw)),
        "quality_labeled_row_count": int(len(labeled)),
        "earliest_signal_timestamp": earliest,
        "latest_signal_timestamp": latest,
        "latest_signal_timestamp_after_guarded_candidate": latest_after,
        "rows_after_guarded_candidate_generated": raw_after,
        "quality_labeled_rows_after_guarded_candidate_generated": labeled_after,
        "max_new_rows_before_stale": int(max_new_rows_before_stale),
        "max_new_labeled_rows_before_stale": int(max_new_labeled_rows_before_stale),
        "guarded_bundle_generated_from_latest_signal_timestamps": bool(generated_from_latest),
        "material_new_data_since_guarded_candidate": bool(material_new_data),
        "dataset_modified_at": dataset_mtime,
        "dataset_modified_after_guarded_candidate_generated": bool(modified_after),
        "dataset_currency_status": status,
    }


def _guarded_population_shift_summary(
    labeled: pd.DataFrame,
    *,
    guarded_ts: pd.Timestamp | None,
    min_shift_sample: int,
    material_hit_rate_delta: float,
    material_probability_delta: float,
    material_distribution_psi: float,
) -> dict[str, Any]:
    shift = _population_shift_summary(
        labeled,
        candidate_ts=guarded_ts,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
    )
    shift["pre_guarded_label_count"] = _safe_int(shift.get("pre_candidate_label_count"))
    shift["post_guarded_label_count"] = _safe_int(shift.get("post_candidate_label_count"))
    shift.pop("pre_candidate_label_count", None)
    shift.pop("post_candidate_label_count", None)
    return shift


def _looks_like_guarded_bundle(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("artifact_type") != "segmented_probability_calibration_candidate_bundle":
        return False
    return bool(
        payload.get("bundle_variant") == "guarded_ev_quarantine_plus_rank_guard"
        or payload.get("guarded_candidate_bundle_status")
        or isinstance(payload.get("rank_preservation_policy"), dict)
    )


def _guarded_supersession_summary(
    *,
    guarded_candidate_bundle_path: str | Path | None,
    guarded_candidate_bundle_search_dir: str | Path | None,
    guarded_generated_at: str | None,
) -> dict[str, Any]:
    guarded_ts = _parse_timestamp(guarded_generated_at)
    if guarded_ts is None:
        return {
            "superseded": False,
            "newer_guarded_candidate_bundle_path": None,
            "newer_guarded_candidate_generated_at": None,
            "guarded_candidate_bundle_search_dir": (
                str(guarded_candidate_bundle_search_dir) if guarded_candidate_bundle_search_dir else None
            ),
            "guarded_candidate_bundle_search_status": "GUARDED_TIMESTAMP_MISSING",
        }

    current_path = Path(guarded_candidate_bundle_path) if guarded_candidate_bundle_path is not None else None
    if guarded_candidate_bundle_search_dir is not None:
        search_dir = Path(guarded_candidate_bundle_search_dir)
    elif current_path is not None:
        search_dir = current_path.parent
    else:
        search_dir = None
    if search_dir is None or not search_dir.exists():
        return {
            "superseded": False,
            "newer_guarded_candidate_bundle_path": None,
            "newer_guarded_candidate_generated_at": None,
            "guarded_candidate_bundle_search_dir": str(search_dir) if search_dir is not None else None,
            "guarded_candidate_bundle_search_status": "SEARCH_DIR_MISSING",
        }

    current_resolved = current_path.resolve(strict=False) if current_path is not None else None
    newer: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(search_dir.glob("*.json")):
        if current_resolved is not None and path.resolve(strict=False) == current_resolved:
            continue
        payload = _load_json_file(path)
        if not _looks_like_guarded_bundle(payload):
            continue
        if _safe_int(payload.get("candidate_count")) <= 0:
            continue
        generated_at = _parse_timestamp(payload.get("generated_at"))
        if generated_at is not None and generated_at > guarded_ts:
            newer.append((generated_at, path))

    if not newer:
        return {
            "superseded": False,
            "newer_guarded_candidate_bundle_path": None,
            "newer_guarded_candidate_generated_at": None,
            "guarded_candidate_bundle_search_dir": str(search_dir),
            "guarded_candidate_bundle_search_status": "NO_NEWER_GUARDED_BUNDLE_FOUND",
        }
    newest_ts, newest_path = max(newer, key=lambda item: item[0])
    return {
        "superseded": True,
        "newer_guarded_candidate_bundle_path": str(newest_path),
        "newer_guarded_candidate_generated_at": newest_ts.isoformat(),
        "guarded_candidate_bundle_search_dir": str(search_dir),
        "guarded_candidate_bundle_search_status": "NEWER_GUARDED_BUNDLE_FOUND",
    }


def _matching_guarded_history(
    history: pd.DataFrame,
    *,
    guarded_candidate_bundle_path: str | Path | None,
    guarded_generated_at: str | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy()
    if guarded_candidate_bundle_path is not None and "guarded_candidate_bundle_path" in frame.columns:
        frame = frame.loc[frame["guarded_candidate_bundle_path"].fillna("").astype(str) == str(guarded_candidate_bundle_path)]
    if guarded_generated_at and "guarded_candidate_generated_at" in frame.columns:
        frame = frame.loc[frame["guarded_candidate_generated_at"].fillna("").astype(str) == str(guarded_generated_at)]
    return frame.copy()


def _guarded_routing_policy_stability(
    history: pd.DataFrame,
    *,
    guarded_candidate_bundle_path: str | Path | None,
    guarded_generated_at: str | None,
    lookback_runs: int,
    min_policy_observations: int,
) -> dict[str, Any]:
    matching = _matching_guarded_history(
        history,
        guarded_candidate_bundle_path=guarded_candidate_bundle_path,
        guarded_generated_at=guarded_generated_at,
    )
    base = {
        "history_count": int(len(history)) if history is not None else 0,
        "matching_history_count": int(len(matching)),
        "lookback_runs": int(lookback_runs),
        "min_policy_observations": int(min_policy_observations),
        "latest_guarded_recommended_routing_policy": None,
        "guarded_routing_policy_changed": False,
        "unique_guarded_recommended_policy_count": 0,
        "guarded_policy_change_count": 0,
    }
    if matching.empty:
        return {
            **base,
            "eligible_true_forward_history_count": 0,
            "policy_stability_status": "NO_GUARDED_SOAK_HISTORY",
            "policy_counts": {},
        }
    if "guarded_strict_forward_row_count" not in matching.columns:
        return {
            **base,
            "eligible_true_forward_history_count": 0,
            "policy_stability_status": "INSUFFICIENT_GUARDED_FORWARD_EVIDENCE",
            "policy_counts": {},
        }
    strict_forward = pd.to_numeric(matching["guarded_strict_forward_row_count"], errors="coerce").fillna(0)
    mode = matching.get("guarded_validation_mode_used", pd.Series(index=matching.index, dtype=object))
    true_forward = mode.fillna("").astype(str).str.lower() == "after_candidate_generated"
    eligible = matching.loc[(strict_forward > 0) & true_forward].tail(max(int(lookback_runs), 1)).copy()
    if eligible.empty:
        latest = matching.iloc[-1].to_dict()
        return {
            **base,
            "eligible_true_forward_history_count": 0,
            "policy_stability_status": "INSUFFICIENT_GUARDED_FORWARD_EVIDENCE",
            "latest_guarded_shadow_status": latest.get("guarded_shadow_status"),
            "latest_soak_status": latest.get("soak_status"),
            "policy_counts": {},
        }
    if "guarded_recommended_routing_policy" not in eligible.columns:
        return {
            **base,
            "eligible_true_forward_history_count": int(len(eligible)),
            "policy_stability_status": "NO_GUARDED_RECOMMENDED_POLICY",
            "policy_counts": {},
        }

    policy = eligible["guarded_recommended_routing_policy"].fillna("").astype(str).str.strip()
    policy = policy.loc[policy != ""]
    if policy.empty:
        return {
            **base,
            "eligible_true_forward_history_count": int(len(eligible)),
            "policy_stability_status": "NO_GUARDED_RECOMMENDED_POLICY",
            "policy_counts": {},
        }
    change_count = int((policy != policy.shift()).sum() - 1) if len(policy) > 1 else 0
    unique_count = int(policy.nunique(dropna=True))
    enough_policy_history = int(len(policy)) >= int(min_policy_observations)
    changed = bool(enough_policy_history and (unique_count > 1 or change_count > 0))
    if not enough_policy_history:
        status = "INSUFFICIENT_GUARDED_POLICY_HISTORY"
    else:
        status = "GUARDED_POLICY_CHANGED" if changed else "GUARDED_POLICY_STABLE"
    latest = eligible.iloc[-1].to_dict()
    return {
        **base,
        "eligible_true_forward_history_count": int(len(eligible)),
        "policy_stability_status": status,
        "latest_guarded_recommended_routing_policy": str(policy.iloc[-1]),
        "guarded_routing_policy_changed": bool(changed),
        "unique_guarded_recommended_policy_count": unique_count,
        "guarded_policy_change_count": max(change_count, 0),
        "latest_guarded_shadow_status": latest.get("guarded_shadow_status"),
        "latest_soak_status": latest.get("soak_status"),
        "policy_counts": {
            str(key): int(value) for key, value in policy.value_counts().sort_index().items()
        },
    }


def _guarded_staleness_actions(status: str, reasons: list[str], *, forward_gap: int) -> list[str]:
    if status == GUARDED_ACTIVE_REVIEW:
        return [
            "Keep the guarded bundle in research-only shadow review and continue monitoring post-guarded labels.",
            "Do not change runtime probabilities, parameter packs, data sources, or execution behavior from this report.",
        ]
    if status == GUARDED_ACCUMULATING_FORWARD_LABELS:
        return [
            f"Keep collecting post-guarded true-forward labels; current guarded forward sample gap is {forward_gap}.",
            "Treat original-candidate routing staleness as context, not as proof that this newer guarded bundle is stale.",
            "No automated runtime adoption is authorized by this research-only report.",
        ]
    if status == GUARDED_SUPERSEDED:
        return [
            "Move research review to the newer guarded candidate bundle before any manual promotion discussion.",
            "Archive this guarded bundle as historical evidence; do not adopt it.",
        ]
    if status == GUARDED_EXPIRED:
        return [
            "Regenerate guarded candidate evidence from the latest quality-approved dataset.",
            "Restart post-guarded forward-shadow accumulation for the regenerated bundle.",
        ]
    if status == GUARDED_BLOCKED:
        return [
            "Stop guarded-bundle review until blocking governance reasons are resolved.",
            "Confirm the guarded bundle is research-only, approval-gated, side-effect clean, and has a rank-preservation policy.",
        ]
    actions = []
    joined = " ".join(reasons)
    if "material_new_data" in joined or "population_shifted" in joined:
        actions.append("Refresh guarded-bundle evidence once post-guarded labels are sufficient for a clean comparison.")
    if "routing_policy_changed" in joined:
        actions.append("Review guarded routing-policy stability using true post-guarded history before advancing the bundle.")
    if "candidate_age" in joined:
        actions.append("Regenerate the guarded bundle if the watch state persists into the expiry window.")
    if not actions:
        actions.append("Keep the guarded bundle under watch and continue collecting forward evidence.")
    actions.append("No automated runtime adoption is authorized by this research-only report.")
    return actions


def build_segmented_probability_guarded_candidate_staleness_report(
    *,
    dataset: pd.DataFrame,
    guarded_candidate_bundle: dict[str, Any],
    guarded_history: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    guarded_candidate_bundle_path: str | Path | None = None,
    guarded_candidate_bundle_search_dir: str | Path | None = None,
    guarded_history_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    stale_after_days: float = 7.0,
    expire_after_days: float = 14.0,
    max_new_rows_before_stale: int = 500,
    max_new_labeled_rows_before_stale: int = 100,
    min_forward_sample: int = 100,
    min_shift_sample: int = 100,
    material_hit_rate_delta: float = 0.10,
    material_probability_delta: float = 0.08,
    material_distribution_psi: float = 0.20,
    policy_lookback_runs: int = 5,
    min_policy_observations: int = 2,
    as_of: Any = None,
) -> dict[str, Any]:
    """Build a staleness report for a guarded segmented-probability bundle."""
    raw = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame()
    bundle = guarded_candidate_bundle if isinstance(guarded_candidate_bundle, dict) else {}
    history = guarded_history if guarded_history is not None else pd.DataFrame()
    candidate_count = _safe_int(bundle.get("candidate_count"))
    quarantined_count = _safe_int(bundle.get("quarantined_candidate_count"))
    generated_at = _guarded_generated_at(bundle)
    generated_ts = _parse_timestamp(generated_at)
    age_days = _candidate_age_days(generated_at, as_of=as_of)
    rank_policy = bundle.get("rank_preservation_policy") if isinstance(bundle.get("rank_preservation_policy"), dict) else {}
    research_only = bundle.get("research_only") is True
    approval_required = bundle.get("approval_required_for_runtime_use") is True
    bundle_side_effects_clean = all(
        bundle.get(field) is False
        for field in ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    )
    bundle_ready = str(bundle.get("guarded_candidate_bundle_status") or bundle.get("calibration_status") or "") == (
        GUARDED_CANDIDATE_BUNDLE_READY
    )

    labeled = _clean_probability_and_label_frame(
        raw,
        probability_field=probability_field,
        label_field=label_field,
    )
    dataset_currency = _guarded_dataset_currency_summary(
        raw,
        labeled,
        dataset_path=dataset_path,
        guarded_ts=generated_ts,
        max_new_rows_before_stale=max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=max_new_labeled_rows_before_stale,
    )
    population_shift = _guarded_population_shift_summary(
        labeled,
        guarded_ts=generated_ts,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
    )
    routing_stability = _guarded_routing_policy_stability(
        history,
        guarded_candidate_bundle_path=guarded_candidate_bundle_path,
        guarded_generated_at=generated_at,
        lookback_runs=policy_lookback_runs,
        min_policy_observations=min_policy_observations,
    )
    supersession = _guarded_supersession_summary(
        guarded_candidate_bundle_path=guarded_candidate_bundle_path,
        guarded_candidate_bundle_search_dir=guarded_candidate_bundle_search_dir,
        guarded_generated_at=generated_at,
    )

    forward_count = _safe_int(dataset_currency.get("quality_labeled_rows_after_guarded_candidate_generated"))
    forward_gap = max(int(min_forward_sample) - forward_count, 0)
    reasons: list[str] = []
    if candidate_count <= 0:
        reasons.append("no_guarded_candidate_routes_available")
    if generated_ts is None:
        reasons.append("guarded_candidate_generated_at_missing_or_invalid")
    if not bundle_ready:
        reasons.append("guarded_candidate_bundle_not_ready")
    if not research_only:
        reasons.append("guarded_candidate_bundle_not_research_only")
    if not approval_required:
        reasons.append("guarded_candidate_bundle_not_approval_gated")
    if not bundle_side_effects_clean:
        reasons.append("guarded_candidate_bundle_side_effect_flags_not_clean")
    if not rank_policy:
        reasons.append("rank_preservation_policy_missing")
    if age_days is not None and age_days > float(expire_after_days):
        reasons.append("guarded_candidate_age_exceeds_expiry_window")
    elif age_days is not None and age_days > float(stale_after_days):
        reasons.append("guarded_candidate_age_exceeds_watch_window")
    if dataset_currency.get("material_new_data_since_guarded_candidate"):
        reasons.append("material_new_data_since_guarded_candidate_generation")
    if population_shift.get("shifted_materially"):
        reasons.append("post_guarded_forward_label_population_shifted")
    if routing_stability.get("guarded_routing_policy_changed"):
        reasons.append("guarded_recommended_routing_policy_changed")
    if supersession.get("superseded"):
        reasons.append("newer_guarded_candidate_bundle_supersedes_current")
    if forward_gap > 0:
        reasons.append("guarded_forward_sample_below_minimum")

    blocking_reason_keys = {
        "no_guarded_candidate_routes_available",
        "guarded_candidate_generated_at_missing_or_invalid",
        "guarded_candidate_bundle_not_ready",
        "guarded_candidate_bundle_not_research_only",
        "guarded_candidate_bundle_not_approval_gated",
        "guarded_candidate_bundle_side_effect_flags_not_clean",
        "rank_preservation_policy_missing",
    }
    if supersession.get("superseded"):
        status = GUARDED_SUPERSEDED
    elif any(reason in blocking_reason_keys for reason in reasons):
        status = GUARDED_BLOCKED
    elif generated_ts is None or (age_days is not None and age_days > float(expire_after_days)):
        status = GUARDED_EXPIRED
    elif any(
        reason
        in {
            "guarded_candidate_age_exceeds_watch_window",
            "material_new_data_since_guarded_candidate_generation",
            "post_guarded_forward_label_population_shifted",
            "guarded_recommended_routing_policy_changed",
        }
        for reason in reasons
    ):
        status = GUARDED_STALE_WATCH
    elif forward_gap > 0:
        status = GUARDED_ACCUMULATING_FORWARD_LABELS
    else:
        status = GUARDED_ACTIVE_REVIEW

    checked_conditions = {
        "guarded_candidate_count_positive": candidate_count > 0,
        "guarded_candidate_generated_at_valid": generated_ts is not None,
        "guarded_candidate_age_within_watch_window": bool(
            age_days is not None and age_days <= float(stale_after_days)
        ),
        "guarded_candidate_age_within_expiry_window": bool(
            age_days is not None and age_days <= float(expire_after_days)
        ),
        "guarded_bundle_not_superseded": not bool(supersession.get("superseded")),
        "guarded_bundle_ready": bool(bundle_ready),
        "guarded_bundle_research_only": bool(research_only),
        "guarded_bundle_approval_required_for_runtime_use": bool(approval_required),
        "guarded_bundle_side_effect_flags_clean": bool(bundle_side_effects_clean),
        "rank_preservation_policy_present": bool(rank_policy),
        "guarded_forward_sample_met": forward_count >= int(min_forward_sample),
        "material_new_data_since_guarded_candidate": bool(
            dataset_currency.get("material_new_data_since_guarded_candidate")
        ),
        "post_guarded_forward_label_population_shifted": bool(population_shift.get("shifted_materially")),
        "guarded_routing_policy_stable": not bool(routing_stability.get("guarded_routing_policy_changed")),
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
    }

    report = {
        "report_type": "segmented_probability_guarded_candidate_staleness",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "guarded_candidate_bundle_path": (
            str(guarded_candidate_bundle_path) if guarded_candidate_bundle_path is not None else None
        ),
        "guarded_candidate_bundle_search_dir": (
            str(guarded_candidate_bundle_search_dir) if guarded_candidate_bundle_search_dir is not None else None
        ),
        "guarded_history_path": str(guarded_history_path) if guarded_history_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "guarded_staleness_status": status,
        "guarded_staleness_reasons": reasons,
        "guarded_candidate_summary": {
            "artifact_type": bundle.get("artifact_type"),
            "bundle_variant": bundle.get("bundle_variant"),
            "guarded_candidate_bundle_status": bundle.get("guarded_candidate_bundle_status"),
            "candidate_count": candidate_count,
            "source_candidate_count": _safe_int(bundle.get("source_candidate_count")),
            "quarantined_candidate_count": quarantined_count,
            "guarded_candidate_generated_at": generated_at,
            "guarded_candidate_age_days": _round_or_none(age_days, 6),
            "stale_after_days": float(stale_after_days),
            "expire_after_days": float(expire_after_days),
            "research_only": bool(research_only),
            "approval_required_for_runtime_use": bool(approval_required),
            "rank_preservation_policy_present": bool(rank_policy),
        },
        "dataset_currency": dataset_currency,
        "forward_label_population_shift": population_shift,
        "guarded_routing_policy_stability": routing_stability,
        "supersession": supersession,
        "checked_conditions": checked_conditions,
        "recommended_next_actions": _guarded_staleness_actions(status, reasons, forward_gap=forward_gap),
    }
    return _sanitize_value(report)


def render_segmented_probability_guarded_candidate_staleness_markdown(report: dict[str, Any]) -> str:
    """Render guarded candidate staleness governance output as Markdown."""
    candidate = report.get("guarded_candidate_summary", {}) or {}
    currency = report.get("dataset_currency", {}) or {}
    shift = report.get("forward_label_population_shift", {}) or {}
    routing = report.get("guarded_routing_policy_stability", {}) or {}
    supersession = report.get("supersession", {}) or {}
    lines = [
        "# Segmented Probability Guarded Candidate Staleness",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Guarded staleness status: **{report.get('guarded_staleness_status')}**",
        f"- Guarded candidate generated at: {candidate.get('guarded_candidate_generated_at')}",
        f"- Guarded candidate age days: {candidate.get('guarded_candidate_age_days')} / {candidate.get('expire_after_days')}",
        f"- Guarded candidate count: {candidate.get('candidate_count')}",
        f"- Quarantined candidate count: {candidate.get('quarantined_candidate_count')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Governance Checks",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Dataset currency status | `{currency.get('dataset_currency_status')}` |",
        f"| Latest signal timestamp | {currency.get('latest_signal_timestamp')} |",
        f"| Rows after guarded candidate | {currency.get('rows_after_guarded_candidate_generated')} |",
        f"| Quality labels after guarded candidate | {currency.get('quality_labeled_rows_after_guarded_candidate_generated')} |",
        f"| Forward-label shift status | `{shift.get('shift_status')}` |",
        f"| Post-guarded labels | {shift.get('post_guarded_label_count')} / {shift.get('min_shift_sample')} |",
        f"| Hit-rate delta | {shift.get('hit_rate_delta')} |",
        f"| Max distribution PSI | {shift.get('max_distribution_psi')} |",
        f"| Guarded routing policy status | `{routing.get('policy_stability_status')}` |",
        f"| Latest guarded routing policy | `{routing.get('latest_guarded_recommended_routing_policy')}` |",
        f"| Supersession status | `{supersession.get('guarded_candidate_bundle_search_status')}` |",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("guarded_staleness_reasons", []) or ["guarded_active_review"]:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append(
        "*Research-only artifact. It does not alter runtime config, parameter packs, "
        "data sources, or execution behavior.*"
    )
    return "\n".join(lines)


def write_segmented_probability_guarded_candidate_staleness_report(
    *,
    dataset_path: str | Path | None = None,
    guarded_candidate_bundle_path: str | Path | None = None,
    guarded_candidate_bundle_search_dir: str | Path | None = None,
    guarded_history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    stale_after_days: float = 7.0,
    expire_after_days: float = 14.0,
    max_new_rows_before_stale: int = 500,
    max_new_labeled_rows_before_stale: int = 100,
    min_forward_sample: int = 100,
    min_shift_sample: int = 100,
    material_hit_rate_delta: float = 0.10,
    material_probability_delta: float = 0.08,
    material_distribution_psi: float = 0.20,
    policy_lookback_runs: int = 5,
    min_policy_observations: int = 2,
    as_of: Any = None,
) -> dict[str, Any]:
    """Read latest guarded artifacts, build staleness report, and write it."""
    dataset = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    bundle_path = (
        Path(guarded_candidate_bundle_path)
        if guarded_candidate_bundle_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH
    )
    search_dir = (
        Path(guarded_candidate_bundle_search_dir)
        if guarded_candidate_bundle_search_dir is not None
        else bundle_path.parent
    )
    history_path = (
        Path(guarded_history_path)
        if guarded_history_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_SOAK_HISTORY_PATH
    )
    raw = _read_dataset(dataset)
    bundle = _load_json_file(bundle_path)
    history = _load_history_frame(history_path)
    report = build_segmented_probability_guarded_candidate_staleness_report(
        dataset=raw,
        guarded_candidate_bundle=bundle,
        guarded_history=history,
        dataset_path=dataset,
        guarded_candidate_bundle_path=bundle_path,
        guarded_candidate_bundle_search_dir=search_dir,
        guarded_history_path=history_path,
        probability_field=probability_field,
        label_field=label_field,
        stale_after_days=stale_after_days,
        expire_after_days=expire_after_days,
        max_new_rows_before_stale=max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=max_new_labeled_rows_before_stale,
        min_forward_sample=min_forward_sample,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
        policy_lookback_runs=policy_lookback_runs,
        min_policy_observations=min_policy_observations,
        as_of=as_of,
    )
    assert_artifact_schema(report, "segmented_probability_guarded_candidate_staleness")
    output = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR
    )
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_JSON_FILENAME
    markdown_path = output / SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_segmented_probability_guarded_candidate_staleness_markdown(report))
    return {
        "guarded_staleness_json_path": str(json_path),
        "guarded_staleness_markdown_path": str(markdown_path),
        "guarded_staleness_report": report,
    }
