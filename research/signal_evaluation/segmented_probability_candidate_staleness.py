"""Candidate staleness governance for segmented probability calibration.

This research-only gate answers whether the latest segmented calibration
candidate bundle is still suitable for shadow review. It does not change
runtime configuration, parameter packs, data-source choices, or execution
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.probability_calibration_experiment import _clean_probability_and_label_frame
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
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
from utils.timestamp_helpers import coerce_timestamp_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_candidate_staleness"
)
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH = (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    / SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME
)

SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME = (
    "latest_segmented_probability_candidate_staleness.json"
)
SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_MARKDOWN_FILENAME = (
    "latest_segmented_probability_candidate_staleness.md"
)

ACTIVE_REVIEW = "ACTIVE_REVIEW"
STALE_WATCH = "STALE_WATCH"
EXPIRED = "EXPIRED"
SUPERSEDED = "SUPERSEDED"
NO_CANDIDATE = "NO_CANDIDATE"


def _load_json_file(path: str | Path | None) -> dict[str, Any]:
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


def _load_history_frame(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _read_dataset(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    return timestamp


def _timestamp_series(frame: pd.DataFrame, column: str = "signal_timestamp") -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series(index=getattr(frame, "index", None), dtype="datetime64[ns, UTC]")
    parsed = coerce_timestamp_series(frame[column], utc=True)
    return parsed if isinstance(parsed, pd.Series) else pd.Series(parsed, index=frame.index)


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


def _candidate_generated_at(candidate_bundle: dict[str, Any]) -> str | None:
    generated_at = candidate_bundle.get("generated_at")
    if generated_at:
        return str(generated_at)
    for candidate in candidate_bundle.get("candidates", []) or []:
        if isinstance(candidate, dict) and candidate.get("generated_at"):
            return str(candidate.get("generated_at"))
    return None


def _candidate_age_days(candidate_generated_at: Any, *, as_of: Any = None) -> float | None:
    candidate_ts = _parse_timestamp(candidate_generated_at)
    if candidate_ts is None:
        return None
    as_of_ts = _parse_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if as_of_ts is None:
        as_of_ts = pd.Timestamp.now(tz="UTC")
    return max(float((as_of_ts - candidate_ts).total_seconds()) / 86400.0, 0.0)


def _frame_time_bounds(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = _timestamp_series(frame)
    valid = timestamps.dropna()
    if valid.empty:
        return {"earliest_signal_timestamp": None, "latest_signal_timestamp": None}
    return {
        "earliest_signal_timestamp": valid.min().isoformat(),
        "latest_signal_timestamp": valid.max().isoformat(),
    }


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


def _dataset_currency_summary(
    raw: pd.DataFrame,
    labeled: pd.DataFrame,
    *,
    dataset_path: str | Path | None,
    candidate_ts: pd.Timestamp | None,
    max_new_rows_before_stale: int,
    max_new_labeled_rows_before_stale: int,
) -> dict[str, Any]:
    raw_timestamps = _timestamp_series(raw)
    labeled_timestamps = _timestamp_series(labeled)
    raw_after = 0
    labeled_after = 0
    latest_after = None
    if candidate_ts is not None:
        raw_after_mask = raw_timestamps > candidate_ts
        labeled_after_mask = labeled_timestamps > candidate_ts
        raw_after = int(raw_after_mask.fillna(False).sum())
        labeled_after = int(labeled_after_mask.fillna(False).sum())
        post = raw_timestamps.loc[raw_after_mask.fillna(False)].dropna()
        latest_after = post.max().isoformat() if not post.empty else None

    bounds = _frame_time_bounds(raw)
    latest_ts = _parse_timestamp(bounds.get("latest_signal_timestamp"))
    latest_at_or_before_candidate = (
        bool(latest_ts <= candidate_ts) if latest_ts is not None and candidate_ts is not None else False
    )
    material_new_data = (
        raw_after >= int(max_new_rows_before_stale)
        or labeled_after >= int(max_new_labeled_rows_before_stale)
    )
    dataset_mtime = _dataset_modified_at(dataset_path)
    mtime_ts = _parse_timestamp(dataset_mtime)
    dataset_modified_after_candidate = (
        bool(mtime_ts > candidate_ts) if mtime_ts is not None and candidate_ts is not None else False
    )
    if candidate_ts is None:
        status = "CANDIDATE_TIMESTAMP_MISSING"
    elif raw_after <= 0:
        status = "CURRENT_AT_CANDIDATE_GENERATION"
    elif material_new_data:
        status = "MATERIAL_NEW_FORWARD_DATA"
    else:
        status = "FORWARD_ROWS_ACCUMULATING"

    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "row_count": int(len(raw)),
        "quality_labeled_row_count": int(len(labeled)),
        "earliest_signal_timestamp": bounds.get("earliest_signal_timestamp"),
        "latest_signal_timestamp": bounds.get("latest_signal_timestamp"),
        "latest_signal_timestamp_after_candidate": latest_after,
        "rows_after_candidate_generated": raw_after,
        "quality_labeled_rows_after_candidate_generated": labeled_after,
        "max_new_rows_before_stale": int(max_new_rows_before_stale),
        "max_new_labeled_rows_before_stale": int(max_new_labeled_rows_before_stale),
        "candidate_bundle_generated_from_latest_signal_timestamps": bool(latest_at_or_before_candidate),
        "material_new_data_since_candidate": bool(material_new_data),
        "dataset_modified_at": dataset_mtime,
        "dataset_modified_after_candidate_generated": bool(dataset_modified_after_candidate),
        "dataset_currency_status": status,
    }


def _categorical_psi(before: pd.Series, after: pd.Series) -> float | None:
    if before.empty or after.empty:
        return None
    before_values = before.fillna("UNKNOWN").astype(str)
    after_values = after.fillna("UNKNOWN").astype(str)
    categories = sorted(set(before_values.unique()) | set(after_values.unique()))
    if not categories:
        return None
    expected = before_values.value_counts(normalize=True)
    actual = after_values.value_counts(normalize=True)
    psi = 0.0
    epsilon = 1e-6
    for category in categories:
        expected_share = max(float(expected.get(category, 0.0)), epsilon)
        actual_share = max(float(actual.get(category, 0.0)), epsilon)
        psi += (actual_share - expected_share) * float(np.log(actual_share / expected_share))
    return float(psi)


def _population_shift_summary(
    labeled: pd.DataFrame,
    *,
    candidate_ts: pd.Timestamp | None,
    min_shift_sample: int,
    material_hit_rate_delta: float,
    material_probability_delta: float,
    material_distribution_psi: float,
) -> dict[str, Any]:
    if labeled.empty or candidate_ts is None or "signal_timestamp" not in labeled.columns:
        return {
            "shift_status": "INSUFFICIENT_FORWARD_LABELS",
            "shifted_materially": False,
            "pre_candidate_label_count": int(len(labeled)),
            "post_candidate_label_count": 0,
            "min_shift_sample": int(min_shift_sample),
        }

    timestamps = _timestamp_series(labeled)
    before = labeled.loc[(timestamps <= candidate_ts).fillna(False)].copy()
    after = labeled.loc[(timestamps > candidate_ts).fillna(False)].copy()
    pre_count = int(len(before))
    post_count = int(len(after))
    summary: dict[str, Any] = {
        "pre_candidate_label_count": pre_count,
        "post_candidate_label_count": post_count,
        "min_shift_sample": int(min_shift_sample),
        "material_hit_rate_delta": float(material_hit_rate_delta),
        "material_probability_delta": float(material_probability_delta),
        "material_distribution_psi": float(material_distribution_psi),
    }
    if pre_count <= 0 or post_count < int(min_shift_sample):
        summary.update(
            {
                "shift_status": "INSUFFICIENT_FORWARD_LABELS",
                "shifted_materially": False,
                "pre_candidate_hit_rate": _round_or_none(
                    before.get("_label", pd.Series(dtype=float)).mean(),
                    6,
                ),
                "post_candidate_hit_rate": _round_or_none(
                    after.get("_label", pd.Series(dtype=float)).mean(),
                    6,
                ),
            }
        )
        return summary

    pre_hit = _safe_float(before["_label"].mean())
    post_hit = _safe_float(after["_label"].mean())
    pre_prob = _safe_float(before["_probability"].mean())
    post_prob = _safe_float(after["_probability"].mean())
    hit_delta = None if pre_hit is None or post_hit is None else post_hit - pre_hit
    prob_delta = None if pre_prob is None or post_prob is None else post_prob - pre_prob
    psi_rows: list[dict[str, Any]] = []
    max_psi = None
    for column in ("direction", "macro_regime", "gamma_regime", "volatility_regime", "global_risk_state"):
        if column not in labeled.columns:
            continue
        psi = _categorical_psi(before[column], after[column])
        if psi is None:
            continue
        psi_rows.append({"field": column, "psi": _round_or_none(psi, 6)})
        max_psi = psi if max_psi is None else max(max_psi, psi)

    shifted = (
        (hit_delta is not None and abs(hit_delta) >= float(material_hit_rate_delta))
        or (prob_delta is not None and abs(prob_delta) >= float(material_probability_delta))
        or (max_psi is not None and max_psi >= float(material_distribution_psi))
    )
    summary.update(
        {
            "shift_status": "MATERIAL_SHIFT_DETECTED" if shifted else "NO_MATERIAL_SHIFT",
            "shifted_materially": bool(shifted),
            "pre_candidate_hit_rate": _round_or_none(pre_hit, 6),
            "post_candidate_hit_rate": _round_or_none(post_hit, 6),
            "hit_rate_delta": _round_or_none(hit_delta, 6),
            "pre_candidate_mean_probability": _round_or_none(pre_prob, 6),
            "post_candidate_mean_probability": _round_or_none(post_prob, 6),
            "mean_probability_delta": _round_or_none(prob_delta, 6),
            "distribution_psi": psi_rows,
            "max_distribution_psi": _round_or_none(max_psi, 6),
        }
    )
    return summary


def _routing_policy_stability(history: pd.DataFrame, *, lookback_runs: int) -> dict[str, Any]:
    if history is None or history.empty or "recommended_routing_policy" not in history.columns:
        return {
            "history_count": 0,
            "lookback_runs": int(lookback_runs),
            "policy_stability_status": "NO_HISTORY",
            "latest_recommended_routing_policy": None,
            "routing_policy_changed": False,
            "unique_recommended_policy_count": 0,
            "policy_change_count": 0,
        }
    lookback = history.tail(max(int(lookback_runs), 1)).copy()
    policy = lookback["recommended_routing_policy"].fillna("").astype(str).str.strip()
    policy = policy.loc[policy != ""]
    if policy.empty:
        return {
            "history_count": int(len(history)),
            "lookback_runs": int(lookback_runs),
            "policy_stability_status": "NO_RECOMMENDED_POLICY",
            "latest_recommended_routing_policy": None,
            "routing_policy_changed": False,
            "unique_recommended_policy_count": 0,
            "policy_change_count": 0,
        }
    change_count = int((policy != policy.shift()).sum() - 1) if len(policy) > 1 else 0
    unique_count = int(policy.nunique(dropna=True))
    changed = unique_count > 1 or change_count > 0
    return {
        "history_count": int(len(history)),
        "lookback_runs": int(lookback_runs),
        "policy_stability_status": "POLICY_CHANGED" if changed else "POLICY_STABLE",
        "latest_recommended_routing_policy": str(policy.iloc[-1]),
        "routing_policy_changed": bool(changed),
        "unique_recommended_policy_count": unique_count,
        "policy_change_count": max(change_count, 0),
        "policy_counts": {
            str(key): int(value) for key, value in policy.value_counts().sort_index().items()
        },
    }


def _looks_like_candidate_bundle(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("artifact_type") == "segmented_probability_calibration_candidate_bundle"
    )


def _supersession_summary(
    *,
    candidate_bundle_path: str | Path | None,
    candidate_bundle_search_dir: str | Path | None,
    candidate_generated_at: str | None,
) -> dict[str, Any]:
    candidate_ts = _parse_timestamp(candidate_generated_at)
    if candidate_ts is None:
        return {
            "superseded": False,
            "newer_candidate_bundle_path": None,
            "newer_candidate_generated_at": None,
            "candidate_bundle_search_dir": (
                str(candidate_bundle_search_dir) if candidate_bundle_search_dir else None
            ),
            "candidate_bundle_search_status": "CANDIDATE_TIMESTAMP_MISSING",
        }

    current_path = Path(candidate_bundle_path) if candidate_bundle_path is not None else None
    if candidate_bundle_search_dir is not None:
        search_dir = Path(candidate_bundle_search_dir)
    elif current_path is not None:
        search_dir = current_path.parent
    else:
        search_dir = None
    if search_dir is None or not search_dir.exists():
        return {
            "superseded": False,
            "newer_candidate_bundle_path": None,
            "newer_candidate_generated_at": None,
            "candidate_bundle_search_dir": str(search_dir) if search_dir is not None else None,
            "candidate_bundle_search_status": "SEARCH_DIR_MISSING",
        }

    current_resolved = current_path.resolve(strict=False) if current_path is not None else None
    newer: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(search_dir.glob("*.json")):
        if current_resolved is not None and path.resolve(strict=False) == current_resolved:
            continue
        payload = _load_json_file(path)
        if not _looks_like_candidate_bundle(payload):
            continue
        if _safe_int(payload.get("candidate_count")) <= 0:
            continue
        generated_at = _parse_timestamp(payload.get("generated_at"))
        if generated_at is not None and generated_at > candidate_ts:
            newer.append((generated_at, path))

    if not newer:
        return {
            "superseded": False,
            "newer_candidate_bundle_path": None,
            "newer_candidate_generated_at": None,
            "candidate_bundle_search_dir": str(search_dir),
            "candidate_bundle_search_status": "NO_NEWER_BUNDLE_FOUND",
        }
    newest_ts, newest_path = max(newer, key=lambda item: item[0])
    return {
        "superseded": True,
        "newer_candidate_bundle_path": str(newest_path),
        "newer_candidate_generated_at": newest_ts.isoformat(),
        "candidate_bundle_search_dir": str(search_dir),
        "candidate_bundle_search_status": "NEWER_BUNDLE_FOUND",
    }


def _staleness_actions(status: str, reasons: list[str]) -> list[str]:
    if status == ACTIVE_REVIEW:
        return [
            "Keep the candidate bundle in shadow review and continue accumulating true forward labels.",
            "Do not change runtime probabilities, parameter packs, data sources, or execution behavior from this report.",
        ]
    if status == SUPERSEDED:
        return [
            "Switch research review to the newer candidate bundle artifact before any manual promotion discussion.",
            "Archive the superseded bundle as historical evidence; do not adopt it.",
        ]
    if status == EXPIRED:
        return [
            "Regenerate the segmented probability calibration experiment on the latest quality-approved dataset.",
            "Restart forward-shadow accumulation for the regenerated candidate bundle.",
        ]
    actions = []
    joined = " ".join(reasons)
    if "material_new_data" in joined or "forward_label_population_shifted" in joined:
        actions.append("Refresh the segmented calibration experiment after enough forward labels have accumulated.")
    if "routing_policy_changed" in joined:
        actions.append("Review routing-policy stability before advancing the candidate bundle.")
    if "candidate_age" in joined:
        actions.append("Regenerate the candidate bundle if the watch state persists into the expiry window.")
    if not actions:
        actions.append("Keep the candidate under watch and continue collecting forward evidence.")
    actions.append("No automated runtime adoption is authorized by this research-only report.")
    return actions


def build_segmented_probability_candidate_staleness_report(
    *,
    dataset: pd.DataFrame,
    candidate_bundle: dict[str, Any],
    history: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    candidate_bundle_path: str | Path | None = None,
    candidate_bundle_search_dir: str | Path | None = None,
    history_path: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    stale_after_days: float = 7.0,
    expire_after_days: float = 14.0,
    max_new_rows_before_stale: int = 500,
    max_new_labeled_rows_before_stale: int = 100,
    min_shift_sample: int = 100,
    material_hit_rate_delta: float = 0.10,
    material_probability_delta: float = 0.08,
    material_distribution_psi: float = 0.20,
    policy_lookback_runs: int = 10,
    as_of: Any = None,
) -> dict[str, Any]:
    """Build a staleness report for a segmented probability candidate bundle."""
    raw = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame()
    bundle = candidate_bundle if isinstance(candidate_bundle, dict) else {}
    hist = history if history is not None else pd.DataFrame()
    candidate_count = _safe_int(bundle.get("candidate_count"))
    generated_at = _candidate_generated_at(bundle)
    generated_ts = _parse_timestamp(generated_at)
    age_days = _candidate_age_days(generated_at, as_of=as_of)
    labeled = _clean_probability_and_label_frame(
        raw,
        probability_field=probability_field,
        label_field=label_field,
    )
    dataset_currency = _dataset_currency_summary(
        raw,
        labeled,
        dataset_path=dataset_path,
        candidate_ts=generated_ts,
        max_new_rows_before_stale=max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=max_new_labeled_rows_before_stale,
    )
    population_shift = _population_shift_summary(
        labeled,
        candidate_ts=generated_ts,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
    )
    routing_stability = _routing_policy_stability(hist, lookback_runs=policy_lookback_runs)
    supersession = _supersession_summary(
        candidate_bundle_path=candidate_bundle_path,
        candidate_bundle_search_dir=candidate_bundle_search_dir,
        candidate_generated_at=generated_at,
    )

    reasons: list[str] = []
    if candidate_count <= 0:
        reasons.append("no_candidate_routes_available")
    if generated_ts is None:
        reasons.append("candidate_generated_at_missing_or_invalid")
    if age_days is not None and age_days > float(expire_after_days):
        reasons.append("candidate_age_exceeds_expiry_window")
    elif age_days is not None and age_days > float(stale_after_days):
        reasons.append("candidate_age_exceeds_watch_window")
    if dataset_currency.get("material_new_data_since_candidate"):
        reasons.append("material_new_data_since_candidate_generation")
    if population_shift.get("shifted_materially"):
        reasons.append("forward_label_population_shifted")
    if routing_stability.get("routing_policy_changed"):
        reasons.append("recommended_routing_policy_changed")
    if supersession.get("superseded"):
        reasons.append("newer_candidate_bundle_supersedes_current")

    if supersession.get("superseded"):
        status = SUPERSEDED
    elif candidate_count <= 0:
        status = NO_CANDIDATE
    elif generated_ts is None or (age_days is not None and age_days > float(expire_after_days)):
        status = EXPIRED
    elif reasons:
        status = STALE_WATCH
    else:
        status = ACTIVE_REVIEW

    checked_conditions = {
        "candidate_count_positive": candidate_count > 0,
        "candidate_generated_at_valid": generated_ts is not None,
        "candidate_age_within_watch_window": bool(age_days is not None and age_days <= float(stale_after_days)),
        "candidate_age_within_expiry_window": bool(age_days is not None and age_days <= float(expire_after_days)),
        "candidate_bundle_generated_from_latest_signal_timestamps": bool(
            dataset_currency.get("candidate_bundle_generated_from_latest_signal_timestamps")
        ),
        "material_new_data_since_candidate": bool(dataset_currency.get("material_new_data_since_candidate")),
        "forward_label_population_shifted": bool(population_shift.get("shifted_materially")),
        "routing_policy_stable": not bool(routing_stability.get("routing_policy_changed")),
        "candidate_bundle_superseded": bool(supersession.get("superseded")),
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
    }

    report = {
        "report_type": "segmented_probability_candidate_staleness",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "candidate_bundle_path": str(candidate_bundle_path) if candidate_bundle_path is not None else None,
        "candidate_bundle_search_dir": str(candidate_bundle_search_dir) if candidate_bundle_search_dir is not None else None,
        "history_path": str(history_path) if history_path is not None else None,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "staleness_status": status,
        "staleness_reasons": reasons,
        "candidate_summary": {
            "artifact_type": bundle.get("artifact_type"),
            "candidate_count": candidate_count,
            "candidate_generated_at": generated_at,
            "candidate_age_days": _round_or_none(age_days, 6),
            "stale_after_days": float(stale_after_days),
            "expire_after_days": float(expire_after_days),
            "calibration_status": bundle.get("calibration_status"),
            "research_only": bool(bundle.get("research_only", True)),
            "approval_required_for_runtime_use": bool(bundle.get("approval_required_for_runtime_use", True)),
        },
        "dataset_currency": dataset_currency,
        "forward_label_population_shift": population_shift,
        "routing_policy_stability": routing_stability,
        "supersession": supersession,
        "checked_conditions": checked_conditions,
        "recommended_next_actions": _staleness_actions(status, reasons),
    }
    return _sanitize_value(report)


def render_segmented_probability_candidate_staleness_markdown(report: dict[str, Any]) -> str:
    """Render candidate staleness governance output as Markdown."""
    candidate = report.get("candidate_summary", {}) or {}
    currency = report.get("dataset_currency", {}) or {}
    shift = report.get("forward_label_population_shift", {}) or {}
    routing = report.get("routing_policy_stability", {}) or {}
    supersession = report.get("supersession", {}) or {}
    lines = [
        "# Segmented Probability Candidate Staleness",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Staleness status: **{report.get('staleness_status')}**",
        f"- Candidate generated at: {candidate.get('candidate_generated_at')}",
        f"- Candidate age days: {candidate.get('candidate_age_days')} / {candidate.get('expire_after_days')}",
        f"- Candidate count: {candidate.get('candidate_count')}",
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
        f"| Rows after candidate | {currency.get('rows_after_candidate_generated')} |",
        f"| Quality labels after candidate | {currency.get('quality_labeled_rows_after_candidate_generated')} |",
        f"| Forward-label shift status | `{shift.get('shift_status')}` |",
        f"| Post-candidate labels | {shift.get('post_candidate_label_count')} / {shift.get('min_shift_sample')} |",
        f"| Hit-rate delta | {shift.get('hit_rate_delta')} |",
        f"| Max distribution PSI | {shift.get('max_distribution_psi')} |",
        f"| Routing policy status | `{routing.get('policy_stability_status')}` |",
        f"| Latest routing policy | `{routing.get('latest_recommended_routing_policy')}` |",
        f"| Supersession status | `{supersession.get('candidate_bundle_search_status')}` |",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("staleness_reasons", []) or ["active_review"]:
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


def write_segmented_probability_candidate_staleness_report(
    *,
    dataset_path: str | Path | None = None,
    candidate_bundle_path: str | Path | None = None,
    candidate_bundle_search_dir: str | Path | None = None,
    history_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    stale_after_days: float = 7.0,
    expire_after_days: float = 14.0,
    max_new_rows_before_stale: int = 500,
    max_new_labeled_rows_before_stale: int = 100,
    min_shift_sample: int = 100,
    material_hit_rate_delta: float = 0.10,
    material_probability_delta: float = 0.08,
    material_distribution_psi: float = 0.20,
    policy_lookback_runs: int = 10,
    as_of: Any = None,
) -> dict[str, Any]:
    """Read latest artifacts, build staleness report, and write JSON/Markdown."""
    dataset = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    bundle_path = (
        Path(candidate_bundle_path)
        if candidate_bundle_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    )
    hist_path = (
        Path(history_path)
        if history_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_PATH
    )
    search_dir = (
        Path(candidate_bundle_search_dir)
        if candidate_bundle_search_dir is not None
        else bundle_path.parent
    )
    raw = _read_dataset(dataset)
    bundle = _load_json_file(bundle_path)
    history = _load_history_frame(hist_path)
    report = build_segmented_probability_candidate_staleness_report(
        dataset=raw,
        candidate_bundle=bundle,
        history=history,
        dataset_path=dataset,
        candidate_bundle_path=bundle_path,
        candidate_bundle_search_dir=search_dir,
        history_path=hist_path,
        probability_field=probability_field,
        label_field=label_field,
        stale_after_days=stale_after_days,
        expire_after_days=expire_after_days,
        max_new_rows_before_stale=max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=max_new_labeled_rows_before_stale,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
        policy_lookback_runs=policy_lookback_runs,
        as_of=as_of,
    )
    assert_artifact_schema(report, "segmented_probability_candidate_staleness")
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME
    markdown_path = output / SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_segmented_probability_candidate_staleness_markdown(report))
    return {
        "staleness_json_path": str(json_path),
        "staleness_markdown_path": str(markdown_path),
        "staleness_report": report,
    }
