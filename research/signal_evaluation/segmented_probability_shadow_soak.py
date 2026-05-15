"""One-command shadow soak workflow for segmented-probability candidates.

This module chains the research-only forward-shadow evidence loop:
accumulation, candidate staleness, EV shadow context, guard-aware validation,
and readiness gating. It only writes advisory artifacts. It never changes
runtime configuration, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.label_quality import select_quality_labeled_rows
from research.signal_evaluation.evaluator import update_signal_dataset_outcomes
from research.signal_evaluation.segmented_probability_candidate_staleness import (
    ACTIVE_REVIEW,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR,
    SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME,
    write_segmented_probability_candidate_staleness_report,
)
from research.signal_evaluation.segmented_probability_ev_shadow_evaluation import (
    DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR,
    SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME,
    write_segmented_probability_ev_shadow_evaluation_report_from_path,
)
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_ROUTING_POLICIES,
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR,
)
from research.signal_evaluation.segmented_probability_forward_shadow_accumulator import (
    ACCUMULATION_TRUE_FORWARD_PASS,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
    write_segmented_probability_forward_shadow_accumulation,
)
from research.signal_evaluation.segmented_probability_forward_shadow_readiness import (
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR,
    FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW,
    SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_JSON_FILENAME,
    write_segmented_probability_forward_shadow_readiness_report,
)
from research.signal_evaluation.segmented_probability_guarded_candidate_staleness import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR,
    GUARDED_STALENESS_NON_BLOCKING,
    SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_JSON_FILENAME,
    write_segmented_probability_guarded_candidate_staleness_report,
)
from research.signal_evaluation.segmented_probability_guarded_shadow_validation import (
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR,
    GUARDED_SHADOW_VALIDATION_PASS,
    GUARDED_SHADOW_VALIDATION_REJECTED,
    SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME,
    write_segmented_probability_guarded_shadow_validation_report_from_path,
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
from data.spot_history import load_spot_history


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_shadow_soak"
)

SEGMENTED_PROBABILITY_SHADOW_SOAK_JSON_FILENAME = "latest_segmented_probability_shadow_soak.json"
SEGMENTED_PROBABILITY_SHADOW_SOAK_MARKDOWN_FILENAME = "latest_segmented_probability_shadow_soak.md"
SEGMENTED_PROBABILITY_SHADOW_SOAK_HISTORY_FILENAME = "segmented_probability_shadow_soak_history.csv"

SOAK_READY_FOR_MANUAL_REVIEW = "SOAK_READY_FOR_MANUAL_REVIEW"
SOAK_HOLDOUT_REPLAY_REVIEWABLE = "SOAK_HOLDOUT_REPLAY_REVIEWABLE"
SOAK_ACCUMULATING_TRUE_FORWARD_LABELS = "SOAK_ACCUMULATING_TRUE_FORWARD_LABELS"
SOAK_GUARDED_VALIDATION_REJECTED = "SOAK_GUARDED_VALIDATION_REJECTED"
SOAK_CANDIDATE_STALENESS_BLOCKED = "SOAK_CANDIDATE_STALENESS_BLOCKED"
SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED = "SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED"
SOAK_SIDE_EFFECT_BLOCKED = "SOAK_SIDE_EFFECT_BLOCKED"
SOAK_READINESS_BLOCKED = "SOAK_READINESS_BLOCKED"

OUTCOME_REFRESH_LOCAL_SPOT_HISTORY = "local_spot_history"
OUTCOME_REFRESH_DEFAULT_PROVIDER = "default_provider"
OUTCOME_REFRESH_SKIP = "skip"


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _csv_tuple(value: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


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


def _file_digest(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_history_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    return _sanitize_value(frame.iloc[-1].to_dict())


def _dataset_summary(path: str | Path, *, label_field: str) -> dict[str, Any]:
    dataset_path = Path(path)
    summary: dict[str, Any] = {
        "dataset_exists": dataset_path.exists(),
        "dataset_size_bytes": None,
        "dataset_modified_at": None,
        "row_count": 0,
        "quality_labeled_row_count": 0,
        "latest_signal_timestamp": None,
        "dataset_read_error": None,
    }
    if not dataset_path.exists():
        return summary
    try:
        stat = dataset_path.stat()
        summary["dataset_size_bytes"] = int(stat.st_size)
        summary["dataset_modified_at"] = pd.Timestamp.fromtimestamp(stat.st_mtime, tz="UTC").isoformat()
        frame = pd.read_csv(dataset_path, low_memory=False)
    except Exception as exc:
        summary["dataset_read_error"] = str(exc)
        return summary

    summary["row_count"] = int(len(frame))
    try:
        summary["quality_labeled_row_count"] = int(len(select_quality_labeled_rows(frame)))
    except Exception:
        labels = pd.to_numeric(frame.get(label_field, pd.Series(dtype=float)), errors="coerce")
        summary["quality_labeled_row_count"] = int(labels.notna().sum())
    if "signal_timestamp" in frame.columns and not frame.empty:
        timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True)
        if timestamps.notna().any():
            summary["latest_signal_timestamp"] = timestamps.max().isoformat()
    return _sanitize_value(summary)


def _candidate_generated_at(candidate_bundle_path: str | Path | None) -> pd.Timestamp | None:
    if candidate_bundle_path is None:
        return None
    try:
        payload = json.loads(Path(candidate_bundle_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    timestamp = pd.to_datetime(payload.get("generated_at"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp


def _post_candidate_label_summary(
    path: str | Path,
    *,
    candidate_bundle_path: str | Path | None,
    label_field: str,
) -> dict[str, Any]:
    dataset_path = Path(path)
    candidate_ts = _candidate_generated_at(candidate_bundle_path)
    summary: dict[str, Any] = {
        "candidate_generated_at": candidate_ts.isoformat() if candidate_ts is not None else None,
        "rows_after_candidate": 0,
        "quality_labeled_rows_after_candidate": 0,
        "raw_labeled_rows_after_candidate": 0,
        "pending_rows_after_candidate": 0,
        "partial_rows_after_candidate": 0,
        "complete_rows_after_candidate": 0,
        "latest_post_candidate_signal_timestamp": None,
        "label_quality_status_counts_after_candidate": {},
        "label_quality_reason_counts_after_candidate": {},
        "summary_error": None,
    }
    if candidate_ts is None or not dataset_path.exists():
        return summary
    try:
        frame = pd.read_csv(dataset_path, low_memory=False)
    except Exception as exc:
        summary["summary_error"] = str(exc)
        return summary
    if frame.empty or "signal_timestamp" not in frame.columns:
        return summary
    timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True)
    post = frame.loc[timestamps > candidate_ts].copy()
    summary["rows_after_candidate"] = int(len(post))
    if post.empty:
        return _sanitize_value(summary)
    latest_ts = pd.to_datetime(post["signal_timestamp"], errors="coerce", utc=True)
    if latest_ts.notna().any():
        summary["latest_post_candidate_signal_timestamp"] = latest_ts.max().isoformat()
    statuses = post.get("outcome_status", pd.Series(index=post.index, dtype=object)).fillna("UNKNOWN").astype(str).str.upper()
    summary["pending_rows_after_candidate"] = int((statuses == "PENDING").sum())
    summary["partial_rows_after_candidate"] = int((statuses == "PARTIAL").sum())
    summary["complete_rows_after_candidate"] = int((statuses == "COMPLETE").sum())
    raw_labels = pd.to_numeric(post.get(label_field, pd.Series(index=post.index, dtype=float)), errors="coerce")
    summary["raw_labeled_rows_after_candidate"] = int(raw_labels.notna().sum())
    try:
        approved = select_quality_labeled_rows(post)
        summary["quality_labeled_rows_after_candidate"] = int(len(approved))
    except Exception:
        summary["quality_labeled_rows_after_candidate"] = int(raw_labels.notna().sum())
    quality_counts = (
        post.get("label_quality_status", pd.Series(index=post.index, dtype=object))
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .value_counts()
    )
    summary["label_quality_status_counts_after_candidate"] = {
        str(key): int(value) for key, value in quality_counts.items()
    }
    reason_counts: dict[str, int] = {}
    if "label_quality_reasons" in post.columns:
        for value in post["label_quality_reasons"].dropna().astype(str):
            for reason in value.split("|"):
                token = reason.strip()
                if token:
                    reason_counts[token] = reason_counts.get(token, 0) + 1
    summary["label_quality_reason_counts_after_candidate"] = dict(
        sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    )
    return _sanitize_value(summary)


def _local_spot_history_fetch(symbol: str, *, start_ts: Any, end_ts: Any, interval: str = "5m") -> pd.DataFrame:
    _ = interval
    return load_spot_history(symbol, start_ts=start_ts, end_ts=end_ts)


def _refresh_outcomes_if_requested(
    *,
    dataset_path: str | Path,
    outcome_refresh_source: str,
    as_of: Any = None,
) -> dict[str, Any]:
    source = str(outcome_refresh_source or OUTCOME_REFRESH_LOCAL_SPOT_HISTORY).strip().lower()
    if source not in {
        OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
        OUTCOME_REFRESH_DEFAULT_PROVIDER,
        OUTCOME_REFRESH_SKIP,
    }:
        source = OUTCOME_REFRESH_LOCAL_SPOT_HISTORY
    summary: dict[str, Any] = {
        "outcome_refresh_source": source,
        "outcome_refresh_attempted": source != OUTCOME_REFRESH_SKIP,
        "outcome_refresh_error": None,
        "outcome_refresh_row_count": 0,
    }
    if source == OUTCOME_REFRESH_SKIP:
        return summary
    try:
        kwargs: dict[str, Any] = {"dataset_path": dataset_path, "as_of": as_of}
        if source == OUTCOME_REFRESH_LOCAL_SPOT_HISTORY:
            kwargs["fetch_spot_history_fn"] = _local_spot_history_fetch
        updated = update_signal_dataset_outcomes(**kwargs)
        summary["outcome_refresh_row_count"] = int(len(updated))
    except Exception as exc:
        summary["outcome_refresh_error"] = str(exc)
    return _sanitize_value(summary)


def _previous_forward_rows(history_before: pd.DataFrame, current_row: dict[str, Any]) -> int | None:
    latest = _latest_history_row(history_before)
    if not latest:
        return None
    same_candidate = str(latest.get("candidate_generated_at")) == str(current_row.get("candidate_generated_at"))
    same_bundle = str(latest.get("candidate_bundle_path")) == str(current_row.get("candidate_bundle_path"))
    if not same_candidate or not same_bundle:
        return None
    return _safe_int(latest.get("strict_forward_row_count"))


def _previous_guarded_forward_rows(
    history_before: pd.DataFrame,
    *,
    guarded_bundle_path: str | Path,
    guarded_generated_at: Any,
) -> int | None:
    if history_before is None or history_before.empty or guarded_generated_at is None:
        return None
    frame = history_before.copy()
    if "guarded_candidate_bundle_path" not in frame.columns or "guarded_candidate_generated_at" not in frame.columns:
        return None
    bundle_values = frame["guarded_candidate_bundle_path"].fillna("").astype(str)
    generated_values = frame["guarded_candidate_generated_at"].fillna("").astype(str)
    matches = frame.loc[
        (bundle_values == str(guarded_bundle_path))
        & (generated_values == str(guarded_generated_at))
    ].copy()
    if matches.empty:
        return None
    latest = _sanitize_value(matches.iloc[-1].to_dict())
    return _safe_int(latest.get("guarded_strict_forward_row_count"))


def append_segmented_probability_shadow_soak_history(
    row: dict[str, Any],
    history_path: str | Path,
) -> pd.DataFrame:
    """Append one guarded shadow-soak observation using atomic CSV persistence."""
    path = Path(history_path)
    incoming = pd.DataFrame([_sanitize_value(row)])
    with _exclusive_file_lock(path):
        if path.exists():
            try:
                existing = pd.read_csv(path, low_memory=False)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()
        history = pd.concat([existing, incoming], ignore_index=True, sort=False) if not existing.empty else incoming
        _atomic_write_csv(history, path)
    return history


def _side_effects_absent(*reports: dict[str, Any]) -> bool:
    fields = ("runtime_config_changed", "parameter_pack_file_changed", "execution_behavior_changed")
    for report in reports:
        if not isinstance(report, dict):
            continue
        for field in fields:
            if bool(report.get(field)):
                return False
    return True


def _guarded_policy_row(guarded_report: dict[str, Any]) -> dict[str, Any]:
    selection = guarded_report.get("selection_summary", {}) if isinstance(guarded_report, dict) else {}
    policy = selection.get("recommended_routing_policy")
    if not policy:
        return {}
    for row in guarded_report.get("policy_results", []) or []:
        if isinstance(row, dict) and str(row.get("route_policy")) == str(policy):
            return row
    return {}


def _soak_reasons(
    *,
    status: str,
    readiness_reasons: list[str],
    staleness_reasons: list[str],
    guarded_staleness_reasons: list[str],
    guarded_staleness_primary: bool,
    no_new_forward_rows: bool,
    no_new_guarded_forward_rows: bool,
) -> list[str]:
    reasons: list[str] = []
    if status == SOAK_READY_FOR_MANUAL_REVIEW:
        return reasons
    if status == SOAK_HOLDOUT_REPLAY_REVIEWABLE:
        reasons.append("guarded_holdout_replay_review_enabled")
    if no_new_forward_rows:
        reasons.append("no_new_true_forward_rows_since_previous_soak")
    if no_new_guarded_forward_rows:
        reasons.append("no_new_post_guarded_true_forward_rows_since_previous_soak")
    filtered_readiness = list(readiness_reasons)
    if guarded_staleness_primary and status != SOAK_CANDIDATE_STALENESS_BLOCKED:
        original_candidate_gate_reasons = {
            "candidate_staleness_status_not_active_review",
            "candidate_staleness_routing_policy_unstable",
            "recommended_routing_policy_changed",
            "recommended_routing_policy_not_stable",
        }
        filtered_readiness = [
            reason
            for reason in readiness_reasons
            if str(reason).strip() not in original_candidate_gate_reasons
        ]
    reasons.extend(str(reason) for reason in filtered_readiness if str(reason).strip())
    if status == SOAK_CANDIDATE_STALENESS_BLOCKED or not guarded_staleness_primary:
        reasons.extend(str(reason) for reason in staleness_reasons if str(reason).strip())
    if status == SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED:
        reasons.extend(str(reason) for reason in guarded_staleness_reasons if str(reason).strip())
    return list(dict.fromkeys(reasons))


def _classify_soak_status(
    *,
    readiness_report: dict[str, Any],
    guarded_report: dict[str, Any],
    staleness_report: dict[str, Any],
    guarded_staleness_report: dict[str, Any],
    accumulation_row: dict[str, Any],
    candidate_bundle_unchanged: bool,
    guarded_bundle_unchanged: bool,
    side_effects_absent: bool,
    allow_holdout_replay_guarded_validation: bool,
) -> str:
    readiness_status = readiness_report.get("readiness_status")
    checks = readiness_report.get("checked_conditions", {}) if isinstance(readiness_report, dict) else {}
    guarded_status = guarded_report.get("guarded_shadow_status")
    staleness_status = staleness_report.get("staleness_status")
    guarded_staleness_status = guarded_staleness_report.get("guarded_staleness_status")
    mode = str(checks.get("validation_mode_used") or accumulation_row.get("validation_mode_used") or "")
    strict_forward = _safe_int(checks.get("strict_forward_row_count", accumulation_row.get("strict_forward_row_count")))
    min_forward = _safe_int(checks.get("min_forward_sample", accumulation_row.get("min_shadow_sample")), 100)

    if not candidate_bundle_unchanged or not guarded_bundle_unchanged or not side_effects_absent:
        return SOAK_SIDE_EFFECT_BLOCKED
    if guarded_status == GUARDED_SHADOW_VALIDATION_REJECTED:
        return SOAK_GUARDED_VALIDATION_REJECTED
    if guarded_staleness_status:
        if guarded_staleness_status not in GUARDED_STALENESS_NON_BLOCKING:
            return SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED
    elif staleness_status and staleness_status != ACTIVE_REVIEW:
        return SOAK_CANDIDATE_STALENESS_BLOCKED
    if readiness_status == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW:
        if allow_holdout_replay_guarded_validation and mode != "after_candidate_generated":
            return SOAK_HOLDOUT_REPLAY_REVIEWABLE
        return SOAK_READY_FOR_MANUAL_REVIEW
    if guarded_status == GUARDED_SHADOW_VALIDATION_PASS and (
        mode != "after_candidate_generated"
        or strict_forward < min_forward
        or accumulation_row.get("accumulation_status") != ACCUMULATION_TRUE_FORWARD_PASS
    ):
        return SOAK_ACCUMULATING_TRUE_FORWARD_LABELS
    return SOAK_READINESS_BLOCKED


def _recommended_actions(report: dict[str, Any]) -> list[str]:
    status = report.get("soak_status")
    progress = report.get("forward_sample_progress", {}) or {}
    guarded_progress = report.get("guarded_forward_sample_progress", {}) or {}
    gap = _safe_int(progress.get("forward_sample_gap"))
    guarded_gap = _safe_int(guarded_progress.get("forward_sample_gap"))
    actions: list[str]
    if guarded_gap > 0:
        actions = [
            (
                "Keep collecting fresh signals after the guarded bundle timestamp; "
                f"current guarded forward sample gap is {guarded_gap}."
            )
        ]
    else:
        actions = []
    if status == SOAK_READY_FOR_MANUAL_REVIEW:
        actions.extend([
            "Open manual calibration-governance review for the candidate; do not apply it to runtime without explicit approval.",
            "Archive the soak report with the readiness artifact used for review.",
        ])
    elif status == SOAK_HOLDOUT_REPLAY_REVIEWABLE:
        actions.extend([
            "Use this only for research review; continue collecting true post-candidate labels before runtime adoption.",
        ])
    elif status == SOAK_ACCUMULATING_TRUE_FORWARD_LABELS:
        actions.extend([
            f"Run this soak command again after new realized outcomes are labeled; current forward sample gap is {gap}.",
            "If the gap does not shrink after market sessions, inspect the outcome-label pipeline.",
        ])
    elif status == SOAK_GUARDED_VALIDATION_REJECTED:
        actions.extend([
            "Do not advance the guarded bundle; return to guarded EV diagnostics or candidate generation.",
        ])
    elif status == SOAK_CANDIDATE_STALENESS_BLOCKED:
        actions.extend([
            "Refresh candidate evidence before review; the staleness gate is no longer active-review clean.",
        ])
    elif status == SOAK_GUARDED_BUNDLE_STALENESS_BLOCKED:
        actions.extend([
            "Refresh guarded-bundle evidence before review; the guarded staleness gate is no longer clean.",
        ])
    elif status == SOAK_SIDE_EFFECT_BLOCKED:
        actions.extend([
            "Stop review and inspect artifact side-effect flags or bundle hashes before trusting this run.",
        ])
    else:
        actions.extend([
            "Inspect readiness reasons and keep the candidate in research-only shadow monitoring.",
        ])
    actions.append("This workflow must not change runtime config, parameter packs, data sources, or execution behavior.")
    return actions


def build_segmented_probability_shadow_soak_report(
    *,
    dataset_path: str | Path,
    candidate_bundle_path: str | Path,
    guarded_candidate_bundle_path: str | Path,
    history_before: pd.DataFrame,
    accumulation_artifact: dict[str, Any],
    staleness_artifact: dict[str, Any],
    guarded_staleness_artifact: dict[str, Any] | None,
    ev_shadow_artifact: dict[str, Any] | None,
    guarded_shadow_artifact: dict[str, Any],
    readiness_artifact: dict[str, Any],
    outcome_refresh_summary: dict[str, Any] | None,
    pre_refresh_label_summary: dict[str, Any] | None,
    post_refresh_label_summary: dict[str, Any] | None,
    pre_guarded_refresh_label_summary: dict[str, Any] | None,
    post_guarded_refresh_label_summary: dict[str, Any] | None,
    guarded_history_before: pd.DataFrame | None,
    guarded_history_path: str | Path | None,
    candidate_bundle_digest_before: str | None,
    candidate_bundle_digest_after: str | None,
    guarded_bundle_digest_before: str | None,
    guarded_bundle_digest_after: str | None,
    min_forward_sample: int,
    allow_holdout_replay_guarded_validation: bool = False,
    refresh_legacy_ev_shadow: bool = True,
    outcome_refresh_source: str = OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> dict[str, Any]:
    """Build a compact soak-status report from freshly generated artifacts."""
    accumulation_row = accumulation_artifact.get("history_row", {}) or {}
    accumulation_dashboard = accumulation_artifact.get("accumulation_dashboard", {}) or {}
    forward_shadow_artifact = accumulation_artifact.get("forward_shadow_artifact", {}) or {}
    staleness_report = staleness_artifact.get("staleness_report", {}) or {}
    guarded_staleness_report = (guarded_staleness_artifact or {}).get("guarded_staleness_report", {}) or {}
    ev_shadow_report = (ev_shadow_artifact or {}).get("report", {}) or {}
    guarded_report = guarded_shadow_artifact.get("report", {}) or {}
    readiness_report = readiness_artifact.get("readiness_report", {}) or {}
    readiness_checks = readiness_report.get("checked_conditions", {}) if isinstance(readiness_report, dict) else {}
    guarded_policy = _guarded_policy_row(guarded_report)
    guarded_window = guarded_report.get("validation_window", {}) if isinstance(guarded_report, dict) else {}
    guarded_strict_forward = _safe_int(guarded_window.get("strict_forward_row_count"))
    guarded_generated_at = (post_guarded_refresh_label_summary or {}).get("candidate_generated_at")
    previous_guarded_rows = _previous_guarded_forward_rows(
        guarded_history_before if guarded_history_before is not None else pd.DataFrame(),
        guarded_bundle_path=guarded_candidate_bundle_path,
        guarded_generated_at=guarded_generated_at,
    )

    previous_rows = _previous_forward_rows(history_before, accumulation_row)
    strict_forward = _safe_int(
        readiness_checks.get("strict_forward_row_count", accumulation_row.get("strict_forward_row_count"))
    )
    new_forward_rows = None if previous_rows is None else max(strict_forward - previous_rows, 0)
    forward_gap = max(int(min_forward_sample) - strict_forward, 0)
    candidate_bundle_unchanged = candidate_bundle_digest_before == candidate_bundle_digest_after
    guarded_bundle_unchanged = guarded_bundle_digest_before == guarded_bundle_digest_after
    clean_side_effects = _side_effects_absent(
        staleness_report,
        guarded_staleness_report,
        ev_shadow_report,
        guarded_report,
        readiness_report,
    )

    status = _classify_soak_status(
        readiness_report=readiness_report,
        guarded_report=guarded_report,
        staleness_report=staleness_report,
        guarded_staleness_report=guarded_staleness_report,
        accumulation_row=accumulation_row,
        candidate_bundle_unchanged=bool(candidate_bundle_unchanged),
        guarded_bundle_unchanged=bool(guarded_bundle_unchanged),
        side_effects_absent=bool(clean_side_effects),
        allow_holdout_replay_guarded_validation=allow_holdout_replay_guarded_validation,
    )
    no_new_forward_rows = bool(previous_rows is not None and new_forward_rows == 0 and strict_forward < int(min_forward_sample))
    guarded_forward_gap = max(int(min_forward_sample) - guarded_strict_forward, 0)
    new_guarded_forward_rows = (
        None
        if previous_guarded_rows is None
        else max(guarded_strict_forward - previous_guarded_rows, 0)
    )
    no_new_guarded_forward_rows = bool(
        previous_guarded_rows is not None
        and new_guarded_forward_rows == 0
        and guarded_strict_forward < int(min_forward_sample)
    )
    reasons = _soak_reasons(
        status=status,
        readiness_reasons=list(readiness_report.get("readiness_reasons", []) or []),
        staleness_reasons=list(staleness_report.get("staleness_reasons", []) or []),
        guarded_staleness_reasons=list(guarded_staleness_report.get("guarded_staleness_reasons", []) or []),
        guarded_staleness_primary=bool(guarded_staleness_report.get("guarded_staleness_status")),
        no_new_forward_rows=no_new_forward_rows,
        no_new_guarded_forward_rows=no_new_guarded_forward_rows,
    )
    dataset_info = _dataset_summary(dataset_path, label_field=label_field)
    report = {
        "report_type": "segmented_probability_shadow_soak",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path),
        "candidate_bundle_path": str(candidate_bundle_path),
        "guarded_candidate_bundle_path": str(guarded_candidate_bundle_path),
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "soak_status": status,
        "soak_reasons": reasons,
        "dataset_summary": dataset_info,
        "outcome_refresh_summary": {
            "outcome_refresh_source": outcome_refresh_source,
            **(outcome_refresh_summary or {}),
            "pre_refresh": pre_refresh_label_summary or {},
            "post_refresh": post_refresh_label_summary or {},
            "new_quality_labeled_rows_after_candidate": max(
                _safe_int((post_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
                - _safe_int((pre_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate")),
                0,
            ),
            "new_raw_labeled_rows_after_candidate": max(
                _safe_int((post_refresh_label_summary or {}).get("raw_labeled_rows_after_candidate"))
                - _safe_int((pre_refresh_label_summary or {}).get("raw_labeled_rows_after_candidate")),
                0,
            ),
            "guarded_pre_refresh": pre_guarded_refresh_label_summary or {},
            "guarded_post_refresh": post_guarded_refresh_label_summary or {},
            "new_quality_labeled_rows_after_guarded_candidate": max(
                _safe_int((post_guarded_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
                - _safe_int((pre_guarded_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate")),
                0,
            ),
            "new_raw_labeled_rows_after_guarded_candidate": max(
                _safe_int((post_guarded_refresh_label_summary or {}).get("raw_labeled_rows_after_candidate"))
                - _safe_int((pre_guarded_refresh_label_summary or {}).get("raw_labeled_rows_after_candidate")),
                0,
            ),
        },
        "forward_sample_progress": {
            "validation_mode_used": accumulation_row.get("validation_mode_used"),
            "strict_forward_row_count": strict_forward,
            "min_forward_sample": int(min_forward_sample),
            "forward_sample_gap": int(forward_gap),
            "forward_sample_progress_ratio": _round_or_none(strict_forward / max(int(min_forward_sample), 1), 6),
            "previous_strict_forward_row_count": previous_rows,
            "new_true_forward_rows_since_previous_soak": new_forward_rows,
            "accumulation_status": accumulation_row.get("accumulation_status"),
            "trend_assessment": accumulation_dashboard.get("trend_assessment"),
        },
        "guarded_forward_sample_progress": {
            "guarded_candidate_generated_at": guarded_generated_at,
            "rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("rows_after_candidate")
            ),
            "quality_labeled_rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate")
            ),
            "raw_labeled_rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("raw_labeled_rows_after_candidate")
            ),
            "pending_rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("pending_rows_after_candidate")
            ),
            "partial_rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("partial_rows_after_candidate")
            ),
            "complete_rows_after_guarded_candidate": _safe_int(
                (post_guarded_refresh_label_summary or {}).get("complete_rows_after_candidate")
            ),
            "latest_post_guarded_signal_timestamp": (
                post_guarded_refresh_label_summary or {}
            ).get("latest_post_candidate_signal_timestamp"),
            "guarded_validation_mode_used": guarded_window.get("validation_mode_used"),
            "guarded_strict_forward_row_count": guarded_strict_forward,
            "min_forward_sample": int(min_forward_sample),
            "forward_sample_gap": int(guarded_forward_gap),
            "forward_sample_progress_ratio": _round_or_none(
                guarded_strict_forward / max(int(min_forward_sample), 1),
                6,
            ),
            "previous_guarded_strict_forward_row_count": previous_guarded_rows,
            "new_post_guarded_true_forward_rows_since_previous_soak": new_guarded_forward_rows,
            "history_path": str(guarded_history_path) if guarded_history_path is not None else None,
        },
        "guarded_validation_summary": {
            "guarded_shadow_status": guarded_report.get("guarded_shadow_status"),
            "validation_mode_used": guarded_window.get("validation_mode_used"),
            "strict_forward_row_count": guarded_window.get("strict_forward_row_count"),
            "min_forward_sample": int(min_forward_sample),
            "forward_sample_gap": max(int(min_forward_sample) - guarded_strict_forward, 0),
            "holdout_replay_row_count": guarded_window.get("holdout_replay_row_count"),
            "recommended_routing_policy": (guarded_report.get("selection_summary", {}) or {}).get(
                "recommended_routing_policy"
            ),
            "recommended_policy_status": (guarded_report.get("selection_summary", {}) or {}).get(
                "recommended_policy_status"
            ),
            "top_bucket_risk_delta_bps": (guarded_report.get("selection_summary", {}) or {}).get(
                "recommended_policy_risk_delta_vs_raw_bps"
            ),
            "top_bucket_hit_rate_delta": (guarded_report.get("selection_summary", {}) or {}).get(
                "recommended_policy_hit_delta_vs_raw"
            ),
            "quarantined_route_top_count": guarded_policy.get("quarantined_route_top_count"),
            "rank_preservation_policy_present": guarded_report.get("rank_preservation_policy_present"),
            "guarded_bundle_research_only": guarded_report.get("guarded_bundle_research_only"),
            "guarded_bundle_approval_required_for_runtime_use": guarded_report.get(
                "guarded_bundle_approval_required_for_runtime_use"
            ),
        },
        "readiness_summary": {
            "readiness_status": readiness_report.get("readiness_status"),
            "readiness_reasons": readiness_report.get("readiness_reasons", []) or [],
            "allow_holdout_replay_guarded_validation": bool(allow_holdout_replay_guarded_validation),
            "recommended_next_actions": readiness_report.get("recommended_next_actions", []) or [],
        },
        "candidate_staleness_summary": {
            "staleness_status": staleness_report.get("staleness_status"),
            "staleness_reasons": staleness_report.get("staleness_reasons", []) or [],
            "candidate_age_days": (staleness_report.get("candidate_summary", {}) or {}).get("candidate_age_days"),
            "post_candidate_label_count": (
                staleness_report.get("forward_label_population_shift", {}) or {}
            ).get("post_candidate_label_count"),
            "routing_policy_status": (staleness_report.get("routing_policy_stability", {}) or {}).get(
                "policy_stability_status"
            ),
        },
        "guarded_candidate_staleness_summary": {
            "guarded_staleness_status": guarded_staleness_report.get("guarded_staleness_status"),
            "guarded_staleness_reasons": guarded_staleness_report.get("guarded_staleness_reasons", []) or [],
            "guarded_candidate_age_days": (
                guarded_staleness_report.get("guarded_candidate_summary", {}) or {}
            ).get("guarded_candidate_age_days"),
            "post_guarded_label_count": (
                guarded_staleness_report.get("forward_label_population_shift", {}) or {}
            ).get("post_guarded_label_count"),
            "guarded_routing_policy_status": (
                guarded_staleness_report.get("guarded_routing_policy_stability", {}) or {}
            ).get("policy_stability_status"),
            "dataset_currency_status": (
                guarded_staleness_report.get("dataset_currency", {}) or {}
            ).get("dataset_currency_status"),
        },
        "legacy_ev_shadow_summary": {
            "refreshed": bool(refresh_legacy_ev_shadow),
            "ev_shadow_status": ev_shadow_report.get("ev_shadow_status"),
            "recommended_routing_policy": (ev_shadow_report.get("selection_summary", {}) or {}).get(
                "recommended_routing_policy"
            ),
            "top_bucket_risk_delta_bps": (ev_shadow_report.get("selection_summary", {}) or {}).get(
                "recommended_policy_risk_adjusted_return_delta_bps"
            ),
            "top_bucket_hit_rate_delta": (ev_shadow_report.get("selection_summary", {}) or {}).get(
                "recommended_policy_hit_rate_delta"
            ),
        },
        "artifact_paths": {
            "history_path": accumulation_artifact.get("history_path"),
            "forward_shadow_json_path": forward_shadow_artifact.get("latest_json_path"),
            "accumulation_dashboard_json_path": accumulation_artifact.get("accumulation_dashboard_json_path"),
            "candidate_staleness_json_path": staleness_artifact.get("staleness_json_path"),
            "guarded_candidate_staleness_json_path": (guarded_staleness_artifact or {}).get(
                "guarded_staleness_json_path"
            ),
            "ev_shadow_json_path": (ev_shadow_artifact or {}).get("latest_json_path"),
            "guarded_shadow_json_path": guarded_shadow_artifact.get("latest_json_path"),
            "readiness_json_path": readiness_artifact.get("readiness_json_path"),
            "guarded_shadow_soak_history_path": str(guarded_history_path) if guarded_history_path is not None else None,
        },
        "checked_conditions": {
            "dataset_exists": bool(dataset_info.get("dataset_exists")),
            "dataset_readable": dataset_info.get("dataset_read_error") is None,
            "outcome_refresh_succeeded": not bool((outcome_refresh_summary or {}).get("outcome_refresh_error")),
            "outcome_refresh_added_quality_labels": (
                _safe_int((post_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
                > _safe_int((pre_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
            ),
            "outcome_refresh_added_guarded_quality_labels": (
                _safe_int((post_guarded_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
                > _safe_int((pre_guarded_refresh_label_summary or {}).get("quality_labeled_rows_after_candidate"))
            ),
            "candidate_bundle_unchanged": bool(candidate_bundle_unchanged),
            "guarded_candidate_bundle_unchanged": bool(guarded_bundle_unchanged),
            "side_effects_absent": bool(clean_side_effects),
            "readiness_ready_for_manual_review": readiness_report.get("readiness_status")
            == FORWARD_SHADOW_READY_FOR_MANUAL_REVIEW,
            "guarded_shadow_passed": guarded_report.get("guarded_shadow_status") == GUARDED_SHADOW_VALIDATION_PASS,
            "guarded_strict_forward_sample_met": guarded_strict_forward >= int(min_forward_sample),
            "candidate_staleness_active_review": staleness_report.get("staleness_status") == ACTIVE_REVIEW,
            "guarded_candidate_staleness_active_or_accumulating": guarded_staleness_report.get(
                "guarded_staleness_status"
            )
            in GUARDED_STALENESS_NON_BLOCKING,
            "strict_forward_sample_met": strict_forward >= int(min_forward_sample),
            "true_forward_accumulation_passed": accumulation_row.get("accumulation_status")
            == ACCUMULATION_TRUE_FORWARD_PASS,
        },
        "recommended_next_actions": [],
    }
    report["recommended_next_actions"] = _recommended_actions(report)
    return _sanitize_value(report)


def render_segmented_probability_shadow_soak_markdown(report: dict[str, Any]) -> str:
    """Render the shadow-soak status as Markdown."""
    progress = report.get("forward_sample_progress", {}) or {}
    outcome = report.get("outcome_refresh_summary", {}) or {}
    guarded_progress = report.get("guarded_forward_sample_progress", {}) or {}
    guarded = report.get("guarded_validation_summary", {}) or {}
    readiness = report.get("readiness_summary", {}) or {}
    staleness = report.get("candidate_staleness_summary", {}) or {}
    guarded_staleness = report.get("guarded_candidate_staleness_summary", {}) or {}
    lines = [
        "# Segmented Probability Shadow Soak",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Soak status: **{report.get('soak_status')}**",
        f"- Dataset path: {report.get('dataset_path')}",
        f"- Candidate bundle: {report.get('candidate_bundle_path')}",
        f"- Guarded candidate bundle: {report.get('guarded_candidate_bundle_path')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Soak Progress",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Validation mode | `{progress.get('validation_mode_used')}` |",
        f"| Strict forward rows | {progress.get('strict_forward_row_count')} / {progress.get('min_forward_sample')} |",
        f"| Forward sample gap | {progress.get('forward_sample_gap')} |",
        f"| New rows since previous soak | {progress.get('new_true_forward_rows_since_previous_soak')} |",
        f"| Accumulation status | `{progress.get('accumulation_status')}` |",
        f"| Trend assessment | `{progress.get('trend_assessment')}` |",
        f"| Outcome refresh source | `{outcome.get('outcome_refresh_source')}` |",
        f"| New quality labels after candidate | {outcome.get('new_quality_labeled_rows_after_candidate')} |",
        f"| New quality labels after guarded candidate | {outcome.get('new_quality_labeled_rows_after_guarded_candidate')} |",
        "",
        "## Post-Guarded Forward Evidence",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Guarded candidate generated at | {guarded_progress.get('guarded_candidate_generated_at')} |",
        f"| Rows after guarded candidate | {guarded_progress.get('rows_after_guarded_candidate')} |",
        f"| Quality labels after guarded candidate | {guarded_progress.get('quality_labeled_rows_after_guarded_candidate')} |",
        f"| Guarded strict forward rows | {guarded_progress.get('guarded_strict_forward_row_count')} / {guarded_progress.get('min_forward_sample')} |",
        f"| Guarded forward sample gap | {guarded_progress.get('forward_sample_gap')} |",
        f"| New guarded rows since previous soak | {guarded_progress.get('new_post_guarded_true_forward_rows_since_previous_soak')} |",
        f"| Latest post-guarded signal | {guarded_progress.get('latest_post_guarded_signal_timestamp')} |",
        "",
        "## Guarded Validation",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Status | `{guarded.get('guarded_shadow_status')}` |",
        f"| Guarded validation mode | `{guarded.get('validation_mode_used')}` |",
        f"| Guarded strict forward rows | {guarded.get('strict_forward_row_count')} / {guarded.get('min_forward_sample')} |",
        f"| Guarded forward sample gap | {guarded.get('forward_sample_gap')} |",
        f"| Recommended policy | `{guarded.get('recommended_routing_policy')}` |",
        f"| Risk delta vs raw | {guarded.get('top_bucket_risk_delta_bps')} |",
        f"| Hit-rate delta vs raw | {guarded.get('top_bucket_hit_rate_delta')} |",
        f"| Quarantined top exposure | {guarded.get('quarantined_route_top_count')} |",
        f"| Rank policy present | {guarded.get('rank_preservation_policy_present')} |",
        "",
        "## Governance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Readiness status | `{readiness.get('readiness_status')}` |",
        f"| Staleness status | `{staleness.get('staleness_status')}` |",
        f"| Candidate age days | {staleness.get('candidate_age_days')} |",
        f"| Post-candidate labels | {staleness.get('post_candidate_label_count')} |",
        f"| Routing policy status | `{staleness.get('routing_policy_status')}` |",
        f"| Guarded staleness status | `{guarded_staleness.get('guarded_staleness_status')}` |",
        f"| Guarded candidate age days | {guarded_staleness.get('guarded_candidate_age_days')} |",
        f"| Post-guarded labels | {guarded_staleness.get('post_guarded_label_count')} |",
        f"| Guarded routing policy status | `{guarded_staleness.get('guarded_routing_policy_status')}` |",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("soak_reasons", []) or ["ready"]:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Recommended Actions", ""])
    for action in report.get("recommended_next_actions", []) or ["No actions recorded."]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("*Research-only artifact. It does not alter runtime config, parameter packs, data sources, or execution behavior.*")
    return "\n".join(lines)


def build_segmented_probability_shadow_soak_history_row(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten the latest soak report into one guarded-forward history row."""
    guarded_progress = report.get("guarded_forward_sample_progress", {}) or {}
    guarded = report.get("guarded_validation_summary", {}) or {}
    outcome = report.get("outcome_refresh_summary", {}) or {}
    staleness = report.get("candidate_staleness_summary", {}) or {}
    guarded_staleness = report.get("guarded_candidate_staleness_summary", {}) or {}
    readiness = report.get("readiness_summary", {}) or {}
    row = {
        "observed_at": report.get("generated_at"),
        "dataset_path": report.get("dataset_path"),
        "guarded_candidate_bundle_path": report.get("guarded_candidate_bundle_path"),
        "guarded_candidate_generated_at": guarded_progress.get("guarded_candidate_generated_at"),
        "latest_post_guarded_signal_timestamp": guarded_progress.get("latest_post_guarded_signal_timestamp"),
        "rows_after_guarded_candidate": guarded_progress.get("rows_after_guarded_candidate"),
        "quality_labeled_rows_after_guarded_candidate": guarded_progress.get(
            "quality_labeled_rows_after_guarded_candidate"
        ),
        "raw_labeled_rows_after_guarded_candidate": guarded_progress.get(
            "raw_labeled_rows_after_guarded_candidate"
        ),
        "pending_rows_after_guarded_candidate": guarded_progress.get("pending_rows_after_guarded_candidate"),
        "partial_rows_after_guarded_candidate": guarded_progress.get("partial_rows_after_guarded_candidate"),
        "complete_rows_after_guarded_candidate": guarded_progress.get("complete_rows_after_guarded_candidate"),
        "guarded_validation_mode_used": guarded_progress.get("guarded_validation_mode_used"),
        "guarded_strict_forward_row_count": guarded_progress.get("guarded_strict_forward_row_count"),
        "min_forward_sample": guarded_progress.get("min_forward_sample"),
        "guarded_forward_sample_gap": guarded_progress.get("forward_sample_gap"),
        "guarded_forward_sample_progress_ratio": guarded_progress.get("forward_sample_progress_ratio"),
        "new_post_guarded_true_forward_rows_since_previous_soak": guarded_progress.get(
            "new_post_guarded_true_forward_rows_since_previous_soak"
        ),
        "guarded_shadow_status": guarded.get("guarded_shadow_status"),
        "guarded_recommended_routing_policy": guarded.get("recommended_routing_policy"),
        "guarded_recommended_policy_status": guarded.get("recommended_policy_status"),
        "guarded_top_bucket_risk_delta_bps": guarded.get("top_bucket_risk_delta_bps"),
        "guarded_top_bucket_hit_rate_delta": guarded.get("top_bucket_hit_rate_delta"),
        "guarded_quarantined_route_top_count": guarded.get("quarantined_route_top_count"),
        "outcome_refresh_source": outcome.get("outcome_refresh_source"),
        "outcome_refresh_error": outcome.get("outcome_refresh_error"),
        "new_quality_labeled_rows_after_guarded_candidate": outcome.get(
            "new_quality_labeled_rows_after_guarded_candidate"
        ),
        "readiness_status": readiness.get("readiness_status"),
        "staleness_status": staleness.get("staleness_status"),
        "guarded_staleness_status": guarded_staleness.get("guarded_staleness_status"),
        "guarded_staleness_routing_policy_status": guarded_staleness.get("guarded_routing_policy_status"),
        "soak_status": report.get("soak_status"),
        "runtime_config_changed": report.get("runtime_config_changed"),
        "parameter_pack_file_changed": report.get("parameter_pack_file_changed"),
        "execution_behavior_changed": report.get("execution_behavior_changed"),
    }
    return _sanitize_value(row)


def write_segmented_probability_shadow_soak_report(
    *,
    dataset_path: str | Path | None = None,
    candidate_bundle_path: str | Path | None = None,
    guarded_candidate_bundle_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    history_filename: str = SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
    accumulation_output_dir: str | Path | None = None,
    forward_shadow_output_dir: str | Path | None = None,
    candidate_staleness_output_dir: str | Path | None = None,
    guarded_candidate_staleness_output_dir: str | Path | None = None,
    ev_shadow_output_dir: str | Path | None = None,
    guarded_shadow_output_dir: str | Path | None = None,
    readiness_output_dir: str | Path | None = None,
    outcome_refresh_source: str = OUTCOME_REFRESH_LOCAL_SPOT_HISTORY,
    refresh_legacy_ev_shadow: bool = True,
    probability_field: str = DEFAULT_PROBABILITY_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    return_field: str = DEFAULT_RETURN_FIELD,
    train_fraction: float = 0.70,
    routing_policies: tuple[str, ...] = DEFAULT_ROUTING_POLICIES,
    regime_fields: tuple[str, ...] = DEFAULT_REGIME_FIELDS,
    min_shadow_sample: int = 100,
    min_forward_sample: int | None = None,
    min_candidate_sample: int = 50,
    min_brier_improvement: float = 0.005,
    max_ece_regression: float = 0.01,
    max_candidate_brier_regression: float = 0.002,
    max_candidate_ece_regression: float = 0.02,
    top_fraction: float | None = None,
    raw_rank_ceiling_multiplier: float | None = None,
    min_ev_sample: int = 100,
    min_top_sample: int = 25,
    min_risk_adjusted_improvement_bps: float = 0.0,
    max_hit_rate_regression: float = 0.02,
    downside_penalty_weight: float = 0.25,
    spread_penalty_per_pct: float = 2.0,
    max_spread_pct: float = 5.0,
    max_liquidity_watch_rate: float = 0.35,
    n_bins: int = 10,
    lookback_runs: int = 20,
    stale_after_days: float = 7.0,
    expire_after_days: float = 14.0,
    max_new_rows_before_stale: int = 500,
    max_new_labeled_rows_before_stale: int = 100,
    min_shift_sample: int = 100,
    material_hit_rate_delta: float = 0.10,
    material_probability_delta: float = 0.08,
    material_distribution_psi: float = 0.20,
    policy_lookback_runs: int = 5,
    min_policy_stability_runs: int = 1,
    max_candidate_age_days: float = 14.0,
    allow_holdout_replay_guarded_validation: bool = False,
    as_of: Any = None,
) -> dict[str, Any]:
    """Run the full daily shadow-soak loop and write a compact status report."""
    dataset = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    bundle = Path(candidate_bundle_path) if candidate_bundle_path is not None else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    guarded_bundle = (
        Path(guarded_candidate_bundle_path)
        if guarded_candidate_bundle_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_BUNDLE_PATH
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_SHADOW_SOAK_DIR
    accumulation_output = (
        Path(accumulation_output_dir)
        if accumulation_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    )
    forward_shadow_output = (
        Path(forward_shadow_output_dir)
        if forward_shadow_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR
    )
    staleness_output = (
        Path(candidate_staleness_output_dir)
        if candidate_staleness_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_DIR
    )
    guarded_staleness_output = (
        Path(guarded_candidate_staleness_output_dir)
        if guarded_candidate_staleness_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_CANDIDATE_STALENESS_DIR
    )
    ev_output = Path(ev_shadow_output_dir) if ev_shadow_output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_EV_SHADOW_DIR
    guarded_output = (
        Path(guarded_shadow_output_dir)
        if guarded_shadow_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_DIR
    )
    readiness_output = (
        Path(readiness_output_dir)
        if readiness_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_READINESS_DIR
    )
    min_forward = int(min_forward_sample if min_forward_sample is not None else min_shadow_sample)
    history_path = accumulation_output / history_filename
    guarded_history_path = output / SEGMENTED_PROBABILITY_SHADOW_SOAK_HISTORY_FILENAME
    history_before = _load_history_frame(history_path)
    guarded_history_before = _load_history_frame(guarded_history_path)
    candidate_digest_before = _file_digest(bundle)
    guarded_digest_before = _file_digest(guarded_bundle)
    routing = _csv_tuple(routing_policies)
    regimes = _csv_tuple(regime_fields)
    pre_refresh_label_summary = _post_candidate_label_summary(
        dataset,
        candidate_bundle_path=bundle,
        label_field=label_field,
    )
    pre_guarded_refresh_label_summary = _post_candidate_label_summary(
        dataset,
        candidate_bundle_path=guarded_bundle,
        label_field=label_field,
    )
    outcome_refresh_summary = _refresh_outcomes_if_requested(
        dataset_path=dataset,
        outcome_refresh_source=outcome_refresh_source,
        as_of=as_of,
    )
    post_refresh_label_summary = _post_candidate_label_summary(
        dataset,
        candidate_bundle_path=bundle,
        label_field=label_field,
    )
    post_guarded_refresh_label_summary = _post_candidate_label_summary(
        dataset,
        candidate_bundle_path=guarded_bundle,
        label_field=label_field,
    )

    accumulation_artifact = write_segmented_probability_forward_shadow_accumulation(
        dataset_path=dataset,
        candidate_bundle_path=bundle,
        shadow_output_dir=forward_shadow_output,
        output_dir=accumulation_output,
        history_filename=history_filename,
        lookback_runs=lookback_runs,
        min_shadow_sample=min_shadow_sample,
        probability_field=probability_field,
        label_field=label_field,
        train_fraction=train_fraction,
        routing_policies=routing,
        min_candidate_sample=min_candidate_sample,
        min_brier_improvement=min_brier_improvement,
        max_ece_regression=max_ece_regression,
        max_candidate_brier_regression=max_candidate_brier_regression,
        max_candidate_ece_regression=max_candidate_ece_regression,
        n_bins=n_bins,
    )
    staleness_artifact = write_segmented_probability_candidate_staleness_report(
        dataset_path=dataset,
        candidate_bundle_path=bundle,
        history_path=history_path,
        output_dir=staleness_output,
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
    guarded_staleness_artifact = write_segmented_probability_guarded_candidate_staleness_report(
        dataset_path=dataset,
        guarded_candidate_bundle_path=guarded_bundle,
        guarded_candidate_bundle_search_dir=guarded_bundle.parent,
        guarded_history_path=guarded_history_path,
        output_dir=guarded_staleness_output,
        probability_field=probability_field,
        label_field=label_field,
        stale_after_days=stale_after_days,
        expire_after_days=expire_after_days,
        max_new_rows_before_stale=max_new_rows_before_stale,
        max_new_labeled_rows_before_stale=max_new_labeled_rows_before_stale,
        min_forward_sample=min_forward,
        min_shift_sample=min_shift_sample,
        material_hit_rate_delta=material_hit_rate_delta,
        material_probability_delta=material_probability_delta,
        material_distribution_psi=material_distribution_psi,
        policy_lookback_runs=policy_lookback_runs,
        min_policy_observations=min_policy_stability_runs,
        as_of=as_of,
    )
    ev_shadow_artifact = None
    ev_shadow_path: str | Path = ev_output / SEGMENTED_PROBABILITY_EV_SHADOW_JSON_FILENAME
    if refresh_legacy_ev_shadow:
        ev_shadow_artifact = write_segmented_probability_ev_shadow_evaluation_report_from_path(
            dataset_path=dataset,
            candidate_bundle_path=bundle,
            probability_field=probability_field,
            label_field=label_field,
            return_field=return_field,
            train_fraction=train_fraction,
            validation_mode="auto",
            routing_policies=routing,
            regime_fields=regimes,
            top_fraction=0.25 if top_fraction is None else top_fraction,
            min_ev_sample=min_ev_sample,
            min_top_sample=min_top_sample,
            min_candidate_sample=min_candidate_sample,
            min_risk_adjusted_improvement_bps=min_risk_adjusted_improvement_bps,
            max_hit_rate_regression=max_hit_rate_regression,
            downside_penalty_weight=downside_penalty_weight,
            spread_penalty_per_pct=spread_penalty_per_pct,
            max_spread_pct=max_spread_pct,
            max_liquidity_watch_rate=max_liquidity_watch_rate,
            output_dir=ev_output,
            write_latest=True,
        )
        ev_shadow_path = ev_shadow_artifact.get("latest_json_path") or ev_shadow_path
    guarded_shadow_artifact = write_segmented_probability_guarded_shadow_validation_report_from_path(
        dataset_path=dataset,
        candidate_bundle_path=guarded_bundle,
        probability_field=probability_field,
        label_field=label_field,
        return_field=return_field,
        train_fraction=train_fraction,
        validation_mode="auto",
        routing_policies=routing,
        regime_fields=regimes,
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
        output_dir=guarded_output,
        write_latest=True,
    )
    readiness_artifact = write_segmented_probability_forward_shadow_readiness_report(
        accumulation_dashboard_path=accumulation_output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME,
        forward_shadow_report_path=forward_shadow_output / "latest_segmented_probability_forward_shadow.json",
        candidate_staleness_path=staleness_output / SEGMENTED_PROBABILITY_CANDIDATE_STALENESS_JSON_FILENAME,
        ev_shadow_path=ev_shadow_path,
        guarded_shadow_path=guarded_output / SEGMENTED_PROBABILITY_GUARDED_SHADOW_VALIDATION_JSON_FILENAME,
        history_path=history_path,
        output_dir=readiness_output,
        min_forward_sample=min_forward,
        min_policy_stability_runs=min_policy_stability_runs,
        policy_lookback_runs=policy_lookback_runs,
        max_candidate_age_days=max_candidate_age_days,
        allow_holdout_replay_guarded_validation=allow_holdout_replay_guarded_validation,
        as_of=as_of,
    )
    candidate_digest_after = _file_digest(bundle)
    guarded_digest_after = _file_digest(guarded_bundle)
    report = build_segmented_probability_shadow_soak_report(
        dataset_path=dataset,
        candidate_bundle_path=bundle,
        guarded_candidate_bundle_path=guarded_bundle,
        history_before=history_before,
        accumulation_artifact=accumulation_artifact,
        staleness_artifact=staleness_artifact,
        guarded_staleness_artifact=guarded_staleness_artifact,
        ev_shadow_artifact=ev_shadow_artifact,
        guarded_shadow_artifact=guarded_shadow_artifact,
        readiness_artifact=readiness_artifact,
        outcome_refresh_summary=outcome_refresh_summary,
        pre_refresh_label_summary=pre_refresh_label_summary,
        post_refresh_label_summary=post_refresh_label_summary,
        pre_guarded_refresh_label_summary=pre_guarded_refresh_label_summary,
        post_guarded_refresh_label_summary=post_guarded_refresh_label_summary,
        guarded_history_before=guarded_history_before,
        guarded_history_path=guarded_history_path,
        candidate_bundle_digest_before=candidate_digest_before,
        candidate_bundle_digest_after=candidate_digest_after,
        guarded_bundle_digest_before=guarded_digest_before,
        guarded_bundle_digest_after=guarded_digest_after,
        min_forward_sample=min_forward,
        allow_holdout_replay_guarded_validation=allow_holdout_replay_guarded_validation,
        refresh_legacy_ev_shadow=refresh_legacy_ev_shadow,
        outcome_refresh_source=outcome_refresh_source,
        label_field=label_field,
    )
    assert_artifact_schema(report, "segmented_probability_shadow_soak")
    history_row = build_segmented_probability_shadow_soak_history_row(report)
    guarded_history = append_segmented_probability_shadow_soak_history(
        history_row,
        guarded_history_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / SEGMENTED_PROBABILITY_SHADOW_SOAK_JSON_FILENAME
    markdown_path = output / SEGMENTED_PROBABILITY_SHADOW_SOAK_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_segmented_probability_shadow_soak_markdown(report))
    return {
        "shadow_soak_json_path": str(json_path),
        "shadow_soak_markdown_path": str(markdown_path),
        "shadow_soak_history_path": str(guarded_history_path),
        "shadow_soak_history_row": history_row,
        "shadow_soak_history_run_count": int(len(guarded_history)),
        "shadow_soak_report": report,
        "accumulation_artifact": {key: value for key, value in accumulation_artifact.items() if key != "accumulation_dashboard"},
        "staleness_artifact": {key: value for key, value in staleness_artifact.items() if key != "staleness_report"},
        "guarded_staleness_artifact": {
            key: value for key, value in guarded_staleness_artifact.items() if key != "guarded_staleness_report"
        },
        "ev_shadow_artifact": None
        if ev_shadow_artifact is None
        else {key: value for key, value in ev_shadow_artifact.items() if key != "report"},
        "guarded_shadow_artifact": {key: value for key, value in guarded_shadow_artifact.items() if key != "report"},
        "readiness_artifact": {key: value for key, value in readiness_artifact.items() if key != "readiness_report"},
    }
