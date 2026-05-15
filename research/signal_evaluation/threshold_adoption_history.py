"""Append-only adoption history for governed threshold rollouts.

The rollout monitor answers "what is true now"; this module records those
answers over time so operators can see whether an approved threshold stayed
unadopted, became active in signal generation, mismatched, or was rolled back.
It is research/ops-only and never changes runtime configuration.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.threshold_adoption_reconciliation import (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import (
    CANDIDATE_SIGNAL_ROLLOUT_BLOCKED,
    DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR,
    THRESHOLD_SIGNAL_ROLLOUT_MONITOR_JSON_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_adoption_history"
)
DEFAULT_ROLLOUT_MONITOR_REPORT_PATH = (
    DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR / THRESHOLD_SIGNAL_ROLLOUT_MONITOR_JSON_FILENAME
)
DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH = (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR / THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME
)
DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH = (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME
)

THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME = "threshold_adoption_history.csv"
THRESHOLD_ADOPTION_HISTORY_JSON_FILENAME = "latest_threshold_adoption_history.json"
THRESHOLD_ADOPTION_HISTORY_MARKDOWN_FILENAME = "latest_threshold_adoption_history.md"

LIFECYCLE_ADOPTED_ACTIVE = "ADOPTED_ACTIVE"
LIFECYCLE_ADOPTED_AWAITING_SIGNALS = "ADOPTED_AWAITING_SIGNALS"
LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING = "ADOPTED_BUT_NOT_SIGNALING"
LIFECYCLE_ADOPTED_MIXED_SIGNALING = "ADOPTED_MIXED_SIGNALING"
LIFECYCLE_APPROVED_BUT_NOT_ADOPTED = "APPROVED_BUT_NOT_ADOPTED"
LIFECYCLE_MISMATCHED = "MISMATCHED"
LIFECYCLE_ROLLED_BACK = "ROLLED_BACK"
LIFECYCLE_UNKNOWN = "UNKNOWN"

RUNTIME_CANDIDATE_SIGNALING = "CANDIDATE_SIGNALING"
RUNTIME_MIXED_SIGNALING = "MIXED_SIGNALING"
RUNTIME_NON_CANDIDATE_SIGNALING = "NON_CANDIDATE_SIGNALING"
RUNTIME_NO_POST_ADOPTION_SIGNALS = "NO_POST_ADOPTION_SIGNALS"
RUNTIME_MISSING_TRACEABILITY = "MISSING_TRACEABILITY"
RUNTIME_UNKNOWN = "UNKNOWN"


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _upper(value: Any) -> str:
    return str(value or "").upper().strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


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


def classify_runtime_signal_status(row: dict[str, Any]) -> str:
    """Classify which parameter pack is actually appearing in signal rows."""
    traceability_status = _upper(row.get("traceability_status"))
    post_count = _safe_int(row.get("post_adoption_signal_count"))
    candidate_count = _safe_int(row.get("candidate_pack_signal_count"))
    non_candidate_count = _safe_int(row.get("non_candidate_pack_signal_count"))
    missing_count = _safe_int(row.get("missing_parameter_pack_count"))

    if traceability_status == "MISSING_PARAMETER_PACK_COLUMN" or missing_count > 0:
        return RUNTIME_MISSING_TRACEABILITY
    if post_count <= 0:
        return RUNTIME_NO_POST_ADOPTION_SIGNALS
    if candidate_count > 0 and non_candidate_count == 0:
        return RUNTIME_CANDIDATE_SIGNALING
    if candidate_count > 0 and non_candidate_count > 0:
        return RUNTIME_MIXED_SIGNALING
    if candidate_count == 0 and non_candidate_count > 0:
        return RUNTIME_NON_CANDIDATE_SIGNALING
    return RUNTIME_UNKNOWN


def classify_adoption_lifecycle_status(row: dict[str, Any]) -> str:
    """Classify threshold adoption lifecycle from reconciliation plus signals."""
    adoption_status = _upper(row.get("adoption_reconciliation_status"))
    rollout_status = _upper(row.get("rollout_status"))
    runtime_status = _upper(row.get("runtime_signal_status"))
    traceability_status = _upper(row.get("traceability_status"))

    if "ROLLBACK" in adoption_status or "ROLLED_BACK" in adoption_status:
        return LIFECYCLE_ROLLED_BACK
    if (
        "MISMATCH" in adoption_status
        or rollout_status == CANDIDATE_SIGNAL_ROLLOUT_BLOCKED
        or runtime_status == RUNTIME_MISSING_TRACEABILITY
        or traceability_status == "MISSING_PARAMETER_PACK_COLUMN"
    ):
        return LIFECYCLE_MISMATCHED
    if adoption_status == LIFECYCLE_APPROVED_BUT_NOT_ADOPTED:
        return LIFECYCLE_APPROVED_BUT_NOT_ADOPTED
    if runtime_status == RUNTIME_NO_POST_ADOPTION_SIGNALS:
        return LIFECYCLE_ADOPTED_AWAITING_SIGNALS if adoption_status.startswith("ADOPTED") else LIFECYCLE_UNKNOWN
    if runtime_status == RUNTIME_NON_CANDIDATE_SIGNALING:
        return (
            LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING
            if adoption_status.startswith("ADOPTED")
            else LIFECYCLE_APPROVED_BUT_NOT_ADOPTED
        )
    if runtime_status == RUNTIME_MIXED_SIGNALING:
        return LIFECYCLE_ADOPTED_MIXED_SIGNALING
    if runtime_status == RUNTIME_CANDIDATE_SIGNALING:
        return LIFECYCLE_ADOPTED_ACTIVE
    return LIFECYCLE_UNKNOWN


def build_threshold_adoption_history_row(
    rollout_report: dict[str, Any],
    *,
    rollout_report_path: str | Path | None = None,
    adoption_reconciliation_report: dict[str, Any] | None = None,
    adoption_reconciliation_report_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    observed_at: Any = None,
) -> dict[str, Any]:
    """Flatten the latest rollout/adoption reports into one history row."""
    rollout = rollout_report if isinstance(rollout_report, dict) else {}
    adoption = adoption_reconciliation_report if isinstance(adoption_reconciliation_report, dict) else {}
    post_monitor = post_promotion_monitor_report if isinstance(post_promotion_monitor_report, dict) else {}
    traceability = rollout.get("post_adoption_traceability", {}) or {}
    readiness = rollout.get("candidate_label_readiness", {}) or {}
    comparison = rollout.get("rollout_comparison", {}) or {}
    execution = rollout.get("execution_side_effects", {}) or {}

    row = {
        "observed_at": str(observed_at or _utc_now()),
        "rollout_generated_at": rollout.get("generated_at"),
        "rollout_report_path": str(rollout_report_path) if rollout_report_path is not None else None,
        "adoption_reconciliation_report_path": (
            str(adoption_reconciliation_report_path) if adoption_reconciliation_report_path is not None else None
        ),
        "post_promotion_monitor_report_path": (
            str(post_promotion_monitor_report_path) if post_promotion_monitor_report_path is not None else None
        ),
        "dataset_path": rollout.get("dataset_path"),
        "baseline_pack_name": rollout.get("baseline_pack_name"),
        "candidate_pack_name": rollout.get("candidate_pack_name"),
        "config_hint": rollout.get("config_hint"),
        "approved_threshold_value": rollout.get("approved_threshold_value"),
        "baseline_runtime_value": rollout.get("baseline_runtime_value"),
        "candidate_runtime_value": rollout.get("candidate_runtime_value"),
        "adoption_start_timestamp": rollout.get("adoption_start_timestamp"),
        "adoption_start_source": rollout.get("adoption_start_source"),
        "runtime_activation_timestamp": rollout.get("runtime_activation_timestamp"),
        "runtime_activation_source": rollout.get("runtime_activation_source"),
        "traceability_window_start_timestamp": rollout.get("traceability_window_start_timestamp"),
        "traceability_window_start_source": rollout.get("traceability_window_start_source"),
        "adoption_reconciliation_status": (
            rollout.get("adoption_reconciliation_status")
            or adoption.get("adoption_status")
        ),
        "post_promotion_monitor_status": (
            rollout.get("post_promotion_monitor_status")
            or post_monitor.get("monitor_status")
        ),
        "rollout_status": rollout.get("rollout_status"),
        "traceability_status": traceability.get("traceability_status"),
        "post_adoption_signal_count": traceability.get("post_adoption_signal_count"),
        "candidate_pack_signal_count": traceability.get("candidate_pack_signal_count"),
        "non_candidate_pack_signal_count": traceability.get("non_candidate_pack_signal_count"),
        "missing_parameter_pack_count": traceability.get("missing_parameter_pack_count"),
        "parameter_pack_values": "|".join(str(value) for value in traceability.get("parameter_pack_values", []) or []),
        "candidate_label_count_60m": readiness.get("label_count_60m"),
        "candidate_outcome_monitoring_status": readiness.get("outcome_monitoring_status"),
        "baseline_signal_count": comparison.get("baseline_signal_count"),
        "candidate_signal_count": comparison.get("candidate_signal_count"),
        "signal_count_delta": comparison.get("signal_count_delta"),
        "execution_side_effect_check_passed": execution.get("execution_side_effect_check_passed"),
        "orders_submitted": execution.get("orders_submitted"),
        "execution_behavior_changed": rollout.get("execution_behavior_changed"),
        "runtime_config_changed": rollout.get("runtime_config_changed"),
    }
    row["runtime_signal_status"] = classify_runtime_signal_status(row)
    row["adoption_lifecycle_status"] = classify_adoption_lifecycle_status(row)
    return _sanitize_value(row)


def append_threshold_adoption_history(
    row: dict[str, Any],
    history_path: str | Path,
) -> pd.DataFrame:
    """Append one adoption history row and return the full history frame."""
    path = Path(history_path)
    incoming = pd.DataFrame([row])
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


def _latest_row(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {}
    return _sanitize_value(history.iloc[-1].to_dict())


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    values = frame[column].fillna("UNKNOWN").astype(str)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _trend_assessment(latest: dict[str, Any]) -> str:
    lifecycle = _upper(latest.get("adoption_lifecycle_status"))
    if not lifecycle:
        return "NO_HISTORY"
    if lifecycle in {LIFECYCLE_MISMATCHED, LIFECYCLE_ROLLED_BACK}:
        return "BLOCKED"
    if lifecycle == LIFECYCLE_ADOPTED_ACTIVE:
        return "ACTIVE"
    if lifecycle in {
        LIFECYCLE_ADOPTED_AWAITING_SIGNALS,
        LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING,
        LIFECYCLE_ADOPTED_MIXED_SIGNALING,
        LIFECYCLE_APPROVED_BUT_NOT_ADOPTED,
    }:
        return "WATCH"
    return "UNKNOWN"


def _operator_message(latest: dict[str, Any], assessment: str) -> str:
    lifecycle = latest.get("adoption_lifecycle_status") or "UNKNOWN"
    if assessment == "ACTIVE":
        return "Approved threshold is active in candidate-pack signal generation."
    if lifecycle == LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING:
        return "Threshold is adopted in the candidate pack, but live post-adoption signals are still coming from another pack."
    if lifecycle == LIFECYCLE_ADOPTED_AWAITING_SIGNALS:
        return "Threshold is adopted, but no post-adoption signals have arrived yet."
    if lifecycle == LIFECYCLE_APPROVED_BUT_NOT_ADOPTED:
        return "Threshold is approved, but the active/manual adoption check does not show it adopted yet."
    if lifecycle == LIFECYCLE_ADOPTED_MIXED_SIGNALING:
        return "Candidate and non-candidate signals are both present after adoption; review runtime selection."
    if lifecycle == LIFECYCLE_MISMATCHED:
        return "Adoption or signal traceability is mismatched and needs review before relying on the rollout."
    if lifecycle == LIFECYCLE_ROLLED_BACK:
        return "The approved threshold appears rolled back."
    return "Adoption lifecycle state is unknown; review latest reconciliation and rollout reports."


def build_threshold_adoption_history_dashboard(
    history: pd.DataFrame,
    *,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Build a compact dashboard from accumulated adoption observations."""
    frame = history.copy() if history is not None else pd.DataFrame()
    if frame.empty:
        return {
            "report_type": "threshold_adoption_history_dashboard",
            "generated_at": _utc_now(),
            "run_count": 0,
            "lookback_runs": int(lookback_runs),
            "trend_assessment": "NO_HISTORY",
            "latest": {},
            "status_counts": {},
            "runtime_signal_status_counts": {},
            "rollout_status_counts": {},
            "lookback_summary": {},
            "operator_message": "No adoption history is available yet.",
        }

    latest = _latest_row(frame)
    lookback = frame.tail(max(int(lookback_runs), 1)).copy()
    assessment = _trend_assessment(latest)
    lifecycle_values = lookback.get("adoption_lifecycle_status", pd.Series(dtype=str)).fillna("").astype(str)
    runtime_values = lookback.get("runtime_signal_status", pd.Series(dtype=str)).fillna("").astype(str)
    rollout_values = lookback.get("rollout_status", pd.Series(dtype=str)).fillna("").astype(str)

    dashboard = {
        "report_type": "threshold_adoption_history_dashboard",
        "generated_at": _utc_now(),
        "run_count": int(len(frame)),
        "lookback_runs": int(lookback_runs),
        "trend_assessment": assessment,
        "latest": latest,
        "status_counts": _value_counts(frame, "adoption_lifecycle_status"),
        "runtime_signal_status_counts": _value_counts(frame, "runtime_signal_status"),
        "rollout_status_counts": _value_counts(frame, "rollout_status"),
        "lookback_summary": {
            "active_runs": int((lifecycle_values == LIFECYCLE_ADOPTED_ACTIVE).sum()),
            "approved_but_not_adopted_runs": int((lifecycle_values == LIFECYCLE_APPROVED_BUT_NOT_ADOPTED).sum()),
            "adopted_but_not_signaling_runs": int((lifecycle_values == LIFECYCLE_ADOPTED_BUT_NOT_SIGNALING).sum()),
            "mixed_signaling_runs": int((lifecycle_values == LIFECYCLE_ADOPTED_MIXED_SIGNALING).sum()),
            "mismatched_runs": int((lifecycle_values == LIFECYCLE_MISMATCHED).sum()),
            "rolled_back_runs": int((lifecycle_values == LIFECYCLE_ROLLED_BACK).sum()),
            "candidate_signaling_runs": int((runtime_values == RUNTIME_CANDIDATE_SIGNALING).sum()),
            "non_candidate_signaling_runs": int((runtime_values == RUNTIME_NON_CANDIDATE_SIGNALING).sum()),
            "blocked_rollout_runs": int((rollout_values == CANDIDATE_SIGNAL_ROLLOUT_BLOCKED).sum()),
        },
        "operator_message": _operator_message(latest, assessment),
    }
    return _sanitize_value(dashboard)


def render_threshold_adoption_history_markdown(dashboard: dict[str, Any]) -> str:
    """Render the adoption history dashboard as Markdown."""
    latest = dashboard.get("latest", {}) or {}
    summary = dashboard.get("lookback_summary", {}) or {}
    lines = [
        "# Threshold Adoption History",
        "",
        f"- Generated at: {dashboard.get('generated_at')}",
        f"- Trend assessment: **{dashboard.get('trend_assessment')}**",
        f"- Run count: {dashboard.get('run_count')}",
        f"- Operator message: {dashboard.get('operator_message')}",
        "",
        "## Latest Observation",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Observed at | {latest.get('observed_at')} |",
        f"| Candidate pack | `{latest.get('candidate_pack_name')}` |",
        f"| Lifecycle status | `{latest.get('adoption_lifecycle_status')}` |",
        f"| Runtime signal status | `{latest.get('runtime_signal_status')}` |",
        f"| Runtime activation | {latest.get('runtime_activation_timestamp') or 'not marked'} |",
        f"| Reconciliation status | `{latest.get('adoption_reconciliation_status')}` |",
        f"| Rollout status | `{latest.get('rollout_status')}` |",
        f"| Post-adoption signals | {latest.get('post_adoption_signal_count')} |",
        f"| Candidate-pack signals | {latest.get('candidate_pack_signal_count')} |",
        f"| Non-candidate signals | {latest.get('non_candidate_pack_signal_count')} |",
        f"| Missing pack values | {latest.get('missing_parameter_pack_count')} |",
        "",
        "## Lookback Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Active runs | {summary.get('active_runs')} |",
        f"| Approved but not adopted runs | {summary.get('approved_but_not_adopted_runs')} |",
        f"| Adopted but not signaling runs | {summary.get('adopted_but_not_signaling_runs')} |",
        f"| Mixed signaling runs | {summary.get('mixed_signaling_runs')} |",
        f"| Mismatched runs | {summary.get('mismatched_runs')} |",
        f"| Rolled-back runs | {summary.get('rolled_back_runs')} |",
        "",
        "*This dashboard is signal-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*",
    ]
    return "\n".join(lines)


def write_threshold_adoption_history_dashboard(
    history: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Write latest adoption history dashboard JSON/Markdown artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR
    output.mkdir(parents=True, exist_ok=True)
    dashboard = build_threshold_adoption_history_dashboard(history, lookback_runs=lookback_runs)
    json_path = output / THRESHOLD_ADOPTION_HISTORY_JSON_FILENAME
    markdown_path = output / THRESHOLD_ADOPTION_HISTORY_MARKDOWN_FILENAME
    assert_artifact_schema(dashboard, "threshold_adoption_history_dashboard")
    _atomic_write_text(json_path, json.dumps(dashboard, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_threshold_adoption_history_markdown(dashboard))
    return {
        "history_dashboard_json_path": str(json_path),
        "history_dashboard_markdown_path": str(markdown_path),
        "history_dashboard": dashboard,
    }


def write_threshold_adoption_history(
    *,
    rollout_report_path: str | Path | None = None,
    adoption_reconciliation_report_path: str | Path | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    history_filename: str = THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME,
    lookback_runs: int = 20,
    observed_at: Any = None,
) -> dict[str, Any]:
    """Append adoption history from latest reports and refresh dashboard."""
    rollout_path = Path(rollout_report_path) if rollout_report_path is not None else DEFAULT_ROLLOUT_MONITOR_REPORT_PATH
    adoption_path = (
        Path(adoption_reconciliation_report_path)
        if adoption_reconciliation_report_path is not None
        else DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH
    )
    post_monitor_path = (
        Path(post_promotion_monitor_report_path)
        if post_promotion_monitor_report_path is not None
        else DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR
    history_path = output / history_filename
    rollout = load_json_file(rollout_path)
    adoption = load_json_file(adoption_path)
    post_monitor = load_json_file(post_monitor_path)
    row = build_threshold_adoption_history_row(
        rollout,
        rollout_report_path=rollout_path,
        adoption_reconciliation_report=adoption,
        adoption_reconciliation_report_path=adoption_path,
        post_promotion_monitor_report=post_monitor,
        post_promotion_monitor_report_path=post_monitor_path,
        observed_at=observed_at,
    )
    history = append_threshold_adoption_history(row, history_path)
    dashboard_artifact = write_threshold_adoption_history_dashboard(
        history,
        output_dir=output,
        lookback_runs=lookback_runs,
    )
    artifact = {
        "history_path": str(history_path),
        "history_row": row,
    }
    artifact.update(dashboard_artifact)
    return artifact
