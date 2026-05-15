"""Append-only accumulation history for segmented probability forward shadow.

This module runs the research-only forward-shadow validator and records whether
the candidate bundle is still relying on holdout replay or has accumulated
enough true post-candidate labels for forward validation. It never changes
runtime configuration, parameter packs, data sources, or execution behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.segmented_probability_forward_shadow import (
    DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH,
    DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR,
    FORWARD_SHADOW_PASS,
    FORWARD_SHADOW_REJECTED,
    FORWARD_SHADOW_WATCH,
    NEEDS_MORE_FORWARD_DATA,
    NO_CANDIDATE_ROUTES,
    SHADOW_REPLAY_PASS,
    write_segmented_probability_forward_shadow_report_from_path,
)
from research.signal_evaluation.signal_quality_model_audit import (
    DEFAULT_PROBABILITY_FIELD,
    _atomic_write_csv,
    _atomic_write_text,
    _round_or_none,
    _sanitize_value,
    _utc_now,
    default_signal_quality_dataset_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "segmented_probability_forward_shadow_accumulation"
)

SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME = "segmented_probability_forward_shadow_history.csv"
SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME = (
    "latest_segmented_probability_forward_shadow_accumulation.json"
)
SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_MARKDOWN_FILENAME = (
    "latest_segmented_probability_forward_shadow_accumulation.md"
)

ACCUMULATION_TRUE_FORWARD_PASS = "TRUE_FORWARD_VALIDATION_PASS"
ACCUMULATION_TRUE_FORWARD_WATCH = "TRUE_FORWARD_VALIDATION_WATCH"
ACCUMULATION_TRUE_FORWARD_REJECTED = "TRUE_FORWARD_VALIDATION_REJECTED"
ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD = "HOLDOUT_REPLAY_PASS_PENDING_FORWARD_LABELS"
ACCUMULATION_HOLDOUT_REPLAY_PENDING_FORWARD = "HOLDOUT_REPLAY_PENDING_FORWARD_LABELS"
ACCUMULATION_NEEDS_MORE_FORWARD_DATA = "NEEDS_MORE_FORWARD_SHADOW_DATA"
ACCUMULATION_NO_CANDIDATE_ROUTES = "NO_CANDIDATE_ROUTES"


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


def _classify_accumulation_status(report: dict[str, Any], *, min_shadow_sample: int) -> str:
    window = report.get("validation_window", {}) or {}
    mode = str(window.get("validation_mode_used") or "")
    shadow_status = str(report.get("shadow_validation_status") or "")
    strict_forward = _safe_int(window.get("strict_forward_row_count"))
    candidate_count = _safe_int(report.get("candidate_count"))

    if candidate_count <= 0 or shadow_status == NO_CANDIDATE_ROUTES:
        return ACCUMULATION_NO_CANDIDATE_ROUTES
    if mode == "after_candidate_generated":
        if strict_forward < int(min_shadow_sample) or shadow_status == NEEDS_MORE_FORWARD_DATA:
            return ACCUMULATION_NEEDS_MORE_FORWARD_DATA
        if shadow_status == FORWARD_SHADOW_PASS:
            return ACCUMULATION_TRUE_FORWARD_PASS
        if shadow_status == FORWARD_SHADOW_WATCH:
            return ACCUMULATION_TRUE_FORWARD_WATCH
        if shadow_status == FORWARD_SHADOW_REJECTED:
            return ACCUMULATION_TRUE_FORWARD_REJECTED
        return ACCUMULATION_TRUE_FORWARD_WATCH
    if mode == "holdout_replay":
        if shadow_status == SHADOW_REPLAY_PASS:
            return ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD
        return ACCUMULATION_HOLDOUT_REPLAY_PENDING_FORWARD
    return ACCUMULATION_NEEDS_MORE_FORWARD_DATA


def build_segmented_probability_forward_shadow_history_row(
    report: dict[str, Any],
    *,
    report_path: str | Path | None = None,
    min_shadow_sample: int = 100,
    observed_at: Any = None,
) -> dict[str, Any]:
    """Flatten the latest forward-shadow report into one accumulation row."""
    shadow = report if isinstance(report, dict) else {}
    window = shadow.get("validation_window", {}) or {}
    selection = shadow.get("selection_summary", {}) or {}
    strict_forward = _safe_int(window.get("strict_forward_row_count"))
    gap = max(int(min_shadow_sample) - strict_forward, 0)
    row = {
        "observed_at": str(observed_at or _utc_now()),
        "forward_shadow_generated_at": shadow.get("generated_at"),
        "forward_shadow_report_path": str(report_path) if report_path is not None else None,
        "dataset_path": shadow.get("dataset_path"),
        "candidate_bundle_path": shadow.get("candidate_bundle_path"),
        "candidate_count": shadow.get("candidate_count"),
        "candidate_generated_at": window.get("candidate_generated_at"),
        "validation_mode_requested": window.get("validation_mode_requested"),
        "validation_mode_used": window.get("validation_mode_used"),
        "fallback_reason": window.get("fallback_reason"),
        "strict_forward_row_count": strict_forward,
        "holdout_replay_row_count": window.get("holdout_replay_row_count"),
        "train_count": window.get("train_count"),
        "min_shadow_sample": int(min_shadow_sample),
        "forward_sample_gap": int(gap),
        "forward_sample_progress_ratio": _round_or_none(strict_forward / max(int(min_shadow_sample), 1), 6),
        "shadow_validation_status": shadow.get("shadow_validation_status"),
        "recommended_routing_policy": selection.get("recommended_routing_policy"),
        "recommended_policy_status": selection.get("recommended_policy_status"),
        "recommended_policy_brier_improvement": selection.get("recommended_policy_brier_improvement"),
        "recommended_policy_ece_change": selection.get("recommended_policy_ece_change"),
        "route_decision_count": shadow.get("route_decision_count"),
        "quality_labeled_row_count": shadow.get("quality_labeled_row_count"),
        "runtime_config_changed": shadow.get("runtime_config_changed"),
        "parameter_pack_file_changed": shadow.get("parameter_pack_file_changed"),
        "execution_behavior_changed": shadow.get("execution_behavior_changed"),
    }
    row["accumulation_status"] = _classify_accumulation_status(shadow, min_shadow_sample=min_shadow_sample)
    return _sanitize_value(row)


def append_segmented_probability_forward_shadow_history(
    row: dict[str, Any],
    history_path: str | Path,
) -> pd.DataFrame:
    """Append one forward-shadow accumulation row and return the full history."""
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
    status = str(latest.get("accumulation_status") or "")
    if status == ACCUMULATION_TRUE_FORWARD_PASS:
        return "READY_FOR_MANUAL_REVIEW"
    if status in {ACCUMULATION_TRUE_FORWARD_REJECTED, ACCUMULATION_NO_CANDIDATE_ROUTES}:
        return "BLOCKED"
    if status in {ACCUMULATION_TRUE_FORWARD_WATCH, ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD}:
        return "WATCH"
    if status in {ACCUMULATION_HOLDOUT_REPLAY_PENDING_FORWARD, ACCUMULATION_NEEDS_MORE_FORWARD_DATA}:
        return "ACCUMULATING"
    return "UNKNOWN"


def _operator_message(latest: dict[str, Any], assessment: str) -> str:
    status = latest.get("accumulation_status") or "UNKNOWN"
    strict_forward = _safe_int(latest.get("strict_forward_row_count"))
    min_shadow = _safe_int(latest.get("min_shadow_sample"), 100)
    gap = max(min_shadow - strict_forward, 0)
    policy = latest.get("recommended_routing_policy") or "unknown"
    if status == ACCUMULATION_TRUE_FORWARD_PASS:
        return f"True forward shadow validation passed for `{policy}`; manual review can inspect the latest report."
    if status == ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD:
        return (
            f"Holdout replay supports `{policy}`, but true forward evidence is still {strict_forward}/{min_shadow}; "
            f"wait for {gap} more quality-approved forward rows."
        )
    if status == ACCUMULATION_TRUE_FORWARD_WATCH:
        return "True forward evidence is present but at least one validation guardrail is in watch mode."
    if status == ACCUMULATION_TRUE_FORWARD_REJECTED:
        return "True forward shadow validation rejected the candidate bundle; do not advance it."
    if status == ACCUMULATION_NO_CANDIDATE_ROUTES:
        return "No candidate route matched the validation rows; regenerate or inspect the candidate bundle."
    return (
        f"Forward shadow evidence is accumulating: {strict_forward}/{min_shadow} true rows are available; "
        f"{gap} more are needed before auto mode stops relying on holdout replay."
    )


def build_segmented_probability_forward_shadow_accumulation_dashboard(
    history: pd.DataFrame,
    *,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Build a compact dashboard from accumulated forward-shadow observations."""
    frame = history.copy() if history is not None else pd.DataFrame()
    if frame.empty:
        return {
            "report_type": "segmented_probability_forward_shadow_accumulation",
            "generated_at": _utc_now(),
            "run_count": 0,
            "lookback_runs": int(lookback_runs),
            "trend_assessment": "NO_HISTORY",
            "latest": {},
            "status_counts": {},
            "validation_mode_counts": {},
            "shadow_status_counts": {},
            "lookback_summary": {},
            "operator_message": "No segmented probability forward-shadow accumulation history is available yet.",
        }

    latest = _latest_row(frame)
    lookback = frame.tail(max(int(lookback_runs), 1)).copy()
    assessment = _trend_assessment(latest)
    status_values = lookback.get("accumulation_status", pd.Series(dtype=str)).fillna("").astype(str)
    mode_values = lookback.get("validation_mode_used", pd.Series(dtype=str)).fillna("").astype(str)
    strict_forward = pd.to_numeric(lookback.get("strict_forward_row_count", pd.Series(dtype=float)), errors="coerce")
    min_shadow = pd.to_numeric(lookback.get("min_shadow_sample", pd.Series(dtype=float)), errors="coerce")

    dashboard = {
        "report_type": "segmented_probability_forward_shadow_accumulation",
        "generated_at": _utc_now(),
        "run_count": int(len(frame)),
        "lookback_runs": int(lookback_runs),
        "trend_assessment": assessment,
        "latest": latest,
        "status_counts": _value_counts(frame, "accumulation_status"),
        "validation_mode_counts": _value_counts(frame, "validation_mode_used"),
        "shadow_status_counts": _value_counts(frame, "shadow_validation_status"),
        "lookback_summary": {
            "true_forward_pass_runs": int((status_values == ACCUMULATION_TRUE_FORWARD_PASS).sum()),
            "true_forward_watch_runs": int((status_values == ACCUMULATION_TRUE_FORWARD_WATCH).sum()),
            "true_forward_rejected_runs": int((status_values == ACCUMULATION_TRUE_FORWARD_REJECTED).sum()),
            "holdout_replay_pending_runs": int(
                status_values.isin(
                    {
                        ACCUMULATION_HOLDOUT_REPLAY_PASS_PENDING_FORWARD,
                        ACCUMULATION_HOLDOUT_REPLAY_PENDING_FORWARD,
                    }
                ).sum()
            ),
            "auto_true_forward_runs": int((mode_values == "after_candidate_generated").sum()),
            "auto_holdout_replay_runs": int((mode_values == "holdout_replay").sum()),
            "latest_strict_forward_row_count": int(_safe_int(latest.get("strict_forward_row_count"))),
            "latest_min_shadow_sample": int(_safe_int(latest.get("min_shadow_sample"), 100)),
            "max_strict_forward_row_count": int(strict_forward.max()) if strict_forward.notna().any() else 0,
            "min_shadow_sample_latest": int(min_shadow.dropna().iloc[-1]) if min_shadow.notna().any() else None,
        },
        "operator_message": _operator_message(latest, assessment),
    }
    return _sanitize_value(dashboard)


def render_segmented_probability_forward_shadow_accumulation_markdown(dashboard: dict[str, Any]) -> str:
    """Render the forward-shadow accumulation dashboard as Markdown."""
    latest = dashboard.get("latest", {}) or {}
    summary = dashboard.get("lookback_summary", {}) or {}
    lines = [
        "# Segmented Probability Forward Shadow Accumulation",
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
        f"| Accumulation status | `{latest.get('accumulation_status')}` |",
        f"| Validation mode | `{latest.get('validation_mode_used')}` |",
        f"| Strict forward rows | {latest.get('strict_forward_row_count')} / {latest.get('min_shadow_sample')} |",
        f"| Forward sample gap | {latest.get('forward_sample_gap')} |",
        f"| Shadow validation status | `{latest.get('shadow_validation_status')}` |",
        f"| Recommended routing policy | `{latest.get('recommended_routing_policy')}` |",
        f"| Brier improvement | {latest.get('recommended_policy_brier_improvement')} |",
        f"| ECE change | {latest.get('recommended_policy_ece_change')} |",
        "",
        "## Lookback Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| True forward pass runs | {summary.get('true_forward_pass_runs')} |",
        f"| True forward watch runs | {summary.get('true_forward_watch_runs')} |",
        f"| True forward rejected runs | {summary.get('true_forward_rejected_runs')} |",
        f"| Holdout replay pending runs | {summary.get('holdout_replay_pending_runs')} |",
        f"| Auto true-forward runs | {summary.get('auto_true_forward_runs')} |",
        f"| Auto holdout-replay runs | {summary.get('auto_holdout_replay_runs')} |",
        "",
        "*This dashboard is research-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*",
    ]
    return "\n".join(lines)


def write_segmented_probability_forward_shadow_accumulation_dashboard(
    history: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    lookback_runs: int = 20,
) -> dict[str, Any]:
    """Write latest forward-shadow accumulation dashboard artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    dashboard = build_segmented_probability_forward_shadow_accumulation_dashboard(
        history,
        lookback_runs=lookback_runs,
    )
    assert_artifact_schema(dashboard, "segmented_probability_forward_shadow_accumulation")
    json_path = output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_JSON_FILENAME
    markdown_path = output / SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(dashboard, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_segmented_probability_forward_shadow_accumulation_markdown(dashboard))
    return {
        "accumulation_dashboard_json_path": str(json_path),
        "accumulation_dashboard_markdown_path": str(markdown_path),
        "accumulation_dashboard": dashboard,
    }


def write_segmented_probability_forward_shadow_accumulation(
    *,
    dataset_path: str | Path | None = None,
    candidate_bundle_path: str | Path | None = None,
    shadow_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    history_filename: str = SEGMENTED_PROBABILITY_FORWARD_SHADOW_HISTORY_FILENAME,
    lookback_runs: int = 20,
    min_shadow_sample: int = 100,
    observed_at: Any = None,
    **shadow_kwargs: Any,
) -> dict[str, Any]:
    """Run forward-shadow validation, append accumulation history, and refresh the dashboard."""
    dataset = Path(dataset_path) if dataset_path is not None else default_signal_quality_dataset_path()
    bundle = (
        Path(candidate_bundle_path)
        if candidate_bundle_path is not None
        else DEFAULT_SEGMENTED_PROBABILITY_CANDIDATE_BUNDLE_PATH
    )
    shadow_output = (
        Path(shadow_output_dir)
        if shadow_output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_DIR
    )
    output = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_SEGMENTED_PROBABILITY_FORWARD_SHADOW_ACCUMULATION_DIR
    )
    shadow_artifact = write_segmented_probability_forward_shadow_report_from_path(
        dataset_path=dataset,
        candidate_bundle_path=bundle,
        validation_mode="auto",
        min_shadow_sample=min_shadow_sample,
        output_dir=shadow_output,
        write_latest=True,
        **shadow_kwargs,
    )
    shadow_report = shadow_artifact.get("report", {}) or {}
    report_path = shadow_artifact.get("latest_json_path") or shadow_artifact.get("json_path")
    row = build_segmented_probability_forward_shadow_history_row(
        shadow_report,
        report_path=report_path,
        min_shadow_sample=min_shadow_sample,
        observed_at=observed_at,
    )
    history_path = output / history_filename
    history = append_segmented_probability_forward_shadow_history(row, history_path)
    dashboard_artifact = write_segmented_probability_forward_shadow_accumulation_dashboard(
        history,
        output_dir=output,
        lookback_runs=lookback_runs,
    )
    artifact = {
        "history_path": str(history_path),
        "history_row": row,
        "forward_shadow_artifact": {
            key: value
            for key, value in shadow_artifact.items()
            if key != "report"
        },
    }
    artifact.update(dashboard_artifact)
    return artifact
