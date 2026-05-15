"""Post-activation verification for governed threshold rollouts.

This module turns the manual post-activation checklist into one read-only
workflow. It validates that the runtime activation marker, active parameter
pack, rollout monitor, and adoption history agree before a candidate threshold
is treated as cleanly adopted in signal generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from config.policy_resolver import get_active_parameter_pack
from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.daily_research_report import (
    DEFAULT_CUMULATIVE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
)
from research.signal_evaluation.threshold_adoption_history import (
    DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR,
    THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME,
    write_threshold_adoption_history,
)
from research.signal_evaluation.threshold_adoption_reconciliation import ADOPTED_MANUALLY
from research.signal_evaluation.threshold_runtime_activation import (
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR,
    THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME,
    load_threshold_runtime_activation_marker,
)
from research.signal_evaluation.threshold_signal_rollout_monitor import (
    CANDIDATE_SIGNAL_ROLLOUT_HEALTHY,
    DEFAULT_BASELINE_PARAMETER_PACK,
    DEFAULT_CANDIDATE_PARAMETER_PACK,
    DEFAULT_CONFIG_HINT,
    DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR,
    write_threshold_signal_rollout_monitor_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_POST_ACTIVATION_VERIFICATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_post_activation_verification"
)
DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH = (
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR / THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME
)

THRESHOLD_POST_ACTIVATION_VERIFICATION_JSON_FILENAME = "latest_threshold_post_activation_verification.json"
THRESHOLD_POST_ACTIVATION_VERIFICATION_MARKDOWN_FILENAME = "latest_threshold_post_activation_verification.md"

POST_ACTIVATION_VERIFICATION_CLEAN = "POST_ACTIVATION_VERIFICATION_CLEAN"
POST_ACTIVATION_VERIFICATION_BLOCKED = "POST_ACTIVATION_VERIFICATION_BLOCKED"


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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _values_match(left: Any, right: Any) -> bool:
    left_float = _safe_float(left)
    right_float = _safe_float(right)
    if left_float is not None and right_float is not None:
        tolerance = max(abs(right_float) * 1e-9, 1e-9)
        return abs(left_float - right_float) <= tolerance
    return str(left).strip().lower() == str(right).strip().lower()


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


def default_signal_dataset_path() -> Path:
    """Return the preferred signal dataset for rollout verification."""
    return DEFAULT_CUMULATIVE_DATASET_PATH if DEFAULT_CUMULATIVE_DATASET_PATH.exists() else DEFAULT_DATASET_PATH


def _active_pack_matches_marker(
    active_parameter_pack: dict[str, Any],
    marker: dict[str, Any],
) -> bool:
    expected_pack = str(marker.get("candidate_pack_name") or marker.get("parameter_pack_name") or "").strip()
    active_name = str(active_parameter_pack.get("name") or "").strip()
    active_layers = {str(layer).strip() for layer in active_parameter_pack.get("layers", []) or []}
    return bool(expected_pack) and (active_name == expected_pack or expected_pack in active_layers)


def build_threshold_post_activation_verification_report(
    *,
    rollout_report: dict[str, Any],
    adoption_history_artifact: dict[str, Any] | None = None,
    runtime_activation_marker: dict[str, Any] | None = None,
    runtime_activation_marker_path: str | Path | None = None,
    active_parameter_pack: dict[str, Any] | None = None,
    candidate_pack_name: str = DEFAULT_CANDIDATE_PARAMETER_PACK,
    min_candidate_label_count: int = 1,
    require_candidate_labels: bool = True,
    require_adopted_reconciliation: bool = True,
) -> dict[str, Any]:
    """Build the final post-activation verification decision."""
    rollout = rollout_report if isinstance(rollout_report, dict) else {}
    history_artifact = adoption_history_artifact if isinstance(adoption_history_artifact, dict) else {}
    marker = runtime_activation_marker if isinstance(runtime_activation_marker, dict) else {}
    active_pack = active_parameter_pack if isinstance(active_parameter_pack, dict) else get_active_parameter_pack()
    traceability = rollout.get("post_adoption_traceability", {}) or {}
    labels = rollout.get("candidate_label_readiness", {}) or {}
    execution = rollout.get("execution_side_effects", {}) or {}
    history_row = history_artifact.get("history_row", {}) or {}
    history_dashboard = history_artifact.get("history_dashboard", {}) or {}

    expected_marker_pack = str(marker.get("candidate_pack_name") or marker.get("parameter_pack_name") or "").strip()
    active_matches_marker = _active_pack_matches_marker(active_pack, marker)
    candidate_label_count = int(labels.get("label_count_60m") or 0)
    non_candidate_count = int(traceability.get("non_candidate_pack_signal_count") or 0)
    missing_pack_count = int(traceability.get("missing_parameter_pack_count") or 0)
    side_effect_count = int(execution.get("total_nonempty_side_effect_fields") or 0)
    approved_threshold = rollout.get("approved_threshold_value")
    candidate_runtime_value = rollout.get("candidate_runtime_value")
    threshold_matches = (
        approved_threshold is not None
        and candidate_runtime_value is not None
        and _values_match(candidate_runtime_value, approved_threshold)
    )

    reasons: list[str] = []
    if not marker:
        reasons.append("Runtime activation marker is missing.")
    if expected_marker_pack and expected_marker_pack != str(candidate_pack_name).strip():
        reasons.append(
            f"Runtime activation marker points to `{expected_marker_pack}`, but verification is for `{candidate_pack_name}`."
        )
    if marker and not active_matches_marker:
        reasons.append(
            f"Active parameter pack `{active_pack.get('name')}` does not match runtime activation marker `{expected_marker_pack}`."
        )
    if rollout.get("rollout_status") != CANDIDATE_SIGNAL_ROLLOUT_HEALTHY:
        reasons.append(f"Rollout monitor status is `{rollout.get('rollout_status')}`.")
    for rollout_reason in rollout.get("rollout_reasons", []) or []:
        if rollout.get("rollout_status") != CANDIDATE_SIGNAL_ROLLOUT_HEALTHY:
            reasons.append(f"Rollout monitor: {rollout_reason}")
    if require_adopted_reconciliation and rollout.get("adoption_reconciliation_status") != ADOPTED_MANUALLY:
        reasons.append(
            f"Adoption reconciliation status is `{rollout.get('adoption_reconciliation_status')}`."
        )
    if not threshold_matches:
        reasons.append(
            f"Candidate runtime threshold `{candidate_runtime_value}` does not match approved threshold `{approved_threshold}`."
        )
    if non_candidate_count > 0:
        reasons.append("Post-activation signal rows include non-candidate parameter-pack signals.")
    if missing_pack_count > 0:
        reasons.append("Post-activation signal rows include missing parameter-pack traceability.")
    if require_candidate_labels and candidate_label_count < int(min_candidate_label_count):
        reasons.append(
            f"Candidate outcome labels are below requirement: {candidate_label_count} < {int(min_candidate_label_count)}."
        )
    if not execution.get("execution_side_effect_check_passed", True) or side_effect_count > 0:
        reasons.append("Order/execution side-effect fields are non-empty.")
    if rollout.get("runtime_config_changed") is True:
        reasons.append("Rollout monitor reported a runtime config mutation.")
    if rollout.get("parameter_pack_file_changed") is True:
        reasons.append("Rollout monitor reported a parameter-pack file mutation.")
    if rollout.get("execution_behavior_changed") is True:
        reasons.append("Rollout monitor reported an execution behavior mutation.")

    status = POST_ACTIVATION_VERIFICATION_BLOCKED if reasons else POST_ACTIVATION_VERIFICATION_CLEAN
    if not reasons:
        reasons = [
            "Post-activation rollout is clean: active pack matches marker, candidate traceability is exclusive, threshold is consistent, labels are ready, and no execution side effects are present."
        ]

    report = {
        "report_type": "threshold_post_activation_verification",
        "generated_at": _utc_now(),
        "verification_status": status,
        "verification_reasons": reasons,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "candidate_pack_name": candidate_pack_name,
        "runtime_activation_marker_path": str(runtime_activation_marker_path) if runtime_activation_marker_path else None,
        "runtime_activation_marker": marker,
        "active_parameter_pack": active_pack,
        "active_pack_matches_marker": active_matches_marker,
        "checked_conditions": {
            "runtime_activation_marker_present": bool(marker),
            "expected_marker_pack": expected_marker_pack or None,
            "active_parameter_pack_name": active_pack.get("name"),
            "active_pack_matches_marker": active_matches_marker,
            "rollout_status": rollout.get("rollout_status"),
            "adoption_reconciliation_status": rollout.get("adoption_reconciliation_status"),
            "approved_threshold_value": approved_threshold,
            "candidate_runtime_value": candidate_runtime_value,
            "threshold_matches": threshold_matches,
            "post_adoption_signal_count": traceability.get("post_adoption_signal_count"),
            "candidate_pack_signal_count": traceability.get("candidate_pack_signal_count"),
            "non_candidate_pack_signal_count": non_candidate_count,
            "missing_parameter_pack_count": missing_pack_count,
            "candidate_label_count_60m": candidate_label_count,
            "min_candidate_label_count": int(min_candidate_label_count),
            "require_candidate_labels": bool(require_candidate_labels),
            "execution_side_effect_check_passed": execution.get("execution_side_effect_check_passed"),
            "total_nonempty_side_effect_fields": side_effect_count,
        },
        "rollout_monitor": {
            "rollout_status": rollout.get("rollout_status"),
            "rollout_reasons": rollout.get("rollout_reasons"),
            "runtime_activation_timestamp": rollout.get("runtime_activation_timestamp"),
            "traceability_window_start_timestamp": rollout.get("traceability_window_start_timestamp"),
            "candidate_label_readiness": labels,
            "post_adoption_traceability": traceability,
            "execution_side_effects": execution,
        },
        "adoption_history": {
            "history_path": history_artifact.get("history_path"),
            "history_dashboard_json_path": history_artifact.get("history_dashboard_json_path"),
            "history_dashboard_markdown_path": history_artifact.get("history_dashboard_markdown_path"),
            "latest_row": history_row,
            "trend_assessment": history_dashboard.get("trend_assessment"),
            "operator_message": history_dashboard.get("operator_message"),
        },
    }
    return _sanitize_value(report)


def render_threshold_post_activation_verification_markdown(report: dict[str, Any]) -> str:
    """Render post-activation verification as Markdown."""
    checks = report.get("checked_conditions", {}) or {}
    rollout = report.get("rollout_monitor", {}) or {}
    history = report.get("adoption_history", {}) or {}
    lines = [
        "# Threshold Post-Activation Verification",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Verification status: **{report.get('verification_status')}**",
        f"- Candidate pack: `{report.get('candidate_pack_name')}`",
        f"- Active pack: `{checks.get('active_parameter_pack_name')}`",
        f"- Active pack matches marker: `{checks.get('active_pack_matches_marker')}`",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("verification_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Value |",
            "| --- | --- |",
            f"| Marker present | {checks.get('runtime_activation_marker_present')} |",
            f"| Marker pack | `{checks.get('expected_marker_pack')}` |",
            f"| Rollout status | `{checks.get('rollout_status')}` |",
            f"| Adoption reconciliation | `{checks.get('adoption_reconciliation_status')}` |",
            f"| Approved threshold | `{checks.get('approved_threshold_value')}` |",
            f"| Candidate runtime threshold | `{checks.get('candidate_runtime_value')}` |",
            f"| Threshold matches | {checks.get('threshold_matches')} |",
            f"| Candidate-pack signals | {checks.get('candidate_pack_signal_count')} |",
            f"| Non-candidate signals | {checks.get('non_candidate_pack_signal_count')} |",
            f"| Missing pack values | {checks.get('missing_parameter_pack_count')} |",
            f"| Candidate 60m labels | {checks.get('candidate_label_count_60m')} |",
            f"| Side-effect fields | {checks.get('total_nonempty_side_effect_fields')} |",
            "",
            "## Downstream Artifacts",
            "",
            f"- Rollout status: `{rollout.get('rollout_status')}`",
            f"- Adoption history: `{history.get('history_path')}`",
            f"- History trend: `{history.get('trend_assessment')}`",
            f"- Operator message: {history.get('operator_message')}",
            "",
            "*This verifier is signal-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*",
        ]
    )
    return "\n".join(lines)


def write_threshold_post_activation_verification_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Write post-activation verification JSON/Markdown artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_POST_ACTIVATION_VERIFICATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_post_activation_verification"
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    latest_json_path = output / THRESHOLD_POST_ACTIVATION_VERIFICATION_JSON_FILENAME
    latest_markdown_path = output / THRESHOLD_POST_ACTIVATION_VERIFICATION_MARKDOWN_FILENAME
    assert_artifact_schema(report, "threshold_post_activation_verification")
    markdown = render_threshold_post_activation_verification_markdown(report)
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, markdown)
    if write_latest:
        _atomic_write_text(latest_json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown_path, markdown)
    return {
        "verification_report": report,
        "verification_json_path": str(json_path),
        "verification_markdown_path": str(markdown_path),
        "latest_verification_json_path": str(latest_json_path),
        "latest_verification_markdown_path": str(latest_markdown_path),
    }


def run_threshold_post_activation_verification(
    frame: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
    baseline_pack_name: str = DEFAULT_BASELINE_PARAMETER_PACK,
    candidate_pack_name: str = DEFAULT_CANDIDATE_PARAMETER_PACK,
    candidate_overrides: dict[str, Any] | None = None,
    config_hint: str = DEFAULT_CONFIG_HINT,
    approved_threshold_value: Any = None,
    adoption_start_at: Any = None,
    adoption_reconciliation_report: dict[str, Any] | None = None,
    adoption_reconciliation_report_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    runtime_activation_at: Any = None,
    runtime_activation_marker: dict[str, Any] | None = None,
    runtime_activation_marker_path: str | Path | None = None,
    rollout_output_dir: str | Path | None = None,
    rollout_report_name: str | None = None,
    history_output_dir: str | Path | None = None,
    history_filename: str = THRESHOLD_ADOPTION_HISTORY_CSV_FILENAME,
    lookback_runs: int = 20,
    verification_output_dir: str | Path | None = None,
    verification_report_name: str | None = None,
    min_candidate_label_count: int = 1,
    require_candidate_labels: bool = True,
    require_adopted_reconciliation: bool = True,
    strict_candidate_pack_required: bool = True,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Run rollout monitor, append history, and write final verification."""
    activation_path = runtime_activation_marker_path or DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH
    marker = runtime_activation_marker
    if marker is None:
        marker = load_threshold_runtime_activation_marker(activation_path)

    rollout_artifact = write_threshold_signal_rollout_monitor_report(
        frame,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        baseline_pack_name=baseline_pack_name,
        candidate_pack_name=candidate_pack_name,
        candidate_overrides=candidate_overrides,
        config_hint=config_hint,
        approved_threshold_value=approved_threshold_value,
        adoption_start_at=adoption_start_at,
        adoption_reconciliation_report=adoption_reconciliation_report,
        adoption_reconciliation_report_path=adoption_reconciliation_report_path,
        post_promotion_monitor_report=post_promotion_monitor_report,
        post_promotion_monitor_report_path=post_promotion_monitor_report_path,
        runtime_activation_at=runtime_activation_at,
        runtime_activation_marker=marker,
        runtime_activation_marker_path=activation_path,
        strict_candidate_pack_required=strict_candidate_pack_required,
        output_dir=rollout_output_dir,
        report_name=rollout_report_name,
        write_latest=write_latest,
    )
    rollout_json_path = rollout_artifact.get("latest_json_path") or rollout_artifact.get("json_path")
    history_artifact = write_threshold_adoption_history(
        rollout_report_path=rollout_json_path,
        adoption_reconciliation_report_path=adoption_reconciliation_report_path,
        post_promotion_monitor_report_path=post_promotion_monitor_report_path,
        output_dir=history_output_dir or DEFAULT_THRESHOLD_ADOPTION_HISTORY_DIR,
        history_filename=history_filename,
        lookback_runs=lookback_runs,
    )
    report = build_threshold_post_activation_verification_report(
        rollout_report=rollout_artifact.get("report", {}),
        adoption_history_artifact=history_artifact,
        runtime_activation_marker=marker,
        runtime_activation_marker_path=activation_path,
        active_parameter_pack=get_active_parameter_pack(),
        candidate_pack_name=candidate_pack_name,
        min_candidate_label_count=min_candidate_label_count,
        require_candidate_labels=require_candidate_labels,
        require_adopted_reconciliation=require_adopted_reconciliation,
    )
    verification_artifact = write_threshold_post_activation_verification_report(
        report,
        output_dir=verification_output_dir,
        report_name=verification_report_name,
        write_latest=write_latest,
    )
    return {
        **verification_artifact,
        "rollout_artifact": rollout_artifact,
        "adoption_history_artifact": history_artifact,
    }


def run_threshold_post_activation_verification_from_paths(
    *,
    dataset_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a signal dataset from disk and run post-activation verification."""
    path = Path(dataset_path) if dataset_path is not None else default_signal_dataset_path()
    frame = pd.read_csv(path, low_memory=False)
    return run_threshold_post_activation_verification(frame, dataset_path=path, **kwargs)
