"""Signal-only rollout monitoring for an adopted threshold candidate pack.

This monitor verifies that the adopted candidate parameter pack remains a
signal-quality rollout, not an execution workflow. It compares baseline versus
candidate signal selection, checks candidate-pack traceability in the signal
dataset, and flags any order/execution side-effect fields if they appear.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from config.policy_resolver import get_active_parameter_pack, temporary_parameter_pack
from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from research.signal_evaluation.artifact_schema_contracts import assert_artifact_schema
from research.signal_evaluation.threshold_adoption_reconciliation import (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR,
    THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME,
)
from research.signal_evaluation.threshold_adoption_replay_gate import (
    PROVENANCE_COLUMNS,
    _frame_metrics,
    _output_structure_report,
    _policy_delta,
    _provenance_report,
    _selection_relationship,
    _values_match,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
)
from research.signal_evaluation.threshold_runtime_activation import (
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR,
    THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME,
)
from tuning.objectives import apply_selection_policy
from tuning.packs import load_parameter_pack


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_signal_rollout_monitor"
)
DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH = (
    DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR / THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME
)
DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH = (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME
)
DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH = (
    DEFAULT_THRESHOLD_RUNTIME_ACTIVATION_DIR / THRESHOLD_RUNTIME_ACTIVATION_JSON_FILENAME
)

THRESHOLD_SIGNAL_ROLLOUT_MONITOR_JSON_FILENAME = "latest_threshold_signal_rollout_monitor.json"
THRESHOLD_SIGNAL_ROLLOUT_MONITOR_MARKDOWN_FILENAME = "latest_threshold_signal_rollout_monitor.md"
THRESHOLD_SIGNAL_ROLLOUT_MONITOR_COMPARISON_FILENAME = "latest_threshold_signal_rollout_monitor_comparison.csv"

CANDIDATE_SIGNAL_ROLLOUT_HEALTHY = "CANDIDATE_SIGNAL_ROLLOUT_HEALTHY"
CANDIDATE_SIGNAL_ROLLOUT_WATCH = "CANDIDATE_SIGNAL_ROLLOUT_WATCH"
CANDIDATE_SIGNAL_ROLLOUT_BLOCKED = "CANDIDATE_SIGNAL_ROLLOUT_BLOCKED"

DEFAULT_BASELINE_PARAMETER_PACK = "baseline_v1"
DEFAULT_CANDIDATE_PARAMETER_PACK = "candidate_v1"
DEFAULT_CONFIG_HINT = "evaluation_thresholds.selection.composite_signal_score_floor"
SELECTION_POLICY_PREFIX = "evaluation_thresholds.selection."

ORDER_SIDE_EFFECT_COLUMNS = (
    "order_id",
    "broker_order_id",
    "execution_id",
    "fill_id",
    "filled_quantity",
    "filled_qty",
    "fill_price",
    "average_fill_price",
    "order_status",
    "order_submitted_at",
    "broker",
    "position_id",
)


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


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _selection_key(config_hint: Any) -> str | None:
    text = str(config_hint or "")
    if text.startswith(SELECTION_POLICY_PREFIX):
        return text[len(SELECTION_POLICY_PREFIX):]
    return None


def _safe_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _candidate_pack_metadata(pack_name: str) -> dict[str, Any]:
    try:
        return dict(load_parameter_pack(pack_name).metadata or {})
    except Exception:
        return {}


def _adoption_start_timestamp(
    *,
    adoption_start_at: Any = None,
    candidate_pack_metadata: dict[str, Any] | None = None,
    adoption_reconciliation_report: dict[str, Any] | None = None,
) -> tuple[pd.Timestamp | None, str | None]:
    explicit = _safe_timestamp(adoption_start_at)
    if explicit is not None:
        return explicit, "explicit_adoption_start_at"
    metadata = candidate_pack_metadata or {}
    metadata_ts = _safe_timestamp(metadata.get("threshold_adoption_generated_at"))
    if metadata_ts is not None:
        return metadata_ts, "candidate_pack_metadata.threshold_adoption_generated_at"
    approval_ts = _safe_timestamp(((adoption_reconciliation_report or {}).get("approval_decision", {}) or {}).get("reviewed_at"))
    if approval_ts is not None:
        return approval_ts, "adoption_reconciliation.approval_decision.reviewed_at"
    return None, None


def _runtime_activation_timestamp(
    *,
    runtime_activation_at: Any = None,
    runtime_activation_marker: dict[str, Any] | None = None,
    candidate_pack_name: str | None = None,
) -> tuple[pd.Timestamp | None, str | None]:
    explicit = _safe_timestamp(runtime_activation_at)
    if explicit is not None:
        return explicit, "explicit_runtime_activation_at"

    marker = runtime_activation_marker if isinstance(runtime_activation_marker, dict) else {}
    if not marker:
        return None, None

    marker_pack = str(marker.get("candidate_pack_name") or marker.get("parameter_pack_name") or "").strip()
    if marker_pack and candidate_pack_name and marker_pack != str(candidate_pack_name).strip():
        return None, None

    marker_ts = _safe_timestamp(marker.get("activated_at") or marker.get("runtime_activation_at"))
    if marker_ts is not None:
        return marker_ts, "runtime_activation_marker.activated_at"
    return None, None


def _filter_after_adoption(frame: pd.DataFrame, adoption_start: pd.Timestamp | None) -> pd.DataFrame:
    if adoption_start is None or "signal_timestamp" not in frame.columns:
        return frame.copy()
    signal_ts = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True)
    return frame.loc[signal_ts.notna() & (signal_ts >= adoption_start)].copy()


def _nonempty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & ~series.astype(str).str.strip().isin({"", "nan", "NaN", "None", "NONE", "<NA>"})


def _execution_side_effect_report(frame: pd.DataFrame) -> dict[str, Any]:
    present = [column for column in ORDER_SIDE_EFFECT_COLUMNS if column in frame.columns]
    counts = {}
    for column in present:
        counts[column] = int(_nonempty_mask(frame[column]).sum())
    total = int(sum(counts.values()))
    return {
        "execution_side_effect_check_passed": total == 0,
        "checked_columns": present,
        "nonempty_counts": counts,
        "total_nonempty_side_effect_fields": total,
        "trading_execution_invoked": False,
        "orders_submitted": total > 0,
    }


def _pack_traceability_report(
    post_adoption_frame: pd.DataFrame,
    *,
    candidate_pack_name: str,
) -> dict[str, Any]:
    post_count = int(len(post_adoption_frame))
    if post_count == 0:
        return {
            "traceability_status": "NO_POST_ADOPTION_SIGNALS_YET",
            "post_adoption_signal_count": 0,
            "candidate_pack_signal_count": 0,
            "non_candidate_pack_signal_count": 0,
            "missing_parameter_pack_count": 0,
            "parameter_pack_column_present": "parameter_pack_name" in post_adoption_frame.columns,
            "parameter_pack_values": [],
        }
    if "parameter_pack_name" not in post_adoption_frame.columns:
        return {
            "traceability_status": "MISSING_PARAMETER_PACK_COLUMN",
            "post_adoption_signal_count": post_count,
            "candidate_pack_signal_count": 0,
            "non_candidate_pack_signal_count": post_count,
            "missing_parameter_pack_count": post_count,
            "parameter_pack_column_present": False,
            "parameter_pack_values": [],
        }
    values = post_adoption_frame["parameter_pack_name"].astype("object")
    normalized = values.astype(str).str.strip()
    missing_mask = values.isna() | normalized.isin({"", "nan", "NaN", "None", "NONE", "<NA>"})
    candidate_mask = normalized == str(candidate_pack_name)
    unique_values = sorted(str(value) for value in normalized.loc[~missing_mask].unique())
    if int(candidate_mask.sum()) == post_count:
        status = "ALL_POST_ADOPTION_SIGNALS_CANDIDATE_PACK"
    elif int(candidate_mask.sum()) > 0:
        status = "MIXED_PARAMETER_PACK_SIGNALS"
    else:
        status = "NO_CANDIDATE_PACK_SIGNALS_YET"
    return {
        "traceability_status": status,
        "post_adoption_signal_count": post_count,
        "candidate_pack_signal_count": int(candidate_mask.sum()),
        "non_candidate_pack_signal_count": int((~candidate_mask & ~missing_mask).sum()),
        "missing_parameter_pack_count": int(missing_mask.sum()),
        "parameter_pack_column_present": True,
        "parameter_pack_values": unique_values,
    }


def _label_readiness_report(candidate_rows: pd.DataFrame) -> dict[str, Any]:
    if candidate_rows.empty:
        return {
            "candidate_signal_count": 0,
            "label_count_60m": 0,
            "outcome_monitoring_status": "NO_CANDIDATE_SIGNALS_YET",
            "post_promotion_monitor_ready": False,
        }
    labels = pd.to_numeric(candidate_rows.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    returns = pd.to_numeric(candidate_rows.get("signed_return_60m_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    ready = int(labels.count()) > 0
    return {
        "candidate_signal_count": int(len(candidate_rows)),
        "label_count_60m": int(labels.count()),
        "avg_signed_return_60m_bps": round(float(returns.mean()), 4) if not returns.empty else None,
        "hit_rate_60m": round(float(labels.mean()), 4) if not labels.empty else None,
        "outcome_monitoring_status": "LABELS_READY" if ready else "AWAITING_OUTCOME_LABELS",
        "post_promotion_monitor_ready": ready,
    }


def build_threshold_signal_rollout_monitor_report(
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
    strict_candidate_pack_required: bool = False,
) -> dict[str, Any]:
    """Build a read-only monitor for candidate-pack signal rollout."""
    adoption_path = adoption_reconciliation_report_path or DEFAULT_ADOPTION_RECONCILIATION_REPORT_PATH
    post_monitor_path = post_promotion_monitor_report_path or DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH
    activation_path = runtime_activation_marker_path or DEFAULT_RUNTIME_ACTIVATION_MARKER_PATH
    adoption = adoption_reconciliation_report or _load_json_if_exists(adoption_path)
    post_monitor = post_promotion_monitor_report or _load_json_if_exists(post_monitor_path)
    activation_marker = runtime_activation_marker or _load_json_if_exists(activation_path)
    comparison = adoption.get("comparison", {}) or {}
    expected_value = approved_threshold_value
    if expected_value is None:
        expected_value = comparison.get("candidate_value")
    effective_hint = str(config_hint or comparison.get("config_hint") or DEFAULT_CONFIG_HINT)
    selection_key = _selection_key(effective_hint)
    candidate_metadata = _candidate_pack_metadata(candidate_pack_name)
    adoption_start, adoption_start_source = _adoption_start_timestamp(
        adoption_start_at=adoption_start_at,
        candidate_pack_metadata=candidate_metadata,
        adoption_reconciliation_report=adoption,
    )
    runtime_activation_start, runtime_activation_source = _runtime_activation_timestamp(
        runtime_activation_at=runtime_activation_at,
        runtime_activation_marker=activation_marker,
        candidate_pack_name=candidate_pack_name,
    )
    traceability_start = runtime_activation_start or adoption_start
    traceability_start_source = runtime_activation_source or adoption_start_source

    active_before = get_active_parameter_pack()
    with temporary_parameter_pack(baseline_pack_name):
        baseline_pack = get_active_parameter_pack()
        baseline_policy = dict(get_signal_evaluation_selection_policy())
    with temporary_parameter_pack(candidate_pack_name, overrides=candidate_overrides):
        candidate_pack = get_active_parameter_pack()
        candidate_policy = dict(get_signal_evaluation_selection_policy())
    active_after = get_active_parameter_pack()
    context_restored = (
        active_before.get("name") == active_after.get("name")
        and dict(active_before.get("overrides", {}) or {}) == dict(active_after.get("overrides", {}) or {})
    )

    baseline_selected = apply_selection_policy(frame, thresholds=baseline_policy)
    candidate_selected = apply_selection_policy(frame, thresholds=candidate_policy)
    policy_delta = _policy_delta(baseline_policy, candidate_policy)
    candidate_runtime_value = candidate_policy.get(selection_key or "")
    baseline_runtime_value = baseline_policy.get(selection_key or "")
    selection_relationship = _selection_relationship(
        selection_key=selection_key or "",
        baseline_value=baseline_runtime_value,
        candidate_value=candidate_runtime_value,
        baseline_selected=baseline_selected,
        candidate_selected=candidate_selected,
    )
    structure = _output_structure_report(frame, baseline_selected, candidate_selected)
    provenance = _provenance_report(frame, baseline_selected, candidate_selected)
    execution_side_effects = _execution_side_effect_report(frame)
    post_adoption = _filter_after_adoption(frame, traceability_start)
    traceability = _pack_traceability_report(
        post_adoption,
        candidate_pack_name=candidate_pack_name,
    )
    traceability["traceability_window_start_timestamp"] = (
        traceability_start.isoformat() if traceability_start is not None else None
    )
    traceability["traceability_window_start_source"] = traceability_start_source
    traceability["runtime_activation_marker_active"] = runtime_activation_start is not None
    if "parameter_pack_name" in post_adoption.columns:
        candidate_rows = post_adoption.loc[
            post_adoption["parameter_pack_name"].astype(str).str.strip() == str(candidate_pack_name)
        ].copy()
    else:
        candidate_rows = post_adoption.iloc[0:0].copy()
    label_readiness = _label_readiness_report(candidate_rows)

    block_reasons: list[str] = []
    watch_reasons: list[str] = []
    if not selection_key:
        block_reasons.append(f"Config hint `{effective_hint}` is not a supported selection-policy key.")
    if expected_value is None:
        block_reasons.append("Approved threshold value is unavailable from adoption reconciliation.")
    elif not _values_match(candidate_runtime_value, expected_value):
        block_reasons.append(
            f"Candidate pack resolved {effective_hint}={candidate_runtime_value}, expected {expected_value}."
        )
    if selection_key and sorted(policy_delta.keys()) != [selection_key]:
        block_reasons.append("Candidate pack changes selection-policy keys beyond the adopted threshold.")
    if not selection_relationship.get("relationship_passed"):
        block_reasons.append("Candidate selected-signal set does not match expected threshold relationship.")
    if not structure.get("baseline_matches_input") or not structure.get("candidate_matches_input"):
        block_reasons.append("Selection replay changed signal dataset output structure.")
    if not provenance.get("provenance_check_passed"):
        block_reasons.append("Selection replay did not preserve data-source/provenance fields.")
    if not execution_side_effects.get("execution_side_effect_check_passed"):
        block_reasons.append("Order/execution side-effect columns contain non-empty values.")
    if not context_restored:
        block_reasons.append("Temporary parameter-pack context was not restored.")
    if traceability.get("traceability_status") == "MISSING_PARAMETER_PACK_COLUMN":
        block_reasons.append("Post-adoption signal rows exist, but parameter_pack_name is not recorded.")
    if traceability.get("missing_parameter_pack_count", 0) > 0:
        block_reasons.append("Post-adoption signal rows include missing parameter_pack_name values.")

    if traceability.get("traceability_status") == "NO_POST_ADOPTION_SIGNALS_YET":
        watch_reasons.append("No post-adoption signals are available yet for candidate-pack traceability.")
    elif traceability.get("traceability_status") == "NO_CANDIDATE_PACK_SIGNALS_YET":
        watch_reasons.append("Post-adoption signals exist, but none were generated with the candidate parameter pack.")
    elif traceability.get("traceability_status") == "MIXED_PARAMETER_PACK_SIGNALS":
        message = "Post-adoption dataset contains a mix of candidate and non-candidate parameter-pack signals."
        if strict_candidate_pack_required:
            block_reasons.append(message)
        else:
            watch_reasons.append(message)
    if (
        traceability.get("candidate_pack_signal_count", 0) > 0
        and label_readiness.get("label_count_60m", 0) == 0
    ):
        watch_reasons.append("Candidate-pack signals are present, but realized outcome labels are not ready yet.")

    if block_reasons:
        status = CANDIDATE_SIGNAL_ROLLOUT_BLOCKED
        reasons = block_reasons
    elif watch_reasons:
        status = CANDIDATE_SIGNAL_ROLLOUT_WATCH
        reasons = watch_reasons
    else:
        status = CANDIDATE_SIGNAL_ROLLOUT_HEALTHY
        reasons = [
            "Candidate signal rollout is traceable, signal-only, threshold-consistent, and ready for outcome monitoring."
        ]

    rollout_comparison = {
        "baseline_signal_count": int(len(baseline_selected)),
        "candidate_signal_count": int(len(candidate_selected)),
        "signal_count_delta": int(len(candidate_selected) - len(baseline_selected)),
        **selection_relationship,
    }
    report = {
        "report_type": "threshold_signal_rollout_monitor",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "rollout_status": status,
        "rollout_reasons": reasons,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_behavior_changed": False,
        "baseline_pack_name": baseline_pack_name,
        "candidate_pack_name": candidate_pack_name,
        "config_hint": effective_hint,
        "selection_key": selection_key,
        "approved_threshold_value": expected_value,
        "baseline_runtime_value": baseline_runtime_value,
        "candidate_runtime_value": candidate_runtime_value,
        "adoption_start_timestamp": adoption_start.isoformat() if adoption_start is not None else None,
        "adoption_start_source": adoption_start_source,
        "runtime_activation_timestamp": (
            runtime_activation_start.isoformat() if runtime_activation_start is not None else None
        ),
        "runtime_activation_source": runtime_activation_source,
        "traceability_window_start_timestamp": (
            traceability_start.isoformat() if traceability_start is not None else None
        ),
        "traceability_window_start_source": traceability_start_source,
        "adoption_reconciliation_report_path": str(adoption_path) if adoption_path is not None else None,
        "post_promotion_monitor_report_path": str(post_monitor_path) if post_monitor_path is not None else None,
        "runtime_activation_marker_path": str(activation_path) if activation_path is not None else None,
        "adoption_reconciliation_status": adoption.get("adoption_status"),
        "post_promotion_monitor_status": post_monitor.get("monitor_status"),
        "runtime_activation_marker": activation_marker,
        "post_promotion_monitor_command": "python scripts/ops/run_threshold_post_promotion_monitor.py",
        "baseline_runtime": {
            "active_parameter_pack": baseline_pack,
            "selection_policy": baseline_policy,
            "selected_metrics": _frame_metrics(baseline_selected),
        },
        "candidate_runtime": {
            "active_parameter_pack": candidate_pack,
            "selection_policy": candidate_policy,
            "selected_metrics": _frame_metrics(candidate_selected),
        },
        "policy_delta": policy_delta,
        "rollout_comparison": rollout_comparison,
        "post_adoption_traceability": traceability,
        "candidate_label_readiness": label_readiness,
        "output_structure_guard": structure,
        "provenance_guard": provenance,
        "execution_side_effects": execution_side_effects,
        "candidate_pack_metadata": candidate_metadata,
        "checked_provenance_columns": [column for column in PROVENANCE_COLUMNS if column in frame.columns],
    }
    return _sanitize_value(report)


def render_threshold_signal_rollout_monitor_markdown(report: dict[str, Any]) -> str:
    """Render signal rollout monitoring as Markdown."""
    comparison = report.get("rollout_comparison", {}) or {}
    traceability = report.get("post_adoption_traceability", {}) or {}
    labels = report.get("candidate_label_readiness", {}) or {}
    execution = report.get("execution_side_effects", {}) or {}
    lines = [
        "# Threshold Signal Rollout Monitor",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Rollout status: **{report.get('rollout_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Execution behavior changed: {report.get('execution_behavior_changed')}",
        f"- Candidate pack: `{report.get('candidate_pack_name')}`",
        f"- Threshold: `{report.get('config_hint')} = {report.get('candidate_runtime_value')}`",
        "",
        "## Rollout Reasons",
        "",
    ]
    for reason in report.get("rollout_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Selection Comparison",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Baseline selected signals | {comparison.get('baseline_signal_count')} |",
            f"| Candidate selected signals | {comparison.get('candidate_signal_count')} |",
            f"| Signal count delta | {comparison.get('signal_count_delta')} |",
            f"| Expected relationship | {comparison.get('expected_relationship')} |",
            f"| Candidate-only signals | {comparison.get('candidate_only_count')} |",
            "",
            "## Traceability",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Adoption start | {report.get('adoption_start_timestamp')} |",
            f"| Runtime activation | {report.get('runtime_activation_timestamp') or 'not marked'} |",
            f"| Traceability window start | {report.get('traceability_window_start_timestamp')} |",
            f"| Traceability-window signals | {traceability.get('post_adoption_signal_count')} |",
            f"| Candidate-pack signals | {traceability.get('candidate_pack_signal_count')} |",
            f"| Missing parameter-pack values | {traceability.get('missing_parameter_pack_count')} |",
            f"| Traceability status | {traceability.get('traceability_status')} |",
            "",
            "## Outcome Readiness",
            "",
            f"- Candidate labels ready: `{labels.get('post_promotion_monitor_ready')}`",
            f"- 60m label count: `{labels.get('label_count_60m')}`",
            f"- Post-promotion monitor status: `{report.get('post_promotion_monitor_status') or 'unknown'}`",
            f"- Post-promotion monitor command: `{report.get('post_promotion_monitor_command')}`",
            "",
            "## Execution Guard",
            "",
            f"- Order/execution side-effect fields present: `{execution.get('checked_columns')}`",
            f"- Non-empty side-effect fields: `{execution.get('total_nonempty_side_effect_fields')}`",
            f"- Orders submitted: `{execution.get('orders_submitted')}`",
            "",
            "*This monitor is signal-only. It does not run the engine, submit orders, alter runtime config, or change parameter packs.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "comparison_csv_path": output / f"{stem}_comparison.csv",
        "latest_json_path": output / THRESHOLD_SIGNAL_ROLLOUT_MONITOR_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_SIGNAL_ROLLOUT_MONITOR_MARKDOWN_FILENAME,
        "latest_comparison_csv_path": output / THRESHOLD_SIGNAL_ROLLOUT_MONITOR_COMPARISON_FILENAME,
    }


def write_threshold_signal_rollout_monitor_report(
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
    strict_candidate_pack_required: bool = False,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write signal-only rollout monitoring artifacts."""
    report = build_threshold_signal_rollout_monitor_report(
        frame,
        dataset_path=dataset_path,
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
        runtime_activation_marker=runtime_activation_marker,
        runtime_activation_marker_path=runtime_activation_marker_path,
        strict_candidate_pack_required=strict_candidate_pack_required,
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_SIGNAL_ROLLOUT_MONITOR_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_signal_rollout_monitor"
    paths = _artifact_paths(output, stem)
    assert_artifact_schema(report, "threshold_signal_rollout_monitor")
    markdown = render_threshold_signal_rollout_monitor_markdown(report)
    comparison = pd.DataFrame([report.get("rollout_comparison", {}) or {}])
    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    _atomic_write_csv(comparison, paths["comparison_csv_path"])
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
        _atomic_write_csv(comparison, paths["latest_comparison_csv_path"])
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact
