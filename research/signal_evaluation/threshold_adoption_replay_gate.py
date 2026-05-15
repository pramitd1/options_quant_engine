"""Pre-adoption replay gate for approved threshold adoption plans.

The gate replays the proposed threshold override in a temporary runtime
parameter context. It is read-only: it does not edit parameter packs, change
active runtime selection, invoke execution, or place trades.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from config.policy_resolver import get_active_parameter_pack, temporary_parameter_pack
from config.signal_evaluation_scoring import get_signal_evaluation_selection_policy
from research.signal_evaluation.threshold_adoption_helper import (
    ADOPTION_PLAN_ALREADY_ACTIVE,
    ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK,
    ADOPTION_PLAN_READY,
    DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR,
    THRESHOLD_ADOPTION_PLAN_JSON_FILENAME,
)
from tuning.objectives import apply_selection_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADOPTION_PLAN_REPORT_PATH = (
    DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR / THRESHOLD_ADOPTION_PLAN_JSON_FILENAME
)
DEFAULT_THRESHOLD_ADOPTION_REPLAY_GATE_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_adoption_replay_gate"
)

THRESHOLD_ADOPTION_REPLAY_GATE_JSON_FILENAME = "latest_threshold_adoption_replay_gate.json"
THRESHOLD_ADOPTION_REPLAY_GATE_MARKDOWN_FILENAME = "latest_threshold_adoption_replay_gate.md"
THRESHOLD_ADOPTION_REPLAY_GATE_COMPARISON_FILENAME = "latest_threshold_adoption_replay_gate_comparison.csv"

ADOPTION_REPLAY_READY = "ADOPTION_REPLAY_READY"
ADOPTION_REPLAY_BLOCKED = "ADOPTION_REPLAY_BLOCKED"

SUPPORTED_PLAN_STATUSES = {
    ADOPTION_PLAN_READY,
    ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK,
    ADOPTION_PLAN_ALREADY_ACTIVE,
}
SELECTION_POLICY_PREFIX = "evaluation_thresholds.selection."
PROVENANCE_COLUMNS = (
    "source",
    "requested_option_source",
    "option_source",
    "spot_source",
    "market_data_source_consistency",
    "market_data_provenance_status",
    "market_data_trade_blocking_status",
    "market_data_timestamp_status",
    "market_data_timestamp_delta_seconds",
    "selected_option_iv_proxy_source",
    "selected_option_delta_proxy_source",
    "saved_spot_snapshot_path",
    "saved_chain_snapshot_path",
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


def _policy_delta(baseline_policy: dict[str, Any], candidate_policy: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(baseline_policy) | set(candidate_policy))
    deltas = {}
    for key in keys:
        baseline_value = baseline_policy.get(key)
        candidate_value = candidate_policy.get(key)
        if not _values_match(baseline_value, candidate_value):
            deltas[key] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
    return deltas


def _frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "signal_count": 0,
            "label_count_60m": 0,
            "hit_rate_60m": None,
            "avg_signed_return_60m_bps": None,
            "first_signal_timestamp": None,
            "last_signal_timestamp": None,
        }
    hit = pd.to_numeric(frame.get("correct_60m", pd.Series(dtype=float)), errors="coerce").dropna()
    returns = pd.to_numeric(frame.get("signed_return_60m_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    timestamps = pd.to_datetime(frame.get("signal_timestamp", pd.Series(dtype=object)), errors="coerce", utc=True).dropna()
    return {
        "signal_count": int(len(frame)),
        "label_count_60m": int(hit.count()),
        "hit_rate_60m": round(float(hit.mean()), 4) if not hit.empty else None,
        "avg_signed_return_60m_bps": round(float(returns.mean()), 4) if not returns.empty else None,
        "first_signal_timestamp": timestamps.min().isoformat() if not timestamps.empty else None,
        "last_signal_timestamp": timestamps.max().isoformat() if not timestamps.empty else None,
    }


def _signal_ids(frame: pd.DataFrame) -> set[str]:
    if "signal_id" not in frame.columns:
        return set()
    values = frame["signal_id"].dropna().astype(str)
    return set(values.tolist())


def _selection_relationship(
    *,
    selection_key: str,
    baseline_value: Any,
    candidate_value: Any,
    baseline_selected: pd.DataFrame,
    candidate_selected: pd.DataFrame,
) -> dict[str, Any]:
    baseline_ids = _signal_ids(baseline_selected)
    candidate_ids = _signal_ids(candidate_selected)
    baseline_float = _safe_float(baseline_value)
    candidate_float = _safe_float(candidate_value)
    expected = "unknown"
    passed = False
    if not baseline_ids and len(baseline_selected) > 0:
        return {
            "expected_relationship": "unverifiable_without_signal_id",
            "relationship_passed": False,
            "baseline_only_count": 0,
            "candidate_only_count": 0,
        }

    if selection_key.endswith("_floor") and baseline_float is not None and candidate_float is not None:
        if candidate_float >= baseline_float:
            expected = "candidate_subset_of_baseline"
            passed = candidate_ids.issubset(baseline_ids)
        else:
            expected = "baseline_subset_of_candidate"
            passed = baseline_ids.issubset(candidate_ids)
    elif selection_key.endswith("_cap") and baseline_float is not None and candidate_float is not None:
        if candidate_float <= baseline_float:
            expected = "candidate_subset_of_baseline"
            passed = candidate_ids.issubset(baseline_ids)
        else:
            expected = "baseline_subset_of_candidate"
            passed = baseline_ids.issubset(candidate_ids)
    elif selection_key == "require_overnight_hold_allowed":
        baseline_bool = str(baseline_value).lower() in {"1", "true", "yes", "on"}
        candidate_bool = str(candidate_value).lower() in {"1", "true", "yes", "on"}
        if baseline_bool == candidate_bool:
            expected = "same_selection"
            passed = baseline_ids == candidate_ids
        elif candidate_bool:
            expected = "candidate_subset_of_baseline"
            passed = candidate_ids.issubset(baseline_ids)
        else:
            expected = "baseline_subset_of_candidate"
            passed = baseline_ids.issubset(candidate_ids)
    return {
        "expected_relationship": expected,
        "relationship_passed": bool(passed),
        "baseline_only_count": int(len(baseline_ids - candidate_ids)),
        "candidate_only_count": int(len(candidate_ids - baseline_ids)),
    }


def _output_structure_report(
    frame: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    candidate_selected: pd.DataFrame,
) -> dict[str, Any]:
    input_columns = list(frame.columns)
    baseline_columns = list(baseline_selected.columns)
    candidate_columns = list(candidate_selected.columns)
    return {
        "input_column_count": int(len(input_columns)),
        "baseline_column_count": int(len(baseline_columns)),
        "candidate_column_count": int(len(candidate_columns)),
        "baseline_matches_input": baseline_columns == input_columns,
        "candidate_matches_input": candidate_columns == input_columns,
        "candidate_matches_baseline": candidate_columns == baseline_columns,
        "candidate_extra_columns": sorted(set(candidate_columns) - set(input_columns)),
        "candidate_missing_columns": sorted(set(input_columns) - set(candidate_columns)),
    }


def _provenance_report(
    frame: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    candidate_selected: pd.DataFrame,
) -> dict[str, Any]:
    columns = [column for column in PROVENANCE_COLUMNS if column in frame.columns]
    if not columns:
        return {
            "provenance_check_passed": True,
            "checked_columns": [],
            "warning": "No provenance columns were present in the dataset.",
            "mismatch_count": 0,
        }
    if "signal_id" not in frame.columns:
        return {
            "provenance_check_passed": False,
            "checked_columns": columns,
            "warning": "signal_id column is required to verify provenance preservation.",
            "mismatch_count": None,
        }
    reference_ids = frame["signal_id"].dropna().astype(str)
    if reference_ids.duplicated().any():
        return {
            "provenance_check_passed": False,
            "checked_columns": columns,
            "warning": "signal_id values must be unique to verify provenance preservation.",
            "mismatch_count": None,
        }

    reference = frame[["signal_id", *columns]].copy()
    reference["_signal_id_key"] = reference["signal_id"].astype(str)
    reference = reference.set_index("_signal_id_key")
    mismatches = 0
    checked_rows = 0
    for subset in (baseline_selected, candidate_selected):
        if subset.empty:
            continue
        comparable = subset[["signal_id", *columns]].copy()
        comparable["_signal_id_key"] = comparable["signal_id"].astype(str)
        comparable = comparable.set_index("_signal_id_key")
        common_ids = comparable.index.intersection(reference.index)
        checked_rows += int(len(common_ids))
        for column in columns:
            left = comparable.loc[common_ids, column].astype("object").where(lambda item: item.notna(), "__NA__")
            right = reference.loc[common_ids, column].astype("object").where(lambda item: item.notna(), "__NA__")
            mismatches += int((left.astype(str) != right.astype(str)).sum())
    return {
        "provenance_check_passed": mismatches == 0,
        "checked_columns": columns,
        "checked_selected_rows": checked_rows,
        "warning": None,
        "mismatch_count": int(mismatches),
    }


def _plan_override_guard(plan: dict[str, Any], *, config_hint: str | None) -> dict[str, Any]:
    before = (plan.get("target_parameter_pack_before", {}) or {}).get("overrides", {}) or {}
    after = (plan.get("target_parameter_pack_after", {}) or {}).get("overrides", {}) or {}
    if not before and not after:
        return {
            "override_guard_passed": True,
            "changed_override_keys": [],
            "warning": "Adoption plan did not include target pack before/after overrides.",
        }
    changed = sorted(
        key
        for key in set(before) | set(after)
        if not _values_match(before.get(key), after.get(key))
    )
    return {
        "override_guard_passed": changed == ([str(config_hint)] if config_hint else []),
        "changed_override_keys": changed,
        "expected_changed_override_keys": [str(config_hint)] if config_hint else [],
        "warning": None,
    }


def build_threshold_adoption_replay_gate_report(
    frame: pd.DataFrame,
    *,
    adoption_plan_report: dict[str, Any] | None = None,
    adoption_plan_report_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a read-only replay gate for a threshold adoption plan."""
    plan_path = adoption_plan_report_path or DEFAULT_ADOPTION_PLAN_REPORT_PATH
    plan = adoption_plan_report or _load_json_if_exists(plan_path)
    proposed = plan.get("proposed_change", {}) or {}
    config_hint = proposed.get("config_hint")
    candidate_value = proposed.get("candidate_value")
    selection_key = _selection_key(config_hint)
    reasons: list[str] = []

    active_before = get_active_parameter_pack()
    baseline_policy = dict(get_signal_evaluation_selection_policy())
    candidate_policy: dict[str, Any] = {}
    candidate_pack: dict[str, Any] = {}
    active_after = active_before
    if selection_key:
        overrides = dict(active_before.get("overrides", {}) or {})
        overrides[str(config_hint)] = candidate_value
        with temporary_parameter_pack(active_before.get("name") or "baseline_v1", overrides=overrides):
            candidate_pack = get_active_parameter_pack()
            candidate_policy = dict(get_signal_evaluation_selection_policy())
        active_after = get_active_parameter_pack()
    else:
        candidate_policy = dict(baseline_policy)

    baseline_selected = apply_selection_policy(frame, thresholds=baseline_policy)
    candidate_selected = apply_selection_policy(frame, thresholds=candidate_policy)
    policy_delta = _policy_delta(baseline_policy, candidate_policy)
    plan_guard = _plan_override_guard(plan, config_hint=str(config_hint) if config_hint else None)
    structure = _output_structure_report(frame, baseline_selected, candidate_selected)
    provenance = _provenance_report(frame, baseline_selected, candidate_selected)
    selection_relationship = _selection_relationship(
        selection_key=selection_key or "",
        baseline_value=baseline_policy.get(selection_key or ""),
        candidate_value=candidate_policy.get(selection_key or ""),
        baseline_selected=baseline_selected,
        candidate_selected=candidate_selected,
    )
    candidate_runtime_value = candidate_policy.get(selection_key or "")
    baseline_runtime_value = baseline_policy.get(selection_key or "")
    context_restored = (
        active_before.get("name") == active_after.get("name")
        and dict(active_before.get("overrides", {}) or {}) == dict(active_after.get("overrides", {}) or {})
    )

    plan_status = plan.get("plan_status")
    if plan_status not in SUPPORTED_PLAN_STATUSES:
        reasons.append(f"Adoption plan status is {plan_status or 'UNKNOWN'}, not replay-ready.")
    if not selection_key:
        reasons.append(f"Config hint `{config_hint or 'none'}` is not a supported selection-policy override.")
    if candidate_value is None:
        reasons.append("Adoption plan does not contain a concrete candidate value.")
    if selection_key and not _values_match(candidate_runtime_value, candidate_value):
        reasons.append("Candidate runtime policy value does not match the adoption plan value.")
    if selection_key and proposed.get("current_active_value") is not None and not _values_match(
        baseline_runtime_value,
        proposed.get("current_active_value"),
    ):
        reasons.append("Baseline runtime policy value no longer matches the adoption plan's active value.")
    if selection_key and sorted(policy_delta.keys()) != [selection_key]:
        reasons.append("Temporary runtime context changed selection-policy keys beyond the planned override.")
    if not plan_guard.get("override_guard_passed"):
        reasons.append("Adoption plan target parameter-pack diff contains override changes beyond the planned threshold.")
    if not structure.get("baseline_matches_input") or not structure.get("candidate_matches_input"):
        reasons.append("Replay output structure differs from the input signal dataset structure.")
    if not provenance.get("provenance_check_passed"):
        reasons.append("Data-source/provenance fields could not be verified as preserved.")
    if not selection_relationship.get("relationship_passed"):
        reasons.append("Candidate replay selection set did not match the expected stricter/looser threshold relationship.")
    if not context_restored:
        reasons.append("Temporary parameter context was not restored after replay.")

    status = ADOPTION_REPLAY_BLOCKED if reasons else ADOPTION_REPLAY_READY
    if not reasons:
        reasons.append("Candidate threshold replay is ready: only the planned selection-policy override changed, output structure stayed stable, provenance fields were preserved, and the temporary runtime context was restored.")

    comparison = {
        "baseline_signal_count": int(len(baseline_selected)),
        "candidate_signal_count": int(len(candidate_selected)),
        "signal_count_delta": int(len(candidate_selected) - len(baseline_selected)),
        **selection_relationship,
    }
    report = {
        "report_type": "threshold_adoption_replay_gate",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "adoption_plan_report_path": str(plan_path) if plan_path is not None else None,
        "replay_status": status,
        "replay_reasons": reasons,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": False,
        "execution_side_effects": {
            "trading_execution_invoked": False,
            "orders_submitted": False,
            "active_parameter_context_restored": context_restored,
            "files_modified": False,
        },
        "adoption_plan_status": plan_status,
        "proposed_change": proposed,
        "selection_key": selection_key,
        "baseline_runtime": {
            "active_parameter_pack": active_before,
            "selection_policy": baseline_policy,
            "selected_metrics": _frame_metrics(baseline_selected),
        },
        "candidate_runtime": {
            "active_parameter_pack": candidate_pack,
            "selection_policy": candidate_policy,
            "selected_metrics": _frame_metrics(candidate_selected),
        },
        "policy_delta": policy_delta,
        "replay_comparison": comparison,
        "output_structure_guard": structure,
        "provenance_guard": provenance,
        "plan_override_guard": plan_guard,
    }
    return _sanitize_value(report)


def render_threshold_adoption_replay_gate_markdown(report: dict[str, Any]) -> str:
    """Render a pre-adoption replay gate report as Markdown."""
    comparison = report.get("replay_comparison", {}) or {}
    change = report.get("proposed_change", {}) or {}
    structure = report.get("output_structure_guard", {}) or {}
    provenance = report.get("provenance_guard", {}) or {}
    lines = [
        "# Threshold Adoption Replay Gate",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Adoption plan: {report.get('adoption_plan_report_path') or 'not supplied'}",
        f"- Replay status: **{report.get('replay_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Selection key: `{report.get('selection_key') or 'none'}`",
        "",
        "## Replay Reasons",
        "",
    ]
    for reason in report.get("replay_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Proposed Override",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Config hint | `{change.get('config_hint') or 'none'}` |",
            f"| Current active value | {change.get('current_active_value')} |",
            f"| Candidate value | {change.get('candidate_value')} |",
            "",
            "## Replay Comparison",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Baseline selected signals | {comparison.get('baseline_signal_count')} |",
            f"| Candidate selected signals | {comparison.get('candidate_signal_count')} |",
            f"| Signal count delta | {comparison.get('signal_count_delta')} |",
            f"| Expected relationship | {comparison.get('expected_relationship')} |",
            f"| Baseline-only signals | {comparison.get('baseline_only_count')} |",
            f"| Candidate-only signals | {comparison.get('candidate_only_count')} |",
            "",
            "## Guardrails",
            "",
            f"- Output structure preserved: `{structure.get('baseline_matches_input') and structure.get('candidate_matches_input')}`",
            f"- Provenance preserved: `{provenance.get('provenance_check_passed')}`",
            f"- Provenance columns checked: `{', '.join(provenance.get('checked_columns', []) or []) or 'none'}`",
            "",
            "*This replay gate is read-only. It uses a temporary parameter context and does not alter runtime configuration, parameter-pack files, or execution behavior.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "comparison_csv_path": output / f"{stem}_comparison.csv",
        "latest_json_path": output / THRESHOLD_ADOPTION_REPLAY_GATE_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_ADOPTION_REPLAY_GATE_MARKDOWN_FILENAME,
        "latest_comparison_csv_path": output / THRESHOLD_ADOPTION_REPLAY_GATE_COMPARISON_FILENAME,
    }


def write_threshold_adoption_replay_gate_report(
    frame: pd.DataFrame,
    *,
    adoption_plan_report: dict[str, Any] | None = None,
    adoption_plan_report_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write pre-adoption replay gate artifacts."""
    report = build_threshold_adoption_replay_gate_report(
        frame,
        adoption_plan_report=adoption_plan_report,
        adoption_plan_report_path=adoption_plan_report_path,
        dataset_path=dataset_path,
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_ADOPTION_REPLAY_GATE_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_adoption_replay_gate"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_adoption_replay_gate_markdown(report)
    comparison = pd.DataFrame([report.get("replay_comparison", {}) or {}])
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
