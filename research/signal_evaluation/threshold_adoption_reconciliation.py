"""Read-only reconciliation of approved threshold adoption state.

This module compares manually approved threshold-promotion decisions against
the active runtime policy view. It never applies, reverts, or edits runtime
configuration; it only reports whether the user's approved threshold appears to
be active.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from config.policy_resolver import get_active_parameter_pack
from config.signal_evaluation_scoring import (
    SIGNAL_EVALUATION_SELECTION_POLICY,
    get_signal_evaluation_selection_policy,
)
from research.signal_evaluation.threshold_governance import CONFIG_HINTS
from research.signal_evaluation.threshold_post_promotion_monitor import (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR,
    THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME,
)
from research.signal_evaluation.threshold_promotion_review import (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    PROMOTION_REVIEW_READY,
    THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME,
    THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_adoption_reconciliation"
)
DEFAULT_PROMOTION_REVIEW_REPORT_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME
)
DEFAULT_PROMOTION_REVIEW_LEDGER_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME
)
DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH = (
    DEFAULT_THRESHOLD_POST_PROMOTION_MONITOR_DIR / THRESHOLD_POST_PROMOTION_MONITOR_JSON_FILENAME
)

THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME = "latest_threshold_adoption_reconciliation.json"
THRESHOLD_ADOPTION_RECONCILIATION_MARKDOWN_FILENAME = "latest_threshold_adoption_reconciliation.md"
THRESHOLD_ADOPTION_RECONCILIATION_COMPARISON_FILENAME = (
    "latest_threshold_adoption_reconciliation_comparison.csv"
)

APPROVED_BUT_NOT_ADOPTED = "APPROVED_BUT_NOT_ADOPTED"
ADOPTED_MANUALLY = "ADOPTED_MANUALLY"
ADOPTION_MISMATCH = "ADOPTION_MISMATCH"
ROLLED_BACK_MANUALLY = "ROLLED_BACK_MANUALLY"
UNKNOWN_ADOPTION_STATE = "UNKNOWN_ADOPTION_STATE"

ADOPTION_STATUSES = {
    APPROVED_BUT_NOT_ADOPTED,
    ADOPTED_MANUALLY,
    ADOPTION_MISMATCH,
    ROLLED_BACK_MANUALLY,
    UNKNOWN_ADOPTION_STATE,
}

ROLLBACK_ACTIONS = {"ROLLED_BACK", "ROLLBACK", "REVERTED"}
ROLLBACK_NOTE_PHRASES = ("rollback", "rolled back", "revert", "reverted")
EVALUATION_SELECTION_PREFIX = "evaluation_thresholds.selection."


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


def _read_ledger(ledger_path: str | Path | None) -> pd.DataFrame:
    if ledger_path is None:
        return pd.DataFrame()
    path = Path(ledger_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _sort_ledger_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    sorted_rows = rows.copy()
    if "reviewed_at" in sorted_rows.columns:
        sorted_rows["_reviewed_at_ts"] = pd.to_datetime(
            sorted_rows["reviewed_at"],
            errors="coerce",
            utc=True,
        )
        sorted_rows = sorted_rows.sort_values("_reviewed_at_ts", na_position="first")
    return sorted_rows


def _row_to_dict(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return _sanitize_value(row.drop(labels=["_reviewed_at_ts"], errors="ignore").to_dict())


def _matching_candidate_rows(
    ledger: pd.DataFrame,
    *,
    candidate_key: str | None = None,
    config_hint: str | None = None,
    threshold_field: str | None = None,
) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    rows = ledger.copy()
    if candidate_key and "candidate_key" in rows.columns:
        keyed = rows.loc[rows["candidate_key"].astype(str) == str(candidate_key)].copy()
        if not keyed.empty:
            return keyed
    if config_hint and "config_hint" in rows.columns:
        hinted = rows.loc[rows["config_hint"].astype(str) == str(config_hint)].copy()
        if not hinted.empty:
            return hinted
    if threshold_field and "threshold_field" in rows.columns:
        fielded = rows.loc[rows["threshold_field"].astype(str) == str(threshold_field)].copy()
        if not fielded.empty:
            return fielded
    return rows


def _latest_approved_decision(
    ledger_path: str | Path | None,
    *,
    candidate_key: str | None = None,
    config_hint: str | None = None,
    threshold_field: str | None = None,
) -> dict[str, Any] | None:
    ledger = _read_ledger(ledger_path)
    if ledger.empty or "review_action" not in ledger.columns:
        return None
    rows = _matching_candidate_rows(
        ledger,
        candidate_key=candidate_key,
        config_hint=config_hint,
        threshold_field=threshold_field,
    )
    rows = rows.loc[rows["review_action"].astype(str).str.upper() == "APPROVED"].copy()
    if rows.empty:
        return None
    sorted_rows = _sort_ledger_rows(rows)
    return _row_to_dict(sorted_rows.iloc[-1])


def _latest_candidate_decision(
    ledger_path: str | Path | None,
    *,
    candidate_key: str | None = None,
    config_hint: str | None = None,
    threshold_field: str | None = None,
) -> dict[str, Any] | None:
    ledger = _read_ledger(ledger_path)
    if ledger.empty:
        return None
    rows = _matching_candidate_rows(
        ledger,
        candidate_key=candidate_key,
        config_hint=config_hint,
        threshold_field=threshold_field,
    )
    if rows.empty:
        return None
    sorted_rows = _sort_ledger_rows(rows)
    return _row_to_dict(sorted_rows.iloc[-1])


def _rollback_decision_after_approval(
    ledger_path: str | Path | None,
    approval: dict[str, Any] | None,
    *,
    candidate_key: str | None = None,
    config_hint: str | None = None,
    threshold_field: str | None = None,
) -> dict[str, Any] | None:
    if not approval:
        return None
    ledger = _read_ledger(ledger_path)
    if ledger.empty or "review_action" not in ledger.columns:
        return None
    rows = _matching_candidate_rows(
        ledger,
        candidate_key=candidate_key,
        config_hint=config_hint,
        threshold_field=threshold_field,
    )
    if rows.empty:
        return None
    approval_ts = pd.to_datetime(approval.get("reviewed_at"), errors="coerce", utc=True)
    if "reviewed_at" in rows.columns and not pd.isna(approval_ts):
        row_ts = pd.to_datetime(rows["reviewed_at"], errors="coerce", utc=True)
        rows = rows.loc[row_ts > approval_ts].copy()
    if rows.empty:
        return None
    actions = rows["review_action"].astype(str).str.upper()
    explicit = rows.loc[actions.isin(ROLLBACK_ACTIONS)].copy()
    if explicit.empty:
        notes = rows.get("review_note", pd.Series([""] * len(rows), index=rows.index)).astype(str).str.lower()
        note_mask = notes.apply(lambda text: any(phrase in text for phrase in ROLLBACK_NOTE_PHRASES))
        explicit = rows.loc[note_mask].copy()
    if explicit.empty:
        return None
    sorted_rows = _sort_ledger_rows(explicit)
    return _row_to_dict(sorted_rows.iloc[-1])


def _deep_get(mapping: Any, dotted_key: str) -> tuple[Any, bool]:
    if not isinstance(mapping, dict):
        return None, False
    if dotted_key in mapping:
        return mapping[dotted_key], True
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _selection_key(config_hint: str | None) -> str | None:
    hint = str(config_hint or "")
    if hint.startswith(EVALUATION_SELECTION_PREFIX):
        return hint[len(EVALUATION_SELECTION_PREFIX):]
    return None


def _resolve_hint_value(
    config_hint: str | None,
    *,
    runtime_config: dict[str, Any] | None,
    parameter_pack: dict[str, Any] | None,
    active_parameter_pack: dict[str, Any],
    active_selection_policy: dict[str, Any],
) -> tuple[Any, str | None]:
    if not config_hint:
        return None, None
    hint = str(config_hint)
    runtime = runtime_config or {}
    pack = parameter_pack or {}

    value, found = _deep_get(runtime, hint)
    if found:
        return value, "runtime_config"
    value, found = _deep_get(runtime.get("overrides", {}), hint)
    if found:
        return value, "runtime_config.overrides"

    value, found = _deep_get(pack.get("overrides", {}), hint)
    if found:
        return value, "parameter_pack_file.overrides"
    value, found = _deep_get(pack, hint)
    if found:
        return value, "parameter_pack_file"

    value, found = _deep_get(active_parameter_pack.get("overrides", {}), hint)
    if found:
        return value, "active_parameter_pack.overrides"

    selection_key = _selection_key(hint)
    if selection_key and selection_key in runtime:
        return runtime[selection_key], "runtime_config.selection_key"
    if selection_key and selection_key in active_selection_policy:
        return active_selection_policy[selection_key], "active_selection_policy"
    return None, None


def _resolve_default_value(config_hint: str | None) -> tuple[Any, str | None]:
    selection_key = _selection_key(config_hint)
    if selection_key and selection_key in SIGNAL_EVALUATION_SELECTION_POLICY:
        return SIGNAL_EVALUATION_SELECTION_POLICY[selection_key], "code_default_selection_policy"
    return None, None


def _expected_candidate_value(candidate: dict[str, Any], approval: dict[str, Any] | None) -> tuple[Any, str | None]:
    config_hint = candidate.get("config_hint") or (approval or {}).get("config_hint")
    overrides = candidate.get("overrides", {}) or {}
    if config_hint and isinstance(overrides, dict) and str(config_hint) in overrides:
        return overrides[str(config_hint)], "promotion_candidate.overrides"
    rule = candidate.get("threshold_rule", {}) or {}
    if rule.get("value") is not None:
        return rule.get("value"), "promotion_candidate.threshold_rule"
    if approval and approval.get("threshold_value") is not None:
        return approval.get("threshold_value"), "approval_decision.threshold_value"
    return None, None


def _candidate_from_package(package: dict[str, Any], approval: dict[str, Any] | None) -> dict[str, Any]:
    candidate = dict(package.get("promotion_candidate", {}) or {})
    rule = dict(candidate.get("threshold_rule", {}) or {})
    if not rule and approval:
        rule = {
            "field": approval.get("threshold_field"),
            "operator": ">=",
            "value": approval.get("threshold_value"),
        }
        candidate["threshold_rule"] = rule
    if not candidate.get("config_hint"):
        candidate["config_hint"] = (approval or {}).get("config_hint") or CONFIG_HINTS.get(
            str(rule.get("field") or ""),
            "research_only.unmapped_threshold",
        )
    if not candidate.get("source_candidate_key"):
        candidate["source_candidate_key"] = (approval or {}).get("candidate_key")
    candidate["runtime_config_changed"] = False
    return candidate


def _recommended_action(status: str, *, post_promotion_status: str | None = None) -> str:
    if status == ADOPTED_MANUALLY:
        if post_promotion_status == "POST_PROMOTION_DETERIORATING":
            return "Approved threshold is active, but post-promotion evidence is deteriorating; open a manual rollback review."
        return "Approved threshold appears active; continue post-promotion monitoring and keep the manual decision ledger current."
    if status == APPROVED_BUT_NOT_ADOPTED:
        return "Approved threshold is not active; either apply it manually through the normal config process or record a new ledger decision."
    if status == ADOPTION_MISMATCH:
        return "Active runtime value differs from both the approved candidate and the default; open a manual config review."
    if status == ROLLED_BACK_MANUALLY:
        return "Ledger indicates a manual rollback or revert; confirm the rollback target and continue shadow evidence collection."
    return "Adoption state is unresolved; verify the promotion ledger, config hint, and active parameter pack."


def _status_and_reasons(
    *,
    approval: dict[str, Any] | None,
    package_status: str | None,
    rollback_decision: dict[str, Any] | None,
    config_hint: str | None,
    expected_value: Any,
    observed_value: Any,
    default_value: Any,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not approval:
        return UNKNOWN_ADOPTION_STATE, ["No APPROVED threshold promotion decision is available in the review ledger."]
    if package_status != PROMOTION_REVIEW_READY:
        return UNKNOWN_ADOPTION_STATE, [
            f"Approved decision does not resolve to a PROMOTION_REVIEW_READY package; package status is {package_status or 'UNKNOWN'}."
        ]
    if rollback_decision:
        return ROLLED_BACK_MANUALLY, [
            "A later ledger entry for the approved candidate indicates rollback or revert intent."
        ]
    if expected_value is None:
        return UNKNOWN_ADOPTION_STATE, ["Approved promotion package does not contain a concrete threshold value."]
    if not config_hint:
        return UNKNOWN_ADOPTION_STATE, ["Approved promotion package does not include a config hint."]
    if observed_value is None:
        return UNKNOWN_ADOPTION_STATE, [f"No active runtime value could be resolved for config hint `{config_hint}`."]
    if _values_match(observed_value, expected_value):
        return ADOPTED_MANUALLY, ["Active runtime value matches the approved threshold candidate."]
    if default_value is not None and _values_match(observed_value, default_value):
        return APPROVED_BUT_NOT_ADOPTED, [
            "Active runtime value still matches the code default rather than the approved threshold candidate."
        ]
    reasons.append("Active runtime value does not match the approved threshold candidate.")
    if default_value is not None:
        reasons.append("Active runtime value also differs from the code default, suggesting an out-of-band policy change.")
    return ADOPTION_MISMATCH, reasons


def build_threshold_adoption_reconciliation_report(
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    parameter_pack: dict[str, Any] | None = None,
    parameter_pack_path: str | Path | None = None,
    candidate_key: str | None = None,
) -> dict[str, Any]:
    """Build a read-only threshold adoption-state reconciliation report."""
    package_path = promotion_package_report_path or DEFAULT_PROMOTION_REVIEW_REPORT_PATH
    ledger = Path(ledger_path) if ledger_path is not None else DEFAULT_PROMOTION_REVIEW_LEDGER_PATH
    post_monitor_path = post_promotion_monitor_report_path or DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH
    package = promotion_package_report or _load_json_if_exists(package_path)
    post_monitor = post_promotion_monitor_report or _load_json_if_exists(post_monitor_path)
    pack_file = parameter_pack or _load_json_if_exists(parameter_pack_path)

    package_candidate = package.get("promotion_candidate", {}) or {}
    package_rule = package_candidate.get("threshold_rule", {}) or {}
    resolved_candidate_key = candidate_key or package_candidate.get("source_candidate_key")
    config_hint = package_candidate.get("config_hint") or CONFIG_HINTS.get(
        str(package_rule.get("field") or ""),
        "research_only.unmapped_threshold",
    )
    approval = _latest_approved_decision(
        ledger,
        candidate_key=resolved_candidate_key,
        config_hint=config_hint,
        threshold_field=package_rule.get("field"),
    )
    if approval and not package:
        package_path = approval.get("report_json") or package_path
        package = _load_json_if_exists(package_path)
        package_candidate = package.get("promotion_candidate", {}) or {}
        package_rule = package_candidate.get("threshold_rule", {}) or {}
        resolved_candidate_key = candidate_key or package_candidate.get("source_candidate_key") or approval.get("candidate_key")
        config_hint = package_candidate.get("config_hint") or approval.get("config_hint") or CONFIG_HINTS.get(
            str(package_rule.get("field") or approval.get("threshold_field") or ""),
            "research_only.unmapped_threshold",
        )

    candidate = _candidate_from_package(package, approval)
    rule = candidate.get("threshold_rule", {}) or {}
    config_hint = candidate.get("config_hint") or config_hint
    resolved_candidate_key = (
        candidate_key
        or candidate.get("source_candidate_key")
        or (approval or {}).get("candidate_key")
    )
    latest_decision = _latest_candidate_decision(
        ledger,
        candidate_key=resolved_candidate_key,
        config_hint=config_hint,
        threshold_field=rule.get("field"),
    )
    rollback_decision = _rollback_decision_after_approval(
        ledger,
        approval,
        candidate_key=resolved_candidate_key,
        config_hint=config_hint,
        threshold_field=rule.get("field"),
    )

    active_parameter_pack = get_active_parameter_pack()
    active_selection_policy = get_signal_evaluation_selection_policy()
    observed_value, observed_source = _resolve_hint_value(
        config_hint,
        runtime_config=runtime_config,
        parameter_pack=pack_file,
        active_parameter_pack=active_parameter_pack,
        active_selection_policy=active_selection_policy,
    )
    default_value, default_source = _resolve_default_value(config_hint)
    expected_value, expected_source = _expected_candidate_value(candidate, approval)
    status, reasons = _status_and_reasons(
        approval=approval,
        package_status=package.get("promotion_review_status"),
        rollback_decision=rollback_decision,
        config_hint=config_hint,
        expected_value=expected_value,
        observed_value=observed_value,
        default_value=default_value,
    )
    comparison = {
        "config_hint": config_hint,
        "threshold_field": rule.get("field"),
        "threshold_operator": rule.get("operator", ">="),
        "candidate_value": expected_value,
        "candidate_value_source": expected_source,
        "observed_runtime_value": observed_value,
        "observed_runtime_source": observed_source,
        "default_runtime_value": default_value,
        "default_runtime_source": default_source,
        "matches_candidate": _values_match(observed_value, expected_value)
        if observed_value is not None and expected_value is not None
        else False,
        "matches_default": _values_match(observed_value, default_value)
        if observed_value is not None and default_value is not None
        else False,
    }
    post_status = post_monitor.get("monitor_status")
    report = {
        "report_type": "threshold_adoption_reconciliation",
        "generated_at": _utc_now(),
        "promotion_review_ledger_path": str(ledger),
        "promotion_package_report_path": str(package_path) if package_path is not None else None,
        "post_promotion_monitor_report_path": str(post_monitor_path) if post_monitor_path is not None else None,
        "parameter_pack_path": str(parameter_pack_path) if parameter_pack_path is not None else None,
        "adoption_status": status,
        "adoption_reasons": reasons,
        "recommended_next_action": _recommended_action(status, post_promotion_status=post_status),
        "runtime_config_changed": False,
        "approval_decision": approval or {},
        "latest_candidate_decision": latest_decision or {},
        "rollback_decision": rollback_decision or {},
        "promotion_candidate": candidate,
        "threshold_rule": {
            "field": rule.get("field"),
            "operator": rule.get("operator", ">="),
            "value": expected_value,
        },
        "comparison": comparison,
        "runtime_state": {
            "active_parameter_pack": {
                "name": active_parameter_pack.get("name"),
                "layers": active_parameter_pack.get("layers", []),
                "override_keys": sorted(str(key) for key in (active_parameter_pack.get("overrides", {}) or {}).keys()),
            },
            "active_selection_policy": active_selection_policy,
            "code_default_selection_policy": dict(SIGNAL_EVALUATION_SELECTION_POLICY),
        },
        "post_promotion_monitor_summary": {
            "monitor_status": post_status,
            "recommended_next_action": post_monitor.get("recommended_next_action"),
            "monitor_reasons": post_monitor.get("monitor_reasons", []),
        },
    }
    return _sanitize_value(report)


def render_threshold_adoption_reconciliation_markdown(report: dict[str, Any]) -> str:
    """Render adoption-state reconciliation as Markdown."""
    comparison = report.get("comparison", {}) or {}
    runtime = report.get("runtime_state", {}) or {}
    active_pack = runtime.get("active_parameter_pack", {}) or {}
    post_monitor = report.get("post_promotion_monitor_summary", {}) or {}
    lines = [
        "# Threshold Adoption Reconciliation",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Promotion ledger: {report.get('promotion_review_ledger_path') or 'not supplied'}",
        f"- Promotion package: {report.get('promotion_package_report_path') or 'not supplied'}",
        f"- Post-promotion monitor: {report.get('post_promotion_monitor_report_path') or 'not supplied'}",
        f"- Adoption status: **{report.get('adoption_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Next action: {report.get('recommended_next_action')}",
        "",
        "## Reconciliation Reasons",
        "",
    ]
    for reason in report.get("adoption_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Runtime Comparison",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Config hint | `{comparison.get('config_hint') or 'none'}` |",
            f"| Candidate value | {comparison.get('candidate_value')} |",
            f"| Candidate source | {comparison.get('candidate_value_source')} |",
            f"| Active runtime value | {comparison.get('observed_runtime_value')} |",
            f"| Active runtime source | {comparison.get('observed_runtime_source')} |",
            f"| Code default value | {comparison.get('default_runtime_value')} |",
            f"| Matches candidate | {comparison.get('matches_candidate')} |",
            f"| Matches default | {comparison.get('matches_default')} |",
            "",
            "## Active Runtime State",
            "",
            f"- Active parameter pack: `{active_pack.get('name') or 'unknown'}`",
            f"- Active pack layers: `{', '.join(str(item) for item in active_pack.get('layers', []) or []) or 'none'}`",
            f"- Active override keys: `{', '.join(str(item) for item in active_pack.get('override_keys', []) or []) or 'none'}`",
            f"- Post-promotion monitor status: `{post_monitor.get('monitor_status') or 'unknown'}`",
            "",
            "*This reconciliation is advisory. It reports adoption state only and never applies, reverts, or edits runtime configuration.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "comparison_csv_path": output / f"{stem}_comparison.csv",
        "latest_json_path": output / THRESHOLD_ADOPTION_RECONCILIATION_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_ADOPTION_RECONCILIATION_MARKDOWN_FILENAME,
        "latest_comparison_csv_path": output / THRESHOLD_ADOPTION_RECONCILIATION_COMPARISON_FILENAME,
    }


def _write_reconciliation_bundle(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_ADOPTION_RECONCILIATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_adoption_reconciliation"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_adoption_reconciliation_markdown(report)
    comparison = pd.DataFrame([report.get("comparison", {}) or {}])
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


def write_threshold_adoption_reconciliation_report(
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    parameter_pack: dict[str, Any] | None = None,
    parameter_pack_path: str | Path | None = None,
    candidate_key: str | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write read-only adoption reconciliation artifacts."""
    report = build_threshold_adoption_reconciliation_report(
        promotion_package_report=promotion_package_report,
        promotion_package_report_path=promotion_package_report_path,
        ledger_path=ledger_path,
        post_promotion_monitor_report=post_promotion_monitor_report,
        post_promotion_monitor_report_path=post_promotion_monitor_report_path,
        runtime_config=runtime_config,
        parameter_pack=parameter_pack,
        parameter_pack_path=parameter_pack_path,
        candidate_key=candidate_key,
    )
    return _write_reconciliation_bundle(
        report,
        output_dir=output_dir,
        report_name=report_name,
        write_latest=write_latest,
    )
