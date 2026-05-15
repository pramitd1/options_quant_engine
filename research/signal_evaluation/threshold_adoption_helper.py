"""Advisory helper for manually adopting approved threshold candidates.

This module builds a concrete parameter-pack patch from an approved promotion
package. It is intentionally advisory by default: dry-run plans and reports do
not edit runtime configuration, active parameter-pack selection, or execution
settings. A parameter-pack file is changed only when the caller explicitly
passes ``apply_changes=True``.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_adoption_reconciliation import (
    ADOPTED_MANUALLY,
    APPROVED_BUT_NOT_ADOPTED,
    ADOPTION_MISMATCH,
    DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH,
    DEFAULT_PROMOTION_REVIEW_LEDGER_PATH,
    DEFAULT_PROMOTION_REVIEW_REPORT_PATH,
    ROLLED_BACK_MANUALLY,
    UNKNOWN_ADOPTION_STATE,
    build_threshold_adoption_reconciliation_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_adoption_plan"
)
DEFAULT_TARGET_PARAMETER_PACK_PATH = PROJECT_ROOT / "config" / "parameter_packs" / "candidate_v1.json"

THRESHOLD_ADOPTION_PLAN_JSON_FILENAME = "latest_threshold_adoption_plan.json"
THRESHOLD_ADOPTION_PLAN_MARKDOWN_FILENAME = "latest_threshold_adoption_plan.md"

ADOPTION_PLAN_READY = "ADOPTION_PLAN_READY"
ADOPTION_PLAN_ALREADY_ACTIVE = "ADOPTION_PLAN_ALREADY_ACTIVE"
ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK = "ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK"
ADOPTION_PLAN_SKIPPED = "ADOPTION_PLAN_SKIPPED"


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


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"


def _parameter_pack_name(path: Path, payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "").strip()
    return name or path.stem


def _base_parameter_pack(path: Path, existing: dict[str, Any]) -> dict[str, Any]:
    if existing:
        pack = dict(existing)
        pack["overrides"] = dict(existing.get("overrides", {}) or {})
        return pack
    return {
        "name": path.stem,
        "version": "1.0.0",
        "description": "Manual threshold adoption candidate generated from promotion review evidence.",
        "parent": "baseline_v1",
        "tags": ["candidate", "manual_threshold_adoption"],
        "metadata": {
            "state": "candidate",
            "owner": "research",
        },
        "overrides": {},
    }


def _patched_parameter_pack(
    *,
    target_path: Path,
    existing_pack: dict[str, Any],
    config_hint: str,
    candidate_value: Any,
) -> dict[str, Any]:
    patched = _base_parameter_pack(target_path, existing_pack)
    overrides = dict(patched.get("overrides", {}) or {})
    overrides[config_hint] = candidate_value
    patched["overrides"] = overrides
    metadata = dict(patched.get("metadata", {}) or {})
    metadata.update(
        {
            "threshold_adoption_config_hint": config_hint,
            "threshold_adoption_candidate_value": candidate_value,
            "threshold_adoption_generated_at": _utc_now(),
        }
    )
    patched["metadata"] = metadata
    return patched


def _diff_text(before: dict[str, Any], after: dict[str, Any], *, target_path: Path) -> str:
    before_lines = _json_text(before).splitlines(keepends=True)
    after_lines = _json_text(after).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{target_path} (current)",
            tofile=f"{target_path} (proposed)",
        )
    )


def _plan_status(reconciliation_status: str | None, *, apply_changes: bool, patchable: bool) -> tuple[str, list[str]]:
    if reconciliation_status == ADOPTED_MANUALLY:
        return ADOPTION_PLAN_ALREADY_ACTIVE, [
            "Active runtime value already matches the approved threshold candidate."
        ]
    if reconciliation_status == ROLLED_BACK_MANUALLY:
        return ADOPTION_PLAN_SKIPPED, [
            "A later ledger entry indicates rollback or revert intent; adoption plan was not built."
        ]
    if reconciliation_status == UNKNOWN_ADOPTION_STATE:
        return ADOPTION_PLAN_SKIPPED, [
            "Adoption reconciliation could not resolve an approved candidate; adoption plan was not built."
        ]
    if not patchable:
        return ADOPTION_PLAN_SKIPPED, [
            "Approved candidate is missing a concrete config hint or threshold value."
        ]
    if apply_changes:
        return ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK, [
            "Parameter-pack file was updated because apply_changes=True was explicitly supplied."
        ]
    if reconciliation_status in {APPROVED_BUT_NOT_ADOPTED, ADOPTION_MISMATCH}:
        return ADOPTION_PLAN_READY, [
            "Approved threshold candidate can be adopted by applying the proposed parameter-pack override."
        ]
    return ADOPTION_PLAN_READY, [
        "Approved threshold candidate resolved; review the proposed parameter-pack override before applying."
    ]


def build_threshold_adoption_plan(
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    parameter_pack: dict[str, Any] | None = None,
    parameter_pack_path: str | Path | None = None,
    target_parameter_pack: dict[str, Any] | None = None,
    target_parameter_pack_path: str | Path | None = None,
    candidate_key: str | None = None,
    apply_changes: bool = False,
) -> dict[str, Any]:
    """Build an advisory parameter-pack patch for an approved threshold."""
    package_path = promotion_package_report_path or DEFAULT_PROMOTION_REVIEW_REPORT_PATH
    ledger = ledger_path or DEFAULT_PROMOTION_REVIEW_LEDGER_PATH
    post_monitor_path = post_promotion_monitor_report_path or DEFAULT_POST_PROMOTION_MONITOR_REPORT_PATH
    target_path = Path(target_parameter_pack_path or parameter_pack_path or DEFAULT_TARGET_PARAMETER_PACK_PATH)
    target_before = target_parameter_pack or _load_json_if_exists(target_path)

    reconciliation = build_threshold_adoption_reconciliation_report(
        promotion_package_report=promotion_package_report,
        promotion_package_report_path=package_path,
        ledger_path=ledger,
        post_promotion_monitor_report=post_promotion_monitor_report,
        post_promotion_monitor_report_path=post_monitor_path,
        runtime_config=runtime_config,
        parameter_pack=parameter_pack,
        parameter_pack_path=parameter_pack_path,
        candidate_key=candidate_key,
    )
    comparison = reconciliation.get("comparison", {}) or {}
    config_hint = comparison.get("config_hint")
    candidate_value = comparison.get("candidate_value")
    current_target_value = None
    if config_hint:
        current_target_value = (target_before.get("overrides", {}) or {}).get(str(config_hint))
    patchable = bool(config_hint) and candidate_value is not None

    target_after = (
        _patched_parameter_pack(
            target_path=target_path,
            existing_pack=target_before,
            config_hint=str(config_hint),
            candidate_value=candidate_value,
        )
        if patchable
        else _base_parameter_pack(target_path, target_before)
    )
    diff = _diff_text(_base_parameter_pack(target_path, target_before), target_after, target_path=target_path) if patchable else ""
    status, reasons = _plan_status(
        reconciliation.get("adoption_status"),
        apply_changes=apply_changes and patchable,
        patchable=patchable,
    )

    parameter_pack_file_changed = False
    if apply_changes and status == ADOPTION_PLAN_APPLIED_TO_PARAMETER_PACK:
        _atomic_write_text(target_path, _json_text(target_after))
        parameter_pack_file_changed = True

    pack_name = _parameter_pack_name(target_path, target_after)
    commands = {
        "dry_run_plan": "python scripts/ops/run_threshold_adoption_helper.py",
        "apply_parameter_pack_patch": (
            "python scripts/ops/run_threshold_adoption_helper.py "
            f"--target-parameter-pack {target_path} --apply"
        ),
        "validate_target_parameter_pack_file": (
            "python scripts/ops/run_threshold_adoption_reconciliation.py "
            f"--parameter-pack {target_path} --require-adopted"
        ),
        "validate_active_runtime_after_selecting_pack": (
            f"OQE_PARAMETER_PACK={pack_name} "
            "python scripts/ops/run_threshold_adoption_reconciliation.py --require-adopted"
        ),
    }
    report = {
        "report_type": "threshold_adoption_plan",
        "generated_at": _utc_now(),
        "plan_status": status,
        "plan_reasons": reasons,
        "runtime_config_changed": False,
        "parameter_pack_file_changed": parameter_pack_file_changed,
        "promotion_package_report_path": str(package_path) if package_path is not None else None,
        "promotion_review_ledger_path": str(ledger) if ledger is not None else None,
        "post_promotion_monitor_report_path": str(post_monitor_path) if post_monitor_path is not None else None,
        "target_parameter_pack_path": str(target_path),
        "target_parameter_pack_name": pack_name,
        "adoption_reconciliation": {
            "adoption_status": reconciliation.get("adoption_status"),
            "adoption_reasons": reconciliation.get("adoption_reasons", []),
            "recommended_next_action": reconciliation.get("recommended_next_action"),
        },
        "promotion_candidate": reconciliation.get("promotion_candidate", {}),
        "threshold_rule": reconciliation.get("threshold_rule", {}),
        "proposed_change": {
            "config_hint": config_hint,
            "current_active_value": comparison.get("observed_runtime_value"),
            "current_active_source": comparison.get("observed_runtime_source"),
            "current_target_pack_value": current_target_value,
            "candidate_value": candidate_value,
            "candidate_value_source": comparison.get("candidate_value_source"),
            "default_runtime_value": comparison.get("default_runtime_value"),
            "operation": "replace" if current_target_value is not None else "add",
        },
        "parameter_pack_patch": {
            "op": "replace" if current_target_value is not None else "add",
            "path": f"/overrides/{config_hint}" if config_hint else None,
            "from": current_target_value,
            "value": candidate_value,
        },
        "target_parameter_pack_before": _base_parameter_pack(target_path, target_before),
        "target_parameter_pack_after": target_after,
        "parameter_pack_diff": diff,
        "exact_commands": commands,
    }
    return _sanitize_value(report)


def render_threshold_adoption_plan_markdown(report: dict[str, Any]) -> str:
    """Render an adoption plan as Markdown."""
    change = report.get("proposed_change", {}) or {}
    commands = report.get("exact_commands", {}) or {}
    lines = [
        "# Threshold Adoption Plan",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Plan status: **{report.get('plan_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Parameter-pack file changed: {report.get('parameter_pack_file_changed')}",
        f"- Target parameter pack: `{report.get('target_parameter_pack_path')}`",
        "",
        "## Plan Reasons",
        "",
    ]
    for reason in report.get("plan_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Proposed Change",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Config hint | `{change.get('config_hint') or 'none'}` |",
            f"| Current active value | {change.get('current_active_value')} |",
            f"| Current active source | {change.get('current_active_source')} |",
            f"| Current target-pack value | {change.get('current_target_pack_value')} |",
            f"| Candidate value | {change.get('candidate_value')} |",
            f"| Default runtime value | {change.get('default_runtime_value')} |",
            f"| Operation | {change.get('operation')} |",
            "",
            "## Commands",
            "",
            f"- Dry run: `{commands.get('dry_run_plan')}`",
            f"- Apply patch: `{commands.get('apply_parameter_pack_patch')}`",
            f"- Validate target pack: `{commands.get('validate_target_parameter_pack_file')}`",
            f"- Validate active runtime: `{commands.get('validate_active_runtime_after_selecting_pack')}`",
            "",
            "## Parameter-Pack Diff",
            "",
            "```diff",
            report.get("parameter_pack_diff") or "",
            "```",
            "",
            "*This plan is advisory by default. It writes a parameter-pack file only when `--apply` is explicitly supplied, and it never changes active runtime pack selection or execution.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "latest_json_path": output / THRESHOLD_ADOPTION_PLAN_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_ADOPTION_PLAN_MARKDOWN_FILENAME,
    }


def write_threshold_adoption_plan(
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    post_promotion_monitor_report: dict[str, Any] | None = None,
    post_promotion_monitor_report_path: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    parameter_pack: dict[str, Any] | None = None,
    parameter_pack_path: str | Path | None = None,
    target_parameter_pack: dict[str, Any] | None = None,
    target_parameter_pack_path: str | Path | None = None,
    candidate_key: str | None = None,
    apply_changes: bool = False,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write an advisory threshold adoption plan."""
    report = build_threshold_adoption_plan(
        promotion_package_report=promotion_package_report,
        promotion_package_report_path=promotion_package_report_path,
        ledger_path=ledger_path,
        post_promotion_monitor_report=post_promotion_monitor_report,
        post_promotion_monitor_report_path=post_promotion_monitor_report_path,
        runtime_config=runtime_config,
        parameter_pack=parameter_pack,
        parameter_pack_path=parameter_pack_path,
        target_parameter_pack=target_parameter_pack,
        target_parameter_pack_path=target_parameter_pack_path,
        candidate_key=candidate_key,
        apply_changes=apply_changes,
    )
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_ADOPTION_PLAN_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_adoption_plan"
    paths = _artifact_paths(output, stem)
    markdown = render_threshold_adoption_plan_markdown(report)
    _atomic_write_text(paths["json_path"], _json_text(report))
    _atomic_write_text(paths["markdown_path"], markdown)
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], _json_text(report))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact
