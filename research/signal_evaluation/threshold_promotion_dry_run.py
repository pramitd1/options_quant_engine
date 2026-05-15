"""Promotion dry-run workflow for real threshold artifacts.

This module rehearses an APPROVED threshold promotion by writing a sandbox
approval ledger, then running post-promotion monitoring and adoption
reconciliation against real artifacts. It never writes to the real promotion
ledger and never changes runtime configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from research.signal_evaluation.threshold_adoption_reconciliation import (
    write_threshold_adoption_reconciliation_report,
)
from research.signal_evaluation.threshold_post_promotion_monitor import (
    write_threshold_post_promotion_monitor_report,
)
from research.signal_evaluation.threshold_promotion_review import (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR,
    PROMOTION_REVIEW_READY,
    THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME,
    THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMOTION_REVIEW_REPORT_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME
)
DEFAULT_PROMOTION_REVIEW_LEDGER_PATH = (
    DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME
)
DEFAULT_THRESHOLD_PROMOTION_DRY_RUN_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_promotion_dry_runs"
)

THRESHOLD_PROMOTION_DRY_RUN_JSON_FILENAME = "latest_threshold_promotion_dry_run.json"
THRESHOLD_PROMOTION_DRY_RUN_MARKDOWN_FILENAME = "latest_threshold_promotion_dry_run.md"
THRESHOLD_PROMOTION_DRY_RUN_LEDGER_FILENAME = "latest_threshold_promotion_dry_run_sandbox_ledger.csv"

PROMOTION_DRY_RUN_COMPLETE = "PROMOTION_DRY_RUN_COMPLETE"
PROMOTION_DRY_RUN_SKIPPED_PACKAGE_NOT_READY = "PROMOTION_DRY_RUN_SKIPPED_PACKAGE_NOT_READY"


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


def _approval_timestamp(frame: pd.DataFrame, explicit_timestamp: str | None = None) -> tuple[str, str]:
    if explicit_timestamp:
        parsed = pd.to_datetime(explicit_timestamp, errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed.isoformat(), "explicit_approval_timestamp"
    if "signal_timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["signal_timestamp"], errors="coerce", utc=True).dropna()
        if not timestamps.empty:
            approval = timestamps.min() - pd.Timedelta(seconds=1)
            return approval.isoformat(), "before_first_signal_timestamp"
    return _utc_now(), "utc_now_no_signal_timestamp"


def _approval_decision(
    promotion_package: dict[str, Any],
    *,
    promotion_package_report_path: str | Path | None,
    reviewed_at: str,
    reviewer: str,
    review_note: str,
) -> dict[str, Any]:
    candidate = promotion_package.get("promotion_candidate", {}) or {}
    rule = candidate.get("threshold_rule", {}) or {}
    return _sanitize_value(
        {
            "reviewed_at": reviewed_at,
            "report_json": str(promotion_package_report_path) if promotion_package_report_path is not None else None,
            "promotion_review_status": promotion_package.get("promotion_review_status"),
            "governance_status": (promotion_package.get("status_chain", {}) or {}).get("governance_status"),
            "policy_experiment_status": (promotion_package.get("status_chain", {}) or {}).get("policy_experiment_status"),
            "shadow_simulation_status": (promotion_package.get("status_chain", {}) or {}).get("shadow_simulation_status"),
            "shadow_review_status": (promotion_package.get("status_chain", {}) or {}).get("shadow_review_status"),
            "candidate_key": candidate.get("source_candidate_key"),
            "threshold_field": rule.get("field"),
            "threshold_value": rule.get("value"),
            "config_hint": candidate.get("config_hint"),
            "review_action": "APPROVED",
            "reviewer": reviewer,
            "review_note": review_note,
            "next_review_at": None,
            "runtime_config_changed": False,
        }
    )


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "sandbox_ledger_path": output / f"{stem}_sandbox_ledger.csv",
        "latest_json_path": output / THRESHOLD_PROMOTION_DRY_RUN_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_PROMOTION_DRY_RUN_MARKDOWN_FILENAME,
        "latest_sandbox_ledger_path": output / THRESHOLD_PROMOTION_DRY_RUN_LEDGER_FILENAME,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    monitor = report.get("post_promotion_monitor", {}) or {}
    reconciliation = report.get("adoption_reconciliation", {}) or {}
    approval = report.get("sandbox_approval_decision", {}) or {}
    lines = [
        "# Threshold Promotion Dry Run",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dry-run status: **{report.get('dry_run_status')}**",
        f"- Dataset: {report.get('dataset_path') or 'not supplied'}",
        f"- Promotion package: {report.get('promotion_package_report_path') or 'not supplied'}",
        f"- Real promotion ledger: {report.get('real_promotion_ledger_path') or 'not supplied'}",
        f"- Sandbox ledger: {report.get('sandbox_ledger_path') or 'not written'}",
        f"- Real ledger changed: {report.get('real_promotion_ledger_changed')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Simulated approval timestamp: {approval.get('reviewed_at')}",
        f"- Approval timestamp strategy: {report.get('approval_timestamp_strategy')}",
        f"- Post-promotion monitor status: {monitor.get('monitor_status')}",
        f"- Adoption reconciliation status: {reconciliation.get('adoption_status')}",
        f"- Next action: {report.get('recommended_next_action')}",
        "",
        "## Dry-Run Reasons",
        "",
    ]
    for reason in report.get("dry_run_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Post-promotion monitor JSON: `{(report.get('post_promotion_monitor_artifact', {}) or {}).get('latest_json_path')}`",
            f"- Adoption reconciliation JSON: `{(report.get('adoption_reconciliation_artifact', {}) or {}).get('latest_json_path')}`",
            "",
            "*This dry run uses a sandbox approval ledger only. It never records a real approval and never mutates runtime thresholds.*",
        ]
    )
    return "\n".join(lines)


def _skip_report(
    *,
    reason: str,
    dataset_path: str | Path | None,
    promotion_package_report_path: str | Path | None,
    real_ledger_path: str | Path | None,
    output_dir: str | Path | None,
    report_name: str | None,
    write_latest: bool,
) -> dict[str, Any]:
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_PROMOTION_DRY_RUN_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_promotion_dry_run"
    paths = _artifact_paths(output, stem)
    report = _sanitize_value(
        {
            "report_type": "threshold_promotion_dry_run",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "promotion_package_report_path": str(promotion_package_report_path) if promotion_package_report_path is not None else None,
            "real_promotion_ledger_path": str(real_ledger_path) if real_ledger_path is not None else None,
            "sandbox_ledger_path": None,
            "dry_run_status": PROMOTION_DRY_RUN_SKIPPED_PACKAGE_NOT_READY,
            "dry_run_reasons": [reason],
            "recommended_next_action": "Run shadow mode until a PROMOTION_REVIEW_READY package is available, then rerun this dry-run.",
            "runtime_config_changed": False,
            "real_promotion_ledger_changed": False,
            "sandbox_approval_decision": {},
            "approval_timestamp_strategy": None,
            "post_promotion_monitor": {},
            "adoption_reconciliation": {},
            "post_promotion_monitor_artifact": {},
            "adoption_reconciliation_artifact": {},
        }
    )
    markdown = _render_markdown(report)
    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def run_threshold_promotion_dry_run(
    frame: pd.DataFrame,
    *,
    promotion_package_report: dict[str, Any] | None = None,
    promotion_package_report_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    real_ledger_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    approval_timestamp: str | None = None,
    reviewer: str = "promotion-dry-run",
    review_note: str = "Dry-run APPROVED decision; sandbox ledger only.",
    candidate_key: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Run the promotion dry-run using a sandbox approval ledger."""
    package_path = promotion_package_report_path or DEFAULT_PROMOTION_REVIEW_REPORT_PATH
    real_ledger = real_ledger_path or DEFAULT_PROMOTION_REVIEW_LEDGER_PATH
    package = promotion_package_report or _load_json_if_exists(package_path)
    if package.get("promotion_review_status") != PROMOTION_REVIEW_READY:
        return _skip_report(
            reason=(
                "Promotion package is not ready for dry-run approval; "
                f"package status is {package.get('promotion_review_status') or 'UNKNOWN'}."
            ),
            dataset_path=dataset_path,
            promotion_package_report_path=package_path,
            real_ledger_path=real_ledger,
            output_dir=output_dir,
            report_name=report_name,
            write_latest=write_latest,
        )

    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_PROMOTION_DRY_RUN_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_promotion_dry_run"
    paths = _artifact_paths(output, stem)
    reviewed_at, timestamp_strategy = _approval_timestamp(frame, approval_timestamp)
    approval = _approval_decision(
        package,
        promotion_package_report_path=package_path,
        reviewed_at=reviewed_at,
        reviewer=reviewer,
        review_note=review_note,
    )
    sandbox_ledger = pd.DataFrame([approval])
    _atomic_write_csv(sandbox_ledger, paths["sandbox_ledger_path"])
    if write_latest:
        _atomic_write_csv(sandbox_ledger, paths["latest_sandbox_ledger_path"])

    monitor_artifact = write_threshold_post_promotion_monitor_report(
        frame,
        promotion_package_report=package,
        promotion_package_report_path=package_path,
        ledger_path=paths["sandbox_ledger_path"],
        approval_decision=approval,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        candidate_key=candidate_key,
        output_dir=output / "post_promotion_monitoring",
        report_name=f"{stem}_post_promotion_monitor",
        write_latest=write_latest,
    )
    monitor_report = monitor_artifact.get("report", {}) or {}
    reconciliation_artifact = write_threshold_adoption_reconciliation_report(
        promotion_package_report=package,
        promotion_package_report_path=package_path,
        ledger_path=paths["sandbox_ledger_path"],
        post_promotion_monitor_report=monitor_report,
        post_promotion_monitor_report_path=monitor_artifact.get("latest_json_path") or monitor_artifact.get("json_path"),
        candidate_key=candidate_key,
        output_dir=output / "adoption_reconciliation",
        report_name=f"{stem}_adoption_reconciliation",
        write_latest=write_latest,
    )
    reconciliation_report = reconciliation_artifact.get("report", {}) or {}
    reasons = [
        "Created an APPROVED decision in a sandbox ledger only.",
        "Ran post-promotion monitoring against the sandbox approval and real signal dataset.",
        "Ran adoption reconciliation against the sandbox approval and active runtime policy.",
    ]
    report = _sanitize_value(
        {
            "report_type": "threshold_promotion_dry_run",
            "generated_at": _utc_now(),
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "promotion_package_report_path": str(package_path) if package_path is not None else None,
            "real_promotion_ledger_path": str(real_ledger) if real_ledger is not None else None,
            "sandbox_ledger_path": str(paths["sandbox_ledger_path"]),
            "latest_sandbox_ledger_path": str(paths["latest_sandbox_ledger_path"]),
            "dry_run_status": PROMOTION_DRY_RUN_COMPLETE,
            "dry_run_reasons": reasons,
            "recommended_next_action": "Review the dry-run monitor and reconciliation artifacts before recording any real ledger decision.",
            "runtime_config_changed": False,
            "real_promotion_ledger_changed": False,
            "sandbox_approval_decision": approval,
            "approval_timestamp_strategy": timestamp_strategy,
            "post_promotion_monitor": {
                "monitor_status": monitor_report.get("monitor_status"),
                "recommended_next_action": monitor_report.get("recommended_next_action"),
                "monitor_reasons": monitor_report.get("monitor_reasons", []),
            },
            "adoption_reconciliation": {
                "adoption_status": reconciliation_report.get("adoption_status"),
                "recommended_next_action": reconciliation_report.get("recommended_next_action"),
                "adoption_reasons": reconciliation_report.get("adoption_reasons", []),
            },
            "post_promotion_monitor_artifact": {key: value for key, value in monitor_artifact.items() if key != "report"},
            "adoption_reconciliation_artifact": {key: value for key, value in reconciliation_artifact.items() if key != "report"},
        }
    )
    markdown = _render_markdown(report)
    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact
