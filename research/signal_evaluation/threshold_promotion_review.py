"""Manual promotion package and review ledger for threshold shadow evidence.

This module packages a PROMOTION_READY shadow review for human decision-making.
It never writes runtime thresholds, parameter packs, or execution settings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import pandas as pd

from research.signal_evaluation.threshold_shadow_review import (
    DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR,
    PROMOTION_READY,
    THRESHOLD_SHADOW_REVIEW_JSON_FILENAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR = (
    PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_promotion_review"
)
DEFAULT_SHADOW_REVIEW_REPORT_PATH = DEFAULT_THRESHOLD_SHADOW_REVIEW_DIR / THRESHOLD_SHADOW_REVIEW_JSON_FILENAME

THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME = "latest_threshold_promotion_review.json"
THRESHOLD_PROMOTION_REVIEW_MARKDOWN_FILENAME = "latest_threshold_promotion_review.md"
THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME = "threshold_promotion_review_ledger.csv"

PROMOTION_REVIEW_READY = "PROMOTION_REVIEW_READY"
SKIPPED_SHADOW_REVIEW_NOT_READY = "SKIPPED_SHADOW_REVIEW_NOT_READY"

REVIEW_ACTIONS = {
    "APPROVED",
    "REJECTED",
    "DEFERRED",
}


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


def _promotion_candidate(shadow_review: dict[str, Any]) -> dict[str, Any]:
    pack = shadow_review.get("candidate_policy_pack", {}) or {}
    rule = shadow_review.get("threshold_rule", {}) or {}
    return {
        "candidate_name": pack.get("name"),
        "source_candidate_key": pack.get("source_candidate_key"),
        "governance_status": pack.get("source_governance_status"),
        "config_hint": pack.get("config_hint"),
        "threshold_rule": rule,
        "overrides": pack.get("overrides", {}),
        "research_only": pack.get("research_only"),
        "runtime_config_changed": False,
    }


def _risk_flags(shadow_review: dict[str, Any]) -> dict[str, Any]:
    impact = shadow_review.get("impact_summary", {}) or {}
    guardrails = shadow_review.get("guardrail_summary", {}) or {}
    segment_failures = list(shadow_review.get("segment_failures", []) or [])
    return {
        "true_positive_lost_count": _safe_int(impact.get("true_positive_lost_count")),
        "true_positive_loss_rate": guardrails.get("true_positive_loss_rate"),
        "false_positive_removed_count": _safe_int(guardrails.get("false_positive_removed_count")),
        "false_positive_removal_rate": guardrails.get("false_positive_removal_rate"),
        "bad_regime_count": _safe_int(guardrails.get("bad_regime_count")),
        "bad_bucket_count": _safe_int(guardrails.get("bad_bucket_count")),
        "segment_failure_count": int(len(segment_failures)),
        "has_segment_failures": bool(segment_failures),
    }


def _monitoring_plan(shadow_review: dict[str, Any]) -> list[str]:
    rule = shadow_review.get("threshold_rule", {}) or {}
    field = rule.get("field") or "candidate threshold field"
    return [
        f"If manually adopted later, track retained/suppressed signal counts for `{field}` daily.",
        "Compare realized hit rate, signed return, true-positive loss, and false-positive removal against this package.",
        "Re-run the automated shadow-mode workflow after each new trading day until post-adoption evidence is stable.",
        "Escalate review if retained avg-return delta turns negative, true positives are repeatedly lost, or a regime bucket deteriorates.",
    ]


def _rollback_notes(shadow_review: dict[str, Any]) -> list[str]:
    candidate = _promotion_candidate(shadow_review)
    return [
        "This package did not change runtime configuration, so no rollback is required from package generation.",
        "If a human later applies this threshold manually, retain the previous parameter pack/config as the rollback target.",
        f"Candidate config hint for manual review: `{candidate.get('config_hint') or 'none'}`.",
        "Rollback trigger: post-adoption evidence violates the same shadow review guardrails or operator confidence degrades.",
    ]


def build_threshold_promotion_review_package(
    shadow_review_report: dict[str, Any] | None,
    *,
    shadow_review_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a human-readable manual promotion package from shadow review evidence."""
    shadow_review = shadow_review_report or {}
    shadow_status = shadow_review.get("shadow_status")
    review_status = shadow_review.get("review_status")
    candidate = _promotion_candidate(shadow_review)
    if review_status != PROMOTION_READY:
        status = SKIPPED_SHADOW_REVIEW_NOT_READY
        reasons = [
            f"Shadow review status is {review_status or 'UNKNOWN'}; manual promotion package is not ready.",
        ]
        manual_review_required = False
        next_action = "Keep running automated shadow mode until the shadow review reaches PROMOTION_READY."
    else:
        status = PROMOTION_REVIEW_READY
        reasons = [
            "Shadow review is PROMOTION_READY; package is ready for human approve/reject/defer decision.",
            "This package is advisory and does not apply the candidate threshold automatically.",
        ]
        manual_review_required = True
        next_action = "Open the promotion review ledger and record APPROVED, REJECTED, or DEFERRED."

    report = {
        "report_type": "threshold_promotion_review",
        "generated_at": _utc_now(),
        "shadow_review_report_path": str(shadow_review_report_path) if shadow_review_report_path is not None else None,
        "dataset_path": shadow_review.get("dataset_path"),
        "promotion_review_status": status,
        "promotion_review_reasons": reasons,
        "recommended_next_action": next_action,
        "manual_review_required": manual_review_required,
        "runtime_config_changed": False,
        "review_decision": None,
        "status_chain": {
            "governance_status": candidate.get("governance_status"),
            "policy_experiment_status": shadow_review.get("policy_experiment_status"),
            "shadow_simulation_status": shadow_status,
            "shadow_review_status": review_status,
        },
        "promotion_candidate": candidate,
        "observation_summary": shadow_review.get("observation_summary", {}),
        "impact_summary": shadow_review.get("impact_summary", {}),
        "guardrail_summary": shadow_review.get("guardrail_summary", {}),
        "baseline_metrics": shadow_review.get("baseline_metrics", {}),
        "retained_metrics": shadow_review.get("retained_metrics", {}),
        "suppressed_metrics": shadow_review.get("suppressed_metrics", {}),
        "retained_vs_baseline_delta": shadow_review.get("retained_vs_baseline_delta", {}),
        "segment_failures": shadow_review.get("segment_failures", []),
        "risk_flags": _risk_flags(shadow_review),
        "recommended_human_actions": [
            "APPROVED: user accepts the candidate for a later manual config/parameter-pack change.",
            "REJECTED: user rejects the candidate; keep collecting evidence before reconsidering.",
            "DEFERRED: user keeps the candidate in review while automated shadow mode continues.",
        ],
        "monitoring_plan": _monitoring_plan(shadow_review),
        "rollback_notes": _rollback_notes(shadow_review),
    }
    return _sanitize_value(report)


def render_threshold_promotion_review_markdown(report: dict[str, Any]) -> str:
    """Render a promotion review package as Markdown."""
    candidate = report.get("promotion_candidate", {}) or {}
    rule = candidate.get("threshold_rule", {}) or {}
    status_chain = report.get("status_chain", {}) or {}
    impact = report.get("impact_summary", {}) or {}
    guardrails = report.get("guardrail_summary", {}) or {}
    risk = report.get("risk_flags", {}) or {}
    delta = report.get("retained_vs_baseline_delta", {}) or {}
    lines = [
        "# Threshold Promotion Review Package",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Shadow review: {report.get('shadow_review_report_path') or 'not supplied'}",
        f"- Promotion review status: **{report.get('promotion_review_status')}**",
        f"- Manual review required: {report.get('manual_review_required')}",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        f"- Next action: {report.get('recommended_next_action')}",
        "",
        "## Status Chain",
        "",
        f"- Governance status: {status_chain.get('governance_status')}",
        f"- Policy experiment status: {status_chain.get('policy_experiment_status')}",
        f"- Shadow simulation status: {status_chain.get('shadow_simulation_status')}",
        f"- Shadow review status: {status_chain.get('shadow_review_status')}",
        "",
        "## Candidate",
        "",
        f"- Candidate: `{candidate.get('source_candidate_key') or candidate.get('candidate_name') or 'none'}`",
        f"- Config hint: `{candidate.get('config_hint') or 'none'}`",
        f"- Rule: `{rule.get('field')} {rule.get('operator', '>=')} {rule.get('value')}`",
        f"- Research only: {candidate.get('research_only')}",
        "",
        "## Review Reasons",
        "",
    ]
    for reason in report.get("promotion_review_reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Eligible signals | {impact.get('eligible_signal_count')} |",
            f"| Retained signals | {impact.get('retained_signal_count')} |",
            f"| Suppressed signals | {impact.get('suppressed_signal_count')} |",
            f"| False positives removed | {risk.get('false_positive_removed_count')} |",
            f"| False-positive removal rate | {guardrails.get('false_positive_removal_rate')} |",
            f"| True positives lost | {risk.get('true_positive_lost_count')} |",
            f"| True-positive loss rate | {guardrails.get('true_positive_loss_rate')} |",
            f"| Retained avg-return delta (bps) | {delta.get('avg_return_delta_bps')} |",
            f"| Retained hit-rate delta | {delta.get('hit_rate_delta')} |",
            f"| Segment failures | {risk.get('segment_failure_count')} |",
            "",
            "## Human Actions",
            "",
        ]
    )
    for action in report.get("recommended_human_actions", []) or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Monitoring Plan", ""])
    for item in report.get("monitoring_plan", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Rollback Notes", ""])
    for item in report.get("rollback_notes", []) or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "*This package is advisory. Approval records user intent only; it does not alter runtime configuration or execution.*",
        ]
    )
    return "\n".join(lines)


def _artifact_paths(output: Path, stem: str) -> dict[str, Path]:
    return {
        "json_path": output / f"{stem}.json",
        "markdown_path": output / f"{stem}.md",
        "latest_json_path": output / THRESHOLD_PROMOTION_REVIEW_JSON_FILENAME,
        "latest_markdown_path": output / THRESHOLD_PROMOTION_REVIEW_MARKDOWN_FILENAME,
        "review_ledger_path": output / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME,
    }


def write_threshold_promotion_review_package(
    shadow_review_report: dict[str, Any] | None,
    *,
    shadow_review_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Build and write manual promotion review JSON/Markdown artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_PROMOTION_REVIEW_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_promotion_review"
    report = build_threshold_promotion_review_package(
        shadow_review_report,
        shadow_review_report_path=shadow_review_report_path,
    )
    markdown = render_threshold_promotion_review_markdown(report)
    paths = _artifact_paths(output, stem)
    _atomic_write_text(paths["json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(paths["markdown_path"], markdown)
    if write_latest:
        _atomic_write_text(paths["latest_json_path"], json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(paths["latest_markdown_path"], markdown)
    artifact = {"report": report}
    artifact.update({key: str(value) for key, value in paths.items()})
    return artifact


def record_threshold_promotion_review_decision(
    *,
    report_json_path: str | Path,
    review_action: str,
    reviewer: str,
    review_note: str = "",
    ledger_path: str | Path | None = None,
    next_review_at: str | None = None,
) -> dict[str, Any]:
    """Append a manual approve/reject/defer decision for a promotion package."""
    action = str(review_action).upper().strip()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"review_action must be one of {sorted(REVIEW_ACTIONS)}, got {review_action!r}")
    report_path = Path(report_json_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report.get("promotion_candidate", {}) or {}
    rule = candidate.get("threshold_rule", {}) or {}
    ledger = Path(ledger_path) if ledger_path is not None else report_path.parent / THRESHOLD_PROMOTION_REVIEW_LEDGER_FILENAME
    row = {
        "reviewed_at": _utc_now(),
        "report_json": str(report_path),
        "promotion_review_status": report.get("promotion_review_status"),
        "governance_status": (report.get("status_chain", {}) or {}).get("governance_status"),
        "policy_experiment_status": (report.get("status_chain", {}) or {}).get("policy_experiment_status"),
        "shadow_simulation_status": (report.get("status_chain", {}) or {}).get("shadow_simulation_status"),
        "shadow_review_status": (report.get("status_chain", {}) or {}).get("shadow_review_status"),
        "candidate_key": candidate.get("source_candidate_key"),
        "threshold_field": rule.get("field"),
        "threshold_value": rule.get("value"),
        "config_hint": candidate.get("config_hint"),
        "review_action": action,
        "reviewer": reviewer,
        "review_note": review_note,
        "next_review_at": next_review_at,
        "runtime_config_changed": False,
    }
    with _exclusive_file_lock(ledger):
        if ledger.exists():
            existing = pd.read_csv(ledger)
            updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            updated = pd.DataFrame([row])
        _atomic_write_csv(updated, ledger)
    return {
        "review_entry": _sanitize_value(row),
        "review_ledger_path": str(ledger),
    }
