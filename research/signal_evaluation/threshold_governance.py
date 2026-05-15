"""Governance workflow for threshold replay candidates.

This module classifies threshold replay output for human review.  It never
writes runtime configuration or changes signal generation behavior.
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

from research.signal_evaluation.threshold_replay import (
    build_threshold_replay_summary,
    run_fixed_threshold_walk_forward_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_GOVERNANCE_DIR = PROJECT_ROOT / "research" / "signal_evaluation" / "reports" / "threshold_governance"

THRESHOLD_GOVERNANCE_JSON_FILENAME = "latest_threshold_governance.json"
THRESHOLD_GOVERNANCE_MARKDOWN_FILENAME = "latest_threshold_governance.md"
THRESHOLD_GOVERNANCE_CANDIDATES_FILENAME = "latest_threshold_governance_candidates.csv"
THRESHOLD_GOVERNANCE_REVIEW_LEDGER_FILENAME = "threshold_governance_review_ledger.csv"

PROMOTE_TO_REVIEW = "PROMOTE_TO_REVIEW"
WATCHLIST = "WATCHLIST"
REJECT_INSUFFICIENT_EVIDENCE = "REJECT_INSUFFICIENT_EVIDENCE"
REJECT_UNSTABLE = "REJECT_UNSTABLE"

REVIEW_ACTIONS = {
    "ACKNOWLEDGED",
    "DEFERRED",
    "REJECTED",
    "PROMOTED_FOR_REVIEW",
    "APPROVED_FOR_POLICY_EXPERIMENT",
}

DEFAULT_GOVERNANCE_POLICY: dict[str, Any] = {
    "min_candidate_labels": 30,
    "min_candidate_signal_dates": 3,
    "min_walk_forward_splits": 3,
    "min_positive_holdout_rate": 0.60,
    "min_avg_holdout_return_bps": 0.0,
    "max_worst_holdout_return_bps": -30.0,
    "max_drawdown_bps": -150.0,
    "adaptive_walk_forward_split_count": 4,
    "top_n_candidates": 10,
}

CONFIG_HINTS = {
    "composite_signal_score": "evaluation_thresholds.selection.composite_signal_score_floor",
    "tradeability_score": "evaluation_thresholds.selection.tradeability_score_floor",
    "trade_strength": "evaluation_thresholds.selection.trade_strength_floor",
    "move_probability": "evaluation_thresholds.selection.move_probability_floor",
    "hybrid_move_probability": "research_only.hybrid_move_probability_floor",
    "ml_rank_score": "research_only.ml_rank_score_floor",
    "ml_confidence_score": "research_only.ml_confidence_score_floor",
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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


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


def _candidate_key(candidate: dict[str, Any]) -> str:
    field = candidate.get("threshold_field") or "unknown"
    value = candidate.get("threshold_value")
    return f"{field}>={value}" if value is not None else str(field)


def _next_action(status: str) -> str:
    if status == PROMOTE_TO_REVIEW:
        return "Run the threshold policy experiment sandbox, then open human review before any policy-pack change."
    if status == WATCHLIST:
        return "Keep collecting labels and review again after additional walk-forward evidence."
    if status == REJECT_UNSTABLE:
        return "Reject for now; revisit only after threshold replay becomes stable out of sample."
    return "Reject for now; evidence does not meet minimum governance thresholds."


def classify_threshold_candidate(
    candidate: dict[str, Any] | None,
    *,
    walk_forward_summary: dict[str, Any] | None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one threshold candidate for human review."""
    rules = {**DEFAULT_GOVERNANCE_POLICY, **(policy or {})}
    reasons: list[str] = []
    candidate = candidate or {}
    walk_forward_summary = walk_forward_summary or {}
    label_count = _safe_int(candidate.get("label_count_60m"))
    signal_dates = _safe_int(candidate.get("distinct_signal_dates"))
    objective = _safe_float(candidate.get("objective_score"))
    sample_quality = str(candidate.get("sample_quality") or "NO_EVIDENCE")
    stability = str(candidate.get("stability_status") or "UNKNOWN")
    max_drawdown = _safe_float(candidate.get("max_drawdown_bps"))

    if not candidate:
        status = REJECT_INSUFFICIENT_EVIDENCE
        reasons.append("No replay candidate is available.")
    elif objective is None or label_count < int(rules["min_candidate_labels"]) or sample_quality in {
        "NO_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
    }:
        status = REJECT_INSUFFICIENT_EVIDENCE
        reasons.append(
            f"Candidate labels {label_count} below minimum {int(rules['min_candidate_labels'])} or objective unavailable."
        )
    elif signal_dates < int(rules["min_candidate_signal_dates"]):
        status = REJECT_INSUFFICIENT_EVIDENCE
        reasons.append(
            f"Candidate appears on only {signal_dates} signal date(s); minimum is {int(rules['min_candidate_signal_dates'])}."
        )
    elif stability == "HOLDOUT_DECAY":
        status = REJECT_UNSTABLE
        reasons.append("Candidate decayed in its internal chronological holdout.")
    elif max_drawdown is not None and max_drawdown < float(rules["max_drawdown_bps"]):
        status = REJECT_UNSTABLE
        reasons.append(
            f"Candidate drawdown {max_drawdown} bps breaches ceiling {float(rules['max_drawdown_bps'])} bps."
        )
    else:
        evaluated_splits = _safe_int(walk_forward_summary.get("evaluated_split_count"))
        robustness = str(walk_forward_summary.get("robustness_status") or "INSUFFICIENT_HISTORY")
        positive_rate = _safe_float(walk_forward_summary.get("positive_holdout_rate"))
        avg_holdout = _safe_float(walk_forward_summary.get("avg_holdout_return_60m_bps"))
        worst_holdout = _safe_float(walk_forward_summary.get("worst_holdout_return_60m_bps"))

        if evaluated_splits < int(rules["min_walk_forward_splits"]):
            status = WATCHLIST
            reasons.append(
                f"Only {evaluated_splits} walk-forward split(s) evaluated; minimum is {int(rules['min_walk_forward_splits'])}."
            )
        elif robustness == "ROBUST" and (
            positive_rate is not None
            and positive_rate >= float(rules["min_positive_holdout_rate"])
            and avg_holdout is not None
            and avg_holdout >= float(rules["min_avg_holdout_return_bps"])
            and (worst_holdout is None or worst_holdout >= float(rules["max_worst_holdout_return_bps"]))
        ):
            status = PROMOTE_TO_REVIEW
            reasons.append("Walk-forward validation meets promotion-to-review thresholds.")
        elif robustness in {"UNSTABLE", "INSUFFICIENT_HOLDOUT"}:
            status = REJECT_UNSTABLE if robustness == "UNSTABLE" else REJECT_INSUFFICIENT_EVIDENCE
            reasons.append(f"Walk-forward robustness status is {robustness}.")
        elif positive_rate is not None and positive_rate < float(rules["min_positive_holdout_rate"]):
            status = REJECT_UNSTABLE
            reasons.append(
                f"Positive holdout rate {positive_rate} below minimum {float(rules['min_positive_holdout_rate'])}."
            )
        elif avg_holdout is not None and avg_holdout < float(rules["min_avg_holdout_return_bps"]):
            status = REJECT_UNSTABLE
            reasons.append(
                f"Average holdout return {avg_holdout} bps below minimum {float(rules['min_avg_holdout_return_bps'])} bps."
            )
        else:
            status = WATCHLIST
            reasons.append("Evidence is promising but not strong enough for promotion review.")

    threshold_field = candidate.get("threshold_field")
    threshold_value = candidate.get("threshold_value")
    return {
        "candidate_key": _candidate_key(candidate),
        "governance_status": status,
        "reasons": reasons,
        "recommended_next_action": _next_action(status),
        "threshold_field": threshold_field,
        "threshold_value": threshold_value,
        "config_hint": CONFIG_HINTS.get(str(threshold_field), "research_only.unmapped_threshold"),
        "candidate": candidate,
        "candidate_walk_forward_summary": walk_forward_summary,
        "requires_manual_review": status in {PROMOTE_TO_REVIEW, WATCHLIST},
        "runtime_config_changed": False,
    }


def _candidate_walk_forward_summary(
    frame: pd.DataFrame | None,
    candidate: dict[str, Any],
    *,
    policy: dict[str, Any],
    fallback_summary: dict[str, Any],
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return fallback_summary
    threshold_field = candidate.get("threshold_field")
    if not threshold_field:
        return fallback_summary
    validation = run_fixed_threshold_walk_forward_validation(
        frame,
        threshold_field=str(threshold_field),
        threshold_value=candidate.get("threshold_value"),
        min_train_labels=int(policy["min_candidate_labels"]),
        min_holdout_labels=max(10, int(int(policy["min_candidate_labels"]) / 3)),
        adaptive_split_count=int(policy["adaptive_walk_forward_split_count"]),
    )
    summary = validation.get("summary", {}) or {}
    summary["validation_type"] = "fixed_candidate_threshold"
    return summary


def build_threshold_governance_report(
    frame: pd.DataFrame | None = None,
    *,
    threshold_summary: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a threshold governance report from replay diagnostics."""
    rules = {**DEFAULT_GOVERNANCE_POLICY, **(policy or {})}
    summary = threshold_summary or build_threshold_replay_summary(frame if frame is not None else pd.DataFrame())
    candidates = list(summary.get("threshold_replay_candidates", []) or [])
    top_n = int(rules["top_n_candidates"])
    walk_forward_summary = (summary.get("walk_forward_validation", {}) or {}).get("summary", {}) or {}
    reviews = []
    for candidate in candidates[:top_n]:
        candidate_walk_forward = _candidate_walk_forward_summary(
            frame,
            candidate,
            policy=rules,
            fallback_summary=walk_forward_summary,
        )
        reviews.append(
            classify_threshold_candidate(
                candidate,
                walk_forward_summary=candidate_walk_forward,
                policy=rules,
            )
        )

    if any(row["governance_status"] == PROMOTE_TO_REVIEW for row in reviews):
        overall_status = PROMOTE_TO_REVIEW
    elif any(row["governance_status"] == WATCHLIST for row in reviews):
        overall_status = WATCHLIST
    elif any(row["governance_status"] == REJECT_UNSTABLE for row in reviews):
        overall_status = REJECT_UNSTABLE
    else:
        overall_status = REJECT_INSUFFICIENT_EVIDENCE

    top_candidate_review = next(
        (row for row in reviews if row["governance_status"] == PROMOTE_TO_REVIEW),
        reviews[0] if reviews else classify_threshold_candidate(None, walk_forward_summary=walk_forward_summary, policy=rules),
    )
    report = {
        "report_type": "threshold_governance",
        "generated_at": _utc_now(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "overall_status": overall_status,
        "runtime_config_changed": False,
        "policy": rules,
        "threshold_replay_config": summary.get("config", {}),
        "walk_forward_summary": walk_forward_summary,
        "top_candidate_review": top_candidate_review,
        "candidate_reviews": reviews,
    }
    return _sanitize_value(report)


def render_threshold_governance_markdown(report: dict[str, Any]) -> str:
    """Render threshold governance as Markdown."""
    top = report.get("top_candidate_review", {}) or {}
    walk = report.get("walk_forward_summary", {}) or {}
    candidate_walk = top.get("candidate_walk_forward_summary", {}) or {}
    lines = [
        "# Threshold Governance Review",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Dataset path: {report.get('dataset_path') or 'unknown'}",
        f"- Overall status: **{report.get('overall_status')}**",
        f"- Runtime config changed: {report.get('runtime_config_changed')}",
        "",
        "## Top Candidate",
        "",
        f"- Candidate: {top.get('candidate_key')}",
        f"- Governance status: {top.get('governance_status')}",
        f"- Config hint: {top.get('config_hint')}",
        f"- Next action: {top.get('recommended_next_action')}",
        "",
        "### Reasons",
        "",
    ]
    for reason in top.get("reasons", []) or ["No reasons recorded."]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## Walk-Forward Summary",
            "",
            "### Selection Walk-Forward",
            "",
            f"- Robustness status: {walk.get('robustness_status')}",
            f"- Split strategy: {walk.get('split_strategy')}",
            f"- Evaluated splits: {walk.get('evaluated_split_count')} / {walk.get('split_count')}",
            f"- Positive holdout rate: {walk.get('positive_holdout_rate')}",
            f"- Avg holdout return 60m (bps): {walk.get('avg_holdout_return_60m_bps')}",
            f"- Worst holdout return 60m (bps): {walk.get('worst_holdout_return_60m_bps')}",
            "",
            "### Top Candidate Fixed-Threshold Walk-Forward",
            "",
            f"- Robustness status: {candidate_walk.get('robustness_status')}",
            f"- Split strategy: {candidate_walk.get('split_strategy')}",
            f"- Evaluated splits: {candidate_walk.get('evaluated_split_count')} / {candidate_walk.get('split_count')}",
            f"- Positive holdout rate: {candidate_walk.get('positive_holdout_rate')}",
            f"- Avg holdout return 60m (bps): {candidate_walk.get('avg_holdout_return_60m_bps')}",
            f"- Worst holdout return 60m (bps): {candidate_walk.get('worst_holdout_return_60m_bps')}",
            "",
            "## Candidate Review Queue",
            "",
            "| Candidate | Status | Labels | Dates | WF Splits | Objective | Config Hint | Next Action |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("candidate_reviews", []) or []:
        candidate = row.get("candidate", {}) or {}
        candidate_wf = row.get("candidate_walk_forward_summary", {}) or {}
        lines.append(
            f"| {row.get('candidate_key')} | {row.get('governance_status')} | "
            f"{candidate.get('label_count_60m')} | {candidate.get('distinct_signal_dates')} | "
            f"{candidate_wf.get('evaluated_split_count')} / {candidate_wf.get('split_count')} | "
            f"{candidate.get('objective_score')} | "
            f"{row.get('config_hint')} | {row.get('recommended_next_action')} |"
        )

    lines.extend(["", "*Human review is required before any threshold policy experiment. No runtime configuration was changed.*"])
    return "\n".join(lines)


def write_threshold_governance_report(
    frame: pd.DataFrame | None = None,
    *,
    threshold_summary: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
    write_latest: bool = True,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write threshold governance JSON, Markdown, and review CSV artifacts."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_THRESHOLD_GOVERNANCE_DIR
    output.mkdir(parents=True, exist_ok=True)
    stem = report_name or "threshold_governance"
    report = build_threshold_governance_report(
        frame,
        threshold_summary=threshold_summary,
        dataset_path=dataset_path,
        policy=policy,
    )
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    candidates_path = output / f"{stem}_candidates.csv"
    _atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
    _atomic_write_text(markdown_path, render_threshold_governance_markdown(report))
    candidates = pd.json_normalize(report.get("candidate_reviews", []) or [])
    _atomic_write_csv(candidates, candidates_path)

    latest_json_path = output / THRESHOLD_GOVERNANCE_JSON_FILENAME
    latest_markdown_path = output / THRESHOLD_GOVERNANCE_MARKDOWN_FILENAME
    latest_candidates_path = output / THRESHOLD_GOVERNANCE_CANDIDATES_FILENAME
    if write_latest:
        _atomic_write_text(latest_json_path, json.dumps(report, indent=2, sort_keys=True, default=str))
        _atomic_write_text(latest_markdown_path, render_threshold_governance_markdown(report))
        _atomic_write_csv(candidates, latest_candidates_path)

    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "candidates_csv_path": str(candidates_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
        "latest_candidates_csv_path": str(latest_candidates_path),
        "review_ledger_path": str(output / THRESHOLD_GOVERNANCE_REVIEW_LEDGER_FILENAME),
    }


def record_threshold_governance_review(
    *,
    report_json_path: str | Path,
    review_action: str,
    reviewer: str,
    review_note: str = "",
    ledger_path: str | Path | None = None,
    next_review_at: str | None = None,
) -> dict[str, Any]:
    """Append a human review action for a threshold governance report."""
    action = str(review_action).upper().strip()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"review_action must be one of {sorted(REVIEW_ACTIONS)}, got {review_action!r}")
    report_path = Path(report_json_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    top = report.get("top_candidate_review", {}) or {}
    ledger = Path(ledger_path) if ledger_path is not None else report_path.parent / THRESHOLD_GOVERNANCE_REVIEW_LEDGER_FILENAME
    row = {
        "reviewed_at": _utc_now(),
        "report_json": str(report_path),
        "governance_status": report.get("overall_status"),
        "candidate_key": top.get("candidate_key"),
        "threshold_field": top.get("threshold_field"),
        "threshold_value": top.get("threshold_value"),
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
