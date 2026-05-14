"""
Signal drift alerting and review ledger workflow.

This module turns the trend dashboard into a compact ops alert and optionally
records human review actions.  It is research/ops-only and never changes
trading decisions or execution behavior.
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

from research.signal_evaluation.drift_monitor import TREND_DASHBOARD_JSON_FILENAME
from research.signal_evaluation.reporting import SIGNAL_EVALUATION_REPORTS_DIR


DRIFT_ALERT_JSON_FILENAME = "latest_signal_drift_alert.json"
DRIFT_ALERT_MARKDOWN_FILENAME = "latest_signal_drift_alert.md"
DRIFT_REVIEW_LEDGER_FILENAME = "signal_drift_review_ledger.csv"

REVIEW_ACTIONS = {"REVIEWED", "ACKNOWLEDGED", "DEFERRED", "RESOLVED"}
OPS_STATUSES = {"NO_HISTORY", "STABLE", "WATCH", "DETERIORATING"}


def default_drift_monitoring_dir() -> Path:
    return SIGNAL_EVALUATION_REPORTS_DIR / "drift_monitoring"


def default_trend_dashboard_path() -> Path:
    return default_drift_monitoring_dir() / TREND_DASHBOARD_JSON_FILENAME


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


def load_json_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_linked_drift_report(dashboard: dict[str, Any]) -> dict[str, Any]:
    latest = dashboard.get("latest", {}) or {}
    report_path = latest.get("report_json")
    if not report_path:
        return {}
    return load_json_file(report_path)


def classify_drift_ops_status(
    dashboard: dict[str, Any],
    *,
    latest_report: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Map trend evidence into a compact daily-ops status."""
    latest_report = latest_report or {}
    latest = dashboard.get("latest", {}) or {}
    assessment = str(dashboard.get("trend_assessment") or "").upper()
    monitor_status = str(latest.get("monitor_status") or latest_report.get("monitor_status") or "").upper()
    run_count = _safe_int(dashboard.get("run_count"), 0)
    warning_count = _safe_int(latest.get("warning_count"), len(latest_report.get("warnings", []) or []))
    reasons: list[str] = []

    if run_count <= 0 or assessment == "NO_HISTORY":
        return "NO_HISTORY", ["No trend history is available yet."]

    if assessment == "DETERIORATING":
        reasons.append("Trend dashboard assessment is DETERIORATING.")
    if monitor_status == "CAUTION":
        reasons.append("Latest drift monitor status is CAUTION.")
    if reasons:
        return "DETERIORATING", reasons

    if assessment == "WATCH":
        reasons.append("Trend dashboard assessment is WATCH.")
    if monitor_status == "WATCH":
        reasons.append("Latest drift monitor status is WATCH.")
    if warning_count > 0:
        reasons.append(f"Latest drift run has {warning_count} warning(s).")
    if reasons:
        return "WATCH", reasons

    return "STABLE", ["No active drift alert conditions triggered."]


def _warning_digest(latest_report: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (latest_report.get("warnings", []) or [])[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "severity": item.get("severity"),
                "category": item.get("category"),
                "message": item.get("message"),
                "value": item.get("value"),
                "threshold": item.get("threshold"),
            }
        )
    return rows


def _latest_review_entry(
    ledger_path: str | Path | None,
    *,
    report_json: str | None,
) -> dict[str, Any] | None:
    if ledger_path is None:
        return None
    path = Path(ledger_path)
    if not path.exists():
        return None
    try:
        ledger = pd.read_csv(path)
    except Exception:
        return None
    if ledger.empty:
        return None
    if report_json and "report_json" in ledger.columns:
        scoped = ledger.loc[ledger["report_json"].astype(str).eq(str(report_json))]
        if not scoped.empty:
            return _sanitize_value(scoped.iloc[-1].to_dict())
    return _sanitize_value(ledger.iloc[-1].to_dict())


def build_signal_drift_alert_summary(
    dashboard: dict[str, Any],
    *,
    trend_dashboard_path: str | Path | None = None,
    latest_report: dict[str, Any] | None = None,
    review_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact alert summary from the latest trend dashboard."""
    latest_report = latest_report if latest_report is not None else load_linked_drift_report(dashboard)
    latest = dashboard.get("latest", {}) or {}
    status, reasons = classify_drift_ops_status(dashboard, latest_report=latest_report)
    report_json = latest.get("report_json")
    summary = {
        "report_type": "signal_drift_alert_summary",
        "generated_at": _utc_now(),
        "ops_status": status,
        "trend_assessment": dashboard.get("trend_assessment"),
        "strict_failure": status == "DETERIORATING",
        "trend_dashboard_path": str(trend_dashboard_path) if trend_dashboard_path else None,
        "review_ledger_path": str(review_ledger_path) if review_ledger_path else None,
        "run_count": dashboard.get("run_count"),
        "lookback_runs": dashboard.get("lookback_runs"),
        "status_counts": dashboard.get("status_counts", {}),
        "lookback_summary": dashboard.get("lookback_summary", {}),
        "latest_run": {
            "generated_at": latest.get("generated_at"),
            "report_name": latest.get("report_name"),
            "monitor_status": latest.get("monitor_status"),
            "dataset_path": latest.get("dataset_path"),
            "recent_start": latest.get("recent_start"),
            "recent_end": latest.get("recent_end"),
            "baseline_start": latest.get("baseline_start"),
            "baseline_end": latest.get("baseline_end"),
            "recent_hit_rate_60m": latest.get("recent_hit_rate_60m"),
            "hit_rate_delta": latest.get("hit_rate_delta"),
            "avg_return_delta_bps": latest.get("avg_return_delta_bps"),
            "warning_count": latest.get("warning_count"),
            "caution_count": latest.get("caution_count"),
            "watch_count": latest.get("watch_count"),
            "report_json": report_json,
            "report_markdown": latest.get("report_markdown"),
        },
        "alert_reasons": reasons,
        "warning_digest": _warning_digest(latest_report),
        "latest_review": _latest_review_entry(review_ledger_path, report_json=report_json),
        "operator_message": _operator_message(status, reasons),
    }
    return _sanitize_value(summary)


def _operator_message(status: str, reasons: list[str]) -> str:
    if status == "DETERIORATING":
        return "Review signal drift before relying on new signals; this is an alert only and does not change execution."
    if status == "WATCH":
        return "Monitor the next drift run and review warning categories if they persist."
    if status == "NO_HISTORY":
        return "Run the drift monitor enough times to build trend history."
    return "No drift review action is required from the latest trend dashboard."


def render_signal_drift_alert_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown alert view."""
    latest = summary.get("latest_run", {}) or {}
    review = summary.get("latest_review") or {}
    lines = [
        "# Signal Drift Alert Summary",
        "",
        f"- Generated at: {summary.get('generated_at')}",
        f"- Ops status: **{summary.get('ops_status')}**",
        f"- Trend assessment: {summary.get('trend_assessment')}",
        f"- Runs tracked: {summary.get('run_count')}",
        f"- Trend dashboard: `{summary.get('trend_dashboard_path')}`",
        "",
        "## Latest Run",
        "",
        f"- Report name: {latest.get('report_name')}",
        f"- Monitor status: {latest.get('monitor_status')}",
        f"- Recent hit rate 60m: {latest.get('recent_hit_rate_60m')}",
        f"- Hit-rate delta: {latest.get('hit_rate_delta')}",
        f"- Avg return delta (bps): {latest.get('avg_return_delta_bps')}",
        f"- Warning count: {latest.get('warning_count')}",
        "",
        "## Alert Reasons",
        "",
    ]
    for reason in summary.get("alert_reasons", []) or []:
        lines.append(f"- {reason}")

    lines.extend(["", "## Warning Digest", ""])
    warnings = summary.get("warning_digest", []) or []
    if warnings:
        for item in warnings:
            lines.append(f"- **{item.get('severity')}** `{item.get('category')}` - {item.get('message')}")
    else:
        lines.append("- None")

    lines.extend(["", "## Latest Review", ""])
    if review:
        lines.extend(
            [
                f"- Action: {review.get('review_action')}",
                f"- Reviewer: {review.get('reviewer') or 'unknown'}",
                f"- Reviewed at: {review.get('reviewed_at')}",
                f"- Note: {review.get('review_note') or ''}",
            ]
        )
    else:
        lines.append("- No review recorded for the latest run.")
    lines.extend(["", "*Research/ops alert only. No execution behavior is changed.*"])
    return "\n".join(lines)


def write_signal_drift_alert_summary(
    *,
    trend_dashboard_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    review_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read latest trend dashboard and write alert JSON/Markdown artifacts."""
    dashboard_path = Path(trend_dashboard_path) if trend_dashboard_path is not None else default_trend_dashboard_path()
    dashboard = load_json_file(dashboard_path)
    output = Path(output_dir) if output_dir is not None else dashboard_path.parent
    ledger_path = Path(review_ledger_path) if review_ledger_path is not None else output / DRIFT_REVIEW_LEDGER_FILENAME
    summary = build_signal_drift_alert_summary(
        dashboard,
        trend_dashboard_path=dashboard_path,
        latest_report=load_linked_drift_report(dashboard),
        review_ledger_path=ledger_path,
    )
    json_path = output / DRIFT_ALERT_JSON_FILENAME
    markdown_path = output / DRIFT_ALERT_MARKDOWN_FILENAME
    _atomic_write_text(json_path, json.dumps(summary, indent=2, default=str))
    _atomic_write_text(markdown_path, render_signal_drift_alert_markdown(summary))
    return {
        "alert_json_path": str(json_path),
        "alert_markdown_path": str(markdown_path),
        "review_ledger_path": str(ledger_path),
        "alert_summary": summary,
    }


def build_signal_drift_review_row(
    summary: dict[str, Any],
    *,
    review_action: str,
    reviewer: str | None = None,
    review_note: str | None = None,
    next_review_at: str | None = None,
) -> dict[str, Any]:
    action = str(review_action or "").upper().strip()
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"review_action must be one of: {', '.join(sorted(REVIEW_ACTIONS))}")
    latest = summary.get("latest_run", {}) or {}
    return _sanitize_value(
        {
            "reviewed_at": _utc_now(),
            "review_action": action,
            "reviewer": reviewer,
            "review_note": review_note,
            "next_review_at": next_review_at,
            "ops_status": summary.get("ops_status"),
            "trend_assessment": summary.get("trend_assessment"),
            "monitor_status": latest.get("monitor_status"),
            "report_name": latest.get("report_name"),
            "report_json": latest.get("report_json"),
            "trend_dashboard_path": summary.get("trend_dashboard_path"),
            "warning_count": latest.get("warning_count"),
            "alert_reasons": " | ".join(summary.get("alert_reasons", []) or []),
        }
    )


def append_signal_drift_review(
    summary: dict[str, Any],
    ledger_path: str | Path,
    *,
    review_action: str,
    reviewer: str | None = None,
    review_note: str | None = None,
    next_review_at: str | None = None,
) -> dict[str, Any]:
    """Append a review row to the drift alert ledger."""
    path = Path(ledger_path)
    row = build_signal_drift_review_row(
        summary,
        review_action=review_action,
        reviewer=reviewer,
        review_note=review_note,
        next_review_at=next_review_at,
    )
    incoming = pd.DataFrame([row])
    with _exclusive_file_lock(path):
        if path.exists():
            try:
                existing = pd.read_csv(path)
            except Exception:
                existing = pd.DataFrame()
            ledger = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            ledger = incoming
        _atomic_write_csv(ledger, path)
    return {
        "review_ledger_path": str(path),
        "review_row": row,
    }


def run_signal_drift_alert_workflow(
    *,
    trend_dashboard_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    review_ledger_path: str | Path | None = None,
    review_action: str | None = None,
    reviewer: str | None = None,
    review_note: str | None = None,
    next_review_at: str | None = None,
) -> dict[str, Any]:
    """Write alert artifacts and optionally append a review ledger row."""
    dashboard_path = Path(trend_dashboard_path) if trend_dashboard_path is not None else default_trend_dashboard_path()
    output = Path(output_dir) if output_dir is not None else dashboard_path.parent
    ledger_path = Path(review_ledger_path) if review_ledger_path is not None else output / DRIFT_REVIEW_LEDGER_FILENAME

    dashboard = load_json_file(dashboard_path)
    summary = build_signal_drift_alert_summary(
        dashboard,
        trend_dashboard_path=dashboard_path,
        latest_report=load_linked_drift_report(dashboard),
        review_ledger_path=ledger_path,
    )
    review_artifact: dict[str, Any] | None = None
    if review_action:
        review_artifact = append_signal_drift_review(
            summary,
            ledger_path,
            review_action=review_action,
            reviewer=reviewer,
            review_note=review_note,
            next_review_at=next_review_at,
        )

    artifact = write_signal_drift_alert_summary(
        trend_dashboard_path=dashboard_path,
        output_dir=output,
        review_ledger_path=ledger_path,
    )
    artifact["review_artifact"] = review_artifact
    return artifact
