"""
Optional notification transport for signal drift alerts.

The notifier reads the drift alert artifact and can deliver a compact message
via webhook and/or local command.  It records every attempt in a delivery ledger.
It is research/ops-only and never changes trading decisions or execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import pandas as pd

from research.signal_evaluation.drift_alerts import (
    DRIFT_ALERT_JSON_FILENAME,
    default_drift_monitoring_dir,
    load_json_file,
)


DRIFT_ALERT_DELIVERY_LEDGER_FILENAME = "signal_drift_alert_delivery_ledger.csv"
WEBHOOK_ENV_VAR = "SIGNAL_DRIFT_ALERT_WEBHOOK_URL"
NOTIFY_COMMAND_ENV_VAR = "SIGNAL_DRIFT_ALERT_NOTIFY_COMMAND"

SUCCESS_STATUSES = {"SENT", "DRY_RUN", "SKIPPED_NOT_TRIGGERED"}


def default_alert_json_path() -> Path:
    return default_drift_monitoring_dir() / DRIFT_ALERT_JSON_FILENAME


def default_delivery_ledger_path() -> Path:
    return default_drift_monitoring_dir() / DRIFT_ALERT_DELIVERY_LEDGER_FILENAME


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


def should_notify_signal_drift_alert(summary: dict[str, Any], *, include_watch: bool = False) -> bool:
    status = str(summary.get("ops_status") or "").upper()
    if status == "DETERIORATING":
        return True
    return bool(include_watch and status == "WATCH")


def build_signal_drift_notification_payload(
    summary: dict[str, Any],
    *,
    alert_json_path: str | Path | None = None,
) -> dict[str, Any]:
    latest = summary.get("latest_run", {}) or {}
    return _sanitize_value(
        {
            "event_type": "signal_drift_alert",
            "generated_at": _utc_now(),
            "ops_status": summary.get("ops_status"),
            "trend_assessment": summary.get("trend_assessment"),
            "operator_message": summary.get("operator_message"),
            "alert_json_path": str(alert_json_path) if alert_json_path else None,
            "run_count": summary.get("run_count"),
            "latest_run": {
                "report_name": latest.get("report_name"),
                "monitor_status": latest.get("monitor_status"),
                "recent_start": latest.get("recent_start"),
                "recent_end": latest.get("recent_end"),
                "recent_hit_rate_60m": latest.get("recent_hit_rate_60m"),
                "hit_rate_delta": latest.get("hit_rate_delta"),
                "avg_return_delta_bps": latest.get("avg_return_delta_bps"),
                "warning_count": latest.get("warning_count"),
                "report_json": latest.get("report_json"),
                "report_markdown": latest.get("report_markdown"),
            },
            "alert_reasons": summary.get("alert_reasons", []),
            "warning_digest": summary.get("warning_digest", [])[:10],
        }
    )


def send_signal_drift_webhook(
    *,
    webhook_url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload, default=str).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(float(timeout_s), 1.0)) as response:
            status = int(getattr(response, "status", 200))
        return {"attempted": True, "sent": True, "status_code": status, "error": None}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        return {"attempted": True, "sent": False, "status_code": None, "error": str(exc)}


def run_signal_drift_notify_command(
    *,
    command: str,
    context: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "SIGNAL_DRIFT_OPS_STATUS": str(context.get("ops_status") or ""),
            "SIGNAL_DRIFT_TREND_ASSESSMENT": str(context.get("trend_assessment") or ""),
            "SIGNAL_DRIFT_ALERT_JSON": str(context.get("alert_json_path") or ""),
            "SIGNAL_DRIFT_REPORT_JSON": str((context.get("latest_run") or {}).get("report_json") or ""),
            "SIGNAL_DRIFT_WARNING_COUNT": str((context.get("latest_run") or {}).get("warning_count") or ""),
        }
    )
    try:
        proc = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=max(float(timeout_s), 1.0),
        )
        return {
            "attempted": True,
            "returncode": int(proc.returncode),
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "error": None,
        }
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return {
            "attempted": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }


def _delivery_status(
    *,
    should_notify: bool,
    dry_run: bool,
    webhook: dict[str, Any],
    notify_command: dict[str, Any],
    destination_count: int,
) -> str:
    if not should_notify:
        return "SKIPPED_NOT_TRIGGERED"
    if dry_run:
        return "DRY_RUN"
    if destination_count == 0:
        return "NO_DESTINATION"

    webhook_attempted = bool(webhook.get("attempted"))
    notify_attempted = bool(notify_command.get("attempted"))
    webhook_ok = (not webhook_attempted) or bool(webhook.get("sent"))
    notify_ok = (not notify_attempted) or (notify_command.get("returncode") == 0)
    if webhook_ok and notify_ok:
        return "SENT"
    if webhook_attempted or notify_attempted:
        return "FAILED"
    return "NO_DESTINATION"


def build_signal_drift_delivery_row(
    *,
    summary: dict[str, Any],
    alert_json_path: str | Path,
    should_notify: bool,
    include_watch: bool,
    dry_run: bool,
    delivery_status: str,
    webhook: dict[str, Any],
    notify_command: dict[str, Any],
) -> dict[str, Any]:
    latest = summary.get("latest_run", {}) or {}
    return _sanitize_value(
        {
            "delivered_at": _utc_now(),
            "alert_generated_at": summary.get("generated_at"),
            "ops_status": summary.get("ops_status"),
            "trend_assessment": summary.get("trend_assessment"),
            "should_notify": bool(should_notify),
            "include_watch": bool(include_watch),
            "dry_run": bool(dry_run),
            "delivery_status": delivery_status,
            "webhook_attempted": bool(webhook.get("attempted")),
            "webhook_sent": bool(webhook.get("sent")),
            "webhook_status_code": webhook.get("status_code"),
            "webhook_error": webhook.get("error"),
            "notify_attempted": bool(notify_command.get("attempted")),
            "notify_returncode": notify_command.get("returncode"),
            "notify_error": notify_command.get("error") or notify_command.get("stderr"),
            "alert_json_path": str(alert_json_path),
            "report_json": latest.get("report_json"),
            "warning_count": latest.get("warning_count"),
            "alert_reasons": " | ".join(summary.get("alert_reasons", []) or []),
        }
    )


def append_signal_drift_delivery_row(row: dict[str, Any], ledger_path: str | Path) -> Path:
    path = Path(ledger_path)
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
    return path


def send_signal_drift_alert_notification(
    *,
    alert_json_path: str | Path | None = None,
    delivery_ledger_path: str | Path | None = None,
    webhook_url: str | None = None,
    notify_command: str | None = None,
    include_watch: bool = False,
    dry_run: bool = False,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """Send an optional drift alert notification and append delivery ledger."""
    alert_path = Path(alert_json_path) if alert_json_path is not None else default_alert_json_path()
    ledger_path = Path(delivery_ledger_path) if delivery_ledger_path is not None else alert_path.parent / DRIFT_ALERT_DELIVERY_LEDGER_FILENAME
    summary = load_json_file(alert_path)
    should_notify = should_notify_signal_drift_alert(summary, include_watch=include_watch)
    payload = build_signal_drift_notification_payload(summary, alert_json_path=alert_path)

    resolved_webhook_url = str(webhook_url if webhook_url is not None else os.getenv(WEBHOOK_ENV_VAR, "")).strip()
    resolved_notify_command = str(
        notify_command if notify_command is not None else os.getenv(NOTIFY_COMMAND_ENV_VAR, "")
    ).strip()
    destination_count = int(bool(resolved_webhook_url)) + int(bool(resolved_notify_command))

    webhook = {"attempted": False, "sent": False, "status_code": None, "error": None}
    command = {"attempted": False, "returncode": None, "stdout": "", "stderr": "", "error": None}
    if should_notify and not dry_run:
        if resolved_webhook_url:
            webhook = send_signal_drift_webhook(
                webhook_url=resolved_webhook_url,
                payload=payload,
                timeout_s=timeout_s,
            )
        if resolved_notify_command:
            command = run_signal_drift_notify_command(
                command=resolved_notify_command,
                context=payload,
                timeout_s=timeout_s,
            )

    delivery_status = _delivery_status(
        should_notify=should_notify,
        dry_run=dry_run,
        webhook=webhook,
        notify_command=command,
        destination_count=destination_count,
    )
    row = build_signal_drift_delivery_row(
        summary=summary,
        alert_json_path=alert_path,
        should_notify=should_notify,
        include_watch=include_watch,
        dry_run=dry_run,
        delivery_status=delivery_status,
        webhook=webhook,
        notify_command=command,
    )
    append_signal_drift_delivery_row(row, ledger_path)
    return {
        "alert_json_path": str(alert_path),
        "delivery_ledger_path": str(ledger_path),
        "should_notify": bool(should_notify),
        "delivery_status": delivery_status,
        "notification_payload": payload,
        "delivery_row": row,
        "webhook": webhook,
        "notify_command": command,
    }
