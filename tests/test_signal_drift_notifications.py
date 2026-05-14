from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pandas as pd

from research.signal_evaluation.drift_notifications import (
    build_signal_drift_notification_payload,
    send_signal_drift_alert_notification,
    send_signal_drift_webhook,
    should_notify_signal_drift_alert,
)


ROOT = Path(__file__).resolve().parents[1]
SEND_SCRIPT = ROOT / "scripts" / "ops" / "send_signal_drift_alert.py"


def _alert_summary(status: str = "DETERIORATING") -> dict:
    return {
        "report_type": "signal_drift_alert_summary",
        "generated_at": "2026-04-08T00:00:00+00:00",
        "ops_status": status,
        "trend_assessment": status,
        "operator_message": "Review signal drift.",
        "run_count": 3,
        "latest_run": {
            "report_name": "unit_signal_drift",
            "monitor_status": "CAUTION" if status == "DETERIORATING" else status,
            "recent_start": "2026-04-07",
            "recent_end": "2026-04-08",
            "recent_hit_rate_60m": 0.2,
            "hit_rate_delta": -0.4,
            "avg_return_delta_bps": -35.0,
            "warning_count": 2,
            "report_json": "/tmp/signal_drift.json",
            "report_markdown": "/tmp/signal_drift.md",
        },
        "alert_reasons": ["Trend dashboard assessment is DETERIORATING."],
        "warning_digest": [
            {"severity": "CAUTION", "category": "outcome_drift", "message": "Recent hit rate fell."}
        ],
    }


def _write_alert(tmp_path: Path, status: str = "DETERIORATING") -> Path:
    path = tmp_path / "latest_signal_drift_alert.json"
    path.write_text(json.dumps(_alert_summary(status)), encoding="utf-8")
    return path


def test_should_notify_only_deteriorating_by_default():
    assert should_notify_signal_drift_alert(_alert_summary("DETERIORATING")) is True
    assert should_notify_signal_drift_alert(_alert_summary("WATCH")) is False
    assert should_notify_signal_drift_alert(_alert_summary("WATCH"), include_watch=True) is True
    assert should_notify_signal_drift_alert(_alert_summary("STABLE"), include_watch=True) is False


def test_build_notification_payload_is_compact():
    payload = build_signal_drift_notification_payload(
        _alert_summary(),
        alert_json_path="/tmp/latest_signal_drift_alert.json",
    )

    assert payload["event_type"] == "signal_drift_alert"
    assert payload["ops_status"] == "DETERIORATING"
    assert payload["latest_run"]["report_name"] == "unit_signal_drift"
    assert payload["alert_json_path"] == "/tmp/latest_signal_drift_alert.json"


def test_send_drift_notification_dry_run_appends_ledger_without_delivery(tmp_path: Path):
    alert_path = _write_alert(tmp_path)
    ledger_path = tmp_path / "delivery.csv"

    artifact = send_signal_drift_alert_notification(
        alert_json_path=alert_path,
        delivery_ledger_path=ledger_path,
        webhook_url="https://example.com/hook",
        notify_command="echo drift",
        dry_run=True,
    )

    ledger = pd.read_csv(ledger_path)
    assert artifact["should_notify"] is True
    assert artifact["delivery_status"] == "DRY_RUN"
    assert artifact["webhook"]["attempted"] is False
    assert artifact["notify_command"]["attempted"] is False
    assert ledger["delivery_status"].iloc[0] == "DRY_RUN"


def test_send_drift_notification_skips_stable_alert(tmp_path: Path):
    alert_path = _write_alert(tmp_path, status="STABLE")
    ledger_path = tmp_path / "delivery.csv"

    artifact = send_signal_drift_alert_notification(
        alert_json_path=alert_path,
        delivery_ledger_path=ledger_path,
        webhook_url="https://example.com/hook",
    )

    ledger = pd.read_csv(ledger_path)
    assert artifact["should_notify"] is False
    assert artifact["delivery_status"] == "SKIPPED_NOT_TRIGGERED"
    assert ledger["delivery_status"].iloc[0] == "SKIPPED_NOT_TRIGGERED"


def test_send_drift_notification_uses_webhook_and_command(monkeypatch, tmp_path: Path):
    alert_path = _write_alert(tmp_path)
    calls = {"webhook": {}, "command": {}}

    def _fake_webhook(**kwargs):
        calls["webhook"] = kwargs
        return {"attempted": True, "sent": True, "status_code": 200, "error": None}

    def _fake_command(**kwargs):
        calls["command"] = kwargs
        return {"attempted": True, "returncode": 0, "stdout": "ok", "stderr": "", "error": None}

    monkeypatch.setattr(
        "research.signal_evaluation.drift_notifications.send_signal_drift_webhook",
        _fake_webhook,
    )
    monkeypatch.setattr(
        "research.signal_evaluation.drift_notifications.run_signal_drift_notify_command",
        _fake_command,
    )

    artifact = send_signal_drift_alert_notification(
        alert_json_path=alert_path,
        delivery_ledger_path=tmp_path / "delivery.csv",
        webhook_url="https://example.com/hook",
        notify_command="echo drift",
        timeout_s=2.5,
    )

    assert artifact["delivery_status"] == "SENT"
    assert calls["webhook"]["webhook_url"] == "https://example.com/hook"
    assert calls["webhook"]["timeout_s"] == 2.5
    assert calls["webhook"]["payload"]["ops_status"] == "DETERIORATING"
    assert calls["command"]["command"] == "echo drift"
    assert calls["command"]["context"]["latest_run"]["report_json"] == "/tmp/signal_drift.json"


def test_send_signal_drift_webhook_handles_url_error(monkeypatch):
    def _raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("research.signal_evaluation.drift_notifications.urllib.request.urlopen", _raise_url_error)

    result = send_signal_drift_webhook(
        webhook_url="https://example.com/hook",
        payload={"ops_status": "DETERIORATING"},
        timeout_s=1.0,
    )

    assert result["attempted"] is True
    assert result["sent"] is False
    assert "network down" in (result["error"] or "")


def test_send_signal_drift_alert_cli_strict_fails_without_destination(tmp_path: Path):
    alert_path = _write_alert(tmp_path)
    ledger_path = tmp_path / "delivery.csv"

    proc = subprocess.run(
        [
            sys.executable,
            str(SEND_SCRIPT),
            "--alert-json",
            str(alert_path),
            "--delivery-ledger",
            str(ledger_path),
            "--strict",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["should_notify"] is True
    assert payload["delivery_status"] == "NO_DESTINATION"
    assert ledger_path.exists()
