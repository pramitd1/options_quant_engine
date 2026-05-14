from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research.signal_evaluation.drift_alerts import (
    build_signal_drift_alert_summary,
    run_signal_drift_alert_workflow,
    write_signal_drift_alert_summary,
)
from research.signal_evaluation.drift_monitor import build_signal_drift_trend_dashboard, write_signal_drift_report


ROOT = Path(__file__).resolve().parents[1]
ALERT_SCRIPT = ROOT / "scripts" / "ops" / "run_signal_drift_alerts.py"


def _drift_frame() -> pd.DataFrame:
    rows = []
    for idx, day in enumerate(pd.date_range("2026-04-01", periods=6, freq="D")):
        rows.append(
            {
                "signal_id": f"base-{idx}",
                "signal_timestamp": f"{day.date()}T09:20:00+05:30",
                "source": "unit",
                "mode": "TEST",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "macro_regime": "RISK_ON",
                "global_risk_state": "CALM",
                "correct_60m": 1,
                "signed_return_60m_bps": 20.0 + idx,
                "calibration_label": 1,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 20.0 + idx,
                "label_quality_status": "CLEAN",
                "hybrid_move_probability": 0.76,
                "ml_confidence_score": 0.74,
                "ml_rank_score": 0.80,
                "trade_strength": 80,
                "composite_signal_score": 78,
                "tradeability_score": 76,
                "sample_policy_decision": "ALLOW",
            }
        )
    for idx in range(3):
        rows.append(
            {
                "signal_id": f"recent-{idx}",
                "signal_timestamp": "2026-04-07T09:20:00+05:30",
                "source": "unit",
                "mode": "TEST",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "macro_regime": "RISK_ON",
                "global_risk_state": "CALM",
                "correct_60m": 0,
                "signed_return_60m_bps": -30.0 - idx,
                "calibration_label": 0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": -30.0 - idx,
                "label_quality_status": "CLEAN",
                "hybrid_move_probability": 0.82,
                "ml_confidence_score": 0.80,
                "ml_rank_score": 0.78,
                "trade_strength": 78,
                "composite_signal_score": 76,
                "tradeability_score": 74,
                "sample_policy_decision": "BLOCK",
            }
        )
    return pd.DataFrame(rows)


def _write_deteriorating_dashboard(tmp_path: Path) -> dict:
    return write_signal_drift_report(
        _drift_frame(),
        dataset_path="unit-alert.csv",
        output_dir=tmp_path,
        report_name="unit_signal_drift_alert",
        recent_days=1,
        baseline_days=10,
        min_recent_labeled=2,
        min_baseline_labeled=3,
        apply_missing_policies=False,
    )


def test_signal_drift_alert_summary_writes_artifacts(tmp_path: Path):
    drift_artifact = _write_deteriorating_dashboard(tmp_path)

    alert_artifact = write_signal_drift_alert_summary(
        trend_dashboard_path=drift_artifact["trend_dashboard_json_path"],
        output_dir=tmp_path,
    )

    summary = alert_artifact["alert_summary"]
    assert Path(alert_artifact["alert_json_path"]).exists()
    assert Path(alert_artifact["alert_markdown_path"]).exists()
    assert summary["ops_status"] == "DETERIORATING"
    assert summary["strict_failure"] is True
    assert summary["latest_run"]["report_name"] == "unit_signal_drift_alert"
    assert any(item["category"] == "outcome_drift" for item in summary["warning_digest"])
    assert "No execution behavior is changed" in Path(alert_artifact["alert_markdown_path"]).read_text(encoding="utf-8")


def test_signal_drift_alert_review_workflow_appends_ledger(tmp_path: Path):
    drift_artifact = _write_deteriorating_dashboard(tmp_path)

    alert_artifact = run_signal_drift_alert_workflow(
        trend_dashboard_path=drift_artifact["trend_dashboard_json_path"],
        output_dir=tmp_path,
        review_action="ACKNOWLEDGED",
        reviewer="unit-test",
        review_note="Reviewed drivers before next session.",
        next_review_at="2026-04-08",
    )

    ledger = pd.read_csv(alert_artifact["review_ledger_path"])
    assert len(ledger) == 1
    assert ledger["review_action"].iloc[0] == "ACKNOWLEDGED"
    assert ledger["reviewer"].iloc[0] == "unit-test"
    assert alert_artifact["alert_summary"]["latest_review"]["review_action"] == "ACKNOWLEDGED"
    assert alert_artifact["alert_summary"]["latest_review"]["review_note"] == "Reviewed drivers before next session."


def test_signal_drift_alert_summary_handles_no_history():
    dashboard = build_signal_drift_trend_dashboard(pd.DataFrame())

    summary = build_signal_drift_alert_summary(dashboard)

    assert summary["ops_status"] == "NO_HISTORY"
    assert summary["strict_failure"] is False
    assert summary["warning_digest"] == []


def test_signal_drift_alert_cli_strict_returns_nonzero_for_deteriorating(tmp_path: Path):
    drift_artifact = _write_deteriorating_dashboard(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(ALERT_SCRIPT),
            "--trend-dashboard",
            str(drift_artifact["trend_dashboard_json_path"]),
            "--output-dir",
            str(tmp_path),
            "--strict",
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ops_status"] == "DETERIORATING"
    assert payload["strict_failure"] is True
    assert Path(payload["alert_json_path"]).exists()
