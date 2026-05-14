from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from config.signal_drift_policy import get_signal_drift_monitor_policy
from research.signal_evaluation.drift_monitor import (
    build_signal_drift_report,
    build_signal_drift_trend_dashboard,
    write_signal_drift_report,
)


def _drift_frame() -> pd.DataFrame:
    rows = []
    for idx, day in enumerate(pd.date_range("2026-04-01", periods=6, freq="D")):
        rows.append(
            {
                "signal_id": f"base-{idx}",
                "signal_timestamp": f"{day.date()}T09:20:00+05:30",
                "source": "broker_a",
                "mode": "LIVE",
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
                "hybrid_move_probability": 0.78,
                "ml_confidence_score": 0.76,
                "ml_rank_score": 0.82,
                "trade_strength": 82,
                "composite_signal_score": 80,
                "tradeability_score": 78,
                "sample_policy_decision": "ALLOW",
            }
        )

    recent_days = list(pd.date_range("2026-04-07", periods=2, freq="D"))
    for idx in range(4):
        day = recent_days[idx % 2]
        approved = idx != 3
        rows.append(
            {
                "signal_id": f"recent-{idx}",
                "signal_timestamp": f"{day.date()}T09:20:00+05:30",
                "source": "broker_a",
                "mode": "LIVE",
                "gamma_regime": "SHORT_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "macro_regime": "RISK_ON",
                "global_risk_state": "CALM",
                "correct_60m": 1 if not approved else 0,
                "signed_return_60m_bps": 100.0 if not approved else -30.0 - idx,
                "calibration_label": 0 if approved else None,
                "calibration_label_available": approved,
                "primary_outcome_return_bps": -30.0 - idx if approved else None,
                "label_quality_status": "CLEAN" if approved else "UNUSABLE",
                "label_quality_reasons": "" if approved else "direction_unresolved",
                "hybrid_move_probability": 0.80,
                "ml_confidence_score": 0.79,
                "ml_rank_score": 0.81,
                "trade_strength": 81,
                "composite_signal_score": 79,
                "tradeability_score": 76,
                "sample_policy_decision": "BLOCK",
            }
        )
    return pd.DataFrame(rows)


def test_signal_drift_report_detects_quality_aware_outcome_and_policy_drift():
    report = build_signal_drift_report(
        _drift_frame(),
        recent_days=2,
        baseline_days=10,
        min_recent_labeled=2,
        min_baseline_labeled=3,
        apply_missing_policies=False,
    )

    assert report["monitor_status"] == "CAUTION"
    assert report["label_quality_summary"]["quality_labeled_rows"] == 9
    assert report["label_quality_summary"]["excluded_labeled_rows"] == 1

    overall = report["overall_outcome_drift"]
    assert overall["baseline"]["hit_rate_60m"] == 1.0
    assert overall["recent"]["hit_rate_60m"] == 0.0
    assert overall["hit_rate_delta"] == -1.0
    assert overall["recent"]["labeled_60m"] == 3

    source_row = report["dimension_outcome_drift"]["source"][0]
    assert source_row["value"] == "broker_a"
    assert source_row["status"] == "DRIFT_DOWN"

    policy_row = report["policy_retention_drift"][0]
    assert policy_row["policy"] == "sample_policy"
    assert policy_row["retention_delta"] == -1.0

    assert any(item["category"] == "outcome_drift" for item in report["warnings"])
    assert any(item["category"] == "policy_retention" for item in report["warnings"])
    assert any(item["category"] == "label_quality" for item in report["warnings"])


def test_signal_drift_report_suppresses_outcome_warning_when_samples_are_thin():
    report = build_signal_drift_report(
        _drift_frame(),
        recent_days=2,
        baseline_days=10,
        min_recent_labeled=5,
        min_baseline_labeled=10,
        apply_missing_policies=False,
    )

    overall = report["overall_outcome_drift"]
    assert overall["hit_rate_delta"] == -1.0
    assert overall["outcome_evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert overall["outcome_guardrail"]["sufficient_evidence"] is False
    assert any(item["category"] == "label_sample" for item in report["warnings"])
    assert not any(item["category"] == "outcome_drift" for item in report["warnings"])


def test_signal_drift_report_writer_emits_artifacts(tmp_path: Path):
    artifact = write_signal_drift_report(
        _drift_frame(),
        dataset_path="unit-test.csv",
        output_dir=tmp_path,
        report_name="unit_signal_drift",
        recent_days=2,
        baseline_days=10,
        min_recent_labeled=2,
        min_baseline_labeled=3,
        apply_missing_policies=False,
    )

    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["markdown_path"]).exists()
    assert "Signal Drift Monitor" in Path(artifact["markdown_path"]).read_text(encoding="utf-8")
    assert Path(artifact["csv_paths"]["dimension_outcome_drift"]).exists()
    assert Path(artifact["trend_history_path"]).exists()
    assert Path(artifact["trend_dashboard_json_path"]).exists()
    assert Path(artifact["trend_dashboard_markdown_path"]).exists()
    assert artifact["report"]["dataset_path"] == "unit-test.csv"
    assert artifact["trend_dashboard"]["latest"]["report_name"] == "unit_signal_drift"
    assert artifact["trend_dashboard"]["trend_assessment"] == "DETERIORATING"


def test_signal_drift_trend_history_appends_and_dashboard_tracks_latest(tmp_path: Path):
    first = write_signal_drift_report(
        _drift_frame(),
        dataset_path="unit-test.csv",
        output_dir=tmp_path,
        report_name="unit_signal_drift_first",
        recent_days=2,
        baseline_days=10,
        min_recent_labeled=2,
        min_baseline_labeled=3,
        apply_missing_policies=False,
    )
    second = write_signal_drift_report(
        _drift_frame(),
        dataset_path="unit-test.csv",
        output_dir=tmp_path,
        report_name="unit_signal_drift_second",
        recent_days=2,
        baseline_days=10,
        min_recent_labeled=2,
        min_baseline_labeled=3,
        apply_missing_policies=False,
    )

    history = pd.read_csv(second["trend_history_path"])
    assert len(history) == 2
    assert history["report_name"].tolist() == ["unit_signal_drift_first", "unit_signal_drift_second"]
    assert history["monitor_status"].tolist() == ["CAUTION", "CAUTION"]
    assert history["report_json"].iloc[0] == first["json_path"]
    assert history["report_json"].iloc[1] == second["json_path"]

    dashboard = json.loads(Path(second["trend_dashboard_json_path"]).read_text(encoding="utf-8"))
    assert dashboard["run_count"] == 2
    assert dashboard["latest"]["report_json"] == second["json_path"]
    assert dashboard["trend_assessment"] == "DETERIORATING"
    assert "Signal Drift Trend Dashboard" in Path(second["trend_dashboard_markdown_path"]).read_text(encoding="utf-8")


def test_signal_drift_trend_dashboard_handles_empty_history():
    dashboard = build_signal_drift_trend_dashboard(pd.DataFrame())

    assert dashboard["run_count"] == 0
    assert dashboard["trend_assessment"] == "NO_HISTORY"


def test_signal_drift_thresholds_resolve_from_governed_policy():
    overrides = {
        "signal_drift.monitor.recent_days": 2,
        "signal_drift.monitor.baseline_days": 10,
        "signal_drift.monitor.min_recent_labeled": 2,
        "signal_drift.monitor.min_baseline_labeled": 3,
        "signal_drift.monitor.hit_rate_drop_warn": 0.33,
        "signal_drift.monitor.return_drop_bps_warn": 44.0,
        "signal_drift.monitor.retention_delta_warn": 0.55,
        "signal_drift.monitor.dimensions": ["source"],
    }

    with temporary_parameter_pack(overrides=overrides):
        policy = get_signal_drift_monitor_policy()
        report = build_signal_drift_report(
            _drift_frame(),
            apply_missing_policies=False,
        )

    assert policy["hit_rate_drop_warn"] == 0.33
    assert policy["dimensions"] == ["source"]
    thresholds = report["thresholds"]
    assert thresholds["policy_source"] == "config.signal_drift_policy"
    assert thresholds["recent_days"] == 2
    assert thresholds["baseline_days"] == 10
    assert thresholds["min_recent_labeled"] == 2
    assert thresholds["min_baseline_labeled"] == 3
    assert thresholds["hit_rate_drop_warn"] == 0.33
    assert thresholds["return_drop_bps_warn"] == 44.0
    assert thresholds["retention_delta_warn"] == 0.55
    assert thresholds["dimensions"] == ["source"]
