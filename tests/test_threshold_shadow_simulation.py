from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.signal_evaluation.threshold_governance import build_threshold_governance_report
from research.signal_evaluation.threshold_policy_experiment import build_threshold_policy_experiment_report
from research.signal_evaluation.threshold_shadow_simulation import (
    SHADOW_SIMULATION_READY,
    SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED,
    build_threshold_shadow_simulation_report,
    write_threshold_shadow_simulation_report,
    write_threshold_shadow_simulation_skip,
)


def _shadow_frame(days: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01 09:20:00+05:30")
    for idx in range(days):
        high_score = idx >= 30
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
                "symbol": "NIFTY",
                "source": "unit",
                "mode": "TEST",
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "trade_status": "TRADE",
                "signal_regime": "EXPANSION_BIAS" if high_score else "CONFLICTED",
                "macro_regime": "RISK_ON" if idx % 3 else "RISK_OFF",
                "gamma_regime": "SHORT_GAMMA_ZONE" if idx % 2 else "LONG_GAMMA_ZONE",
                "volatility_regime": "NORMAL",
                "global_risk_state": "CALM",
                "composite_signal_score": 82.0 if high_score else 52.0,
                "tradeability_score": 78.0 if high_score else 45.0,
                "move_probability": 0.72 if high_score else 0.48,
                "ml_confidence_score": 0.74 if high_score else 0.42,
                "correct_60m": 1.0 if high_score else 0.0,
                "signed_return_60m_bps": 24.0 if high_score else -8.0,
                "calibration_label": 1.0 if high_score else 0.0,
                "calibration_label_available": True,
                "primary_outcome_return_bps": 24.0 if high_score else -8.0,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def _approved_policy_experiment(frame: pd.DataFrame) -> dict:
    governance = build_threshold_governance_report(frame, dataset_path="unit.csv")
    return build_threshold_policy_experiment_report(
        frame,
        governance_report=governance,
        dataset_path="unit.csv",
        governance_report_path="governance.json",
    )


def test_shadow_simulation_splits_retained_and_suppressed_signal_stream():
    frame = _shadow_frame()
    policy_experiment = _approved_policy_experiment(frame)

    report = build_threshold_shadow_simulation_report(
        frame,
        policy_experiment_report=policy_experiment,
        dataset_path="unit.csv",
        policy_experiment_report_path="policy_experiment.json",
    )

    assert report["shadow_status"] == SHADOW_SIMULATION_READY
    assert report["runtime_config_changed"] is False
    assert report["threshold_rule"]["field"] == "composite_signal_score"
    assert report["impact_summary"]["eligible_signal_count"] == 120
    assert report["impact_summary"]["retained_signal_count"] == 90
    assert report["impact_summary"]["suppressed_signal_count"] == 30
    assert report["impact_summary"]["false_positive_removed_count"] == 30
    assert report["impact_summary"]["true_positive_lost_count"] == 0
    assert report["retained_vs_baseline_delta"]["avg_return_delta_bps"] > 0
    assert report["regime_shadow"]
    assert report["bucket_shadow"]


def test_shadow_simulation_skips_when_policy_experiment_not_approved():
    report = build_threshold_shadow_simulation_report(
        _shadow_frame(),
        policy_experiment_report={"experiment_status": "REVIEW_REQUIRED"},
    )

    assert report["shadow_status"] == SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED
    assert report["runtime_config_changed"] is False
    assert "not approved" in report["shadow_reasons"][0]


def test_shadow_simulation_writer_outputs_signal_classification_csvs(tmp_path: Path):
    frame = _shadow_frame()
    policy_experiment = _approved_policy_experiment(frame)
    artifact = write_threshold_shadow_simulation_report(
        frame,
        policy_experiment_report=policy_experiment,
        dataset_path="unit.csv",
        policy_experiment_report_path="policy_experiment.json",
        output_dir=tmp_path,
        report_name="unit_threshold_shadow_simulation",
    )

    for key in [
        "json_path",
        "markdown_path",
        "retained_signals_csv_path",
        "suppressed_signals_csv_path",
        "regimes_csv_path",
        "buckets_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_retained_signals_csv_path",
        "latest_suppressed_signals_csv_path",
        "latest_regimes_csv_path",
        "latest_buckets_csv_path",
    ]:
        assert Path(artifact[key]).exists()

    payload = json.loads(Path(artifact["json_path"]).read_text(encoding="utf-8"))
    retained = pd.read_csv(artifact["retained_signals_csv_path"])
    suppressed = pd.read_csv(artifact["suppressed_signals_csv_path"])
    assert payload["shadow_status"] == SHADOW_SIMULATION_READY
    assert "TRUE_POSITIVE_RETAINED" in set(retained["shadow_outcome_classification"])
    assert "FALSE_POSITIVE_REMOVED" in set(suppressed["shadow_outcome_classification"])


def test_shadow_simulation_skip_writer_prevents_stale_latest(tmp_path: Path):
    artifact = write_threshold_shadow_simulation_skip(
        reason="Policy experiment not approved.",
        policy_experiment_report={"experiment_status": "REVIEW_REQUIRED"},
        output_dir=tmp_path,
        report_name="unit_threshold_shadow_simulation_skip",
    )

    payload = json.loads(Path(artifact["latest_json_path"]).read_text(encoding="utf-8"))
    assert payload["shadow_status"] == SKIPPED_POLICY_EXPERIMENT_NOT_APPROVED
    assert payload["runtime_config_changed"] is False
    assert Path(artifact["latest_markdown_path"]).exists()
    assert Path(artifact["latest_suppressed_signals_csv_path"]).exists()
