from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.signal_quality_model_audit import (
    build_signal_quality_model_audit_report,
    write_signal_quality_model_audit_report,
)


def _sample_frame() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-03-01T09:20:00+05:30")
    probabilities = [0.35, 0.42, 0.55, 0.62, 0.70, 0.76, 0.84, 0.90]
    labels = [0, 0, 1, 0, 1, 1, 1, 1]
    returns = [-18, -12, 20, -8, 26, 32, 44, 55]
    for idx, probability in enumerate(probabilities):
        rows.append(
            {
                "signal_id": f"sig-{idx}",
                "signal_timestamp": (base + pd.Timedelta(days=idx)).isoformat(),
                "direction": "CALL" if idx % 2 == 0 else "PUT",
                "hybrid_move_probability": probability,
                "rule_move_probability": probability - 0.03,
                "signal_confidence_score": probability * 100,
                "composite_signal_score": 45 + idx * 6,
                "trade_strength": 50 + idx * 4,
                "tradeability_score": 55 + idx * 3,
                "target_reachability_score": 50 + idx * 5,
                "premium_efficiency_score": 45 + idx * 4,
                "strike_efficiency_score": 52 + idx * 3,
                "option_efficiency_score": 48 + idx * 5,
                "global_risk_score": 20 + idx,
                "gamma_vol_acceleration_score": 40 + idx * 2,
                "dealer_hedging_pressure_score": 35 + idx,
                "macro_event_risk_score": 5,
                "data_quality_score": 90,
                "selected_option_ba_spread_pct": 1.0 + idx * 0.1,
                "selected_option_volume": 1000 + idx * 100,
                "selected_option_open_interest": 5000 + idx * 50,
                "selected_option_iv": 12 + idx,
                "macro_regime": "RISK_ON" if idx >= 4 else "NEUTRAL",
                "gamma_regime": "POSITIVE_GAMMA" if idx >= 3 else "NEGATIVE_GAMMA",
                "volatility_regime": "NORMAL_VOL",
                "global_risk_state": "GLOBAL_NEUTRAL",
                "correct_60m": labels[idx],
                "calibration_label": labels[idx],
                "calibration_label_available": True,
                "signed_return_60m_bps": returns[idx],
                "primary_outcome_return_bps": returns[idx],
                "mae_60m_bps": abs(min(returns[idx], 0)) + 5,
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_signal_quality_model_audit_builds_calibration_regimes_and_rankings():
    report = build_signal_quality_model_audit_report(
        _sample_frame(),
        dataset_path="unit.csv",
        min_label_sample=4,
        strong_label_sample=8,
        min_regime_sample=2,
    )

    assert report["report_type"] == "signal_quality_model_audit"
    assert report["runtime_config_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["quality_labeled_row_count"] == 8
    assert report["calibration_summary"]["label_count"] == 8
    assert report["calibration_summary"]["brier_score"] is not None
    assert report["calibration_bins"]
    assert report["regime_calibration"]
    assert report["feature_stability"]
    assert report["ranking_feature_audit"]
    assert any(row["feature"] == "composite_signal_score" for row in report["ranking_feature_audit"])
    assert report["recommended_next_actions"]


def test_signal_quality_model_audit_writer_outputs_all_artifacts(tmp_path: Path):
    artifact = write_signal_quality_model_audit_report(
        _sample_frame(),
        dataset_path="unit.csv",
        min_label_sample=4,
        strong_label_sample=8,
        min_regime_sample=2,
        output_dir=tmp_path,
        report_name="unit_signal_quality_model_audit",
    )

    for key in [
        "json_path",
        "markdown_path",
        "calibration_csv_path",
        "regime_csv_path",
        "feature_csv_path",
        "ranking_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_calibration_csv_path",
        "latest_regime_csv_path",
        "latest_feature_csv_path",
        "latest_ranking_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["calibration_summary"]["sample_quality"] in {
        "MODERATE_EVIDENCE",
        "STRONG_EVIDENCE",
    }


def test_signal_quality_model_audit_handles_unlabeled_frame():
    frame = _sample_frame()
    frame["calibration_label_available"] = False
    frame["calibration_label"] = pd.NA
    frame["correct_60m"] = pd.NA

    report = build_signal_quality_model_audit_report(frame, min_label_sample=4)

    assert report["quality_labeled_row_count"] == 0
    assert report["calibration_summary"]["calibration_status"] == "NO_EVIDENCE"
    assert "collect more quality-approved labels" in report["recommended_next_actions"][0]
