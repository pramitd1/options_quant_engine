from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.signal_evaluation.historical_modifier_monitor import (
    build_historical_modifier_monitor_report,
    write_historical_modifier_monitor_report,
)


def _sample_frame() -> pd.DataFrame:
    base = pd.Timestamp("2026-05-18T09:30:00+05:30")
    rows = [
        {
            "signal_id": "support-help",
            "historical_context_score_adjustment": 4,
            "historical_context_probability_adjustment": 0.02,
            "historical_context_trade_strength_threshold_adjustment": -1,
            "historical_context_size_multiplier": 1.0,
            "historical_context_reasons": "historical_global_prior_aligned",
            "historical_interaction_count": 1,
            "historical_interaction_score_adjustment": 2,
            "historical_interaction_reasons": "expiry_x_pcr_aligned_call",
            "historical_interaction_bucket_state": '{"expiry_bucket":"2-3d","pcr_oi_bucket":"high","weekday":"Monday"}',
            "signed_return_15m_bps": 9.0,
            "signed_return_30m_bps": 16.0,
            "signed_return_60m_bps": 24.0,
            "correct_15m": 1,
            "correct_30m": 1,
            "correct_60m": 1,
        },
        {
            "signal_id": "support-hurt",
            "historical_context_score_adjustment": 3,
            "historical_context_probability_adjustment": 0.01,
            "historical_context_trade_strength_threshold_adjustment": 0,
            "historical_context_size_multiplier": 1.0,
            "historical_context_reasons": "historical_high_pcr_supports_call",
            "historical_interaction_count": 0,
            "historical_interaction_bucket_state": '{"expiry_bucket":"0-1d","pcr_oi_bucket":"high","weekday":"Tuesday"}',
            "signed_return_15m_bps": -3.0,
            "signed_return_30m_bps": -9.0,
            "signed_return_60m_bps": -12.0,
            "correct_15m": 0,
            "correct_30m": 0,
            "correct_60m": 0,
        },
        {
            "signal_id": "restrict-help",
            "historical_context_score_adjustment": -5,
            "historical_context_probability_adjustment": -0.03,
            "historical_context_trade_strength_threshold_adjustment": 2,
            "historical_context_size_multiplier": 0.75,
            "historical_context_reasons": "historical_global_prior_conflict",
            "historical_interaction_count": 1,
            "historical_interaction_score_adjustment": -2,
            "historical_interaction_reasons": "india_vix_x_trend_conflicts_call",
            "historical_interaction_bucket_state": '{"india_vix_bucket":"high","trend_20d_bucket":"selloff","pcr_basis":"OPEN_INTEREST"}',
            "signed_return_15m_bps": -4.0,
            "signed_return_30m_bps": -10.0,
            "signed_return_60m_bps": -18.0,
            "correct_15m": 0,
            "correct_30m": 0,
            "correct_60m": 0,
        },
        {
            "signal_id": "restrict-hurt",
            "historical_context_score_adjustment": -4,
            "historical_context_probability_adjustment": -0.02,
            "historical_context_trade_strength_threshold_adjustment": 1,
            "historical_context_size_multiplier": 0.85,
            "historical_context_reasons": "historical_max_pain_friction",
            "historical_interaction_count": 0,
            "historical_interaction_bucket_state": '{"india_vix_bucket":"q4","weekday":"Wednesday"}',
            "signed_return_15m_bps": 5.0,
            "signed_return_30m_bps": 8.0,
            "signed_return_60m_bps": 14.0,
            "correct_15m": 1,
            "correct_30m": 1,
            "correct_60m": 1,
        },
        {
            "signal_id": "override-help",
            "historical_context_score_adjustment": 2,
            "historical_context_probability_adjustment": 0.01,
            "historical_context_trade_strength_threshold_adjustment": 0,
            "historical_context_size_multiplier": 1.0,
            "historical_context_direction_override": "PUT",
            "historical_context_reasons": "historical_global_prior_direction_fallback",
            "historical_interaction_count": 0,
            "historical_interaction_bucket_state": "{}",
            "signed_return_15m_bps": 7.0,
            "signed_return_30m_bps": 11.0,
            "signed_return_60m_bps": 19.0,
            "correct_15m": 1,
            "correct_30m": 1,
            "correct_60m": 1,
        },
        {
            "signal_id": "no-modifier",
            "historical_context_score_adjustment": 0,
            "historical_context_probability_adjustment": 0.0,
            "historical_context_trade_strength_threshold_adjustment": 0,
            "historical_context_size_multiplier": 1.0,
            "historical_interaction_count": 0,
            "historical_interaction_bucket_state": "{}",
            "signed_return_15m_bps": 6.0,
            "signed_return_30m_bps": 7.0,
            "signed_return_60m_bps": 8.0,
            "correct_15m": 1,
            "correct_30m": 1,
            "correct_60m": 1,
        },
    ]
    for idx, row in enumerate(rows):
        row.update(
            {
                "signal_timestamp": (base + pd.Timedelta(minutes=idx)).isoformat(),
                "direction": "CALL",
                "trade_status": "TRADE",
                "calibration_label": row["correct_60m"],
                "calibration_label_available": True,
                "primary_outcome_return_bps": row["signed_return_60m_bps"],
                "label_quality_status": "CLEAN",
            }
        )
    return pd.DataFrame(rows)


def test_historical_modifier_monitor_builds_help_hurt_summary():
    report = build_historical_modifier_monitor_report(
        _sample_frame(),
        dataset_path="unit.csv",
        min_label_sample=5,
    )

    assert report["report_type"] == "historical_modifier_monitor"
    assert report["runtime_config_changed"] is False
    assert report["execution_behavior_changed"] is False
    assert report["assessment_basis"] == "alignment_proxy_not_counterfactual_replay"
    assert report["modifier_row_count"] == 5
    assert report["score_adjustment_nonzero_count"] == 5
    assert report["probability_adjustment_nonzero_count"] == 5
    assert report["direction_override_count"] == 1
    assert report["interaction_nonzero_count"] == 2
    assert report["horizon_summary"]["60m"]["label_count"] == 5
    assert report["horizon_summary"]["60m"]["helped_count"] == 3
    assert report["horizon_summary"]["60m"]["hurt_count"] == 2
    assert report["horizon_summary"]["60m"]["help_rate"] == 0.6
    assert report["monitor_status"] == "HISTORICAL_LAYER_HELPFUL"
    assert any(row["component"] == "direction_override_used" for row in report["component_summary"])
    assert any(row["reason"] == "historical_global_prior_conflict" for row in report["reason_summary"])
    assert any(row["bucket_field"] == "pcr_oi_bucket" and row["bucket_value"] == "high" for row in report["bucket_summary"])


def test_historical_modifier_monitor_writer_outputs_artifacts(tmp_path: Path):
    artifact = write_historical_modifier_monitor_report(
        _sample_frame(),
        dataset_path="unit.csv",
        min_label_sample=5,
        output_dir=tmp_path,
        report_name="unit_historical_modifier_monitor",
    )

    for key in [
        "json_path",
        "markdown_path",
        "component_csv_path",
        "reason_csv_path",
        "bucket_csv_path",
        "row_csv_path",
        "latest_json_path",
        "latest_markdown_path",
        "latest_component_csv_path",
        "latest_reason_csv_path",
        "latest_bucket_csv_path",
        "latest_row_csv_path",
    ]:
        assert Path(artifact[key]).exists()
    assert artifact["report"]["modifier_row_count"] == 5


def test_historical_modifier_monitor_handles_unlabeled_rows():
    frame = _sample_frame()
    frame["calibration_label_available"] = False
    frame["calibration_label"] = pd.NA
    frame["correct_60m"] = pd.NA
    frame["signed_return_60m_bps"] = pd.NA
    frame["primary_outcome_return_bps"] = pd.NA

    report = build_historical_modifier_monitor_report(frame, min_label_sample=5)

    assert report["horizon_summary"]["60m"]["label_count"] == 0
    assert report["monitor_status"] == "NO_EVIDENCE"
    assert "Keep collecting labeled outcomes" in report["recommended_next_actions"][0]
