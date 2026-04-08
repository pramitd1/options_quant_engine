from __future__ import annotations

import json
from pathlib import Path

from strategy.direction_probability_head import compute_direction_probability_head
from strategy.direction_probability_head import get_direction_head_calibration_metrics
from strategy.direction_probability_head import reset_direction_head_calibration_metrics
from strategy.score_calibration import ScoreCalibrator


def test_direction_probability_head_outputs_probabilities_and_uncertainty():
    out = compute_direction_probability_head(
        final_flow_signal="BULLISH_FLOW",
        spot_vs_flip="ABOVE_FLIP",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="GAMMA_SQUEEZE",
        gamma_regime="NEGATIVE_GAMMA",
        macro_regime="RISK_OFF",
        volatility_regime="VOL_EXPANSION",
        oi_velocity_score=0.25,
        rr_value=-0.8,
        rr_momentum="FALLING_PUT_SKEW",
        volume_pcr_atm=0.78,
        hybrid_move_probability=0.62,
        vote_bull_probability=0.63,
        provider_health_summary="CAUTION",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.52,
        core_one_sided_quote_ratio=0.22,
        core_quote_integrity_health="CAUTION",
        apply_calibration=False,
    )

    assert 0.0 <= out["probability_up_raw"] <= 1.0
    assert 0.0 <= out["probability_up"] <= 1.0
    assert 0.0 <= out["probability_down"] <= 1.0
    assert 0.0 <= out["uncertainty"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["calibration_applied"] is False


def test_direction_probability_head_applies_calibrator(tmp_path: Path):
    calibrator = ScoreCalibrator(method="isotonic", n_bins=10)
    calibrator.fit([10, 30, 50, 70, 90], [0, 0, 0, 1, 1])
    path = tmp_path / "direction_probability_calibrator.json"
    calibrator.save_to_file(str(path))

    out = compute_direction_probability_head(
        final_flow_signal="BEARISH_FLOW",
        spot_vs_flip="BELOW_FLIP",
        hedging_bias="DOWNSIDE_ACCELERATION",
        gamma_event="NONE",
        gamma_regime="NEGATIVE_GAMMA",
        macro_regime="RISK_OFF",
        volatility_regime="VOL_EXPANSION",
        hybrid_move_probability=0.40,
        vote_bull_probability=0.35,
        provider_health_summary="GOOD",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.80,
        core_one_sided_quote_ratio=0.05,
        core_quote_integrity_health="GOOD",
        calibrator_path=str(path),
        apply_calibration=True,
    )

    assert out["calibrator_loaded"] is True
    assert out["calibration_applied"] is True
    assert 0.0 <= out["probability_up"] <= 1.0


def test_direction_probability_head_segmented_calibrator_and_fallback(tmp_path: Path):
    reset_direction_head_calibration_metrics()
    global_cal = ScoreCalibrator(method="isotonic", n_bins=10)
    global_cal.fit([10, 30, 50, 70, 90], [0, 0, 0, 1, 1])
    global_path = tmp_path / "direction_probability_calibrator.json"
    global_cal.save_to_file(str(global_path))

    gamma_cal = ScoreCalibrator(method="isotonic", n_bins=10)
    gamma_cal.fit([10, 30, 50, 70, 90], [0, 0, 1, 1, 1])
    segmented_payload = {
        "meta": {
            "method": "isotonic",
            "n_bins": 10,
            "regime_column": "gamma_regime",
            "min_group_samples": 10,
            "fallback_global_calibrator": str(global_path),
        },
        "groups": {
            "NEGATIVE_GAMMA": gamma_cal.to_state(),
        },
    }
    segmented_path = tmp_path / "direction_probability_calibrator_gamma_regime_segments.json"
    segmented_path.write_text(json.dumps(segmented_payload), encoding="utf-8")

    base_kwargs = dict(
        final_flow_signal="BULLISH_FLOW",
        spot_vs_flip="ABOVE_FLIP",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="NONE",
        macro_regime="RISK_OFF",
        volatility_regime="VOL_EXPANSION",
        hybrid_move_probability=0.60,
        vote_bull_probability=0.62,
        provider_health_summary="GOOD",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.80,
        core_one_sided_quote_ratio=0.05,
        core_quote_integrity_health="GOOD",
        calibrator_path=str(global_path),
        segmented_calibrator_path=str(segmented_path),
        apply_calibration=True,
    )

    out_segment = compute_direction_probability_head(gamma_regime="NEGATIVE_GAMMA", **base_kwargs)
    assert out_segment["calibrator_loaded"] is True
    assert out_segment["calibration_applied"] is True
    assert out_segment["calibration_segment"] == "NEGATIVE_GAMMA"

    out_fallback = compute_direction_probability_head(gamma_regime="POSITIVE_GAMMA", **base_kwargs)
    assert out_fallback["calibrator_loaded"] is True
    assert out_fallback["calibration_applied"] is True
    assert out_fallback["calibration_segment"] is None

    metrics = get_direction_head_calibration_metrics()
    assert metrics["total"] >= 2
    assert metrics["segment_hits"] >= 1
    assert metrics["fallback_hits"] >= 1


def test_rr_unit_handling_points_vs_decimal_consistency():
    common = dict(
        final_flow_signal="BULLISH_FLOW",
        spot_vs_flip="ABOVE_FLIP",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="NONE",
        gamma_regime="NEGATIVE_GAMMA",
        macro_regime="RISK_OFF",
        volatility_regime="VOL_EXPANSION",
        oi_velocity_score=0.0,
        rr_momentum="STABLE",
        volume_pcr_atm=1.0,
        hybrid_move_probability=0.55,
        vote_bull_probability=0.55,
        provider_health_summary="GOOD",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.8,
        core_one_sided_quote_ratio=0.0,
        core_quote_integrity_health="GOOD",
        apply_calibration=False,
    )

    out_points = compute_direction_probability_head(rr_value=-2.0, rr_unit="VOL_POINTS", **common)
    out_decimal = compute_direction_probability_head(rr_value=-0.02, rr_unit="DECIMAL", **common)
    assert out_points["probability_up_raw"] == out_decimal["probability_up_raw"]


def test_rr_points_transform_monotonic_without_early_saturation():
    common = dict(
        final_flow_signal="NEUTRAL_FLOW",
        spot_vs_flip="AT_FLIP",
        hedging_bias="PINNING",
        gamma_event="NONE",
        gamma_regime="NEUTRAL_GAMMA",
        macro_regime="RISK_OFF",
        volatility_regime="NORMAL_VOL",
        oi_velocity_score=0.0,
        rr_momentum="STABLE",
        volume_pcr_atm=1.0,
        hybrid_move_probability=0.5,
        vote_bull_probability=0.5,
        provider_health_summary="GOOD",
        provider_health_blocking_status="PASS",
        core_effective_priced_ratio=0.9,
        core_one_sided_quote_ratio=0.0,
        core_quote_integrity_health="GOOD",
        apply_calibration=False,
    )

    p_neg_small = compute_direction_probability_head(rr_value=-0.5, rr_unit="VOL_POINTS", **common)["probability_up_raw"]
    p_neg_big = compute_direction_probability_head(rr_value=-1.5, rr_unit="VOL_POINTS", **common)["probability_up_raw"]
    p_pos_small = compute_direction_probability_head(rr_value=0.5, rr_unit="VOL_POINTS", **common)["probability_up_raw"]
    p_pos_big = compute_direction_probability_head(rr_value=1.5, rr_unit="VOL_POINTS", **common)["probability_up_raw"]

    assert p_neg_big > p_neg_small
    assert p_pos_big < p_pos_small
