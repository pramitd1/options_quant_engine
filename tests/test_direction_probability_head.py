from __future__ import annotations

import json
from pathlib import Path

from strategy.direction_probability_head import compute_direction_probability_head
from strategy.score_calibration import ScoreCalibrator


def test_direction_probability_head_outputs_probabilities_and_uncertainty():
    out = compute_direction_probability_head(
        final_flow_signal="BULLISH_FLOW",
        spot_vs_flip="ABOVE_FLIP",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_event="GAMMA_SQUEEZE",
        gamma_regime="NEGATIVE_GAMMA",
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
