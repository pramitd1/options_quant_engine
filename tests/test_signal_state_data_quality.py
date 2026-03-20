from __future__ import annotations

from engine.trading_support.signal_state import _compute_data_quality


def test_data_quality_provider_health_is_case_insensitive():
    quality = _compute_data_quality(
        spot_validation={"is_valid": True, "is_stale": False},
        option_chain_validation={
            "is_valid": True,
            "is_stale": False,
            "provider_health": {"summary_status": "caution"},
        },
        analytics_state={
            "flip": 23000,
            "gamma_regime": "SHORT_GAMMA_ZONE",
            "final_flow_signal": "BULLISH_FLOW",
            "dealer_pos": "Short Gamma",
            "hedging_bias": "UPSIDE_ACCELERATION",
            "vol_regime": "VOL_EXPANSION",
        },
        probability_state={
            "rule_move_probability": 0.55,
            "ml_move_probability": 0.52,
            "hybrid_move_probability": 0.54,
        },
    )

    assert "provider_health_caution" in quality["reasons"]
