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


def test_data_quality_score_with_none_spot_validation_is_robust():
    quality = _compute_data_quality(
        spot_validation=None,
        option_chain_validation={"is_valid": True, "is_stale": False, "provider_health": {"summary_status": "GOOD"}},
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

    assert isinstance(quality["score"], int)
    assert quality["fatal"] is False


def test_data_quality_score_missing_provider_health_and_malformed_payload():
    quality = _compute_data_quality(
        spot_validation={"is_valid": True, "is_stale": False},
        option_chain_validation={"is_valid": True, "is_stale": False, "provider_health": "bad_payload"},
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

    assert quality["fatal"] is False
    assert "provider_health_caution" not in quality["reasons"]
    assert "weak_provider_health" not in quality["reasons"]


def test_data_quality_score_missing_analytics_keys_penalizes_and_records_reason():
    quality = _compute_data_quality(
        spot_validation={"is_valid": True, "is_stale": False},
        option_chain_validation={"is_valid": True, "is_stale": False},
        analytics_state={"flip": 23000},
        probability_state={
            "rule_move_probability": 0.55,
            "ml_move_probability": 0.52,
            "hybrid_move_probability": 0.54,
        },
    )

    assert quality["score"] < 100
    assert any(str(reason).startswith("missing_critical_analytics:") for reason in quality["reasons"])
    assert quality["analytics_quality"]["critical_missing_count"] > 0


def test_data_quality_score_missing_hybrid_probability_adds_specific_reason():
    quality = _compute_data_quality(
        spot_validation={"is_valid": True, "is_stale": False},
        option_chain_validation={"is_valid": True, "is_stale": False},
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
            "hybrid_move_probability": None,
        },
    )

    assert "missing_hybrid_move_probability" in quality["reasons"]


def test_data_quality_complete_analytics_layer_failure_fallback_is_safe():
    quality = _compute_data_quality(
        spot_validation={"is_valid": True, "is_stale": False},
        option_chain_validation={"is_valid": True, "is_stale": False},
        analytics_state=None,
        probability_state={
            "rule_move_probability": 0.55,
            "ml_move_probability": 0.52,
            "hybrid_move_probability": 0.54,
        },
    )

    assert isinstance(quality, dict)
    assert quality["score"] < 100
    assert quality["fatal"] is False
