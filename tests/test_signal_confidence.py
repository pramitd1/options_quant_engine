from __future__ import annotations

from analytics.signal_confidence import compute_signal_confidence


def _high_confidence_trade(**overrides):
    trade = {
        "trade_status": "TRADE",
        "direction": "CALL",
        "trade_strength": 94,
        "hybrid_move_probability": 0.86,
        "confirmation_status": "STRONG_CONFIRMATION",
        "confirmation_breakdown": {"flow": 1.0, "dealer": 1.0, "macro": 1.0},
        "macro_regime": "RISK_ON",
        "global_risk_state": "LOW_RISK",
        "market_volatility_shock_score": 4,
        "gamma_vol_acceleration_score_normalized": 6,
        "data_quality_status": "GOOD",
        "provider_health_summary": "GOOD",
        "provider_health": {"summary_status": "GOOD", "market_data_readiness_score": 92.0},
        "option_efficiency_score": 92,
        "premium_efficiency_score": 90,
        "net_oi_change_bias": -100.0,
        "dealer_hedging_bias": "UPSIDE_ACCELERATION",
        "ta_direction": "CALL",
        "ta_confidence": 0.86,
        "gamma_regime": "NEGATIVE_GAMMA",
        "volatility_regime": "VOL_EXPANSION",
        "runtime_thresholds": {
            "confidence_guard_min_calibration_trades": 80,
            "confidence_guard_max_calibration_staleness_days": 5,
        },
    }
    trade.update(overrides)
    return trade


def test_signal_strength_component_uses_0_to_100_trade_strength_scale():
    base_trade = {
        "trade_strength": 80,
        "hybrid_move_probability": 0.50,
        "confirmation_status": "CONFIRMED",
        "confirmation_breakdown": {"a": 1.0, "b": 1.0},
        "macro_regime": "MACRO_NEUTRAL",
        "global_risk_state": "MODERATE_RISK",
        "market_volatility_shock_score": 20,
        "gamma_vol_acceleration_score_normalized": 35,
        "data_quality_status": "GOOD",
        "provider_health": {"summary_status": "GOOD"},
        "option_efficiency_score": 60,
        "premium_efficiency_score": 60,
    }

    result = compute_signal_confidence(base_trade)

    # With 0-100 normalization, component should stay below saturation for
    # trade_strength=80 and hybrid probability=0.50.
    assert result["signal_strength_component"] == 68.0


def test_confidence_recalibration_caps_watchlist_with_provider_and_reason_blocks():
    trade = {
        "trade_status": "WATCHLIST",
        "direction": "PUT",
        "trade_strength": 95,
        "hybrid_move_probability": 0.92,
        "confirmation_status": "STRONG_CONFIRMATION",
        "confirmation_breakdown": {"a": 2.0, "b": 1.0},
        "macro_regime": "RISK_ON",
        "global_risk_state": "LOW_RISK",
        "market_volatility_shock_score": 5,
        "gamma_vol_acceleration_score_normalized": 8,
        "data_quality_status": "GOOD",
        "provider_health_summary": "WEAK",
        "option_efficiency_score": 92,
        "premium_efficiency_score": 90,
        "no_trade_reason_code": "UPSTREAM_BLOCK",
    }

    result = compute_signal_confidence(trade)
    assert result["confidence_score"] <= 42.0
    assert "status_watchlist_or_blocked" in result["confidence_recalibration_guards"]
    assert "provider_health_weak" in result["confidence_recalibration_guards"]
    assert "explicit_no_trade_reason" in result["confidence_recalibration_guards"]


def test_confidence_recalibration_caps_direction_unresolved_conflict_setup():
    trade = {
        "trade_status": "WATCHLIST",
        "direction": None,
        "trade_strength": 88,
        "hybrid_move_probability": 0.70,
        "confirmation_status": "NO_DIRECTION",
        "data_quality_status": "CAUTION",
        "provider_health_summary": "CAUTION",
        "option_efficiency_score": 85,
        "premium_efficiency_score": 80,
    }

    result = compute_signal_confidence(trade)
    assert result["confidence_score"] <= 52.0
    assert "direction_unresolved" in result["confidence_recalibration_guards"]
    assert "confirmation_no_direction" in result["confidence_recalibration_guards"]


def test_feature_reliability_weights_reduce_confidence_components():
    strong_trade = {
        "trade_status": "TRADE",
        "direction": "CALL",
        "trade_strength": 88,
        "hybrid_move_probability": 0.76,
        "confirmation_status": "STRONG_CONFIRMATION",
        "confirmation_breakdown": {"flow": 1.0, "dealer": 1.0},
        "macro_regime": "RISK_ON",
        "global_risk_state": "LOW_RISK",
        "market_volatility_shock_score": 8,
        "gamma_vol_acceleration_score_normalized": 14,
        "data_quality_status": "GOOD",
        "provider_health_summary": "GOOD",
        "option_efficiency_score": 84,
        "premium_efficiency_score": 82,
        "feature_reliability_weights": {
            "flow": 1.0,
            "vol_surface": 1.0,
            "greeks": 1.0,
            "liquidity": 1.0,
            "macro": 1.0,
        },
    }
    fragile_trade = dict(strong_trade)
    fragile_trade["feature_reliability_weights"] = {
        "flow": 0.35,
        "vol_surface": 0.25,
        "greeks": 0.30,
        "liquidity": 0.28,
        "macro": 0.80,
    }

    strong = compute_signal_confidence(strong_trade)
    fragile = compute_signal_confidence(fragile_trade)

    assert fragile["confidence_score"] < strong["confidence_score"]
    assert fragile["signal_strength_component"] < strong["signal_strength_component"]
    assert fragile["market_stability_component"] < strong["market_stability_component"]
    assert fragile["option_efficiency_component"] < strong["option_efficiency_component"]


def test_directional_bias_correction_penalizes_conflicting_put_pressure_for_call_signals():
    trade = {
        "trade_status": "TRADE",
        "direction": "CALL",
        "trade_strength": 88,
        "hybrid_move_probability": 0.76,
        "confirmation_status": "STRONG_CONFIRMATION",
        "confirmation_breakdown": {"flow": 1.0, "dealer": 1.0},
        "macro_regime": "RISK_ON",
        "global_risk_state": "LOW_RISK",
        "market_volatility_shock_score": 8,
        "gamma_vol_acceleration_score_normalized": 14,
        "data_quality_status": "GOOD",
        "provider_health_summary": "GOOD",
        "option_efficiency_score": 84,
        "premium_efficiency_score": 82,
        "net_oi_change_bias": 120.0,
        "dealer_hedging_bias": "DOWNSIDE_ACCELERATION",
        "ta_direction": "PUT",
    }

    result = compute_signal_confidence(trade)

    assert result["directional_bias_component"] < 50.0
    assert result["directional_bias_multiplier"] < 1.0
    assert "directional_bias_correction" in result["confidence_recalibration_guards"]


def test_calibration_guardrail_caps_thin_live_calibration_sample():
    trade = _high_confidence_trade(
        live_calibration_gate={
            "ok": True,
            "verdict": "CAUTION",
            "completed_trades": 12,
            "reason": "insufficient_completed_trades",
        }
    )

    result = compute_signal_confidence(trade)

    assert result["confidence_score"] <= 66.0
    assert result["calibration_status"] == "CAUTION"
    assert result["calibration_sample_size"] == 12
    assert "calibration_sample_insufficient" in result["confidence_recalibration_guards"]
    assert result["calibration_guardrail"]["min_sample_size"] == 80


def test_calibration_guardrail_caps_stale_completed_trade_history():
    trade = _high_confidence_trade(
        live_calibration_gate={
            "ok": True,
            "verdict": "CAUTION",
            "completed_trades": 160,
            "reason": "stale_completed_trade_history",
            "days_since_last_completed_trade": 9.5,
        }
    )

    result = compute_signal_confidence(trade)

    assert result["confidence_score"] <= 64.0
    assert result["calibration_status"] == "WEAK"
    assert result["calibration_guardrail"]["recency_days"] == 9.5
    assert "calibration_history_stale" in result["confidence_recalibration_guards"]


def test_calibration_guardrail_caps_mismatched_score_calibration_segment():
    trade = _high_confidence_trade(
        runtime_composite_score=82,
        score_calibration_enabled=True,
        score_calibration_applied=True,
        score_calibration_segment_key="direction=PUT",
        score_calibration_segment_context={"direction": "PUT"},
    )

    result = compute_signal_confidence(trade)

    assert result["confidence_score"] <= 60.0
    assert result["calibration_status"] == "WEAK"
    assert result["calibration_regime_match"] == "MISMATCH"
    assert "score_calibration_segment_mismatch" in result["confidence_recalibration_guards"]


def test_calibration_guardrail_reports_healthy_full_segment_without_capping():
    trade = _high_confidence_trade(
        runtime_composite_score=82,
        score_calibration_enabled=True,
        score_calibration_applied=True,
        score_calibration_segment_key="direction=CALL|gamma_regime=NEGATIVE_GAMMA|vol_regime=VOL_EXPANSION",
        score_calibration_segment_context={
            "direction": "CALL",
            "gamma_regime": "NEGATIVE_GAMMA",
            "vol_regime": "VOL_EXPANSION",
        },
        live_calibration_gate={
            "ok": True,
            "verdict": "GO",
            "completed_trades": 180,
            "ece": 0.05,
            "brier": 0.18,
            "top_decile_overconfidence": 0.04,
        },
        regime_segment_guard={
            "enabled": True,
            "verdict": "PASS",
            "sample_size": 96,
            "segment_key": "direction=CALL|gamma_regime=NEGATIVE_GAMMA|vol_regime=VOL_EXPANSION",
        },
    )

    result = compute_signal_confidence(trade)

    assert result["calibration_status"] == "PASS"
    assert result["calibration_regime_match"] == "FULL"
    assert result["calibration_guardrail"]["confidence_cap"] is None
    assert "score_calibration_segment_mismatch" not in result["confidence_recalibration_guards"]
