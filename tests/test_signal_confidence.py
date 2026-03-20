from __future__ import annotations

from analytics.signal_confidence import compute_signal_confidence


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
    assert "confirmation_conflict_or_no_direction" in result["confidence_recalibration_guards"]
