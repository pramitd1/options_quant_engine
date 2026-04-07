from __future__ import annotations

from datetime import datetime

import engine.trading_support.signal_state as signal_state


def _base_market_state() -> dict:
    return {
        "dealer_pos": "SHORT_GAMMA",
        "vol_regime": "ELEVATED",
        "gamma_regime": "NEGATIVE_GAMMA",
        "gamma_event": "GAMMA_SQUEEZE",
        "probability_confidence": "HIGH",
        "price_move_15m_pct": 0.25,
        "oi_change_pct": 4.0,
        "underlying_price": 100.0,
        "open_price": 99.0,
        "last_updated": datetime.now(),
        "position_score": 1.2,
        "hybrid_move_probability": 0.64,
        "confidence_score": 78,
        "gamma_acceleration": "UP",
        "squeeze_signal": "CALL_SQUEEZE",
        "max_pain": 95.0,
        "max_pain_distance_pct": 5.0,
        "max_pain_magnet_score": 0.25,
        "max_pain_bias": "BULLISH",
        "flow_imbalance": "CALL_PRESSURE",
        "flow_intensity": 0.72,
        "smart_money_flow_signal": "BULLISH_SWEEP",
        "aggressive_flow_ratio": 0.67,
        "smart_money_conviction": 0.7,
        "liquidity_regime": "NORMAL",
        "liquidity_warning": False,
        "vacuum_state": "STABLE",
        "intraday_gamma_state": "STABLE",
        "support_wall": 98.0,
        "resistance_wall": 102.0,
        "final_flow_signal": "BULLISH_FLOW",
        "direction_signal": "UP",
        "gamma_flip": 98.0,
        "spot_vs_flip": "ABOVE_FLIP",
        "gamma_flip_distance_pct": 2.0,
        "gamma_flip_context": "ABOVE_FLIP_STABLE",
        "gamma_flip_drift_5m": 0.15,
        "wall_density": "HIGH",
        "hedging_pressure": "ACCELERATING",
        "hedging_bias": "UPSIDE_ACCELERATION",
        "hedging_intensity": 0.75,
        "hedging_speed": 0.8,
        "hedging_acceleration": 0.15,
        "rr_signal": "BULLISH",
        "rr_value": -0.7,
        "rr_momentum": "FALLING_PUT_SKEW",
        "rr_stretch": 0.4,
        "oi_velocity_signal": "BUILDUP",
        "oi_velocity_score": 0.3,
        "oi_velocity_put_call_skew": 0.82,
        "pcr_signal": "BULLISH",
        "volume_pcr_atm": 0.79,
        "volume_pcr_total": 0.84,
        "volume_pcr_bias": "CALL_HEAVY",
        "recommended_ttl_minutes": 45,
        "regime": "HIGH_CONFIDENCE",
        "final_signal": "BUY_CALL",
        "data_quality": "LIVE",
        "provider_health_status": "GOOD",
        "provider_health_reasons": [],
        "provider_health_symbol_reasons": {},
        "source": "LIVE",
        "flow_signal_value": 0.7,
        "smart_money_signal_value": 0.66,
        "void_signal": "NONE",
        "max_pain_dist": 3.0,
        "max_pain_zone": "NEAR",
        "days_to_expiry": 0,
        "volume_pcr_regime": "NEUTRAL",
        "greek_exposures": {
            "vanna_regime": "NEUTRAL",
            "charm_regime": "NEUTRAL",
        },
        "dealer_liquidity_map": {
            "next_support": 98.0,
            "next_resistance": 102.0,
            "gamma_squeeze_zone": "UP",
        },
    }


def _base_option_chain_validation() -> dict:
    return {
        "provider_health": {
            "summary_status": "GOOD",
            "trade_blocking_status": "PASS",
            "core_effective_priced_ratio": 0.8,
            "core_one_sided_quote_ratio": 0.08,
            "core_quote_integrity_health": "GOOD",
        }
    }


def test_direction_head_can_set_direction_without_vote(monkeypatch):
    def _mock_decide_direction(**kwargs):
        return (None, None, 0.5, 0.5, False, None, 0.0)

    monkeypatch.setattr(signal_state, "decide_direction", _mock_decide_direction)
    monkeypatch.setattr(signal_state, "compute_trade_strength", lambda **kwargs: (68, {}))
    monkeypatch.setattr(
        signal_state,
        "compute_confirmation_filters",
        lambda **kwargs: {"status": "CONFIRMED", "veto": False, "reasons": [], "score_adjustment": 0, "breakdown": {}},
    )
    monkeypatch.setattr(
        signal_state,
        "get_trade_runtime_thresholds",
        lambda: {
            "enable_probabilistic_direction_head": 1,
            "direction_head_call_threshold": 0.55,
            "direction_head_put_threshold": 0.45,
            "direction_head_min_confidence": 0.10,
            "direction_head_allow_vote_override": 1,
            "direction_head_override_min_confidence": 0.55,
            "direction_probability_calibrator_path": "",
            "reversal_stage_min_vote_count": 3,
            "reversal_stage_min_breakout_votes": 1,
        },
    )
    monkeypatch.setattr(
        signal_state,
        "compute_direction_probability_head",
        lambda **kwargs: {
            "probability_up": 0.72,
            "probability_up_raw": 0.70,
            "probability_down": 0.28,
            "confidence": 0.80,
            "uncertainty": 0.20,
            "disagreement_with_vote": 0.22,
            "microstructure_friction_score": 0.15,
            "calibration_applied": False,
        },
    )

    out = signal_state._compute_signal_state(
        spot=100.0,
        symbol="NIFTY",
        previous_direction=None,
        reversal_age=None,
        day_open=99.0,
        prev_close=98.5,
        intraday_range_pct=1.2,
        backtest_mode=False,
        market_state=_base_market_state(),
        probability_state={
            "hybrid_move_probability": 0.64,
            "ml_move_probability": 0.59,
            "components": {"gamma_flip_distance_pct": 2.0},
        },
        option_chain_validation=_base_option_chain_validation(),
    )

    assert out["direction"] == "CALL"
    assert out["direction_source"] == "DIRECTION_HEAD"
    assert out["direction_vote_shadow"] is None
    assert out["direction_head_probability_up"] >= 0.55


def test_direction_head_can_override_vote_when_confident(monkeypatch):
    def _mock_decide_direction(**kwargs):
        return ("PUT", "FLOW", 0.42, 0.58, False, None, 0.0)

    monkeypatch.setattr(signal_state, "decide_direction", _mock_decide_direction)
    monkeypatch.setattr(signal_state, "compute_trade_strength", lambda **kwargs: (68, {}))
    monkeypatch.setattr(
        signal_state,
        "compute_confirmation_filters",
        lambda **kwargs: {"status": "CONFIRMED", "veto": False, "reasons": [], "score_adjustment": 0, "breakdown": {}},
    )
    monkeypatch.setattr(
        signal_state,
        "get_trade_runtime_thresholds",
        lambda: {
            "enable_probabilistic_direction_head": 1,
            "direction_head_call_threshold": 0.55,
            "direction_head_put_threshold": 0.45,
            "direction_head_min_confidence": 0.10,
            "direction_head_allow_vote_override": 1,
            "direction_head_override_min_confidence": 0.10,
            "direction_probability_calibrator_path": "",
            "reversal_stage_min_vote_count": 3,
            "reversal_stage_min_breakout_votes": 1,
        },
    )
    monkeypatch.setattr(
        signal_state,
        "compute_direction_probability_head",
        lambda **kwargs: {
            "probability_up": 0.74,
            "probability_up_raw": 0.72,
            "probability_down": 0.26,
            "confidence": 0.85,
            "uncertainty": 0.15,
            "disagreement_with_vote": 0.32,
            "microstructure_friction_score": 0.12,
            "calibration_applied": False,
        },
    )

    out = signal_state._compute_signal_state(
        spot=100.0,
        symbol="NIFTY",
        previous_direction=None,
        reversal_age=None,
        day_open=99.0,
        prev_close=98.5,
        intraday_range_pct=1.2,
        backtest_mode=False,
        market_state=_base_market_state(),
        probability_state={
            "hybrid_move_probability": 0.64,
            "ml_move_probability": 0.59,
            "components": {"gamma_flip_distance_pct": 2.0},
        },
        option_chain_validation=_base_option_chain_validation(),
    )

    assert out["direction_vote_shadow"] == "PUT"
    assert out["direction"] == "CALL"
    assert out["direction_source"] == "FLOW+DIRECTION_HEAD_OVERRIDE"
    assert float(out["direction_head_disagreement_with_vote"] or 0.0) > 0.0


def test_direction_head_calibration_segment_is_propagated(monkeypatch):
    def _mock_decide_direction(**kwargs):
        return ("CALL", "FLOW", 0.58, 0.42, False, None, 0.0)

    monkeypatch.setattr(signal_state, "decide_direction", _mock_decide_direction)
    monkeypatch.setattr(signal_state, "compute_trade_strength", lambda **kwargs: (68, {}))
    monkeypatch.setattr(
        signal_state,
        "compute_confirmation_filters",
        lambda **kwargs: {"status": "CONFIRMED", "veto": False, "reasons": [], "score_adjustment": 0, "breakdown": {}},
    )
    monkeypatch.setattr(
        signal_state,
        "get_trade_runtime_thresholds",
        lambda: {
            "enable_probabilistic_direction_head": 1,
            "direction_head_call_threshold": 0.55,
            "direction_head_put_threshold": 0.45,
            "direction_head_min_confidence": 0.10,
            "direction_head_allow_vote_override": 1,
            "direction_head_override_min_confidence": 0.10,
            "direction_probability_calibrator_path": "",
            "direction_probability_segmented_calibrator_path": "",
            "reversal_stage_min_vote_count": 3,
            "reversal_stage_min_breakout_votes": 1,
        },
    )
    monkeypatch.setattr(
        signal_state,
        "compute_direction_probability_head",
        lambda **kwargs: {
            "probability_up": 0.74,
            "probability_up_raw": 0.72,
            "probability_down": 0.26,
            "confidence": 0.85,
            "uncertainty": 0.15,
            "disagreement_with_vote": 0.16,
            "microstructure_friction_score": 0.12,
            "calibration_applied": True,
            "calibration_segment": "RISK_OFF",
            "calibration_segment_metrics": {
                "total": 10,
                "segment_hits": 7,
                "fallback_hits": 3,
                "segment_hit_rate": 0.7,
                "fallback_rate": 0.3,
            },
        },
    )

    out = signal_state._compute_signal_state(
        spot=100.0,
        symbol="NIFTY",
        previous_direction=None,
        reversal_age=None,
        day_open=99.0,
        prev_close=98.5,
        intraday_range_pct=1.2,
        backtest_mode=False,
        market_state=_base_market_state(),
        probability_state={
            "hybrid_move_probability": 0.64,
            "ml_move_probability": 0.59,
            "components": {"gamma_flip_distance_pct": 2.0},
        },
        option_chain_validation=_base_option_chain_validation(),
    )

    assert out["direction_head_calibration_segment"] == "RISK_OFF"
