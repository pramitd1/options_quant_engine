from __future__ import annotations

import pandas as pd

from engine.signal_engine import (
    _build_decision_explainability,
    _collect_neutralization_states,
    _evaluate_historical_outcome_guard,
    _evaluate_regime_segment_guard,
    _evaluate_session_risk_governor,
    _evaluate_trade_slot_governor,
    _evaluate_trade_promotion_governor,
)


def test_directionless_two_sided_setup_is_explicitly_ambiguous_watchlist():
    payload = {
        "direction": None,
        "flow_signal": "BEARISH_FLOW",
        "smart_money_flow": "NEUTRAL_FLOW",
        "confirmation_status": "NO_DIRECTION",
        "signal_quality": "VERY_WEAK",
        "directional_convexity_state": "TWO_SIDED_VOLATILITY_RISK",
        "dealer_flow_state": "PINNING_DOMINANT",
        "dealer_hedging_bias": "DOWNSIDE_PINNING",
        "trade_strength": 24,
        "support_wall": 23000,
        "resistance_wall": 23250,
        "gamma_flip": 23120,
        "spot": 23110,
        "macro_adjustment_reasons": ["macro_news_neutral_fallback"],
        "global_risk_diagnostics": {"fallback": True},
        "expected_move_points": None,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="NO_SIGNAL",
        min_trade_strength=45,
    )

    assert explainability["decision_classification"] == "DIRECTIONALLY_AMBIGUOUS"
    assert explainability["setup_state"] == "DIRECTION_PENDING"
    assert explainability["watchlist_flag"] is True
    assert explainability["no_trade_reason_code"] == "TWO_SIDED_VOLATILITY_WITHOUT_EDGE"
    assert "missing_directional_consensus" in explainability["missing_signal_requirements"]
    assert "insufficient_trade_strength" in explainability["missing_signal_requirements"]


def test_data_invalid_maps_to_data_blocked_taxonomy():
    payload = {
        "direction": None,
        "signal_quality": "VERY_WEAK",
        "trade_strength": 0,
        "spot": 0,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="DATA_INVALID",
        min_trade_strength=45,
    )

    assert explainability["decision_classification"] == "DATA_BLOCKED"
    assert explainability["setup_state"] == "DATA_BLOCKED"
    assert explainability["no_trade_reason_code"] == "DATA_QUALITY_INSUFFICIENT"
    assert "data_quality" in explainability["blocked_by"]


def test_trade_ready_case_keeps_no_trade_fields_empty():
    payload = {
        "direction": "CALL",
        "flow_signal": "BULLISH_FLOW",
        "smart_money_flow": "BULLISH_FLOW",
        "confirmation_status": "CONFIRMED",
        "signal_quality": "STRONG",
        "trade_strength": 78,
        "spot": 23000,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="TRADE",
        min_trade_strength=45,
    )

    assert explainability["decision_classification"] == "TRADE_READY"
    assert explainability["setup_state"] == "NONE"
    assert explainability["watchlist_flag"] is False
    assert explainability["no_trade_reason"] is None


def test_directionless_low_activation_setup_is_classified_as_inactive():
    payload = {
        "direction": None,
        "flow_signal": "NEUTRAL_FLOW",
        "smart_money_flow": "NEUTRAL_FLOW",
        "confirmation_status": "NO_DIRECTION",
        "signal_quality": "VERY_WEAK",
        "directional_convexity_state": "NO_CONVEXITY_EDGE",
        "dealer_flow_state": "HEDGING_NEUTRAL",
        "trade_strength": 7,
        "spot": 23100,
        "hybrid_move_probability": 0.42,
        "expected_move_points": None,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="NO_SIGNAL",
        min_trade_strength=45,
    )

    assert explainability["decision_classification"] == "DEAD_INACTIVE"
    assert explainability["watchlist_flag"] is False
    assert explainability["setup_activation_score"] < 35
    assert explainability["explainability_confidence"] in {"LOW", "MEDIUM", "HIGH"}


def test_direction_none_with_confirmed_status_is_marked_as_contradiction():
    payload = {
        "direction": None,
        "flow_signal": "BULLISH_FLOW",
        "smart_money_flow": "BULLISH_FLOW",
        "confirmation_status": "CONFIRMED",
        "signal_quality": "MEDIUM",
        "directional_convexity_state": "NO_CONVEXITY_EDGE",
        "dealer_flow_state": "UPSIDE_HEDGING_ACCELERATION",
        "trade_strength": 38,
        "spot": 23120,
        "hybrid_move_probability": 0.57,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="NO_SIGNAL",
        min_trade_strength=45,
    )

    assert "direction_confirmation_conflict" in explainability["missing_signal_requirements"]
    assert any("direction is unresolved" in detail for detail in explainability["no_trade_reason_details"])


def test_watchlist_provider_health_classification_does_not_depend_on_message_text():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 73,
        "confirmation_status": "STRONG_CONFIRMATION",
        "provider_health_summary": "CAUTION",
        "data_quality_status": "GOOD",
        "message": "Execution routed to watchlist",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert explainability["no_trade_reason_code"] == "PROVIDER_HEALTH_CAUTION_BLOCK"
    assert "provider_health" in explainability["blocked_by"]


def test_watchlist_keeps_primary_reason_and_tracks_low_strength_as_secondary_detail():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 48,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "WEAK",
        "data_quality_status": "GOOD",
        "message": "Provider health weak blocks trade execution",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["no_trade_reason_code"] == "PROVIDER_HEALTH_WEAK_BLOCK"
    details = explainability["no_trade_reason_details"]
    assert any("secondary_blocker" in detail for detail in details)


def test_watchlist_uses_structured_provider_health_reason_code_when_summary_is_missing():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 72,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": None,
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "PROVIDER_HEALTH_WEAK_BLOCK",
        "message": "Routed to watchlist",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert explainability["setup_state"] == "RISK_BLOCKED"
    assert explainability["no_trade_reason_code"] == "PROVIDER_HEALTH_WEAK_BLOCK"
    assert "provider_health" in explainability["blocked_by"]


def test_watchlist_low_strength_message_does_not_override_provider_health_block():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 48,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "WEAK",
        "data_quality_status": "GOOD",
        "message": "LOW STRENGTH watchlist route",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert explainability["setup_state"] == "RISK_BLOCKED"
    assert explainability["no_trade_reason_code"] == "PROVIDER_HEALTH_WEAK_BLOCK"
    assert "provider_health" in explainability["blocked_by"]


def test_preserves_preexisting_no_trade_reason_code_and_reason_from_payload():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 52,
        "confirmation_status": "CONFLICT",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "UPSTREAM_REASON_CODE",
        "no_trade_reason": "Upstream policy vetoed this trade",
        "message": "watchlist",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["no_trade_reason_code"] == "UPSTREAM_REASON_CODE"
    assert explainability["no_trade_reason"] == "Upstream policy vetoed this trade"


def test_option_efficiency_neutralization_uses_feature_state_not_payload_field():
    payload = {
        "expected_move_points": None,
        "option_efficiency_features": {
            "neutral_fallback": False,
            "expected_move_quality": "DIRECT",
            "expected_move_points": 142.7,
        },
        "option_efficiency_diagnostics": {"warnings": []},
        "option_efficiency_reasons": ["option_efficiency_balanced"],
    }

    neutralization = _collect_neutralization_states(payload)

    assert neutralization["option_efficiency_status"] == "AVAILABLE"
    assert neutralization["option_efficiency_reason"] == "features_available"


def test_option_efficiency_neutralization_marks_missing_features_unavailable():
    payload = {
        "expected_move_points": 115.0,
        "option_efficiency_features": {},
        "option_efficiency_diagnostics": {"warnings": []},
        "option_efficiency_reasons": [],
    }

    neutralization = _collect_neutralization_states(payload)

    assert neutralization["option_efficiency_status"] == "UNAVAILABLE_NEUTRALIZED"
    assert neutralization["option_efficiency_reason"] == "option_efficiency_features_missing"


def test_historical_outcome_guard_blocks_fragile_regime_profile():
    payload = {
        "symbol": "NIFTY",
        "direction": "CALL",
        "gamma_regime": "POSITIVE_GAMMA",
        "macro_regime": "MACRO_NEUTRAL",
        "spot_vs_flip": "ABOVE_FLIP",
        "signal_regime": "EXPANSION_BIAS",
    }
    history = pd.DataFrame(
        [
            {
                "symbol": "NIFTY",
                "direction": "CALL",
                "gamma_regime": "POSITIVE_GAMMA",
                "macro_regime": "MACRO_NEUTRAL",
                "spot_vs_flip": "ABOVE_FLIP",
                "signal_regime": "EXPANSION_BIAS",
                "signed_return_60m_bps": 32.0,
                "signed_return_session_close_bps": -18.0,
                "tradeability_score": 41.0,
                "horizon_edge_label": "EARLY_ALPHA_DECAY",
                "exit_quality_label": "MISSED_EXIT",
                "correct_60m": 1,
                "correct_session_close": 0,
            }
            for _ in range(12)
        ]
    )

    guard = _evaluate_historical_outcome_guard(
        payload=payload,
        history_frame=history,
        runtime_thresholds={
            "enable_historical_outcome_guard": 1,
            "historical_outcome_guard_min_samples": 8,
            "historical_outcome_guard_min_tradeability_score": 55.0,
            "historical_outcome_guard_min_session_close_bps": -5.0,
            "historical_outcome_guard_early_decay_share_threshold": 0.50,
            "historical_outcome_guard_stopout_share_threshold": 0.30,
        },
    )

    assert guard["verdict"] == "BLOCK"
    assert guard["exit_bias"] == "TAKE_PROFIT_EARLY"
    assert guard["best_horizon"] in {"5m", "15m", "30m", "60m"}


def test_watchlist_surfaces_historical_outcome_guard_as_blocker():
    payload = {
        "direction": "CALL",
        "signal_quality": "MEDIUM",
        "trade_strength": 72,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "HISTORICAL_OUTCOME_GUARD",
        "no_trade_reason": "Historical outcome guard downgraded this setup",
        "historical_outcome_guard": {
            "verdict": "BLOCK",
            "reason": "Historical outcome guard: similar setups have weak realized tradeability",
            "sample_size": 18,
            "best_horizon": "15m",
        },
        "message": "Historical outcome guard routed TRADE to WATCHLIST",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "historical_outcome_guard" in explainability["blocked_by"]
    assert any("historical outcome guard" in detail.lower() for detail in explainability["no_trade_reason_details"])


def test_regime_segment_guard_blocks_underperforming_segment():
    payload = {
        "symbol": "NIFTY",
        "direction": "PUT",
        "gamma_regime": "NEGATIVE_GAMMA",
        "volatility_regime": "VOL_EXPANSION",
        "macro_regime": "RISK_OFF",
        "score_calibration_segment_context": {
            "direction": "PUT",
            "gamma_regime": "NEGATIVE_GAMMA",
            "vol_regime": "VOL_EXPANSION",
        },
    }
    history = pd.DataFrame(
        [
            {
                "symbol": "NIFTY",
                "direction": "PUT",
                "gamma_regime": "NEGATIVE_GAMMA",
                "volatility_regime": "VOL_EXPANSION",
                "macro_regime": "RISK_OFF",
                "tradeability_score": 42.0,
                "signed_return_60m_bps": -12.0,
                "signed_return_session_close_bps": -28.0,
                "correct_60m": 0,
            }
            for _ in range(14)
        ]
    )

    guard = _evaluate_regime_segment_guard(
        payload=payload,
        history_frame=history,
        runtime_thresholds={
            "enable_regime_segment_guard": 1,
            "regime_segment_guard_min_samples": 10,
            "regime_segment_guard_min_hit_rate_60m": 0.48,
            "regime_segment_guard_min_tradeability_score": 55.0,
            "regime_segment_guard_min_avg_close_bps": -5.0,
        },
    )

    assert guard["verdict"] == "BLOCK"
    assert guard["sample_size"] == 14
    assert "NEGATIVE_GAMMA" in str(guard["segment_key"])


def test_watchlist_surfaces_regime_segment_guard_as_blocker():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 74,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "REGIME_SEGMENT_GUARD",
        "no_trade_reason": "Regime segment guard downgraded this setup",
        "regime_segment_guard": {
            "verdict": "BLOCK",
            "reason": "Segment underperforms historically",
            "sample_size": 14,
            "segment_key": "direction=PUT|gamma_regime=NEGATIVE_GAMMA|vol_regime=VOL_EXPANSION",
        },
        "message": "Regime segment guard routed TRADE to WATCHLIST",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "regime_segment_guard" in explainability["blocked_by"]


def test_watchlist_surfaces_portfolio_concentration_guard_as_blocker():
    payload = {
        "direction": "CALL",
        "signal_quality": "MEDIUM",
        "trade_strength": 76,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "portfolio_concentration_guard": {
            "verdict": "WATCHLIST",
            "reason": "Trade downgraded due to a concentrated same-way options book",
            "recent_signal_count": 6,
            "same_direction_count": 5,
            "same_direction_share": 0.8333,
            "heat_score": 86,
            "heat_label": "CRITICAL",
        },
        "message": "Trade downgraded due to a concentrated same-way options book",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "portfolio_concentration_guard" in explainability["blocked_by"]
    assert any("same-way options book" in detail.lower() for detail in explainability["no_trade_reason_details"])
    assert any("critical heat" in detail.lower() for detail in explainability["no_trade_reason_details"])


def test_session_risk_governor_blocks_after_recent_stopout_streak():
    payload = {
        "symbol": "NIFTY",
        "direction": "CALL",
        "valuation_time": "2026-03-14T14:30:00+05:30",
    }
    history = pd.DataFrame(
        [
            {
                "symbol": "NIFTY",
                "direction": "CALL",
                "timestamp": f"2026-03-14T10:{10 + i:02d}:00+05:30",
                "signed_return_session_close_bps": value,
                "tradeability_score": 42.0,
                "exit_quality_label": label,
            }
            for i, (value, label) in enumerate([
                (-18.0, "STOPPED_OUT"),
                (-22.0, "STOPPED_OUT"),
                (-15.0, "MISSED_EXIT"),
                (8.0, "USABLE_EXIT"),
            ])
        ]
    )

    guard = _evaluate_session_risk_governor(
        payload=payload,
        history_frame=history,
        runtime_thresholds={
            "enable_session_risk_governor": 1,
            "session_risk_lookback_signals": 4,
            "session_risk_min_samples": 3,
            "session_risk_max_stopout_streak": 2,
            "session_risk_max_loss_share": 0.60,
            "session_risk_min_avg_close_bps": -5.0,
            "session_risk_block_size_cap": 0.25,
            "session_risk_cooldown_minutes": 30,
        },
    )

    assert guard["verdict"] == "BLOCK"
    assert guard["stopout_streak"] >= 2
    assert guard["cooldown_active"] is True
    assert guard["budget_remaining_pct"] <= 35


def test_watchlist_surfaces_session_risk_governor_as_blocker():
    payload = {
        "direction": "PUT",
        "signal_quality": "MEDIUM",
        "trade_strength": 75,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "session_risk_governor": {
            "verdict": "BLOCK",
            "reason": "Session risk governor: recent stop-out streak and drawdown require cooldown",
            "recent_signal_count": 5,
            "stopout_streak": 3,
            "cooldown_active": True,
            "budget_remaining_pct": 20.0,
        },
        "message": "Session risk governor routed TRADE to WATCHLIST",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "session_risk_governor" in explainability["blocked_by"]
    assert any("cooldown" in detail.lower() for detail in explainability["no_trade_reason_details"])


def test_trade_slot_governor_blocks_when_symbol_book_is_full():
    payload = {
        "symbol": "NIFTY",
        "direction": "CALL",
        "valuation_time": "2026-03-14T14:30:00+05:30",
    }
    history = pd.DataFrame(
        [
            {
                "symbol": "NIFTY",
                "direction": direction,
                "timestamp": f"2026-03-14T11:{10 + i:02d}:00+05:30",
                "signed_return_session_close_bps": value,
                "trade_status": "TRADE",
            }
            for i, (direction, value) in enumerate([
                ("CALL", 12.0),
                ("CALL", -4.0),
                ("CALL", 8.0),
                ("PUT", 6.0),
            ])
        ]
    )

    guard = _evaluate_trade_slot_governor(
        payload=payload,
        history_frame=history,
        runtime_thresholds={
            "enable_trade_slot_governor": 1,
            "trade_slot_lookback_signals": 5,
            "trade_slot_min_samples": 3,
            "trade_slot_max_total_signals": 3,
            "trade_slot_max_same_direction_signals": 2,
            "trade_slot_caution_size_cap": 0.55,
            "trade_slot_override_size_cap": 0.35,
            "enable_operator_override_controls": 1,
        },
    )

    assert guard["verdict"] == "BLOCK"
    assert guard["active_signal_count"] >= 4
    assert guard["same_direction_count"] >= 3
    assert guard["operator_override_active"] is False


def test_trade_slot_governor_respects_governed_operator_override():
    payload = {
        "symbol": "NIFTY",
        "direction": "CALL",
        "valuation_time": "2026-03-14T14:30:00+05:30",
        "operator_control_state": {
            "slot_override": "ALLOW",
            "override_reason": "desk hedge already reduced elsewhere",
        },
    }
    history = pd.DataFrame(
        [
            {
                "symbol": "NIFTY",
                "direction": direction,
                "timestamp": f"2026-03-14T12:{10 + i:02d}:00+05:30",
                "signed_return_session_close_bps": value,
                "trade_status": "TRADE",
            }
            for i, (direction, value) in enumerate([
                ("CALL", 10.0),
                ("CALL", -2.0),
                ("CALL", 5.0),
                ("PUT", 4.0),
            ])
        ]
    )

    guard = _evaluate_trade_slot_governor(
        payload=payload,
        history_frame=history,
        runtime_thresholds={
            "enable_trade_slot_governor": 1,
            "trade_slot_lookback_signals": 5,
            "trade_slot_min_samples": 3,
            "trade_slot_max_total_signals": 3,
            "trade_slot_max_same_direction_signals": 2,
            "trade_slot_caution_size_cap": 0.55,
            "trade_slot_override_size_cap": 0.30,
            "enable_operator_override_controls": 1,
        },
    )

    assert guard["verdict"] == "CAUTION"
    assert guard["operator_override_active"] is True
    assert guard["size_cap"] == 0.3
    assert "override" in guard["reason"].lower()


def test_watchlist_surfaces_trade_slot_governor_as_blocker():
    payload = {
        "direction": "CALL",
        "signal_quality": "MEDIUM",
        "trade_strength": 77,
        "confirmation_status": "CONFIRMED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "TRADE_SLOT_GOVERNOR",
        "trade_slot_governor": {
            "verdict": "BLOCK",
            "reason": "Trade slot governor: symbol book already has too many same-way ideas",
            "active_signal_count": 4,
            "same_direction_count": 3,
            "operator_override_active": False,
        },
        "message": "Trade slot governor routed TRADE to WATCHLIST",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "trade_slot_governor" in explainability["blocked_by"]
    assert any("slot" in detail.lower() for detail in explainability["no_trade_reason_details"])


def test_trade_promotion_governor_blocks_when_replay_validation_is_weak():
    payload = {
        "direction": "CALL",
        "trade_strength": 69,
        "runtime_composite_score": 66,
        "signal_success_probability": 0.59,
        "confirmation_status": "MIXED",
        "data_quality_status": "GOOD",
        "live_calibration_gate": {"verdict": "BLOCK", "reason": "recent live calibration drift"},
        "live_directional_gate": {"verdict": "PASS"},
    }

    guard = _evaluate_trade_promotion_governor(
        payload=payload,
        runtime_thresholds={
            "enable_trade_promotion_governor": 1,
            "min_trade_strength": 62,
            "min_composite_score": 58,
            "trade_promotion_min_probability": 0.60,
            "trade_promotion_caution_size_cap": 0.50,
            "trade_promotion_hold_cap_minutes": 20,
            "trade_promotion_require_confirmed_status": 1,
        },
    )

    assert guard["verdict"] == "BLOCK"
    assert guard["replay_validation_required"] is True
    assert guard["promotion_state"] == "REPLAY_REQUIRED"


def test_trade_promotion_governor_passes_clean_confirmed_setup():
    payload = {
        "direction": "PUT",
        "trade_strength": 81,
        "runtime_composite_score": 74,
        "signal_success_probability": 0.68,
        "confirmation_status": "CONFIRMED",
        "data_quality_status": "GOOD",
        "live_calibration_gate": {"verdict": "PASS"},
        "live_directional_gate": {"verdict": "PASS"},
    }

    guard = _evaluate_trade_promotion_governor(
        payload=payload,
        runtime_thresholds={
            "enable_trade_promotion_governor": 1,
            "min_trade_strength": 62,
            "min_composite_score": 58,
            "trade_promotion_min_probability": 0.60,
            "trade_promotion_caution_size_cap": 0.50,
            "trade_promotion_hold_cap_minutes": 20,
            "trade_promotion_require_confirmed_status": 1,
        },
    )

    assert guard["verdict"] == "PASS"
    assert guard["replay_validation_required"] is False
    assert guard["promotion_state"] == "PROMOTE"


def test_watchlist_surfaces_trade_promotion_governor_as_blocker():
    payload = {
        "direction": "CALL",
        "signal_quality": "MEDIUM",
        "trade_strength": 76,
        "confirmation_status": "MIXED",
        "provider_health_summary": "GOOD",
        "data_quality_status": "GOOD",
        "no_trade_reason_code": "TRADE_PROMOTION_GOVERNOR",
        "trade_promotion_governor": {
            "verdict": "BLOCK",
            "reason": "Trade promotion governor: replay validation is required before live promotion",
            "promotion_state": "REPLAY_REQUIRED",
            "replay_validation_required": True,
        },
        "message": "Trade promotion governor routed TRADE to WATCHLIST",
        "spot": 23100,
    }

    explainability = _build_decision_explainability(
        payload,
        trade_status="WATCHLIST",
        min_trade_strength=60,
    )

    assert explainability["decision_classification"] == "BLOCKED_SETUP"
    assert "trade_promotion_governor" in explainability["blocked_by"]
    assert any("replay" in detail.lower() for detail in explainability["no_trade_reason_details"])
