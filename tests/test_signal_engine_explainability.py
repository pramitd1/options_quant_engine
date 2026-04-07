from __future__ import annotations

from engine.signal_engine import _build_decision_explainability, _collect_neutralization_states


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
