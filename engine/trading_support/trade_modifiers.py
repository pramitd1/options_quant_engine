"""
Module: trade_modifiers.py

Purpose:
    Provide trade modifiers helpers used during market-state, probability, or signal assembly.

Role in the System:
    Part of the signal engine that turns analytics, probability estimates, and overlays into final trade decisions.

Key Outputs:
    Trade decisions, intermediate state bundles, and signal diagnostics.

Downstream Usage:
    Consumed by the live runtime loop, backtests, shadow mode, and signal-evaluation logging.
"""
from __future__ import annotations

from config.global_risk_policy import get_global_risk_policy_config
from config.signal_policy import get_trade_modifier_policy_config

from .common import _clip, _safe_float


def derive_global_risk_trade_modifiers(global_risk_state):
    """
    Purpose:
        Derive global risk trade modifiers from the supplied inputs.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        global_risk_state (Any): Structured state payload for global risk.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_global_risk_policy_config()
    global_risk_state = global_risk_state if isinstance(global_risk_state, dict) else {}
    features = global_risk_state.get("global_risk_features", {})
    features = features if isinstance(features, dict) else {}

    base_adjustment_score = int(_safe_float(global_risk_state.get("global_risk_adjustment_score"), 0.0))
    feature_adjustment_score = 0
    adjustment_reasons = []

    volatility_explosion_probability = _safe_float(features.get("volatility_explosion_probability"), 0.0)
    oil_shock_score = _safe_float(features.get("oil_shock_score"), 0.0)

    # These penalties sit on top of the base risk regime classification and
    # capture specific catalysts that should further de-risk the trade.
    if volatility_explosion_probability > cfg.volatility_explosion_penalty_threshold:
        feature_adjustment_score += int(cfg.volatility_explosion_penalty_score)
        adjustment_reasons.append("volatility_explosion_probability_high")

    if oil_shock_score >= cfg.oil_shock_penalty_threshold:
        feature_adjustment_score += int(cfg.oil_shock_penalty_score)
        adjustment_reasons.append("oil_shock_score_high")

    effective_adjustment_score = base_adjustment_score + feature_adjustment_score
    overnight_hold_allowed = bool(global_risk_state.get("overnight_hold_allowed", True))

    global_risk_state_label = str(global_risk_state.get("global_risk_state", "")).upper().strip()
    force_no_trade = (
        global_risk_state_label == "EVENT_LOCKDOWN"
        or bool(global_risk_state.get("global_risk_veto", False))
    )

    return {
        "base_adjustment_score": base_adjustment_score,
        "feature_adjustment_score": feature_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "oil_shock_score": oil_shock_score,
        "commodity_risk_score": _safe_float(features.get("commodity_risk_score"), 0.0),
        "volatility_shock_score": _safe_float(features.get("volatility_shock_score"), 0.0),
        "volatility_explosion_probability": volatility_explosion_probability,
        "overnight_hold_allowed": overnight_hold_allowed,
        "overnight_hold_reason": str(global_risk_state.get("overnight_hold_reason", "overnight_risk_contained")),
        "overnight_risk_penalty": int(_safe_float(global_risk_state.get("overnight_risk_penalty"), 0.0)),
        "overnight_trade_block": not overnight_hold_allowed,
        "force_no_trade": force_no_trade,
    }


def derive_gamma_vol_trade_modifiers(gamma_vol_state, direction=None):
    """
    Purpose:
        Derive gamma vol trade modifiers from the supplied inputs.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        gamma_vol_state (Any): Structured state payload for gamma vol.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_trade_modifier_policy_config()
    gamma_vol_state = gamma_vol_state if isinstance(gamma_vol_state, dict) else {}
    base_adjustment_score = int(_safe_float(gamma_vol_state.get("gamma_vol_adjustment_score"), 0.0))
    alignment_adjustment_score = 0
    adjustment_reasons = []

    directional_convexity_state = str(gamma_vol_state.get("directional_convexity_state", "NO_CONVEXITY_EDGE")).upper().strip()
    squeeze_risk_state = str(gamma_vol_state.get("squeeze_risk_state", "LOW_ACCELERATION_RISK")).upper().strip()
    direction = str(direction or "").upper().strip()

    if direction == "CALL":
        if directional_convexity_state == "UPSIDE_SQUEEZE_RISK":
            alignment_adjustment_score += cfg.gamma_alignment_score
            adjustment_reasons.append("upside_squeeze_alignment")
        elif directional_convexity_state == "DOWNSIDE_AIRPOCKET_RISK":
            alignment_adjustment_score += cfg.gamma_conflict_penalty
            adjustment_reasons.append("downside_convexity_conflict")
        elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK" and squeeze_risk_state in {"HIGH_ACCELERATION_RISK", "EXTREME_ACCELERATION_RISK"}:
            alignment_adjustment_score += cfg.gamma_two_sided_score
            adjustment_reasons.append("two_sided_volatility_convexity")
    elif direction == "PUT":
        if directional_convexity_state == "DOWNSIDE_AIRPOCKET_RISK":
            alignment_adjustment_score += cfg.gamma_alignment_score
            adjustment_reasons.append("downside_airpocket_alignment")
        elif directional_convexity_state == "UPSIDE_SQUEEZE_RISK":
            alignment_adjustment_score += cfg.gamma_conflict_penalty
            adjustment_reasons.append("upside_convexity_conflict")
        elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK" and squeeze_risk_state in {"HIGH_ACCELERATION_RISK", "EXTREME_ACCELERATION_RISK"}:
            alignment_adjustment_score += cfg.gamma_two_sided_score
            adjustment_reasons.append("two_sided_volatility_convexity")

    effective_adjustment_score = int(
        _clip(base_adjustment_score + alignment_adjustment_score, cfg.alignment_score_floor, cfg.alignment_score_cap)
    )
    overnight_hold_allowed = bool(gamma_vol_state.get("overnight_hold_allowed", True))

    return {
        "base_adjustment_score": base_adjustment_score,
        "alignment_adjustment_score": alignment_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "gamma_vol_acceleration_score": int(_safe_float(gamma_vol_state.get("gamma_vol_acceleration_score"), 0.0)),
        "squeeze_risk_state": str(gamma_vol_state.get("squeeze_risk_state", "LOW_ACCELERATION_RISK")),
        "directional_convexity_state": str(gamma_vol_state.get("directional_convexity_state", "NO_CONVEXITY_EDGE")),
        "upside_squeeze_risk": _safe_float(gamma_vol_state.get("upside_squeeze_risk"), 0.0),
        "downside_airpocket_risk": _safe_float(gamma_vol_state.get("downside_airpocket_risk"), 0.0),
        "overnight_convexity_risk": _safe_float(gamma_vol_state.get("overnight_convexity_risk"), 0.0),
        "overnight_hold_allowed": overnight_hold_allowed,
        "overnight_hold_reason": str(gamma_vol_state.get("overnight_hold_reason", "overnight_convexity_contained")),
        "overnight_convexity_penalty": int(_safe_float(gamma_vol_state.get("overnight_convexity_penalty"), 0.0)),
        "overnight_convexity_boost": int(_safe_float(gamma_vol_state.get("overnight_convexity_boost"), 0.0)),
    }


def derive_dealer_pressure_trade_modifiers(dealer_pressure_state, direction=None):
    """
    Purpose:
        Derive dealer pressure trade modifiers from the supplied inputs.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        dealer_pressure_state (Any): Structured state payload for dealer pressure.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_trade_modifier_policy_config()
    dealer_pressure_state = dealer_pressure_state if isinstance(dealer_pressure_state, dict) else {}
    base_adjustment_score = int(_safe_float(dealer_pressure_state.get("dealer_pressure_adjustment_score"), 0.0))
    alignment_adjustment_score = 0
    adjustment_reasons = []

    dealer_flow_state = str(dealer_pressure_state.get("dealer_flow_state", "HEDGING_NEUTRAL")).upper().strip()
    direction = str(direction or "").upper().strip()

    # Pinning dampens long-option opportunities, while aligned acceleration is
    # rewarded because dealer hedging can reinforce the directional move.
    if dealer_flow_state == "PINNING_DOMINANT":
        alignment_adjustment_score += cfg.dealer_pinning_penalty
        adjustment_reasons.append("pinning_dampens_option_buying")
    elif dealer_flow_state == "TWO_SIDED_INSTABILITY":
        alignment_adjustment_score += cfg.dealer_instability_penalty
        adjustment_reasons.append("two_sided_hedging_instability")
    elif direction == "CALL" and dealer_flow_state == "UPSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += cfg.dealer_alignment_score
        adjustment_reasons.append("upside_hedging_alignment")
    elif direction == "PUT" and dealer_flow_state == "DOWNSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += cfg.dealer_alignment_score
        adjustment_reasons.append("downside_hedging_alignment")
    elif direction == "CALL" and dealer_flow_state == "DOWNSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += cfg.dealer_conflict_penalty
        adjustment_reasons.append("downside_hedging_conflict")
    elif direction == "PUT" and dealer_flow_state == "UPSIDE_HEDGING_ACCELERATION":
        alignment_adjustment_score += cfg.dealer_conflict_penalty
        adjustment_reasons.append("upside_hedging_conflict")

    effective_adjustment_score = int(
        _clip(base_adjustment_score + alignment_adjustment_score, cfg.alignment_score_floor, cfg.alignment_score_cap)
    )

    return {
        "base_adjustment_score": base_adjustment_score,
        "alignment_adjustment_score": alignment_adjustment_score,
        "effective_adjustment_score": effective_adjustment_score,
        "adjustment_reasons": adjustment_reasons,
        "dealer_hedging_pressure_score": int(_safe_float(dealer_pressure_state.get("dealer_hedging_pressure_score"), 0.0)),
        "dealer_flow_state": str(dealer_pressure_state.get("dealer_flow_state", "HEDGING_NEUTRAL")),
        "upside_hedging_pressure": _safe_float(dealer_pressure_state.get("upside_hedging_pressure"), 0.0),
        "downside_hedging_pressure": _safe_float(dealer_pressure_state.get("downside_hedging_pressure"), 0.0),
        "pinning_pressure_score": _safe_float(dealer_pressure_state.get("pinning_pressure_score"), 0.0),
        "overnight_hedging_risk": _safe_float(dealer_pressure_state.get("overnight_hedging_risk"), 0.0),
        "overnight_hold_allowed": bool(dealer_pressure_state.get("overnight_hold_allowed", True)),
        "overnight_hold_reason": str(dealer_pressure_state.get("overnight_hold_reason", "overnight_hedging_contained")),
        "overnight_dealer_pressure_penalty": int(_safe_float(dealer_pressure_state.get("overnight_dealer_pressure_penalty"), 0.0)),
        "overnight_dealer_pressure_boost": int(_safe_float(dealer_pressure_state.get("overnight_dealer_pressure_boost"), 0.0)),
    }


def derive_option_efficiency_trade_modifiers(option_efficiency_state):
    """
    Purpose:
        Derive option efficiency trade modifiers from the supplied inputs.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_efficiency_state (Any): Structured state payload for option efficiency.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    option_efficiency_state = option_efficiency_state if isinstance(option_efficiency_state, dict) else {}
    adjustment_score = int(_safe_float(option_efficiency_state.get("option_efficiency_adjustment_score"), 0.0))
    return {
        "effective_adjustment_score": adjustment_score,
        "expected_move_points": option_efficiency_state.get("expected_move_points"),
        "expected_move_pct": option_efficiency_state.get("expected_move_pct"),
        "expected_move_quality": str(option_efficiency_state.get("expected_move_quality", "UNAVAILABLE")),
        "target_reachability_score": int(_safe_float(option_efficiency_state.get("target_reachability_score"), 50.0)),
        "premium_efficiency_score": int(_safe_float(option_efficiency_state.get("premium_efficiency_score"), 50.0)),
        "strike_efficiency_score": int(_safe_float(option_efficiency_state.get("strike_efficiency_score"), 50.0)),
        "option_efficiency_score": int(_safe_float(option_efficiency_state.get("option_efficiency_score"), 50.0)),
        "option_efficiency_adjustment_score": adjustment_score,
        "overnight_hold_allowed": bool(option_efficiency_state.get("overnight_hold_allowed", True)),
        "overnight_hold_reason": str(option_efficiency_state.get("overnight_hold_reason", "overnight_option_efficiency_contained")),
        "overnight_option_efficiency_penalty": int(_safe_float(option_efficiency_state.get("overnight_option_efficiency_penalty"), 0.0)),
        "strike_moneyness_bucket": str(option_efficiency_state.get("strike_moneyness_bucket", "UNKNOWN")),
        "strike_distance_from_spot": option_efficiency_state.get("strike_distance_from_spot"),
        "payoff_efficiency_hint": str(option_efficiency_state.get("payoff_efficiency_hint", "unknown")),
    }
