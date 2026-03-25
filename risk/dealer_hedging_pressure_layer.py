"""
Module: dealer_hedging_pressure_layer.py

Purpose:
    Assemble the dealer hedging pressure overlay decision from features and policy rules.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

from risk.dealer_hedging_pressure_features import build_dealer_hedging_pressure_features
from risk.dealer_hedging_pressure_regime import classify_dealer_hedging_pressure_state


def build_dealer_hedging_pressure_state(
    *,
    spot=None,
    gamma_regime=None,
    spot_vs_flip=None,
    gamma_flip_distance_pct=None,
    dealer_position=None,
    dealer_hedging_bias=None,
    dealer_hedging_flow=None,
    market_gamma=None,
    gamma_clusters=None,
    liquidity_levels=None,
    support_wall=None,
    resistance_wall=None,
    liquidity_vacuum_state=None,
    intraday_gamma_state=None,
    intraday_range_pct=None,
    flow_signal=None,
    smart_money_flow=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    volatility_explosion_probability=None,
    gamma_vol_acceleration_score=None,
    holding_profile="AUTO",
    max_pain_dist=None,
    max_pain_zone=None,
    days_to_expiry=None,
):
    """
    Purpose:
        Build the dealer hedging pressure state used by downstream components.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        gamma_regime (Any): Input associated with gamma regime.
        spot_vs_flip (Any): Input associated with spot vs flip.
        gamma_flip_distance_pct (Any): Input associated with gamma flip distance percentage.
        dealer_position (Any): Input associated with dealer position.
        dealer_hedging_bias (Any): Input associated with dealer hedging bias.
        dealer_hedging_flow (Any): Input associated with dealer hedging flow.
        market_gamma (Any): Input associated with market gamma.
        gamma_clusters (Any): Input associated with gamma clusters.
        liquidity_levels (Any): Input associated with liquidity levels.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
        liquidity_vacuum_state (Any): Structured state payload for liquidity vacuum.
        intraday_gamma_state (Any): Structured state payload for intraday gamma.
        intraday_range_pct (Any): Input associated with intraday range percentage.
        flow_signal (Any): Input associated with flow signal.
        smart_money_flow (Any): Input associated with smart money flow.
        macro_event_risk_score (Any): Macro-event risk score used by fallback or overlay logic.
        global_risk_state (Any): Structured state payload for global risk.
        volatility_explosion_probability (Any): Input associated with volatility explosion probability.
        gamma_vol_acceleration_score (Any): Score value for gamma vol acceleration.
        holding_profile (Any): Holding intent that determines whether overnight rules should be considered.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    features = build_dealer_hedging_pressure_features(
        spot=spot,
        gamma_regime=gamma_regime,
        spot_vs_flip=spot_vs_flip,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        dealer_position=dealer_position,
        dealer_hedging_bias=dealer_hedging_bias,
        dealer_hedging_flow=dealer_hedging_flow,
        market_gamma=market_gamma,
        gamma_clusters=gamma_clusters,
        liquidity_levels=liquidity_levels,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        liquidity_vacuum_state=liquidity_vacuum_state,
        intraday_gamma_state=intraday_gamma_state,
        intraday_range_pct=intraday_range_pct,
        flow_signal=flow_signal,
        smart_money_flow=smart_money_flow,
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=volatility_explosion_probability,
        gamma_vol_acceleration_score=gamma_vol_acceleration_score,
        holding_profile=holding_profile,
        max_pain_dist=max_pain_dist,
        max_pain_zone=max_pain_zone,
        days_to_expiry=days_to_expiry,
    )
    return classify_dealer_hedging_pressure_state(features).to_dict()
