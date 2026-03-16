"""
Module: gamma_vol_acceleration_layer.py

Purpose:
    Assemble the gamma vol acceleration overlay decision from features and policy rules.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

from risk.gamma_vol_acceleration_features import build_gamma_vol_acceleration_features
from risk.gamma_vol_acceleration_regime import classify_gamma_vol_acceleration_state


def build_gamma_vol_acceleration_state(
    *,
    gamma_regime=None,
    spot_vs_flip=None,
    gamma_flip_distance_pct=None,
    dealer_hedging_bias=None,
    liquidity_vacuum_state=None,
    intraday_range_pct=None,
    volatility_compression_score=None,
    volatility_shock_score=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    volatility_explosion_probability=None,
    holding_profile="AUTO",
    support_wall=None,
    resistance_wall=None,
):
    """
    Purpose:
        Build the gamma vol acceleration state used by downstream components.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        gamma_regime (Any): Input associated with gamma regime.
        spot_vs_flip (Any): Input associated with spot vs flip.
        gamma_flip_distance_pct (Any): Input associated with gamma flip distance percentage.
        dealer_hedging_bias (Any): Input associated with dealer hedging bias.
        liquidity_vacuum_state (Any): Structured state payload for liquidity vacuum.
        intraday_range_pct (Any): Input associated with intraday range percentage.
        volatility_compression_score (Any): Score value for volatility compression.
        volatility_shock_score (Any): Score value for volatility shock.
        macro_event_risk_score (Any): Macro-event risk score used by fallback or overlay logic.
        global_risk_state (Any): Structured state payload for global risk.
        volatility_explosion_probability (Any): Input associated with volatility explosion probability.
        holding_profile (Any): Holding intent that determines whether overnight rules should be considered.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    features = build_gamma_vol_acceleration_features(
        gamma_regime=gamma_regime,
        spot_vs_flip=spot_vs_flip,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        dealer_hedging_bias=dealer_hedging_bias,
        liquidity_vacuum_state=liquidity_vacuum_state,
        intraday_range_pct=intraday_range_pct,
        volatility_compression_score=volatility_compression_score,
        volatility_shock_score=volatility_shock_score,
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=volatility_explosion_probability,
        holding_profile=holding_profile,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
    )
    return classify_gamma_vol_acceleration_state(features).to_dict()
