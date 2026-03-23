"""
Module: options_flow_imbalance.py

Purpose:
    Compute options flow imbalance analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd

from config.analytics_feature_policy import get_flow_imbalance_policy_config
from analytics.flow_utils import front_expiry_atm_slice


def calculate_flow_imbalance(option_chain, spot=None):
    """
    Purpose:
        Calculate flow imbalance from the supplied inputs.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        spot (Any): Input associated with spot.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_flow_imbalance_policy_config()
    if option_chain is None or option_chain.empty:
        return cfg.neutral_default_imbalance

    df = front_expiry_atm_slice(option_chain, spot=spot, strike_window_steps=4)
    if df is None or df.empty:
        return cfg.neutral_default_imbalance

    calls = df[df["OPTION_TYP"] == "CE"].copy()
    puts = df[df["OPTION_TYP"] == "PE"].copy()

    def _delta_adjusted_notional(side_df) -> float:
        """Volume * price * |delta| (fallback to notional when DELTA absent)."""
        vol = pd.to_numeric(side_df.get("totalTradedVolume"), errors="coerce").fillna(0.0)
        price = pd.to_numeric(side_df.get("lastPrice"), errors="coerce").fillna(0.0)
        notional = vol * price
        if "DELTA" in side_df.columns:
            delta_abs = pd.to_numeric(side_df["DELTA"], errors="coerce").abs().fillna(0.5)
            return float((notional * delta_abs).sum())
        return float(notional.sum())

    call_notional = _delta_adjusted_notional(calls)
    put_notional = _delta_adjusted_notional(puts)

    if call_notional <= 0 and put_notional <= 0:
        return cfg.neutral_default_imbalance

    if put_notional <= 0:
        return cfg.no_put_flow_fallback_imbalance

    return float(call_notional / put_notional)


def flow_signal(option_chain, spot=None):
    """
    Purpose:
        Process flow signal for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        spot (Any): Input associated with spot.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_flow_imbalance_policy_config()
    imbalance = calculate_flow_imbalance(option_chain, spot=spot)

    if imbalance >= cfg.bullish_threshold:
        return "BULLISH_FLOW"

    if imbalance <= cfg.bearish_threshold:
        return "BEARISH_FLOW"

    return "NEUTRAL_FLOW"
