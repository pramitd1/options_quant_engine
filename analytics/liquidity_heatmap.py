"""
Module: liquidity_heatmap.py

Purpose:
    Compute liquidity heatmap analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd


def build_liquidity_heatmap(option_chain):

    """
    Purpose:
        Build the liquidity heatmap used by downstream components.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain.empty:
        return {}

    grouped = option_chain.groupby("strikePrice")[
        "openInterest"
    ].sum()

    heatmap = grouped.sort_values(ascending=False)

    return heatmap


def strongest_liquidity_levels(option_chain, top_n=5):

    """
    Purpose:
        Process strongest liquidity levels for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        top_n (Any): Input associated with top n.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    heatmap = build_liquidity_heatmap(option_chain)

    levels = heatmap.head(top_n).index.tolist()

    return levels


def liquidity_signal(spot, levels):

    """
    Purpose:
        Process liquidity signal for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        levels (Any): Input associated with levels.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if not levels:
        return None

    closest = min(levels, key=lambda x: abs(x - spot))

    distance = abs(spot - closest)

    if distance < 50:
        return "STRONG_LEVEL_NEAR"

    return "LEVEL_FAR"