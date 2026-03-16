"""
Module: liquidity_void.py

Purpose:
    Compute liquidity void analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd


def detect_liquidity_voids(option_chain, threshold=50):

    """
    Purpose:
        Detect liquidity voids from the available inputs.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        threshold (Any): Input associated with threshold.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain.empty:
        return []

    df = option_chain.copy()

    grouped = df.groupby("strikePrice")["openInterest"].sum()

    voids = []

    for strike, oi in grouped.items():

        if oi < threshold:
            voids.append(strike)

    return voids


def nearest_liquidity_void(spot, voids):

    """
    Purpose:
        Process nearest liquidity void for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        voids (Any): Input associated with voids.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if not voids:
        return None

    closest = min(voids, key=lambda x: abs(x - spot))

    return closest


def liquidity_void_signal(spot, voids):

    """
    Purpose:
        Process liquidity void signal for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        voids (Any): Input associated with voids.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    nearest = nearest_liquidity_void(spot, voids)

    if nearest is None:
        return None

    distance = abs(spot - nearest)

    if distance < 50:
        return "VOID_NEAR"

    return "VOID_FAR"