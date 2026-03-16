"""
Module: flow_utils.py

Purpose:
    Compute flow utils analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

from __future__ import annotations

import pandas as pd

from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry


def _safe_float(value, default=0.0):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `flow utils` module. The module sits in the analytics layer that turns option-chain and market-structure data into tradable features.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def infer_strike_step(option_chain: pd.DataFrame):
    """
    Purpose:
        Infer strike step from the available inputs.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (pd.DataFrame): Input associated with option chain.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain is None or option_chain.empty or "strikePrice" not in option_chain.columns:
        return None

    strikes = []
    for value in option_chain["strikePrice"].tolist():
        strike = _safe_float(value, None)
        if strike is not None:
            strikes.append(float(strike))

    strikes = sorted(set(strikes))
    if len(strikes) < 2:
        return None

    diffs = [round(strikes[idx] - strikes[idx - 1], 6) for idx in range(1, len(strikes))]
    diffs = [diff for diff in diffs if diff > 0]

    if not diffs:
        return None

    return min(diffs)


def front_expiry_atm_slice(option_chain: pd.DataFrame, spot=None, strike_window_steps: int = 4):
    """
    Purpose:
        Process front expiry ATM slice for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (pd.DataFrame): Input associated with option chain.
        spot (Any): Input associated with spot.
        strike_window_steps (int): Input associated with strike window steps.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain is None or option_chain.empty:
        return option_chain

    selected_expiry = resolve_selected_expiry(option_chain)
    df = filter_option_chain_by_expiry(option_chain, selected_expiry)

    if df is None or df.empty or spot is None or "strikePrice" not in df.columns:
        return df

    strike_step = infer_strike_step(df)
    if strike_step in (None, 0):
        return df

    lower_bound = float(spot) - (strike_window_steps * strike_step)
    upper_bound = float(spot) + (strike_window_steps * strike_step)

    window = df[
        (pd.to_numeric(df["strikePrice"], errors="coerce") >= lower_bound) &
        (pd.to_numeric(df["strikePrice"], errors="coerce") <= upper_bound)
    ].copy()

    return window if not window.empty else df
