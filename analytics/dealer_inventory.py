"""
Module: dealer_inventory.py

Purpose:
    Compute dealer inventory analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd

from utils.numerics import safe_float as _safe_float  # noqa: F401


def dealer_inventory_metrics(option_chain: pd.DataFrame):
    """
    Purpose:
        Compute dealer inventory metrics from the supplied inputs.

    Context:
        Function inside the `dealer inventory` module. The module sits in the analytics layer that turns option-chain and market-structure data into tradable features.

    Inputs:
        option_chain (pd.DataFrame): Option-chain snapshot in dataframe form.

    Returns:
        dict: Metric bundle returned by the current calculation.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    if option_chain is None or option_chain.empty:
        return {
            "position": "Unknown",
            "basis": "NO_DATA",
            "call_oi": 0.0,
            "put_oi": 0.0,
            "call_oi_change": 0.0,
            "put_oi_change": 0.0,
            "net_oi_change_bias": 0.0,
        }

    df = option_chain.copy()

    calls = df[df["OPTION_TYP"] == "CE"].copy()
    puts = df[df["OPTION_TYP"] == "PE"].copy()

    call_oi = _safe_float(calls.get("openInterest", pd.Series(dtype=float)).sum(), 0.0)
    put_oi = _safe_float(puts.get("openInterest", pd.Series(dtype=float)).sum(), 0.0)

    if "changeinOI" in df.columns:
        call_oi_change = _safe_float(calls.get("changeinOI", pd.Series(dtype=float)).sum(), 0.0)
        put_oi_change = _safe_float(puts.get("changeinOI", pd.Series(dtype=float)).sum(), 0.0)
    else:
        call_oi_change = 0.0
        put_oi_change = 0.0

    total_abs_change = abs(call_oi_change) + abs(put_oi_change)

    if total_abs_change > 0:
        # More put OI build than call OI build indicates downside/bearish dealer
        # positioning: dealers sold more puts and are short gamma on the put side.
        # More call OI build means more calls sold, i.e., long gamma/neutral regime.
        # Keep this consistent with OPEN_INTEREST basis: call_oi > put_oi => Long Gamma.
        net_oi_change_bias = put_oi_change - call_oi_change
        position = "Short Gamma" if net_oi_change_bias >= 0 else "Long Gamma"
        basis = "OI_CHANGE"
    else:
        net_oi_change_bias = put_oi - call_oi
        position = "Long Gamma" if call_oi > put_oi else "Short Gamma"
        basis = "OPEN_INTEREST"

    return {
        "position": position,
        "basis": basis,
        "call_oi": round(call_oi, 2),
        "put_oi": round(put_oi, 2),
        "call_oi_change": round(call_oi_change, 2),
        "put_oi_change": round(put_oi_change, 2),
        "net_oi_change_bias": round(net_oi_change_bias, 2),
    }


def dealer_inventory_position(option_chain: pd.DataFrame):
    """
    Purpose:
        Process dealer inventory position for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (pd.DataFrame): Input associated with option chain.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return dealer_inventory_metrics(option_chain)["position"]
