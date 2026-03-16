"""
Module: dealer_hedging_flow.py

Purpose:
    Compute dealer hedging flow analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import numpy as np


def dealer_hedging_flow(option_chain):

    """
    Estimate hedging pressure from dealer positions.
    """

    if option_chain is None or len(option_chain) == 0:
        return "SELL_FUTURES"

    df = option_chain.copy()
    delta = np.nan_to_num(df.get("DELTA", 0.0), nan=0.0)
    open_int = np.nan_to_num(df.get("OPEN_INT", df.get("openInterest", 0.0)), nan=0.0)

    flow = float((delta * open_int).sum())

    if flow > 0:

        return "BUY_FUTURES"

    else:

        return "SELL_FUTURES"
