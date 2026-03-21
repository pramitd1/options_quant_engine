"""
Module: gamma_walls.py

Purpose:
    Compute gamma walls analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import pandas as pd


def detect_gamma_walls(option_chain, top_n=3):
    """
    Detect strikes with highest open interest.

    These act as gamma walls (strong support/resistance).
    """

    if option_chain is None or option_chain.empty:
        return []

    oi_by_strike = option_chain.groupby(
        "STRIKE_PR"
    )["OPEN_INT"].sum()

    walls = oi_by_strike.sort_values(
        ascending=False
    ).head(top_n)

    return list(walls.index)


def classify_walls(option_chain):
    """
    Classify support and resistance walls.
    """

    if option_chain is None or option_chain.empty:
        return {}

    call_oi = option_chain[
        option_chain["OPTION_TYP"] == "CE"
    ].groupby("STRIKE_PR")["OPEN_INT"].sum()

    put_oi = option_chain[
        option_chain["OPTION_TYP"] == "PE"
    ].groupby("STRIKE_PR")["OPEN_INT"].sum()

    if call_oi.empty or put_oi.empty:
        return {}

    resistance = call_oi.idxmax()

    support = put_oi.idxmax()

    return {

        "support_wall": support,
        "resistance_wall": resistance
    }