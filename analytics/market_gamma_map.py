"""
Module: market_gamma_map.py

Purpose:
    Compute market gamma map analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import pandas as pd


def calculate_market_gamma(option_chain):
    """
    Calculate strike-wise signed gamma exposure proxy.
    """
    if option_chain is None or len(option_chain) == 0:
        return pd.Series(dtype=float)

    df = option_chain.copy()
    strike_col = "STRIKE_PR" if "STRIKE_PR" in df.columns else "strikePrice"
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else "openInterest"

    gamma = pd.to_numeric(df.get("GAMMA"), errors="coerce").fillna(0.0)
    oi = pd.to_numeric(df.get(oi_col), errors="coerce").fillna(0.0)
    strikes = pd.to_numeric(df.get(strike_col), errors="coerce").fillna(0.0)
    option_type = df.get("OPTION_TYP", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    signed = option_type.map({"CE": 1.0, "PE": -1.0}).fillna(0.0)

    df["GAMMA_EXPOSURE"] = gamma * oi * strikes * signed

    return df.groupby(strike_col)["GAMMA_EXPOSURE"].sum()


def market_gamma_regime(gex):
    """
    Determine overall gamma regime
    """

    if gex is None or len(gex) == 0:
        return "UNKNOWN"

    total_gex = gex.sum()
    gross_gex = gex.abs().sum()

    if gross_gex == 0 or abs(total_gex) <= gross_gex * 0.05:
        return "NEUTRAL_GAMMA"

    if total_gex > 0:
        return "POSITIVE_GAMMA"

    return "NEGATIVE_GAMMA"


def largest_gamma_strikes(gex, top_n=5, spot=None, max_distance_pct=0.10):
    """
    Find strikes with largest gamma concentration.
    When *spot* is given, restrict to strikes within *max_distance_pct* of spot
    so that deep OTM hedging strikes do not dominate near-spot clusters.
    """
    if spot is not None and spot > 0:
        lower = spot * (1 - max_distance_pct)
        upper = spot * (1 + max_distance_pct)
        gex = gex[(gex.index >= lower) & (gex.index <= upper)]

    if gex.empty:
        return []

    walls = gex.abs().sort_values(
        ascending=False
    ).head(top_n)

    return list(walls.index)
