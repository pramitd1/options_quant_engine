"""
Module: gamma_exposure.py

Purpose:
    Compute gamma exposure analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import numpy as np
import pandas as pd


def approximate_gamma(strike, spot):
    """
    Rough gamma approximation based on distance from ATM.

    Real gamma requires a full options model. For structural
    positioning analysis, a simple inverse-distance proxy is enough.
    """

    distance = abs(strike - spot)
    return 1 / (1 + distance)


def calculate_gamma_exposure(option_chain: pd.DataFrame, spot=None):
    """
    Estimate total gamma exposure from an option chain.

    Expected columns
    ----------------
    strikePrice
    OPTION_TYP
    openInterest

    Parameters
    ----------
    option_chain : pd.DataFrame
    spot : float, optional

    Returns
    -------
    float
        Net gamma exposure proxy
    """

    if option_chain is None or option_chain.empty:
        return 0.0

    df = option_chain.copy()

    strike_col = "strikePrice" if "strikePrice" in df.columns else "STRIKE_PR"
    oi_col = "openInterest" if "openInterest" in df.columns else "OPEN_INT"
    type_col = "OPTION_TYP"

    if strike_col not in df.columns or oi_col not in df.columns or type_col not in df.columns:
        return 0.0

    strikes = pd.to_numeric(df[strike_col], errors="coerce")
    oi = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)
    option_type = df[type_col].astype(str).str.upper()

    if spot is None:
        spot = float(strikes.median()) if strikes.notna().any() else 0.0

    if "GAMMA" in df.columns:
        gamma = pd.to_numeric(df["GAMMA"], errors="coerce").fillna(0.0)
    else:
        distance = (strikes - float(spot)).abs().fillna(np.inf)
        gamma = 1.0 / (1.0 + distance)

    signed = option_type.map({"CE": 1.0, "PE": -1.0}).fillna(0.0)
    exposure = gamma * oi * signed
    return float(np.nansum(exposure.values))


def gamma_signal(option_chain: pd.DataFrame, spot=None):
    """
    Convert gamma exposure into a simple regime label.
    """

    gamma = calculate_gamma_exposure(option_chain, spot)

    if gamma > 0:
        return "LONG_GAMMA"

    return "SHORT_GAMMA"


# --------------------------------------------------
# Backward-compatible alias used by older modules
# --------------------------------------------------

def calculate_gex(option_chain: pd.DataFrame, spot=None):
    """
    Alias for calculate_gamma_exposure().

    Kept to support modules like intraday_gamma_shift.py
    that still import calculate_gex.
    """
    return calculate_gamma_exposure(option_chain, spot)
