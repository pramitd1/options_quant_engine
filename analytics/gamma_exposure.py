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

    if spot is None:
        spot = df["strikePrice"].median()

    if "GAMMA" in df.columns:
        gamma = pd.to_numeric(df["GAMMA"], errors="coerce").fillna(0.0)
        oi = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0)
        signed = np.where(df["OPTION_TYP"].astype(str).eq("PE"), -1.0, 1.0)
        return float(np.sum(gamma * oi * signed))

    exposures = []

    for _, row in df.iterrows():
        strike = row["strikePrice"]
        oi = row["openInterest"]

        gamma = approximate_gamma(strike, spot)
        exposure = gamma * oi

        if row["OPTION_TYP"] == "PE":
            exposure *= -1

        exposures.append(exposure)

    return float(np.sum(exposures))


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
