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


def _infer_dealer_gamma_sign(df: pd.DataFrame, *, type_col: str, oi_col: str) -> float:
    """
    Infer whether positioning is more likely long-gamma (+1) or short-gamma (-1).
    """
    change_col = "changeinOI" if "changeinOI" in df.columns else ("CHG_IN_OI" if "CHG_IN_OI" in df.columns else None)

    option_type = df[type_col].astype(str).str.upper()
    if change_col is not None:
        change = pd.to_numeric(df[change_col], errors="coerce").fillna(0.0)
        call_change = float(change[option_type == "CE"].sum())
        put_change = float(change[option_type == "PE"].sum())
        total_change = abs(call_change) + abs(put_change)
        if total_change > 0:
            return 1.0 if (put_change - call_change) >= 0 else -1.0

    oi = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)
    call_oi = float(oi[option_type == "CE"].sum())
    put_oi = float(oi[option_type == "PE"].sum())
    return 1.0 if put_oi >= call_oi else -1.0


def approximate_gamma(strike, spot):
    """
    Rough gamma approximation based on distance from ATM.

    Real gamma requires a full options model. For structural
    positioning analysis, a simple inverse-distance proxy is enough.
    """

    strike = float(strike)
    spot = float(spot)
    if spot <= 0:
        return 0.0
    moneyness_distance = abs(strike - spot) / spot
    return 1.0 / (1.0 + moneyness_distance)


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
    if spot is None:
        spot = float(strikes.median()) if strikes.notna().any() else 0.0

    if "GAMMA" in df.columns:
        gamma = pd.to_numeric(df["GAMMA"], errors="coerce").fillna(0.0)
    else:
        distance = (strikes - float(spot)).abs() / max(float(spot), 1e-6)
        gamma = 1.0 / (1.0 + distance)

    raw_exposure = gamma * oi
    finite_gamma = gamma.replace([np.inf, -np.inf], np.nan).dropna()
    gamma_has_intrinsic_sign = bool((finite_gamma < 0).any() and (finite_gamma > 0).any())

    if gamma_has_intrinsic_sign:
        net_exposure = float(np.nansum(raw_exposure.values))
    else:
        dealer_gamma_sign = _infer_dealer_gamma_sign(df, type_col=type_col, oi_col=oi_col)
        net_exposure = float(dealer_gamma_sign * np.nansum(np.abs(raw_exposure.values)))

    return net_exposure


def gamma_signal(option_chain: pd.DataFrame, spot=None):
    """
    Convert gamma exposure into a simple regime label.
    """

    if option_chain is None or option_chain.empty:
        return "NEUTRAL_GAMMA"

    df = option_chain.copy()
    strike_col = "strikePrice" if "strikePrice" in df.columns else "STRIKE_PR"
    oi_col = "openInterest" if "openInterest" in df.columns else "OPEN_INT"
    type_col = "OPTION_TYP"

    if strike_col not in df.columns or oi_col not in df.columns or type_col not in df.columns:
        return "NEUTRAL_GAMMA"

    strikes = pd.to_numeric(df[strike_col], errors="coerce")
    oi = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)
    if spot is None:
        spot = float(strikes.median()) if strikes.notna().any() else 0.0

    if "GAMMA" in df.columns:
        gamma = pd.to_numeric(df["GAMMA"], errors="coerce").fillna(0.0)
    else:
        distance = (strikes - float(spot)).abs() / max(float(spot), 1e-6)
        gamma = 1.0 / (1.0 + distance.fillna(np.inf))

    raw_exposure = gamma * oi
    finite_gamma = gamma.replace([np.inf, -np.inf], np.nan).dropna()
    gamma_has_intrinsic_sign = bool((finite_gamma < 0).any() and (finite_gamma > 0).any())

    if gamma_has_intrinsic_sign:
        net_gamma = float(np.nansum(raw_exposure.values))
    else:
        dealer_gamma_sign = _infer_dealer_gamma_sign(df, type_col=type_col, oi_col=oi_col)
        net_gamma = float(dealer_gamma_sign * np.nansum(np.abs(raw_exposure.values)))
    gross_gamma = float(np.nansum(np.abs(raw_exposure.values)))

    if gross_gamma <= 0 or abs(net_gamma) <= gross_gamma * 0.05:
        return "NEUTRAL_GAMMA"
    if net_gamma > 0:
        # Keep legacy output labels for gamma_signal() to preserve API
        # compatibility with existing tests and downstream consumers.
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
