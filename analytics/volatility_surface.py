"""
Module: volatility_surface.py

Purpose:
    Compute volatility surface analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import pandas as pd
import numpy as np


def build_vol_surface(option_chain):
    """
    Build implied volatility surface
    across strikes and expiries.
    """

    clean_chain = option_chain.copy()
    clean_chain["IV"] = pd.to_numeric(clean_chain["IV"], errors="coerce")
    clean_chain = clean_chain[clean_chain["IV"] > 0]

    surface = clean_chain.pivot_table(
        values="IV",
        index="STRIKE_PR",
        columns="EXPIRY_DT",
        aggfunc="mean"
    )

    return surface


def atm_vol(option_chain, spot):
    """
    Compute ATM implied volatility.
    """
    clean_chain = option_chain.copy()
    clean_chain["STRIKE_PR"] = pd.to_numeric(clean_chain["STRIKE_PR"], errors="coerce")
    clean_chain["IV"] = pd.to_numeric(clean_chain["IV"], errors="coerce")
    clean_chain = clean_chain.dropna(subset=["STRIKE_PR", "IV"])
    clean_chain = clean_chain[clean_chain["IV"] > 0]

    if clean_chain.empty:
        return None

    clean_chain["DIST"] = abs(clean_chain["STRIKE_PR"] - spot)
    atm_row = clean_chain.sort_values("DIST").iloc[0]
    return float(atm_row["IV"])


def vol_regime(atm_iv):
    """
    Determine volatility regime.
    """

    if atm_iv is None:
        return "UNKNOWN"

    if atm_iv > 25:
        return "HIGH_VOL"

    if atm_iv < 15:
        return "LOW_VOL"

    return "NORMAL_VOL"


def compute_risk_reversal(option_chain, spot: float, delta_target: float = 0.25) -> dict:
    """
    Compute the 25-delta risk reversal for the front expiry.

    The risk reversal (RR) = IV(25-delta put) – IV(25-delta call).
    A positive RR means put skew dominates (hedging demand / downside fear).
    A negative RR means call skew dominates (upside demand / melt-up positioning).

    Parameters
    ----------
    option_chain : DataFrame
        Must have columns: STRIKE_PR, OPTION_TYP, IV, EXPIRY_DT.
    spot : float
        Current underlying spot price.
    delta_target : float
        Absolute delta level to target.  Default 0.25 (25-delta).

    Returns
    -------
    dict with keys:
        rr_value        – float (put_iv_25d - call_iv_25d); None if unavailable
        put_iv_25d      – float; None if unavailable
        call_iv_25d     – float; None if unavailable
        rr_regime       – "PUT_SKEW" | "CALL_SKEW" | "BALANCED" | "UNAVAILABLE"
    """
    result = {
        "rr_value": None,
        "put_iv_25d": None,
        "call_iv_25d": None,
        "rr_regime": "UNAVAILABLE",
    }

    if option_chain is None or option_chain.empty or spot is None or spot <= 0:
        return result

    df = option_chain.copy()
    for col in ("STRIKE_PR", "IV"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["STRIKE_PR", "IV"])
    df = df[df["IV"] > 0]

    if df.empty:
        return result

    # Restrict to front expiry
    if "EXPIRY_DT" in df.columns:
        try:
            df["_expiry_dt"] = pd.to_datetime(df["EXPIRY_DT"], errors="coerce")
            front_exp = df["_expiry_dt"].dropna().min()
            if pd.notna(front_exp):
                df = df[df["_expiry_dt"] == front_exp]
        except Exception:
            pass

    # Identify 25-delta put: OTM put closest to delta_target below spot
    # For puts: delta_target ≈ 0.25 corresponds to strike ~ spot * exp(-z*sigma*sqrt(T))
    # Since we don't have delta directly, we approximate via moneyness:
    # 25-delta put → strike roughly at spot * (1 - 1.5 * atm_vol_fraction)
    # 25-delta call → strike roughly at spot * (1 + 1.5 * atm_vol_fraction)
    # Instead we just pick the strike closest to the 25-delta moneyness fraction.

    def _nearest_iv(side_df, target_moneyness_sign: float) -> float | None:
        """Return IV at the strike closest to the moneyness implied by delta_target."""
        if side_df.empty:
            return None
        # Approximate 25-delta strike via log-moneyness: ln(K/S) ≈ ±1.28 * atm_vol * sqrt(T)
        # We use a simplified proxy: 1.5% OTM for 25-delta at typical index vol.
        moneyness_offset = spot * 0.015 * (1.0 / delta_target)  # scales with target delta
        target_strike = spot + target_moneyness_sign * moneyness_offset
        side_df = side_df.copy()
        side_df["_dist"] = (side_df["STRIKE_PR"] - target_strike).abs()
        nearest = side_df.sort_values("_dist").iloc[0]
        iv_val = float(nearest["IV"])
        return iv_val if iv_val > 0 else None

    calls = df[df["OPTION_TYP"] == "CE"]
    puts = df[df["OPTION_TYP"] == "PE"]

    call_iv = _nearest_iv(calls, target_moneyness_sign=+1.0)
    put_iv = _nearest_iv(puts, target_moneyness_sign=-1.0)

    if call_iv is None or put_iv is None:
        return result

    rr = round(put_iv - call_iv, 4)
    result["rr_value"] = rr
    result["put_iv_25d"] = round(put_iv, 4)
    result["call_iv_25d"] = round(call_iv, 4)
    if rr > 0.5:
        result["rr_regime"] = "PUT_SKEW"
    elif rr < -0.5:
        result["rr_regime"] = "CALL_SKEW"
    else:
        result["rr_regime"] = "BALANCED"

    return result


def risk_reversal_velocity(
    current_rr: float | None,
    prev_rr: float | None,
    seconds_elapsed: float = 300.0,
) -> dict:
    """
    Compute the rate of change of the risk reversal (RR velocity).

    RR velocity > 0 means skew is shifting toward puts (growing fear).
    RR velocity < 0 means skew is collapsing or shifting toward calls.

    Returns dict with keys: rr_velocity, rr_momentum ("RISING_PUT_SKEW"
    | "FALLING_PUT_SKEW" | "STABLE").
    """
    if current_rr is None or prev_rr is None or seconds_elapsed <= 0:
        return {"rr_velocity": None, "rr_momentum": "UNAVAILABLE"}

    velocity = (current_rr - prev_rr) / max(seconds_elapsed / 300.0, 1e-6)
    momentum = "STABLE"
    if velocity > 0.20:
        momentum = "RISING_PUT_SKEW"
    elif velocity < -0.20:
        momentum = "FALLING_PUT_SKEW"

    return {"rr_velocity": round(velocity, 4), "rr_momentum": momentum}

