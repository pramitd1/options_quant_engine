"""
Module: max_pain.py

Purpose:
    Compute the max pain strike — the price level at which aggregate
    option-holder payout is minimised at expiry.

Role in the System:
    Analytics layer. Provides an expiry-gravity signal that is distinct from
    the GEX-based gamma flip. On expiry days (especially weekly NIFTY Thursday
    expiry) max pain acts as a strong price magnet because option writers
    collectively profit most when spot settles at or near that level, creating
    an incentive for market-maker hedging to steer price there.

Key Outputs:
    - max_pain          : float | None  — max pain strike
    - max_pain_dist     : float | None  — spot − max_pain (+ = max pain below spot)
    - max_pain_zone     : "ABOVE_SPOT" | "BELOW_SPOT" | "AT_SPOT" | "UNAVAILABLE"
    - total_writer_pain : float         — aggregate option-holder payout at max pain level
    - pain_curve        : dict          — {strike: total_payout} for all candidate strikes

Downstream Usage:
    Consumed by market-state assembly, terminal output, and signal diagnostics.
"""
from __future__ import annotations

import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry


def _resolve_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (strike_col, oi_col) for the supplied dataframe."""
    strike_col = "STRIKE_PR" if "STRIKE_PR" in df.columns else (
        "strikePrice" if "strikePrice" in df.columns else None
    )
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else (
        "openInterest" if "openInterest" in df.columns else None
    )
    return strike_col, oi_col


def compute_max_pain(option_chain: pd.DataFrame, spot: float | None = None) -> dict:
    """
    Compute the max pain strike for the front expiry.

    Algorithm
    ---------
    For each candidate strike S (all strikes in the front-expiry chain):
        total_holder_payout(S) =
            Σ_calls  max(0, S − K_i) × CE_OI_i   (in-the-money calls)
          + Σ_puts   max(0, K_i − S) × PE_OI_i   (in-the-money puts)

    max_pain = argmin total_holder_payout(S)

    The strike that minimises holder payout = maximises writer profit = the
    level at which the aggregate option book "wants" to settle.

    Parameters
    ----------
    option_chain : DataFrame
        Normalised option chain with (at minimum):
        STRIKE_PR or strikePrice, OPTION_TYP, OPEN_INT or openInterest.
    spot : float | None
        Current underlying spot price — used to annotate the max pain zone.

    Returns
    -------
    dict with keys:
        max_pain          – float | None
        max_pain_dist     – float | None  (spot − max_pain; positive = below spot)
        max_pain_zone     – "ABOVE_SPOT" | "BELOW_SPOT" | "AT_SPOT" | "UNAVAILABLE"
        total_writer_pain – float
        pain_curve        – dict {strike: total_payout}
    """
    _empty = {
        "max_pain": None,
        "max_pain_dist": None,
        "max_pain_zone": "UNAVAILABLE",
        "total_writer_pain": 0.0,
        "pain_curve": {},
    }

    if option_chain is None or option_chain.empty:
        return _empty

    strike_col, oi_col = _resolve_columns(option_chain)
    if strike_col is None or oi_col is None:
        return _empty

    # Restrict to front expiry to stay relevant intraday.
    try:
        selected_expiry = resolve_selected_expiry(option_chain)
        df = filter_option_chain_by_expiry(option_chain, selected_expiry)
        if df is None or df.empty:
            df = option_chain.copy()
    except Exception:
        df = option_chain.copy()

    df = df.copy()
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=[strike_col])
    df = df[df[oi_col] >= 0]

    if df.empty:
        return _empty

    # Build per-side OI series indexed by strike.
    ce_df = df[df["OPTION_TYP"] == "CE"][[strike_col, oi_col]].groupby(strike_col, as_index=True)[oi_col].sum()
    pe_df = df[df["OPTION_TYP"] == "PE"][[strike_col, oi_col]].groupby(strike_col, as_index=True)[oi_col].sum()

    all_strikes = sorted(set(ce_df.index.tolist()) | set(pe_df.index.tolist()))
    if len(all_strikes) < 2:
        return _empty

    # For each candidate expiry price, compute total holder payout.
    pain_curve: dict[float, float] = {}
    for s in all_strikes:
        call_pain = sum(
            max(0.0, float(s) - float(k)) * float(oi)
            for k, oi in ce_df.items()
        )
        put_pain = sum(
            max(0.0, float(k) - float(s)) * float(oi)
            for k, oi in pe_df.items()
        )
        pain_curve[float(s)] = round(call_pain + put_pain, 2)

    if not pain_curve:
        return _empty

    max_pain_strike = float(min(pain_curve, key=pain_curve.__getitem__))
    total_writer_pain = pain_curve[max_pain_strike]

    max_pain_dist: float | None = None
    max_pain_zone = "UNAVAILABLE"
    if spot is not None:
        max_pain_dist = round(float(spot) - max_pain_strike, 2)
        tolerance = 25.0  # within ±25 pts considered "AT_SPOT" for NIFTY
        if abs(max_pain_dist) <= tolerance:
            max_pain_zone = "AT_SPOT"
        elif max_pain_dist > 0:
            max_pain_zone = "BELOW_SPOT"
        else:
            max_pain_zone = "ABOVE_SPOT"

    return {
        "max_pain": max_pain_strike,
        "max_pain_dist": max_pain_dist,
        "max_pain_zone": max_pain_zone,
        "total_writer_pain": total_writer_pain,
        "pain_curve": pain_curve,
    }
