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
    - max_pain              : float | None  — max pain strike
    - max_pain_dist         : float | None  — spot − max_pain (+ = max pain below spot)
    - max_pain_zone         : "ABOVE_SPOT" | "BELOW_SPOT" | "AT_SPOT" | "UNAVAILABLE"
    - min_aggregate_payout  : float         — minimum total holder payout at max pain level
                                             (this is the level of maximum writer profit, not
                                              writer pain — the old key `total_writer_pain` was
                                              semantically inverted and has been corrected)
    - total_writer_pain     : float         — deprecated alias kept for backward compatibility;
                                             equal to min_aggregate_payout
    - pain_curve            : dict          — {str(strike): total_payout} — keys are strings so
                                             the dict survives a JSON serialisation round-trip

Downstream Usage:
    Consumed by market-state assembly, terminal output, and signal diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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

    Implementation note — vectorised numpy
    ----------------------------------------
    The pain curve is computed via numpy broadcasting instead of O(N²) Python
    loops.  For a 150-strike NIFTY chain this reduces ~22 500 Python-level
    ``max()`` calls to a single pair of vectorised ``np.maximum`` operations,
    typically 100× faster.

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
        max_pain             – float | None
        max_pain_dist        – float | None  (spot − max_pain; positive = below spot)
        max_pain_zone        – "ABOVE_SPOT" | "BELOW_SPOT" | "AT_SPOT" | "UNAVAILABLE"
        min_aggregate_payout – float  (minimum total holder payout at max pain level;
                                       equivalently, the level of maximum writer profit)
        pain_curve           – dict {str(strike): total_payout}  — string keys are safe
                               across json.dumps / json.loads round-trips
    """
    _empty = {
        "max_pain": None,
        "max_pain_dist": None,
        "max_pain_zone": "UNAVAILABLE",
        "min_aggregate_payout": 0.0,
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
    ce_df = (
        df[df["OPTION_TYP"] == "CE"][[strike_col, oi_col]]
        .groupby(strike_col, as_index=True)[oi_col]
        .sum()
    )
    pe_df = (
        df[df["OPTION_TYP"] == "PE"][[strike_col, oi_col]]
        .groupby(strike_col, as_index=True)[oi_col]
        .sum()
    )

    all_strikes_set = set(ce_df.index.tolist()) | set(pe_df.index.tolist())
    if len(all_strikes_set) < 2:
        return _empty

    # Vectorised pain-curve computation via numpy broadcasting.
    # S has shape (N,); K_ce/K_pe have shape (M_ce,) and (M_pe,) respectively.
    S = np.array(sorted(all_strikes_set), dtype=float)       # candidate prices (N,)

    # Call side: payoff = max(0, S - K) * OI  for each CE strike K.
    K_ce = ce_df.index.to_numpy(dtype=float)                  # (M_ce,)
    OI_ce = ce_df.values.astype(float)                        # (M_ce,)
    # Broadcast (N,1) - (1,M_ce) → (N,M_ce); then dot with OI_ce → (N,)
    call_pain: np.ndarray = np.maximum(0.0, S[:, None] - K_ce[None, :]) @ OI_ce

    # Put side: payoff = max(0, K - S) * OI  for each PE strike K.
    K_pe = pe_df.index.to_numpy(dtype=float)                  # (M_pe,)
    OI_pe = pe_df.values.astype(float)                        # (M_pe,)
    put_pain: np.ndarray = np.maximum(0.0, K_pe[None, :] - S[:, None]) @ OI_pe

    total_pain: np.ndarray = call_pain + put_pain             # (N,)

    # pain_curve: string keys for JSON round-trip safety.
    pain_curve: dict[str, float] = {
        str(s): round(float(p), 2) for s, p in zip(S.tolist(), total_pain.tolist())
    }

    min_idx = int(np.argmin(total_pain))
    max_pain_strike = float(S[min_idx])
    min_aggregate_payout = float(total_pain[min_idx])

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
        "min_aggregate_payout": round(min_aggregate_payout, 2),
        # Deprecated compatibility alias for historical consumers.
        "total_writer_pain": round(min_aggregate_payout, 2),
        "pain_curve": pain_curve,
    }
