"""
Module: volume_pcr.py

Purpose:
    Compute volume-based Put/Call Ratio (PCR) for the option chain.

Role in the System:
    Analytics layer. Volume PCR is a faster, more reactive sentiment signal
    than OI-based PCR. It reflects real-time order flow rather than
    accumulated positioning, making it useful for detecting intraday
    capitulation, panic buying, or forced short-covering.

    Unlike the flow imbalance module (which uses delta-adjusted notional),
    raw volume PCR is a simple, widely-understood market breadth indicator.
    Readings above ~1.3 with falling spot = capitulation signal; readings
    below ~0.7 with rising spot = call-buying conviction.

Key Outputs:
    volume_pcr          : float | None  — total put volume / total call volume (full chain)
    volume_pcr_atm      : float | None  — PCR for near-ATM front-expiry strikes only
    volume_pcr_regime   : "PUT_DOMINANT" | "NEUTRAL" | "CALL_DOMINANT" | "UNAVAILABLE"
    call_volume_total   : float
    put_volume_total    : float
    call_volume_atm     : float
    put_volume_atm      : float

Downstream Usage:
    Consumed by market-state assembly, signal enrichment, and terminal output.
"""
from __future__ import annotations

import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from data.expiry_resolver import filter_option_chain_by_expiry, resolve_selected_expiry


_EMPTY = {
    "volume_pcr": None,
    "volume_pcr_atm": None,
    "volume_pcr_regime": "UNAVAILABLE",
    "call_volume_total": 0.0,
    "put_volume_total": 0.0,
    "call_volume_atm": 0.0,
    "put_volume_atm": 0.0,
}

# PCR thresholds — standard values used by NIFTY practitioners:
#   PCR > 1.3  → PUT_DOMINANT (bearish sentiment / potential capitulation)
#   PCR < 0.75 → CALL_DOMINANT (bullish sentiment / potential complacency)
_PCR_BULLISH_THRESHOLD = 0.75
_PCR_BEARISH_THRESHOLD = 1.30


def _vol_col(df: pd.DataFrame) -> str | None:
    """Return the volume column name present in df."""
    for name in ("VOLUME", "totalTradedVolume", "volume"):
        if name in df.columns:
            return name
    return None


def compute_volume_pcr(option_chain: pd.DataFrame, spot: float | None = None) -> dict:
    """
    Compute full-chain and near-ATM volume PCR.

    Two readings are provided:
    - Full front-expiry chain PCR  (all strikes, front expiry)
    - Near-ATM PCR                 (±4 strike steps, front expiry)

    Near-ATM PCR is more sensitive: it filters out far-OTM hedging noise
    and reflects active directional flow in the most liquid strikes.

    Parameters
    ----------
    option_chain : DataFrame
        Normalised option chain. Must contain OPTION_TYP and a volume column.
    spot : float | None
        Current underlying spot price (used for ATM windowing).

    Returns
    -------
    dict with keys: volume_pcr, volume_pcr_atm, volume_pcr_regime,
    call_volume_total, put_volume_total, call_volume_atm, put_volume_atm.
    """
    if option_chain is None or option_chain.empty:
        return dict(_EMPTY)

    vol_col = _vol_col(option_chain)
    if vol_col is None:
        return dict(_EMPTY)

    # ── Full front-expiry PCR ────────────────────────────────────────────
    try:
        selected_expiry = resolve_selected_expiry(option_chain)
        front_df = filter_option_chain_by_expiry(option_chain, selected_expiry)
        if front_df is None or front_df.empty:
            front_df = option_chain.copy()
    except Exception:
        front_df = option_chain.copy()

    front_df = front_df.copy()
    front_df[vol_col] = pd.to_numeric(front_df[vol_col], errors="coerce").fillna(0.0)

    call_vol_total = float(front_df.loc[front_df["OPTION_TYP"] == "CE", vol_col].sum())
    put_vol_total  = float(front_df.loc[front_df["OPTION_TYP"] == "PE", vol_col].sum())

    full_pcr: float | None = None
    if call_vol_total > 0:
        full_pcr = round(put_vol_total / call_vol_total, 4)

    # ── Near-ATM PCR ─────────────────────────────────────────────────────
    atm_df = front_expiry_atm_slice(option_chain, spot=spot, strike_window_steps=4)
    call_vol_atm = 0.0
    put_vol_atm  = 0.0
    if atm_df is not None and not atm_df.empty:
        vol_col_atm = _vol_col(atm_df)
        if vol_col_atm:
            atm_df = atm_df.copy()
            atm_df[vol_col_atm] = pd.to_numeric(atm_df[vol_col_atm], errors="coerce").fillna(0.0)
            call_vol_atm = float(atm_df.loc[atm_df["OPTION_TYP"] == "CE", vol_col_atm].sum())
            put_vol_atm  = float(atm_df.loc[atm_df["OPTION_TYP"] == "PE", vol_col_atm].sum())

    atm_pcr: float | None = None
    if call_vol_atm > 0:
        atm_pcr = round(put_vol_atm / call_vol_atm, 4)

    # ── Regime classification (prefer ATM PCR if available) ──────────────
    reference_pcr = atm_pcr if atm_pcr is not None else full_pcr
    if reference_pcr is None:
        regime = "UNAVAILABLE"
    elif reference_pcr >= _PCR_BEARISH_THRESHOLD:
        regime = "PUT_DOMINANT"
    elif reference_pcr <= _PCR_BULLISH_THRESHOLD:
        regime = "CALL_DOMINANT"
    else:
        regime = "NEUTRAL"

    return {
        "volume_pcr": full_pcr,
        "volume_pcr_atm": atm_pcr,
        "volume_pcr_regime": regime,
        "call_volume_total": call_vol_total,
        "put_volume_total": put_vol_total,
        "call_volume_atm": call_vol_atm,
        "put_volume_atm": put_vol_atm,
    }
