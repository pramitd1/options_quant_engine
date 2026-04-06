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
from __future__ import annotations

import math

import pandas as pd
import numpy as np

from utils.regime_normalization import normalize_iv_decimal
from utils.math_helpers import norm_cdf as _norm_cdf


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

    iv_decimal = normalize_iv_decimal(atm_iv, default=None)
    if iv_decimal is None:
        return "UNKNOWN"

    if iv_decimal > 0.25:
        return "HIGH_VOL"

    if iv_decimal < 0.15:
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
    # Normalize IV to decimal internally so outputs are unit-invariant.
    df["IV"] = df["IV"].apply(lambda value: normalize_iv_decimal(value, default=np.nan))
    df = df.dropna(subset=["IV"])
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

    def _infer_tte_years(data: pd.DataFrame) -> float:
        for col in ("TTE", "expiry_days", "days_to_expiry"):
            if col in data.columns:
                vals = pd.to_numeric(data[col], errors="coerce").dropna()
                if len(vals) > 0:
                    value = float(vals.median())
                    if col == "TTE":
                        return max(value, 1.0 / 365.0)
                    return max(value / 365.0, 1.0 / 365.0)
        return 30.0 / 365.0

    def _bs_delta(option_type: str, strike: float, sigma: float, t_years: float) -> float | None:
        if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
            return None
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / max(sigma * sqrt_t, 1e-9)
        call_delta = _norm_cdf(d1)
        if option_type == "CE":
            return float(call_delta)
        return float(call_delta - 1.0)

    def _nearest_iv(side_df, option_type: str, target_moneyness_sign: float) -> float | None:
        """Return IV at the strike closest to target delta, with moneyness fallback."""
        if side_df.empty:
            return None

        side_df = side_df.copy()
        side_df["STRIKE_PR"] = pd.to_numeric(side_df["STRIKE_PR"], errors="coerce")
        side_df = side_df.dropna(subset=["STRIKE_PR", "IV"])
        if side_df.empty:
            return None

        t_years = _infer_tte_years(side_df)
        atm_iv_decimal = normalize_iv_decimal(atm_vol(df, spot), default=None)
        if atm_iv_decimal is not None and atm_iv_decimal > 0 and t_years > 0:
            target_abs_delta = max(min(float(delta_target), 0.49), 0.01)

            # Vectorised Black-Scholes delta path (replaces iterrows hot path).
            sigma = float(atm_iv_decimal)
            sqrt_t = math.sqrt(float(t_years))
            denom = max(sigma * sqrt_t, 1e-9)

            strikes = side_df["STRIKE_PR"].to_numpy(dtype=float)
            valid_strike_mask = strikes > 0.0
            abs_delta = np.full_like(strikes, np.nan, dtype=float)
            if valid_strike_mask.any() and spot > 0:
                d1 = (np.log(float(spot) / strikes[valid_strike_mask]) + 0.5 * sigma * sigma * float(t_years)) / denom
                call_delta = np.vectorize(_norm_cdf, otypes=[float])(d1)
                if option_type == "CE":
                    deltas = call_delta
                else:
                    deltas = call_delta - 1.0
                abs_delta[valid_strike_mask] = np.abs(deltas)

            side_df["_abs_delta"] = abs_delta
            valid = side_df.dropna(subset=["_abs_delta"])
            if not valid.empty:
                valid = valid.copy()
                valid["_delta_dist"] = (valid["_abs_delta"] - target_abs_delta).abs()
                nearest = valid.sort_values(["_delta_dist", "STRIKE_PR"]).iloc[0]
                iv_val = float(nearest["IV"])
                if iv_val > 0:
                    return iv_val

        # Fallback moneyness proxy when delta path is unavailable.
        target_delta = max(float(delta_target), 1e-6)
        moneyness_offset = spot * 0.015 * (0.25 / target_delta)
        target_strike = spot + target_moneyness_sign * moneyness_offset
        side_df["_dist"] = (side_df["STRIKE_PR"] - target_strike).abs()
        nearest = side_df.sort_values("_dist").iloc[0]
        iv_val = float(nearest["IV"])
        return iv_val if iv_val > 0 else None

    calls = df[df["OPTION_TYP"] == "CE"]
    puts = df[df["OPTION_TYP"] == "PE"]

    call_iv = _nearest_iv(calls, option_type="CE", target_moneyness_sign=+1.0)
    put_iv = _nearest_iv(puts, option_type="PE", target_moneyness_sign=-1.0)

    if call_iv is None or put_iv is None:
        return result

    # Report RR and component IVs in volatility points to preserve familiar
    # thresholds (e.g., +/-0.5 vol points).
    put_iv_points = put_iv * 100.0
    call_iv_points = call_iv * 100.0
    # Convention used throughout this codebase:
    #   rr = IV(25d put) − IV(25d call)
    # Positive RR → put skew dominates (hedging demand / downside fear)
    # Negative RR → call skew dominates (upside demand / melt-up)
    # Note: this is the inverse of the Bloomberg/dealer convention
    # (call − put), which reports positive RR for call skew.
    # The direction_probability_head inversion (-rr) corrects for this.
    rr = round(put_iv_points - call_iv_points, 4)
    result["rr_value"] = rr
    result["put_iv_25d"] = round(put_iv_points, 4)
    result["call_iv_25d"] = round(call_iv_points, 4)
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
) -> dict:  # noqa: E501
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


def compute_atm_straddle_price(option_chain: pd.DataFrame, spot: float) -> dict:
    """
    Compute the ATM straddle price from the front-expiry option chain.

    The straddle price = ATM call last price + ATM put last price.
    This is the market's direct estimate of how many index points it expects
    the underlying to move by expiry. It is more grounded than IV-derived
    expected move because it uses actual traded premiums, not implied vol.

    Returns
    -------
    dict with keys:
        atm_straddle_price  – float | None  (call_price + put_price)
        atm_call_price      – float | None
        atm_put_price       – float | None
        expected_move_up    – float | None  (spot + straddle)
        expected_move_down  – float | None  (spot - straddle)
        expected_move_pct   – float | None  (straddle / spot as percentage)
    """
    _empty = {
        "atm_straddle_price": None,
        "atm_call_price": None,
        "atm_put_price": None,
        "expected_move_up": None,
        "expected_move_down": None,
        "expected_move_pct": None,
    }

    if option_chain is None or option_chain.empty or spot is None or spot <= 0:
        return _empty

    df = option_chain.copy()

    # Infer price column.
    price_col = None
    for col in ("LAST_PRICE", "lastPrice", "LTP", "ltp"):
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        return _empty

    strike_col = "STRIKE_PR" if "STRIKE_PR" in df.columns else (
        "strikePrice" if "strikePrice" in df.columns else None
    )
    if strike_col is None:
        return _empty

    # Restrict to front expiry.
    if "EXPIRY_DT" in df.columns:
        try:
            df["_expiry_dt"] = pd.to_datetime(df["EXPIRY_DT"], errors="coerce")
            front_exp = df["_expiry_dt"].dropna().min()
            if pd.notna(front_exp):
                df = df[df["_expiry_dt"] == front_exp]
        except Exception:
            pass

    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[strike_col, price_col])
    df = df[df[price_col] > 0]
    if df.empty:
        return _empty

    # Find ATM strike — the one closest to spot.
    df["_dist"] = (df[strike_col] - float(spot)).abs()
    # Use the actual strike value at minimum distance row.
    atm_row = df.sort_values("_dist").iloc[0]
    atm_strike = float(atm_row[strike_col])

    calls = df[(df["OPTION_TYP"] == "CE") & (df[strike_col] == atm_strike)]
    puts  = df[(df["OPTION_TYP"] == "PE") & (df[strike_col] == atm_strike)]

    if calls.empty or puts.empty:
        # Fallback must remain a true straddle: pick nearest strike that has
        # both CE and PE prices, rather than mixing different strikes.
        counts = (
            df.pivot_table(index=strike_col, columns="OPTION_TYP", values=price_col, aggfunc="count")
            .fillna(0)
        )
        valid_strikes = counts[(counts.get("CE", 0) > 0) & (counts.get("PE", 0) > 0)].index.tolist()
        if not valid_strikes:
            return _empty
        best_strike = min(valid_strikes, key=lambda s: abs(float(s) - float(spot)))
        calls = df[(df["OPTION_TYP"] == "CE") & (df[strike_col] == best_strike)]
        puts = df[(df["OPTION_TYP"] == "PE") & (df[strike_col] == best_strike)]
        if calls.empty or puts.empty:
            return _empty

    atm_call_price = round(float(calls[price_col].iloc[0]), 2)
    atm_put_price  = round(float(puts[price_col].iloc[0]), 2)
    straddle_price = round(atm_call_price + atm_put_price, 2)

    expected_move_pct = round(straddle_price / float(spot) * 100.0, 3) if spot > 0 else None

    return {
        "atm_straddle_price": straddle_price,
        "atm_call_price": atm_call_price,
        "atm_put_price": atm_put_price,
        "expected_move_up":   round(float(spot) + straddle_price, 2),
        "expected_move_down": round(float(spot) - straddle_price, 2),
        "expected_move_pct":  expected_move_pct,
    }


def iv_hv_spread(atm_iv: float | None, realized_hv: float | None) -> dict:
    """
    Compute the IV − HV spread to assess option richness vs cheapness.

    A positive spread means implied vol > recent realised vol:
    options are 'expensive' — sellers have an edge.
    A negative spread means implied vol < realised vol:
    options are 'cheap' — buyers have a slight structural edge.

    Both inputs should be annualised decimal fractions (e.g., 0.18 for 18%).

    Returns
    -------
    dict with keys:
        iv_hv_spread    – float | None  (positive = IV rich, negative = IV cheap)
        iv_hv_regime    – "IV_RICH" | "IV_FAIR" | "IV_CHEAP" | "UNAVAILABLE"
        atm_iv_pct      – float | None  (ATM IV as percentage)
        realized_hv_pct – float | None
    """
    _empty = {
        "iv_hv_spread": None,
        "iv_hv_regime": "UNAVAILABLE",
        "atm_iv_pct": None,
        "realized_hv_pct": None,
    }

    if atm_iv is None or realized_hv is None or realized_hv <= 0:
        return _empty

    # Normalise: both should be decimal fractions (0–1 range).
    from utils.regime_normalization import normalize_iv_decimal
    atm_iv_dec = normalize_iv_decimal(atm_iv, default=None)
    hv_dec = normalize_iv_decimal(realized_hv, default=None)

    if atm_iv_dec is None or hv_dec is None or hv_dec <= 0:
        return _empty

    spread = round(atm_iv_dec - hv_dec, 4)

    # ±2 vol points tolerance band for "FAIR"
    if spread > 0.02:
        regime = "IV_RICH"
    elif spread < -0.02:
        regime = "IV_CHEAP"
    else:
        regime = "IV_FAIR"

    return {
        "iv_hv_spread": spread,
        "iv_hv_regime": regime,
        "atm_iv_pct": round(atm_iv_dec * 100.0, 2),
        "realized_hv_pct": round(hv_dec * 100.0, 2),
    }

