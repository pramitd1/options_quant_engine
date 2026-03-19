"""
Module: enhanced_strike_scoring.py

Purpose:
    Compute institutional-grade strike scoring factors that complement the
    base strike ranking model.

Role in the System:
    Part of the strategy layer. Sits alongside strike_selector.py to enrich
    each candidate strike with market-microstructure factors and tradeability
    diagnostics.

Key Outputs:
    Per-strike scores for liquidity gravity, gamma magnetism, dealer hedging
    pressure, volatility convexity, and premium efficiency, plus a composite
    enhanced_strike_score and tradeability flags.

Downstream Usage:
    Consumed by strike_selector.py (merged into ranked_strike_candidates) and
    displayed by the Streamlit operator interface.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from utils.numerics import clip, safe_float


# ---------------------------------------------------------------------------
# Default weights for the composite enhanced score
# ---------------------------------------------------------------------------

ENHANCED_SCORE_WEIGHTS = {
    "liquidity": 0.30,
    "gamma_magnetism": 0.25,
    "dealer_pressure": 0.20,
    "volatility_convexity": 0.15,
    "premium_efficiency": 0.10,
}


# ---------------------------------------------------------------------------
# 1. Liquidity Gravity
# ---------------------------------------------------------------------------

def _rank_normalize(series: pd.Series) -> pd.Series:
    """Rank-normalize a series to [0, 1] using average rank."""
    if series.empty:
        return pd.Series(0.5, index=series.index)

    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
    n = values.size
    if n <= 1:
        return pd.Series(0.5, index=series.index)

    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-12:
        return pd.Series(0.5, index=series.index)

    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Assign average ranks for tie groups to preserve previous behavior.
    tie_bounds = np.flatnonzero(
        np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1], [True]))
    )
    for idx in range(len(tie_bounds) - 1):
        start = tie_bounds[idx]
        end = tie_bounds[idx + 1]
        if end - start > 1:
            avg_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg_rank

    normalized = (ranks - 1.0) / (n - 1.0)
    return pd.Series(normalized, index=series.index)


def _safe_series(rows: pd.DataFrame, *col_names) -> pd.Series:
    """Return the first matching column as a numeric Series, or zeros."""
    for name in col_names:
        if name in rows.columns:
            return pd.to_numeric(rows[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=rows.index)


def compute_liquidity_gravity(rows: pd.DataFrame) -> pd.Series:
    volume = _safe_series(rows, "_normalized_volume", "totalTradedVolume", "VOLUME")
    oi = _safe_series(rows, "_normalized_open_interest", "openInterest", "OPEN_INT")
    oi_change = _safe_series(rows, "changeinOI", "CHANGE_IN_OI").abs()

    vol_rank = _rank_normalize(volume)
    oi_rank = _rank_normalize(oi)
    oi_change_rank = _rank_normalize(oi_change)

    return (0.4 * vol_rank + 0.4 * oi_rank + 0.2 * oi_change_rank).round(4)


# ---------------------------------------------------------------------------
# 2. Gamma Magnetism
# ---------------------------------------------------------------------------

def compute_gamma_magnetism(
    strikes: pd.Series,
    gamma_clusters: list | None,
) -> pd.Series:
    if not gamma_clusters:
        return pd.Series(0.5, index=strikes.index)

    strikes_f = strikes.astype(float)
    cluster_arr = np.array([float(c) for c in gamma_clusters if c is not None])
    if len(cluster_arr) == 0:
        return pd.Series(0.5, index=strikes.index)

    # Distance to nearest gamma cluster
    distances = np.abs(strikes_f.values[:, None] - cluster_arr[None, :]).min(axis=1)
    raw = 1.0 / (1.0 + distances)

    # Normalize to [0, 1]
    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-9:
        return pd.Series(0.5, index=strikes.index)
    normalized = (raw - rmin) / (rmax - rmin)
    return pd.Series(np.round(normalized, 4), index=strikes.index)


# ---------------------------------------------------------------------------
# 3. Dealer Hedging Pressure
# ---------------------------------------------------------------------------

_GAMMA_REGIME_SCORES = {
    "SHORT_GAMMA_ZONE": 0.9,
    "NEGATIVE_GAMMA": 0.85,
    "NEUTRAL_GAMMA": 0.5,
    "LONG_GAMMA_ZONE": 0.2,
    "POSITIVE_GAMMA": 0.15,
}

_HEDGING_BIAS_SCORES = {
    "DOWNSIDE_HEDGING_ACCELERATION": 0.9,
    "UPSIDE_HEDGING_ACCELERATION": 0.85,
    "TWO_SIDED_INSTABILITY": 0.8,
    "PINNING_DOMINANT": 0.3,
    "DOWNSIDE_PINNING": 0.35,
    "UPSIDE_PINNING": 0.35,
    "NEUTRAL": 0.5,
}


def compute_dealer_pressure(
    strikes: pd.Series,
    *,
    gamma_regime: str | None = None,
    spot_vs_flip: str | None = None,
    dealer_hedging_bias: str | None = None,
    gamma_flip_distance_pct: float | None = None,
    dealer_gamma_exposure: float | None = None,
) -> pd.Series:
    regime_score = _GAMMA_REGIME_SCORES.get(
        str(gamma_regime or "").upper().strip(), 0.5
    )

    # Flip proximity: higher when closer to flip level
    flip_dist = safe_float(gamma_flip_distance_pct, 5.0)
    flip_proximity = clip(1.0 - flip_dist / 5.0, 0.0, 1.0)

    # Hedging bias amplification
    bias_score = _HEDGING_BIAS_SCORES.get(
        str(dealer_hedging_bias or "").upper().strip(), 0.5
    )

    # Combine components
    base_pressure = (
        0.40 * regime_score
        + 0.30 * flip_proximity
        + 0.30 * bias_score
    )

    return pd.Series(round(clip(base_pressure, 0.0, 1.0), 4), index=strikes.index)


# ---------------------------------------------------------------------------
# 4. Volatility Convexity
# ---------------------------------------------------------------------------

def compute_volatility_convexity(rows: pd.DataFrame) -> pd.Series:
    gamma = _safe_series(rows, "GAMMA")
    vega = _safe_series(rows, "VEGA")

    raw = (gamma * vega).abs()

    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-12:
        return pd.Series(0.5, index=rows.index)
    normalized = (raw - rmin) / (rmax - rmin)
    return normalized.round(4)


# ---------------------------------------------------------------------------
# 5. Premium Efficiency
# ---------------------------------------------------------------------------

def compute_premium_efficiency(
    rows: pd.DataFrame,
    *,
    spot: float,
    atm_iv: float | None,
    days_to_expiry: float | None,
    expected_move: float | None = None,
) -> pd.Series:
    if expected_move is None:
        iv = safe_float(atm_iv, 0.15)
        if iv > 1.5:
            iv = iv / 100.0
        dte = max(safe_float(days_to_expiry, 1.0), 0.1)
        expected_move = float(spot) * iv * math.sqrt(dte / 365.0)

    premium = _safe_series(rows, "_normalized_last_price", "lastPrice", "LAST_PRICE")

    raw = expected_move / premium.clip(lower=0.01)

    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-9:
        return pd.Series(0.5, index=rows.index)
    normalized = (raw - rmin) / (rmax - rmin)
    return normalized.round(4)


# ---------------------------------------------------------------------------
# 6. Payoff Efficiency — composite strike efficiency for execution quality
# ---------------------------------------------------------------------------

_PAYOFF_WEIGHTS = {
    "premium_efficiency": 0.35,
    "delta_alignment": 0.25,
    "liquidity_score": 0.20,
    "distance_to_target": 0.10,
    "iv_efficiency": 0.10,
}


def compute_payoff_efficiency(
    rows: pd.DataFrame,
    *,
    spot: float,
    direction: str,
    atm_iv: float | None,
    days_to_expiry: float | None,
    support_wall: float | None = None,
    resistance_wall: float | None = None,
    expected_move: float | None = None,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Compute per-strike payoff efficiency and sub-component scores.

    Returns
    -------
    payoff_score : pd.Series
        0–100 composite payoff efficiency score.
    components : dict[str, pd.Series]
        Keys: pe_premium_eff, pe_delta_align, pe_liquidity,
        pe_dist_target, pe_iv_eff (each 0–100).
    """
    if expected_move is None:
        iv_val = safe_float(atm_iv, 0.15)
        if iv_val > 1.5:
            iv_val = iv_val / 100.0
        dte = max(safe_float(days_to_expiry, 1.0), 0.1)
        expected_move = float(spot) * iv_val * math.sqrt(dte / 365.0)

    premium = _safe_series(rows, "_normalized_last_price", "lastPrice", "LAST_PRICE")
    strikes = pd.to_numeric(
        rows.get("_normalized_strike", rows.get("strikePrice")),
        errors="coerce",
    ).fillna(float(spot))
    delta = _safe_series(rows, "DELTA")
    volume = _safe_series(rows, "_normalized_volume", "totalTradedVolume", "VOLUME")
    oi = _safe_series(rows, "_normalized_open_interest", "openInterest", "OPEN_INT")
    iv_col = _safe_series(rows, "_normalized_iv", "impliedVolatility", "IV")

    # 1. Premium efficiency: expected_move / premium
    pe_raw = expected_move / premium.clip(lower=0.01)
    pe_norm = _rank_normalize(pe_raw) * 100

    # 2. Delta alignment: prefer |delta| in [0.35, 0.55]
    delta_abs = delta.abs()
    # Score peaks at 0.45 centre, falls off outside [0.35, 0.55]
    delta_ideal = 0.45
    delta_band = 0.10
    delta_dist = (delta_abs - delta_ideal).abs()
    pe_delta = (1.0 - (delta_dist / max(delta_ideal, 1e-6)).clip(upper=1.0)) * 100

    # 3. Liquidity: rank-normalised blend of volume + OI
    liq_blend = 0.5 * _rank_normalize(volume) + 0.5 * _rank_normalize(oi)
    pe_liq = liq_blend * 100

    # 4. Distance to target: penalise strikes far from expected move endpoint
    if direction == "CALL":
        target_level = float(spot) + expected_move
    else:
        target_level = float(spot) - expected_move
    target_dist = (strikes - target_level).abs()
    pe_dist = (1.0 - _rank_normalize(target_dist)) * 100

    # 5. IV efficiency: penalise excessively high IV; rank-invert
    pe_iv = (1.0 - _rank_normalize(iv_col)) * 100

    w = _PAYOFF_WEIGHTS
    composite = (
        w["premium_efficiency"] * pe_norm
        + w["delta_alignment"] * pe_delta
        + w["liquidity_score"] * pe_liq
        + w["distance_to_target"] * pe_dist
        + w["iv_efficiency"] * pe_iv
    ).round(1)

    components = {
        "pe_premium_eff": pe_norm.round(1),
        "pe_delta_align": pe_delta.round(1),
        "pe_liquidity": pe_liq.round(1),
        "pe_dist_target": pe_dist.round(1),
        "pe_iv_eff": pe_iv.round(1),
    }

    return composite, components


# ---------------------------------------------------------------------------
# Tradeability flags
# ---------------------------------------------------------------------------

_MIN_INTRADAY_VOLUME = 500
_MIN_OVERNIGHT_VOLUME = 2000
_MIN_LIQUIDITY_OI = 10000
_MAX_PREMIUM_RATIO = 5.0  # premium / expected_move


def compute_tradeability_flags(
    rows: pd.DataFrame,
    *,
    spot: float,
    atm_iv: float | None,
    days_to_expiry: float | None,
    expected_move: float | None = None,
) -> dict[str, pd.Series]:
    volume = _safe_series(rows, "_normalized_volume", "totalTradedVolume", "VOLUME")
    oi = _safe_series(rows, "_normalized_open_interest", "openInterest", "OPEN_INT")
    premium = _safe_series(rows, "_normalized_last_price", "lastPrice", "LAST_PRICE")

    if expected_move is None:
        iv_val = safe_float(atm_iv, 0.15)
        if iv_val > 1.5:
            iv_val = iv_val / 100.0
        dte = max(safe_float(days_to_expiry, 1.0), 0.1)
        expected_move = float(spot) * iv_val * math.sqrt(dte / 365.0)

    flags = {
        "tradable_intraday": volume >= _MIN_INTRADAY_VOLUME,
        "tradable_overnight": volume >= _MIN_OVERNIGHT_VOLUME,
        "liquidity_ok": oi >= _MIN_LIQUIDITY_OI,
        "premium_reasonable": premium <= max(expected_move * _MAX_PREMIUM_RATIO, 1.0),
    }

    return flags


# ---------------------------------------------------------------------------
# Composite enhanced scoring
# ---------------------------------------------------------------------------

def compute_enhanced_strike_scores(
    rows: pd.DataFrame,
    *,
    spot: float,
    direction: str,
    gamma_clusters: list | None = None,
    gamma_regime: str | None = None,
    spot_vs_flip: str | None = None,
    dealer_hedging_bias: str | None = None,
    gamma_flip_distance_pct: float | None = None,
    dealer_gamma_exposure: float | None = None,
    atm_iv: float | None = None,
    days_to_expiry: float | None = None,
    vol_surface_regime: str | None = None,
    weights: dict[str, float] | None = None,
    support_wall: float | None = None,
    resistance_wall: float | None = None,
) -> pd.DataFrame:
    """Compute all enhanced scoring factors and the composite score.

    Returns a DataFrame aligned with ``rows`` containing per-strike factor
    scores, tradeability flags, context fields, and the weighted composite
    ``enhanced_strike_score``.
    """
    if rows.empty:
        return pd.DataFrame()

    w = weights or ENHANCED_SCORE_WEIGHTS
    strikes = pd.to_numeric(
        rows.get("_normalized_strike", rows.get("strikePrice")),
        errors="coerce",
    ).fillna(0.0)

    # Factor scores
    liquidity = compute_liquidity_gravity(rows)
    gamma_mag = compute_gamma_magnetism(strikes, gamma_clusters)
    dealer = compute_dealer_pressure(
        strikes,
        gamma_regime=gamma_regime,
        spot_vs_flip=spot_vs_flip,
        dealer_hedging_bias=dealer_hedging_bias,
        gamma_flip_distance_pct=gamma_flip_distance_pct,
        dealer_gamma_exposure=dealer_gamma_exposure,
    )
    convexity = compute_volatility_convexity(rows)
    # Compute expected_move once for all sub-functions
    _iv = safe_float(atm_iv, 0.15)
    if _iv > 1.5:
        _iv = _iv / 100.0
    _dte = max(safe_float(days_to_expiry, 1.0), 0.1)
    _expected_move = float(spot) * _iv * math.sqrt(_dte / 365.0)

    prem_eff = compute_premium_efficiency(
        rows, spot=spot, atm_iv=atm_iv, days_to_expiry=days_to_expiry,
        expected_move=_expected_move,
    )

    # Composite
    composite = (
        w.get("liquidity", 0.30) * liquidity
        + w.get("gamma_magnetism", 0.25) * gamma_mag
        + w.get("dealer_pressure", 0.20) * dealer
        + w.get("volatility_convexity", 0.15) * convexity
        + w.get("premium_efficiency", 0.10) * prem_eff
    )
    # Scale to 0-100
    enhanced_score = (composite * 100).round(0).astype(int)

    # Tradeability
    flags = compute_tradeability_flags(
        rows, spot=spot, atm_iv=atm_iv, days_to_expiry=days_to_expiry,
        expected_move=_expected_move,
    )

    # Distance from spot
    spot_f = float(spot)
    dist_pts = (strikes - spot_f).round(2)
    dist_pct = ((strikes - spot_f) / max(spot_f, 1e-6) * 100).round(2)

    # Payoff efficiency
    payoff_score, payoff_components = compute_payoff_efficiency(
        rows,
        spot=spot,
        direction=direction,
        atm_iv=atm_iv,
        days_to_expiry=days_to_expiry,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        expected_move=_expected_move,
    )

    result_data = {
        "liquidity_score": liquidity,
        "gamma_magnetism": gamma_mag,
        "dealer_pressure": dealer,
        "convexity_score": convexity,
        "premium_efficiency": prem_eff,
        "enhanced_strike_score": enhanced_score,
        "payoff_efficiency_score": payoff_score,
        **payoff_components,
        "distance_from_spot_pts": dist_pts,
        "distance_from_spot_pct": dist_pct,
        "gamma_regime": gamma_regime or "",
        "spot_vs_flip": spot_vs_flip or "",
        "dealer_hedging_bias": dealer_hedging_bias or "",
        "vol_surface_regime": vol_surface_regime or "",
        "tradable_intraday": flags["tradable_intraday"],
        "tradable_overnight": flags["tradable_overnight"],
        "liquidity_ok": flags["liquidity_ok"],
        "premium_reasonable": flags["premium_reasonable"],
    }

    return pd.DataFrame(result_data, index=rows.index)
