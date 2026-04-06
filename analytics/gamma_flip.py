"""
Module: gamma_flip.py

Purpose:
    Compute gamma flip analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from analytics.market_gamma_map import calculate_market_gamma
from config.analytics_feature_policy import get_gamma_flip_policy_config


def _interpolate_zero_crossing(left_strike, left_value, right_strike, right_value):
    """
    Purpose:
        Linearly interpolate the point where a signed series crosses zero.

    Context:
        Used within the gamma flip workflow. The module sits in the analytics layer that turns option-chain structure into features for the signal engine and overlays.

    Inputs:
        left_strike (Any): Strike on the left side of the interpolation bracket.
        left_value (Any): Signed metric value at the left interpolation boundary.
        right_strike (Any): Strike on the right side of the interpolation bracket.
        right_value (Any): Signed metric value at the right interpolation boundary.

    Returns:
        float | None: Interpolated crossing value, or `None` when interpolation is not possible.

    Notes:
        Internal helper that keeps the surrounding trading logic compact and readable.
    """
    try:
        left_strike = float(left_strike)
        right_strike = float(right_strike)
        left_value = float(left_value)
        right_value = float(right_value)
    except Exception:
        return None

    denominator = right_value - left_value
    if denominator == 0:
        return None

    weight = -left_value / denominator
    if 0.0 <= weight <= 1.0:
        return left_strike + weight * (right_strike - left_strike)
    return None


def gamma_flip_level(option_chain, spot=None, strike_window_steps: int = 10):
    """
    Purpose:
        Process gamma flip level for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        spot (Any): Input associated with spot.
        strike_window_steps (int): Input associated with strike window steps.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain is None or option_chain.empty:
        return None

    df = option_chain.copy()
    if spot is None and "strikePrice" in df.columns:
        try:
            spot = float(pd.to_numeric(df["strikePrice"], errors="coerce").median())
        except Exception:
            spot = None

    df = front_expiry_atm_slice(df, spot=spot, strike_window_steps=strike_window_steps)
    gex = calculate_market_gamma(df)
    if gex is None or len(gex) == 0:
        return None

    gex = gex.sort_index()
    gex = gex[gex.abs() > 1e-9]
    if len(gex) == 0:
        return None

    sign_changes = []
    prev_strike = None
    prev_value = None
    for strike, value in gex.items():
        if prev_value is not None and prev_value * value < 0:
            # Linear interpolation gives a smoother estimate than simply taking
            # the midpoint between two neighboring strikes with opposite signs.
            interpolated = _interpolate_zero_crossing(prev_strike, prev_value, strike, value)
            if interpolated is None:
                interpolated = (float(prev_strike) + float(strike)) / 2.0
            sign_changes.append(float(interpolated))
        prev_strike = strike
        prev_value = value

    if sign_changes:
        if spot is None:
            return round(float(sign_changes[0]), 2)
        return round(min(sign_changes, key=lambda x: abs(x - float(spot))), 2)

    # If the signed series never crosses zero cleanly, fall back to the strike
    # where cumulative gamma is closest to neutral.
    cumulative = gex.cumsum()
    if len(cumulative) == 0:
        return None

    if spot is None:
        return round(float(cumulative.abs().idxmin()), 2)

    candidate_strikes = cumulative.index.tolist()
    return round(
        float(
            min(
                candidate_strikes,
                key=lambda strike: (
                    abs(float(cumulative.loc[strike])),
                    abs(float(strike) - float(spot)),
                ),
            )
        ),
        2,
    )


def gamma_flip_distance(spot, flip):

    """
    Purpose:
        Process gamma flip distance for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        flip (Any): Input associated with flip.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if flip is None:
        return None

    return spot - flip


def gamma_regime(spot, flip):

    """
    Classify the current gamma regime as POSITIVE_GAMMA, NEUTRAL_GAMMA, or
    NEGATIVE_GAMMA relative to the estimated gamma-flip level.

    A configurable neutral band (``neutral_band_pct`` in
    ``GammaFlipPolicyConfig``) prevents the classifier from flipping between
    positive and negative on every tick when spot is hovering around the flip
    level.  The band is expressed as a percentage of the flip price:

        neutral when  |spot - flip| / flip * 100  <=  neutral_band_pct

    Returns
    -------
    str
        ``"POSITIVE_GAMMA"``  – spot is above the neutral band
        ``"NEUTRAL_GAMMA"``   – spot is within the neutral band around flip
        ``"NEGATIVE_GAMMA"``  – spot is below the neutral band
        ``"UNKNOWN"``         – flip level is unavailable
    """
    if flip is None:
        return "UNKNOWN"

    cfg = get_gamma_flip_policy_config()
    flip_f = float(flip)
    spot_f = float(spot)

    if flip_f != 0.0:
        distance_pct = abs(spot_f - flip_f) / abs(flip_f) * 100.0
    else:
        distance_pct = 0.0

    if distance_pct <= cfg.neutral_band_pct:
        return "NEUTRAL_GAMMA"

    if spot_f > flip_f:
        return "POSITIVE_GAMMA"

    return "NEGATIVE_GAMMA"
