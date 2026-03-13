"""
Gamma Flip Level Detection

Uses signed strike-wise gamma exposure so the flip and regime framework are
consistent with the rest of the engine's Greek stack.
"""

import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from analytics.market_gamma_map import calculate_market_gamma


def _interpolate_zero_crossing(left_strike, left_value, right_strike, right_value):
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

    if flip is None:
        return None

    return spot - flip


def gamma_regime(spot, flip):

    if flip is None:
        return "UNKNOWN"

    if spot > flip:
        return "POSITIVE_GAMMA"

    return "NEGATIVE_GAMMA"
