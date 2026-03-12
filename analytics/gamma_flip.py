"""
Gamma Flip Level Detection

Gamma flip = strike where dealers switch
from LONG gamma to SHORT gamma.

Below flip:
    dealers short gamma
    volatility expands

Above flip:
    dealers long gamma
    volatility suppressed
"""

import pandas as pd


def gamma_flip_level(option_chain):

    if option_chain.empty:
        return None

    df = option_chain.copy()

    grouped = df.groupby("strikePrice")["openInterest"].sum()

    grouped = grouped.sort_index()

    cumulative = grouped.cumsum()

    total_oi = grouped.sum()

    threshold = total_oi / 2

    for strike, value in cumulative.items():

        if value >= threshold:
            return strike

    return None


def gamma_flip_distance(spot, flip):

    if flip is None:
        return None

    return spot - flip


def gamma_regime(spot, flip):

    if flip is None:
        return "UNKNOWN"

    if spot > flip:
        return "LONG_GAMMA_ZONE"

    return "SHORT_GAMMA_ZONE"