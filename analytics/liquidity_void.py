"""
Liquidity Void Detector

Identifies areas with extremely low open interest.

When price enters these regions,
markets often move quickly because
there is little dealer liquidity.

This is a core concept used by options desks
to identify potential fast moves.
"""

import pandas as pd


def detect_liquidity_voids(option_chain, threshold=50):

    if option_chain.empty:
        return []

    df = option_chain.copy()

    grouped = df.groupby("strikePrice")["openInterest"].sum()

    voids = []

    for strike, oi in grouped.items():

        if oi < threshold:
            voids.append(strike)

    return voids


def nearest_liquidity_void(spot, voids):

    if not voids:
        return None

    closest = min(voids, key=lambda x: abs(x - spot))

    return closest


def liquidity_void_signal(spot, voids):

    nearest = nearest_liquidity_void(spot, voids)

    if nearest is None:
        return None

    distance = abs(spot - nearest)

    if distance < 50:
        return "VOID_NEAR"

    return "VOID_FAR"