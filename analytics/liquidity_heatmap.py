"""
Liquidity Heatmap Generator

Identifies high-liquidity strike zones.

These zones act as support and resistance.
"""

import pandas as pd


def build_liquidity_heatmap(option_chain):

    if option_chain.empty:
        return {}

    grouped = option_chain.groupby("strikePrice")[
        "openInterest"
    ].sum()

    heatmap = grouped.sort_values(ascending=False)

    return heatmap


def strongest_liquidity_levels(option_chain, top_n=5):

    heatmap = build_liquidity_heatmap(option_chain)

    levels = heatmap.head(top_n).index.tolist()

    return levels


def liquidity_signal(spot, levels):

    if not levels:
        return None

    closest = min(levels, key=lambda x: abs(x - spot))

    distance = abs(spot - closest)

    if distance < 50:
        return "STRONG_LEVEL_NEAR"

    return "LEVEL_FAR"