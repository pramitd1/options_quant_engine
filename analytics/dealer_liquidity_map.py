"""
Module: dealer_liquidity_map.py

Purpose:
    Compute dealer liquidity map analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd


def _to_sorted_levels(levels):
    """
    Purpose:
        Convert sorted levels into the representation expected downstream.
    
    Context:
        Internal helper within the analytics layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        levels (Any): Input associated with levels.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if levels is None:
        return []

    if isinstance(levels, pd.Series):
        levels = list(levels.index)

    if isinstance(levels, pd.Index):
        levels = list(levels)

    if not isinstance(levels, list):
        try:
            levels = list(levels)
        except Exception:
            return []

    cleaned = []
    for x in levels:
        try:
            cleaned.append(float(x))
        except Exception:
            continue

    return sorted(set(cleaned))


def nearest_support_resistance(spot, levels):
    """
    Split levels into support / resistance relative to spot.
    """

    levels = _to_sorted_levels(levels)

    supports = [lvl for lvl in levels if lvl <= spot]
    resistances = [lvl for lvl in levels if lvl >= spot]

    next_support = max(supports) if supports else None
    next_resistance = min(resistances) if resistances else None

    return next_support, next_resistance


def estimate_squeeze_zone(spot, gamma_flip, gamma_clusters):
    """
    Estimate a likely squeeze / acceleration zone.

    Preference:
    1. nearest gamma cluster ahead of spot
    2. gamma flip
    """

    gamma_clusters = _to_sorted_levels(gamma_clusters)

    if gamma_clusters:
        above = [lvl for lvl in gamma_clusters if lvl >= spot]
        below = [lvl for lvl in gamma_clusters if lvl <= spot]

        if above:
            return min(above)

        if below:
            return max(below)

    return gamma_flip


def summarize_vacuum(vacuum_zones):
    """
    Keep only a compact trader-friendly subset of vacuum zones.
    """

    if not vacuum_zones:
        return []

    cleaned = []
    for zone in vacuum_zones:
        try:
            low, high = zone
            cleaned.append((float(low), float(high)))
        except Exception:
            continue

    return cleaned[:5]


def predict_large_move_band(
    spot,
    gamma_flip,
    next_support,
    next_resistance,
    vacuum_zones
):
    """
    Very simple structural move-band estimate.

    If spot is near a vacuum or far below/above flip, estimate
    the nearest likely impulse band.
    """

    if vacuum_zones:
        for low, high in vacuum_zones:
            if low <= spot <= high:
                return {
                    "expected_band_low": low,
                    "expected_band_high": high,
                    "band_reason": "INSIDE_VACUUM"
                }

    if gamma_flip is not None:
        if spot < gamma_flip and next_resistance is not None:
            return {
                "expected_band_low": spot,
                "expected_band_high": next_resistance,
                "band_reason": "MOVE_TO_RESISTANCE"
            }

        if spot > gamma_flip and next_support is not None:
            return {
                "expected_band_low": next_support,
                "expected_band_high": spot,
                "band_reason": "MOVE_TO_SUPPORT"
            }

    return {
        "expected_band_low": next_support,
        "expected_band_high": next_resistance,
        "band_reason": "STRUCTURAL_RANGE"
    }


def build_dealer_liquidity_map(
    spot,
    gamma_flip,
    liquidity_levels,
    support_wall,
    resistance_wall,
    gamma_clusters,
    vacuum_zones
):
    """
    Build a compact structural market map for traders.
    """

    combined_levels = []

    for bucket in [liquidity_levels, gamma_clusters]:
        if bucket:
            combined_levels.extend(_to_sorted_levels(bucket))

    if support_wall is not None:
        combined_levels.append(float(support_wall))

    if resistance_wall is not None:
        combined_levels.append(float(resistance_wall))

    combined_levels = sorted(set(combined_levels))

    next_support, next_resistance = nearest_support_resistance(
        spot,
        combined_levels
    )

    squeeze_zone = estimate_squeeze_zone(
        spot,
        gamma_flip,
        gamma_clusters
    )

    compact_vacuums = summarize_vacuum(vacuum_zones)

    move_band = predict_large_move_band(
        spot,
        gamma_flip,
        next_support,
        next_resistance,
        compact_vacuums
    )

    return {
        "next_support": next_support,
        "next_resistance": next_resistance,
        "dealer_flip": gamma_flip,
        "gamma_squeeze_zone": squeeze_zone,
        "vacuum_zones_compact": compact_vacuums,
        "expected_band_low": move_band["expected_band_low"],
        "expected_band_high": move_band["expected_band_high"],
        "band_reason": move_band["band_reason"]
    }