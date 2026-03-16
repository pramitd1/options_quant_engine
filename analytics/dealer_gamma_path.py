"""
Module: dealer_gamma_path.py

Purpose:
    Compute dealer gamma path analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import numpy as np
import pandas as pd


def simulate_gamma_path(option_chain, spot, step=25, range_points=None):
    """
    Purpose:
        Simulate gamma path across the requested scenario space.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        spot (Any): Input associated with spot.
        step (Any): Input associated with step.
        range_points (Any): Input associated with range points.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain is None or len(option_chain) == 0 or spot is None:
        return np.array([]), []

    strike_col = "strikePrice" if "strikePrice" in option_chain.columns else "STRIKE_PR"
    oi_col = "openInterest" if "openInterest" in option_chain.columns else "OPEN_INT"

    df = option_chain.copy()
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[oi_col] = pd.to_numeric(df.get(oi_col), errors="coerce")
    df["GAMMA"] = pd.to_numeric(df.get("GAMMA"), errors="coerce")
    df["OPTION_TYP"] = df.get("OPTION_TYP", "").astype(str).str.upper()

    valid = (
        df[strike_col].notna()
        & df[oi_col].notna()
        & df["GAMMA"].notna()
        & np.isfinite(df[strike_col])
        & np.isfinite(df[oi_col])
        & np.isfinite(df["GAMMA"])
        & (df[oi_col] > 0)
        & (df["GAMMA"] > 0)
        & df["OPTION_TYP"].isin(["CE", "PE"])
    )
    df = df.loc[valid, [strike_col, oi_col, "GAMMA", "OPTION_TYP"]].copy()

    strikes = np.sort(df[strike_col].unique())
    if len(strikes) == 0:
        return np.array([]), []

    inferred_step = None
    if len(strikes) > 1:
        diffs = np.diff(strikes)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            inferred_step = float(np.median(diffs))

    if inferred_step:
        step = max(1, int(round(inferred_step)))

    if range_points is None:
        spot_band = max(float(spot) * 0.035, step * 12)
        strike_band = step * min(max(len(strikes) // 2, 12), 40)
        range_points = int(max(spot_band, strike_band))

    prices = np.arange(
        spot - range_points,
        spot + range_points,
        step
    )

    records = df.to_dict("records")
    gamma_curve = []

    for price in prices:

        gamma_total = 0.0

        for row in records:
            strike = float(row[strike_col])
            oi = float(row[oi_col])
            gamma = float(row["GAMMA"])
            option_type = row["OPTION_TYP"]
            signed = 1.0 if option_type == "CE" else -1.0 if option_type == "PE" else 0.0

            # Decay strike influence as the hypothetical price moves away.
            width = max(abs(strike) * 0.01, float(step), 1.0)
            distance = (strike - float(price)) / width
            localized_gamma = gamma * np.exp(-0.5 * distance * distance)
            gamma_total += localized_gamma * oi * strike * signed

        gamma_curve.append(float(gamma_total))

    return prices, gamma_curve


def detect_gamma_squeeze(prices, gamma_curve):

    """
    Purpose:
        Detect gamma squeeze from the available inputs.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        prices (Any): Input associated with prices.
        gamma_curve (Any): Input associated with gamma curve.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if len(gamma_curve) < 2:
        return None

    prices = np.asarray(prices, dtype=float)
    curve = np.asarray(gamma_curve, dtype=float)

    if len(prices) != len(curve) or len(prices) < 2:
        return None

    finite = np.isfinite(prices) & np.isfinite(curve)
    prices = prices[finite]
    curve = curve[finite]

    if len(prices) < 2:
        return "NORMAL"

    slope = np.gradient(curve, prices)
    if not np.isfinite(slope).any():
        return "NORMAL"

    max_slope = float(np.nanmax(np.abs(slope)))
    gross_curve = float(np.nansum(np.abs(curve)))

    if gross_curve <= 0:
        return "NORMAL"

    normalized_slope = max_slope / gross_curve

    if normalized_slope > 0.015:
        return "GAMMA_SQUEEZE"

    return "NORMAL"