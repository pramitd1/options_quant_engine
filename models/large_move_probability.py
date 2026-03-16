"""
Module: large_move_probability.py

Purpose:
    Implement large move probability modeling logic used by predictive or heuristic components.

Role in the System:
    Part of the modeling layer that builds statistical features and predictive estimates.

Key Outputs:
    Model-ready feature sets, fitted estimators, or predictive outputs.

Downstream Usage:
    Consumed by analytics, the probability stack, and research workflows.
"""
from __future__ import annotations

from typing import Optional

from config.large_move_policy import get_large_move_probability_config


def _clip(x: float, lo: float, hi: float) -> float:
    """
    Purpose:
        Clamp a numeric value to the configured bounds.

    Context:
        Used within the large move probability workflow. The module sits in the modeling layer that turns features into scores and probabilities.

    Inputs:
        x (float): Raw scalar input supplied by the caller.
        lo (float): Inclusive lower bound for the returned value.
        hi (float): Inclusive upper bound for the returned value.

    Returns:
        float | int: Bounded value returned by the helper.

    Notes:
        Internal helper that keeps the surrounding trading logic compact and readable.
    """
    return max(lo, min(hi, x))


def _safe_float(x, default: float = 0.0) -> float:
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Used within the large move probability workflow. The module sits in the modeling layer that turns features into scores and probabilities.

    Inputs:
        x (Any): Raw scalar input supplied by the caller.
        default (float): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding trading logic compact and readable.
    """
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def large_move_probability(
    gamma_regime: str,
    vacuum_state: str,
    hedging_bias: str,
    smart_money_flow: str,
    *,
    gamma_flip_distance_pct: Optional[float] = None,
    vacuum_strength: Optional[float] = None,
    hedging_flow_ratio: Optional[float] = None,
    smart_money_flow_score: Optional[float] = None,
    atm_iv_percentile: Optional[float] = None,
    intraday_range_pct: Optional[float] = None,
) -> float:
    """
    Purpose:
        Estimate the probability of a large intraday move with a bounded
        additive evidence model.

    Context:
        This model is the rule-based probability leg used by the signal engine.
        It translates structural gamma, liquidity, hedging, flow, and
        volatility cues into a move-probability estimate even when the ML model
        is unavailable.

    Inputs:
        gamma_regime (str): Gamma regime classification such as
        `NEGATIVE_GAMMA` or `LONG_GAMMA_ZONE`.
        vacuum_state (str): Liquidity-vacuum regime around spot.
        hedging_bias (str): Dealer-hedging bias classification.
        smart_money_flow (str): Directional institutional-flow regime.
        gamma_flip_distance_pct (Optional[float]): `abs(spot - gamma_flip) /
        spot * 100`; smaller values imply spot is closer to the unstable flip.
        vacuum_strength (Optional[float]): Normalized vacuum score from 0 to 1.
        hedging_flow_ratio (Optional[float]): Normalized hedging-flow ratio,
        usually bounded to `-1..1`.
        smart_money_flow_score (Optional[float]): Normalized smart-money-flow
        score, usually bounded to `-1..1`.
        atm_iv_percentile (Optional[float]): Normalized ATM-IV percentile from
        0 to 1.
        intraday_range_pct (Optional[float]): Realized range normalized by the
        expected baseline range.

    Returns:
        float: Probability estimate clipped to the configured floor and ceiling.

    Notes:
        The model is intentionally additive and bounded so each feature can be
        interpreted as evidence for or against expansion rather than as a black
        box score.
    """

    cfg = get_large_move_probability_config()

    # Start from a calibrated prior and add evidence from each market feature.
    # This keeps the model interpretable while allowing the config to tune the
    # relative influence of each signal source.
    prob = cfg["base_probability"]

    # --- Categorical regime effects ---
    if gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}:
        prob += cfg["short_gamma_bonus"]
    elif gamma_regime in {"POSITIVE_GAMMA", "LONG_GAMMA_ZONE"}:
        prob += cfg["long_gamma_penalty"]
    elif gamma_regime == "NEUTRAL_GAMMA":
        prob += 0.0

    if vacuum_state == "BREAKOUT_ZONE":
        prob += cfg["breakout_zone_bonus"]
    elif vacuum_state in ("NEAR_VACUUM", "VACUUM_WATCH"):
        prob += cfg["near_vacuum_bonus"]

    if hedging_bias in ("UPSIDE_ACCELERATION", "DOWNSIDE_ACCELERATION"):
        prob += cfg["acceleration_bias_bonus"]
    elif hedging_bias in ("UPSIDE_PINNING", "DOWNSIDE_PINNING", "PINNING"):
        prob += cfg["pinning_bias_penalty"]

    if smart_money_flow in ("BULLISH_FLOW", "BEARISH_FLOW"):
        prob += cfg["directional_flow_bonus"]
    elif smart_money_flow in ("NEUTRAL_FLOW", "MIXED_FLOW"):
        prob += cfg["neutral_flow_penalty"]

    # --- Continuous refinements ---
    if gamma_flip_distance_pct is not None:
        d = _clip(_safe_float(gamma_flip_distance_pct), 0.0, 2.0)
        # Distance is inverted because being closer to the flip usually means
        # a less stable dealer-hedging regime and therefore more move potential.
        prob += cfg["gamma_flip_distance_weight"] * (1.0 - d / 2.0)

    if vacuum_strength is not None:
        v = _clip(_safe_float(vacuum_strength), 0.0, 1.0)
        prob += cfg["vacuum_strength_weight"] * v

    if hedging_flow_ratio is not None:
        h = abs(_clip(_safe_float(hedging_flow_ratio), -1.0, 1.0))
        # Move-size probability is driven more by hedging intensity than by
        # direction, so the magnitude rather than the sign is used here.
        prob += cfg["hedging_flow_ratio_weight"] * h

    if smart_money_flow_score is not None:
        s = abs(_clip(_safe_float(smart_money_flow_score), -1.0, 1.0))
        # Strong one-sided institutional flow often matters even if the
        # direction is already encoded elsewhere in the signal pipeline.
        prob += cfg["smart_money_flow_weight"] * s

    if atm_iv_percentile is not None:
        ivp = _clip(_safe_float(atm_iv_percentile), 0.0, 1.0)
        prob += cfg["atm_iv_percentile_weight"] * ivp

    if intraday_range_pct is not None:
        r = _clip(_safe_float(intraday_range_pct), 0.0, 1.0)
        prob += cfg["intraday_range_weight"] * r

    # --- Conflict penalties ---
    bullish_flow = smart_money_flow == "BULLISH_FLOW"
    bearish_flow = smart_money_flow == "BEARISH_FLOW"
    upside_accel = hedging_bias == "UPSIDE_ACCELERATION"
    downside_accel = hedging_bias == "DOWNSIDE_ACCELERATION"

    # Penalize directional disagreement because conflicting flow and dealer
    # pressure usually imply chop rather than clean expansion.
    if (bullish_flow and downside_accel) or (bearish_flow and upside_accel):
        prob += cfg["directional_conflict_penalty"]

    if gamma_regime == "POSITIVE_GAMMA" and vacuum_state == "BREAKOUT_ZONE":
        prob += cfg["positive_gamma_breakout_penalty"]
    elif gamma_regime == "NEUTRAL_GAMMA" and vacuum_state == "BREAKOUT_ZONE":
        prob += cfg["neutral_gamma_breakout_bonus"]

    return round(_clip(prob, cfg["probability_floor"], cfg["probability_ceiling"]), 2)
