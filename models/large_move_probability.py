from __future__ import annotations

from typing import Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x, default: float = 0.0) -> float:
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
    Estimate probability of a large intraday move (roughly 150-300 points)
    using a bounded evidence model.

    Backward compatible with the old 4 categorical inputs, but improved by
    allowing continuous intraday features that can change throughout the day.

    Optional inputs:
    - gamma_flip_distance_pct: abs(spot - gamma_flip) / spot * 100
      Smaller distance -> more unstable regime -> higher probability
    - vacuum_strength: 0..1
      Higher means stronger/cleaner liquidity vacuum
    - hedging_flow_ratio: -1..1
      Magnitude captures dealer acceleration pressure
    - smart_money_flow_score: -1..1
      Magnitude captures directional institutional flow
    - atm_iv_percentile: 0..1
      Higher IV percentile -> more move potential
    - intraday_range_pct: 0..1+
      Normalized realized/expected range expansion
    """

    prob = 0.22

    # --- Categorical regime effects ---
    if gamma_regime == "NEGATIVE_GAMMA":
        prob += 0.14
    elif gamma_regime == "POSITIVE_GAMMA":
        prob -= 0.08
    elif gamma_regime == "NEUTRAL_GAMMA":
        prob += 0.0

    if vacuum_state == "BREAKOUT_ZONE":
        prob += 0.16
    elif vacuum_state in ("NEAR_VACUUM", "VACUUM_WATCH"):
        prob += 0.07

    if hedging_bias in ("UPSIDE_ACCELERATION", "DOWNSIDE_ACCELERATION"):
        prob += 0.14
    elif hedging_bias in ("UPSIDE_PINNING", "DOWNSIDE_PINNING", "PINNING"):
        prob -= 0.06

    if smart_money_flow in ("BULLISH_FLOW", "BEARISH_FLOW"):
        prob += 0.08
    elif smart_money_flow in ("NEUTRAL_FLOW", "MIXED_FLOW"):
        prob -= 0.02

    # --- Continuous refinements ---
    if gamma_flip_distance_pct is not None:
        d = _clip(_safe_float(gamma_flip_distance_pct), 0.0, 2.0)
        prob += 0.12 * (1.0 - d / 2.0)

    if vacuum_strength is not None:
        v = _clip(_safe_float(vacuum_strength), 0.0, 1.0)
        prob += 0.12 * v

    if hedging_flow_ratio is not None:
        h = abs(_clip(_safe_float(hedging_flow_ratio), -1.0, 1.0))
        prob += 0.10 * h

    if smart_money_flow_score is not None:
        s = abs(_clip(_safe_float(smart_money_flow_score), -1.0, 1.0))
        prob += 0.08 * s

    if atm_iv_percentile is not None:
        ivp = _clip(_safe_float(atm_iv_percentile), 0.0, 1.0)
        prob += 0.07 * ivp

    if intraday_range_pct is not None:
        r = _clip(_safe_float(intraday_range_pct), 0.0, 1.0)
        prob += 0.08 * r

    # --- Conflict penalties ---
    bullish_flow = smart_money_flow == "BULLISH_FLOW"
    bearish_flow = smart_money_flow == "BEARISH_FLOW"
    upside_accel = hedging_bias == "UPSIDE_ACCELERATION"
    downside_accel = hedging_bias == "DOWNSIDE_ACCELERATION"

    if (bullish_flow and downside_accel) or (bearish_flow and upside_accel):
        prob -= 0.10

    if gamma_regime == "POSITIVE_GAMMA" and vacuum_state == "BREAKOUT_ZONE":
        prob -= 0.05
    elif gamma_regime == "NEUTRAL_GAMMA" and vacuum_state == "BREAKOUT_ZONE":
        prob += 0.02

    return round(_clip(prob, 0.05, 0.95), 2)
