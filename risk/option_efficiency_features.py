"""
Module: option_efficiency_features.py

Purpose:
    Build option efficiency features used by the risk overlay.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

import math

import pandas as pd

from config.option_efficiency_policy import get_option_efficiency_policy_config
from utils.regime_normalization import detect_iv_unit, normalize_iv_decimal
from utils.numerics import clip as _clip, safe_float as _safe_float  # noqa: F401


def _holding_context(global_risk_state, holding_profile):
    """
    Purpose:
        Process holding context for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        global_risk_state (Any): Structured state payload for global risk.
        holding_profile (Any): Holding intent that determines whether overnight rules should be considered.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    global_risk_state = global_risk_state if isinstance(global_risk_state, dict) else {}
    holding_context = global_risk_state.get("holding_context", {})
    holding_context = holding_context if isinstance(holding_context, dict) else {}
    profile = str(holding_profile or holding_context.get("holding_profile") or "AUTO").upper().strip() or "AUTO"
    return {
        "holding_profile": profile,
        "overnight_relevant": bool(
            holding_context.get("overnight_relevant", False)
            or profile in {"OVERNIGHT", "SWING"}
        ),
        "market_session": holding_context.get("market_session", "UNKNOWN"),
        "minutes_to_close": holding_context.get("minutes_to_close"),
    }


def _normalize_iv(value):
    """
    Purpose:
        Normalize IV into the repository-standard representation.
    
    Context:
        Internal helper in the `option efficiency features` module. It isolates one overlay heuristic so risk adjustments remain auditable.
    
    Inputs:
        value (Any): Raw value supplied by the caller.
    
    Returns:
        Any: Value returned by the current workflow step.
    
    Notes:
        The helper intentionally produces bounded, serializable values so overlays can be inspected alongside the final trade decision.
    """
    cfg = get_option_efficiency_policy_config()
    iv_decimal = normalize_iv_decimal(
        value,
        percent_unit_threshold=cfg.iv_percent_unit_threshold,
        default=None,
    )
    if iv_decimal is None:
        return None, None

    return iv_decimal, detect_iv_unit(value, percent_unit_threshold=cfg.iv_percent_unit_threshold)


def _parse_time_to_expiry_years(expiry_value=None, valuation_time=None, tte=None):
    """
    Purpose:
        Process parse time to expiry years for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        expiry_value (Any): Input associated with expiry value.
        valuation_time (Any): Input associated with valuation time.
        tte (Any): Input associated with tte.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    tte_value = _safe_float(tte, None)
    if tte_value not in (None, 0):
        return max(tte_value, cfg.minimum_time_to_expiry_years), "DIRECT_TTE"

    if expiry_value is None:
        return None, None

    expiry_ts = pd.to_datetime(expiry_value, errors="coerce", utc=True)
    if pd.isna(expiry_ts):
        expiry_ts = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(expiry_ts):
        return None, None

    valuation_ts = pd.to_datetime(valuation_time, errors="coerce", utc=True)
    if pd.isna(valuation_ts):
        valuation_ts = pd.Timestamp.utcnow()

    time_years = (expiry_ts - valuation_ts).total_seconds() / (365.0 * 24.0 * 3600.0)
    return max(float(time_years), cfg.minimum_time_to_expiry_years), "PARSED_EXPIRY"


def _expected_move_quality(iv_source, dte_source):
    """
    Purpose:
        Process expected move quality for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        iv_source (Any): Input associated with IV source.
        dte_source (Any): Input associated with dte source.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if iv_source is None or dte_source is None:
        return "UNAVAILABLE"
    if iv_source == "ATM_IV" and dte_source in {"DIRECT_TTE", "PARSED_EXPIRY"}:
        return "DIRECT"
    return "FALLBACK"


def _effective_delta(delta):
    """
    Purpose:
        Process effective delta for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        delta (Any): Input associated with delta.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    value = abs(_safe_float(delta, 0.0))
    if value <= 0:
        return None
    return _clip(value, cfg.min_effective_delta, cfg.max_effective_delta)


def _strike_moneyness_bucket(direction, strike, spot):
    """
    Purpose:
        Process strike moneyness bucket for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        strike (Any): Input associated with strike.
        spot (Any): Input associated with spot.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    spot_value = _safe_float(spot, None)
    strike_value = _safe_float(strike, None)
    if spot_value in (None, 0) or strike_value is None:
        return "UNKNOWN"

    distance_pct = abs(strike_value - spot_value) / spot_value * 100.0
    if distance_pct <= cfg.strike_moneyness_atm_distance_pct:
        return "ATM"

    if direction == "CALL":
        return "ITM" if strike_value < spot_value else "OTM"
    if direction == "PUT":
        return "ITM" if strike_value > spot_value else "OTM"
    return "UNKNOWN"


def _payoff_hint(strike_moneyness_bucket, strike_distance_ratio, premium_ratio):
    """
    Purpose:
        Process payoff hint for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        strike_moneyness_bucket (Any): Input associated with strike moneyness bucket.
        strike_distance_ratio (Any): Input associated with strike distance ratio.
        premium_ratio (Any): Input associated with premium ratio.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    if strike_moneyness_bucket == "OTM" and strike_distance_ratio is not None and strike_distance_ratio > cfg.payoff_far_otm_distance_ratio:
        return "far_otm_requires_large_move"
    if strike_moneyness_bucket == "ITM" and premium_ratio is not None and premium_ratio < cfg.payoff_deep_itm_premium_ratio:
        return "deep_itm_premium_heavy"
    if strike_moneyness_bucket == "ATM":
        return "atm_convexity_balanced"
    return "near_spot_payoff_reasonable"


def build_option_efficiency_features(
    *,
    spot=None,
    atm_iv=None,
    india_vix_level=None,
    india_vix_change_24h=None,
    fallback_iv=None,
    expiry_value=None,
    valuation_time=None,
    time_to_expiry_years=None,
    direction=None,
    strike=None,
    option_type=None,
    entry_price=None,
    target=None,
    stop_loss=None,
    trade_strength=None,
    hybrid_move_probability=None,
    rule_move_probability=None,
    ml_move_probability=None,
    gamma_regime=None,
    volatility_regime=None,
    volatility_shock_score=None,
    volatility_compression_score=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    gamma_vol_acceleration_score=None,
    dealer_hedging_pressure_score=None,
    liquidity_vacuum_state=None,
    support_wall=None,
    resistance_wall=None,
    delta=None,
    holding_profile="AUTO",
):
    """
    Purpose:
        Build the option efficiency features used by downstream components.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        atm_iv (Any): Input associated with ATM IV.
        india_vix_level (Any): India VIX level used as a local-volatility
            fallback when the option chain lacks a stable ATM IV.
        india_vix_change_24h (Any): One-day India VIX change carried through as
            local volatility context for diagnostics.
        fallback_iv (Any): Input associated with fallback IV.
        expiry_value (Any): Input associated with expiry value.
        valuation_time (Any): Input associated with valuation time.
        time_to_expiry_years (Any): Input associated with time to expiry years.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        strike (Any): Input associated with strike.
        option_type (Any): Input associated with option type.
        entry_price (Any): Input associated with entry price.
        target (Any): Input associated with target.
        stop_loss (Any): Input associated with stop loss.
        trade_strength (Any): Input associated with trade strength.
        hybrid_move_probability (Any): Input associated with hybrid move probability.
        rule_move_probability (Any): Input associated with rule move probability.
        ml_move_probability (Any): Input associated with ML move probability.
        gamma_regime (Any): Input associated with gamma regime.
        volatility_regime (Any): Input associated with volatility regime.
        volatility_shock_score (Any): Score value for volatility shock.
        volatility_compression_score (Any): Score value for volatility compression.
        macro_event_risk_score (Any): Macro-event risk score used by fallback or overlay logic.
        global_risk_state (Any): Structured state payload for global risk.
        gamma_vol_acceleration_score (Any): Score value for gamma vol acceleration.
        dealer_hedging_pressure_score (Any): Score value for dealer hedging pressure.
        liquidity_vacuum_state (Any): Structured state payload for liquidity vacuum.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
        delta (Any): Input associated with delta.
        holding_profile (Any): Holding intent that determines whether overnight rules should be considered.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_option_efficiency_policy_config()
    holding_context = _holding_context(global_risk_state, holding_profile)
    iv_decimal, iv_unit = _normalize_iv(atm_iv)
    iv_source = "ATM_IV" if iv_decimal is not None else None
    if iv_decimal is None:
        iv_decimal, iv_unit = _normalize_iv(india_vix_level)
        if iv_decimal is not None:
            iv_source = "INDIA_VIX"
    if iv_decimal is None:
        iv_decimal, iv_unit = _normalize_iv(fallback_iv)
        if iv_decimal is not None:
            iv_source = "FALLBACK_IV"

    dte_years, dte_source = _parse_time_to_expiry_years(
        expiry_value=expiry_value,
        valuation_time=valuation_time,
        tte=time_to_expiry_years,
    )
    expected_move_quality = _expected_move_quality(iv_source, dte_source)

    spot_value = _safe_float(spot, None)
    expected_move_points = None
    expected_move_pct = None
    if spot_value not in (None, 0) and iv_decimal is not None and dte_years is not None:
        expected_move_points = round(spot_value * iv_decimal * math.sqrt(dte_years), 2)
        expected_move_pct = round((expected_move_points / spot_value) * 100.0, 4)

    effective_delta = _effective_delta(delta)
    if effective_delta is None:
        effective_delta = cfg.fallback_delta
    option_gain_target = None
    target_distance_points = None
    target_distance_pct = None
    expected_move_coverage_ratio = None

    entry_value = _safe_float(entry_price, None)
    target_value = _safe_float(target, None)
    strike_value = _safe_float(strike, None)
    direction = str(direction or "").upper().strip()
    if entry_value not in (None, 0) and target_value is not None and spot_value not in (None, 0):
        option_gain_target = max(target_value - entry_value, 0.0)
        delta_for_target = effective_delta if effective_delta is not None else cfg.fallback_delta
        target_distance_points = option_gain_target / max(delta_for_target, cfg.target_delta_floor)

        intrinsic_hurdle = 0.0
        if strike_value is not None:
            if direction == "CALL":
                intrinsic_hurdle = max(strike_value - spot_value, 0.0) * cfg.target_intrinsic_hurdle_multiplier
            elif direction == "PUT":
                intrinsic_hurdle = max(spot_value - strike_value, 0.0) * cfg.target_intrinsic_hurdle_multiplier
        target_distance_points = round(max(target_distance_points, intrinsic_hurdle), 2)
        target_distance_pct = round((target_distance_points / spot_value) * 100.0, 4)
        if expected_move_points not in (None, 0):
            expected_move_coverage_ratio = round(expected_move_points / max(target_distance_points, 1e-6), 4)

    trade_prob = _clip(_safe_float(hybrid_move_probability, 0.5), cfg.trade_probability_floor, cfg.trade_probability_ceiling)
    gamma_vol_norm = _clip(_safe_float(gamma_vol_acceleration_score, 0.0) / 100.0, 0.0, 1.0)
    dealer_pressure_norm = _clip(_safe_float(dealer_hedging_pressure_score, 0.0) / 100.0, 0.0, 1.0)
    volatility_shock_norm = _clip(_safe_float(volatility_shock_score, 0.0), 0.0, 1.0)
    volatility_compression_norm = _clip(_safe_float(volatility_compression_score, 0.0), 0.0, 1.0)
    macro_event_norm = _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0)
    convexity_multiplier = (
        cfg.convexity_base
        + (cfg.convexity_gamma_vol_weight * gamma_vol_norm)
        + (cfg.convexity_dealer_pressure_weight * dealer_pressure_norm)
    )
    if str(liquidity_vacuum_state or "").upper().strip() in {"BREAKOUT_ZONE", "NEAR_VACUUM"}:
        convexity_multiplier += cfg.convexity_liquidity_vacuum_bonus

    expected_option_move_value = None
    premium_coverage_ratio = None
    if expected_move_points is not None and entry_value not in (None, 0):
        delta_for_efficiency = effective_delta if effective_delta is not None else cfg.fallback_delta
        expected_option_move_value = round(
            expected_move_points
            * delta_for_efficiency
            * convexity_multiplier
            * (cfg.option_move_probability_base + cfg.option_move_probability_weight * trade_prob),
            2,
        )
        premium_coverage_ratio = round(expected_option_move_value / max(entry_value, 1e-6), 4)

    strike_distance_from_spot = None
    strike_distance_ratio = None
    strike_moneyness_bucket = _strike_moneyness_bucket(direction, strike_value, spot_value)
    if strike_value is not None and spot_value not in (None, 0):
        strike_distance_from_spot = round(abs(strike_value - spot_value), 2)
        if expected_move_points not in (None, 0):
            strike_distance_ratio = round(strike_distance_from_spot / expected_move_points, 4)

    neutral_fallback = expected_move_quality == "UNAVAILABLE"
    warnings = []
    if neutral_fallback:
        warnings.append("expected_move_unavailable")
    elif iv_source == "INDIA_VIX":
        warnings.append("india_vix_used")
    elif iv_source == "FALLBACK_IV":
        warnings.append("fallback_iv_used")
    if _effective_delta(delta) is None:
        warnings.append("fallback_delta_used")

    return {
        "spot": spot_value,
        "direction": direction,
        "strike": strike_value,
        "option_type": option_type,
        "entry_price": entry_value,
        "target": target_value,
        "stop_loss": _safe_float(stop_loss, None),
        "trade_strength": _safe_float(trade_strength, None),
        "atm_iv": _safe_float(atm_iv, None),
        "india_vix_level": _safe_float(india_vix_level, None),
        "india_vix_change_24h": _safe_float(india_vix_change_24h, None),
        "fallback_iv": _safe_float(fallback_iv, None),
        "iv_source": iv_source,
        "iv_unit": iv_unit,
        "selected_expiry": expiry_value,
        "time_to_expiry_years": dte_years,
        "dte_source": dte_source,
        "expected_move_points": expected_move_points,
        "expected_move_pct": expected_move_pct,
        "expected_move_quality": expected_move_quality,
        "target_distance_points": target_distance_points,
        "target_distance_pct": target_distance_pct,
        "expected_move_coverage_ratio": expected_move_coverage_ratio,
        "effective_delta": effective_delta,
        "option_gain_target": option_gain_target,
        "expected_option_move_value": expected_option_move_value,
        "premium_coverage_ratio": premium_coverage_ratio,
        "strike_distance_from_spot": strike_distance_from_spot,
        "strike_distance_ratio": strike_distance_ratio,
        "strike_moneyness_bucket": strike_moneyness_bucket,
        "payoff_efficiency_hint": _payoff_hint(strike_moneyness_bucket, strike_distance_ratio, premium_coverage_ratio),
        "hybrid_move_probability": _safe_float(hybrid_move_probability, None),
        "rule_move_probability": _safe_float(rule_move_probability, None),
        "ml_move_probability": _safe_float(ml_move_probability, None),
        "gamma_regime": gamma_regime,
        "volatility_regime": volatility_regime,
        "volatility_shock_score": volatility_shock_norm,
        "volatility_compression_score": volatility_compression_norm,
        "macro_event_risk_score": _safe_float(macro_event_risk_score, None),
        "global_risk_state": (
            global_risk_state.get("global_risk_state")
            if isinstance(global_risk_state, dict)
            else global_risk_state
        ),
        "gamma_vol_acceleration_score": _safe_float(gamma_vol_acceleration_score, None),
        "dealer_hedging_pressure_score": _safe_float(dealer_hedging_pressure_score, None),
        "liquidity_vacuum_state": liquidity_vacuum_state,
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "convexity_multiplier": round(convexity_multiplier, 4),
        "holding_context": holding_context,
        "neutral_fallback": neutral_fallback,
        "warnings": warnings,
    }
