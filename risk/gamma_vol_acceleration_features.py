"""
Deterministic raw feature model for the gamma-vol acceleration overlay.
"""

from __future__ import annotations

from config.gamma_vol_acceleration_policy import get_gamma_vol_acceleration_policy_config


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _holding_context(global_risk_state, holding_profile):
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


def _gamma_regime_score(gamma_regime):
    cfg = get_gamma_vol_acceleration_policy_config()
    gamma_regime = str(gamma_regime or "").upper().strip()
    if gamma_regime in {"SHORT_GAMMA_ZONE", "NEGATIVE_GAMMA"}:
        return cfg.gamma_regime_short_score
    if gamma_regime in {"LONG_GAMMA_ZONE", "POSITIVE_GAMMA"}:
        return cfg.gamma_regime_long_score
    return 0.0


def _flip_proximity_score(gamma_flip_distance_pct, spot_vs_flip):
    cfg = get_gamma_vol_acceleration_policy_config()
    spot_vs_flip = str(spot_vs_flip or "").upper().strip()
    distance = _safe_float(gamma_flip_distance_pct, None)

    if spot_vs_flip == "AT_FLIP":
        return cfg.flip_at_score

    if distance is None:
        if spot_vs_flip in {"ABOVE_FLIP", "BELOW_FLIP"}:
            return cfg.flip_unknown_context_score
        return 0.0

    if distance <= cfg.flip_distance_1_pct:
        return cfg.flip_distance_1_score
    if distance <= cfg.flip_distance_2_pct:
        return cfg.flip_distance_2_score
    if distance <= cfg.flip_distance_3_pct:
        return cfg.flip_distance_3_score
    if distance <= cfg.flip_distance_4_pct:
        return cfg.flip_distance_4_score
    return cfg.flip_distance_far_score


def _volatility_transition_score(volatility_compression_score, volatility_shock_score, volatility_explosion_probability):
    cfg = get_gamma_vol_acceleration_policy_config()
    compression = _clip(_safe_float(volatility_compression_score, 0.0), 0.0, 1.0)
    shock = _clip(_safe_float(volatility_shock_score, 0.0), 0.0, 1.0)
    explosion = _clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0)
    return round(_clip(
        (cfg.vol_transition_compression_weight * compression)
        + (cfg.vol_transition_shock_weight * shock)
        + (cfg.vol_transition_explosion_weight * explosion),
        0.0,
        1.0,
    ), 4)


def _liquidity_vacuum_score(liquidity_vacuum_state):
    cfg = get_gamma_vol_acceleration_policy_config()
    mapping = {
        "BREAKOUT_ZONE": cfg.liquidity_breakout_score,
        "NEAR_VACUUM": cfg.liquidity_near_vacuum_score,
        "VACUUM_WATCH": cfg.liquidity_watch_score,
        "VACUUM_CONTAINED": cfg.liquidity_contained_score,
        "NO_VACUUM": 0.0,
    }
    return mapping.get(str(liquidity_vacuum_state or "").upper().strip(), 0.0)


def _hedging_bias_score(dealer_hedging_bias):
    cfg = get_gamma_vol_acceleration_policy_config()
    bias = str(dealer_hedging_bias or "").upper().strip()
    if bias == "UPSIDE_ACCELERATION":
        return cfg.hedging_upside_acceleration_score
    if bias == "DOWNSIDE_ACCELERATION":
        return -cfg.hedging_upside_acceleration_score
    if bias == "UPSIDE_PINNING":
        return cfg.hedging_upside_pinning_score
    if bias == "DOWNSIDE_PINNING":
        return cfg.hedging_downside_pinning_score
    if bias == "PINNING":
        return 0.0
    return 0.0


def _pinning_dampener(dealer_hedging_bias):
    cfg = get_gamma_vol_acceleration_policy_config()
    bias = str(dealer_hedging_bias or "").upper().strip()
    if bias == "PINNING":
        return cfg.pinning_bias_full_dampener
    if bias in {"UPSIDE_PINNING", "DOWNSIDE_PINNING"}:
        return cfg.pinning_bias_partial_dampener
    return 0.0


def _intraday_extension_score(intraday_range_pct):
    cfg = get_gamma_vol_acceleration_policy_config()
    value = _safe_float(intraday_range_pct, None)
    if value is None:
        return 0.0
    if value <= cfg.intraday_extension_low_threshold:
        return 0.0
    if value <= cfg.intraday_extension_mid_threshold:
        return cfg.intraday_extension_mid_score
    if value <= cfg.intraday_extension_high_threshold:
        return cfg.intraday_extension_high_score
    return cfg.intraday_extension_extreme_score


def _macro_global_boost(macro_event_risk_score, global_risk_state, volatility_explosion_probability):
    cfg = get_gamma_vol_acceleration_policy_config()
    event_norm = _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0)
    state = str(global_risk_state or "").upper().strip()
    global_state_boost = {
        "VOL_SHOCK": cfg.macro_global_state_vol_shock,
        "EVENT_LOCKDOWN": cfg.macro_global_state_event_lockdown,
        "RISK_OFF": cfg.macro_global_state_risk_off,
        "GLOBAL_NEUTRAL": cfg.macro_global_state_neutral,
        "RISK_ON": 0.0,
    }.get(state, cfg.macro_global_state_unknown)
    explosion = _clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0)
    return round(_clip(
        (cfg.macro_global_event_weight * event_norm)
        + (cfg.macro_global_state_weight * global_state_boost)
        + (cfg.macro_global_explosion_weight * explosion),
        0.0,
        1.0,
    ), 4)


def build_gamma_vol_acceleration_features(
    *,
    gamma_regime=None,
    spot_vs_flip=None,
    gamma_flip_distance_pct=None,
    dealer_hedging_bias=None,
    liquidity_vacuum_state=None,
    intraday_range_pct=None,
    volatility_compression_score=None,
    volatility_shock_score=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    volatility_explosion_probability=None,
    holding_profile="AUTO",
    support_wall=None,
    resistance_wall=None,
):
    cfg = get_gamma_vol_acceleration_policy_config()
    holding_context = _holding_context(global_risk_state, holding_profile)
    global_state_label = (
        global_risk_state.get("global_risk_state")
        if isinstance(global_risk_state, dict)
        else global_risk_state
    )

    feature_inputs = {
        "gamma_regime": gamma_regime,
        "spot_vs_flip": spot_vs_flip,
        "gamma_flip_distance_pct": gamma_flip_distance_pct,
        "dealer_hedging_bias": dealer_hedging_bias,
        "liquidity_vacuum_state": liquidity_vacuum_state,
        "intraday_range_pct": intraday_range_pct,
        "volatility_compression_score": volatility_compression_score,
        "volatility_shock_score": volatility_shock_score,
        "macro_event_risk_score": macro_event_risk_score,
        "global_risk_state": global_state_label,
        "volatility_explosion_probability": volatility_explosion_probability,
    }
    input_availability = {
        key: value not in (None, "", "UNKNOWN")
        for key, value in feature_inputs.items()
    }
    coverage_ratio = round(sum(1 for value in input_availability.values() if value) / max(len(input_availability), 1), 4)
    feature_confidence = _clip(coverage_ratio, 0.0, 1.0)

    gamma_regime_score = _gamma_regime_score(gamma_regime)
    flip_proximity_score = _flip_proximity_score(gamma_flip_distance_pct, spot_vs_flip)
    volatility_transition_score = _volatility_transition_score(
        volatility_compression_score,
        volatility_shock_score,
        volatility_explosion_probability,
    )
    liquidity_vacuum_score = _liquidity_vacuum_score(liquidity_vacuum_state)
    hedging_bias_score = _hedging_bias_score(dealer_hedging_bias)
    pinning_dampener = _pinning_dampener(dealer_hedging_bias)
    intraday_extension_score = _intraday_extension_score(intraday_range_pct)
    macro_global_boost = _macro_global_boost(
        macro_event_risk_score,
        global_state_label,
        volatility_explosion_probability,
    )

    positive_gamma_pressure = max(gamma_regime_score, 0.0)
    gamma_dampener = max(-gamma_regime_score, 0.0)
    acceleration_core = _clip(
        (cfg.acceleration_gamma_weight * positive_gamma_pressure)
        + (cfg.acceleration_flip_weight * flip_proximity_score)
        + (cfg.acceleration_vol_weight * volatility_transition_score)
        + (cfg.acceleration_liquidity_weight * liquidity_vacuum_score)
        + (cfg.acceleration_hedging_weight * abs(hedging_bias_score))
        + (cfg.acceleration_intraday_weight * intraday_extension_score)
        + (cfg.acceleration_macro_weight * macro_global_boost),
        0.0,
        1.0,
    )
    dampening_core = _clip(
        (cfg.dampening_gamma_weight * gamma_dampener)
        + (cfg.dampening_pinning_weight * pinning_dampener),
        0.0,
        1.0,
    )
    normalized_acceleration = _clip((acceleration_core - (cfg.acceleration_dampening_weight * dampening_core)) * feature_confidence, 0.0, 1.0)

    spot_vs_flip_label = str(spot_vs_flip or "").upper().strip()
    upside_alignment = 0.0
    downside_alignment = 0.0
    if spot_vs_flip_label == "ABOVE_FLIP":
        upside_alignment += cfg.alignment_above_flip_boost
    elif spot_vs_flip_label == "BELOW_FLIP":
        downside_alignment += cfg.alignment_below_flip_boost
    elif spot_vs_flip_label == "AT_FLIP":
        upside_alignment += cfg.alignment_at_flip_boost
        downside_alignment += cfg.alignment_at_flip_boost

    if hedging_bias_score > 0:
        upside_alignment += min(cfg.alignment_bias_cap, hedging_bias_score * cfg.alignment_bias_weight)
    elif hedging_bias_score < 0:
        downside_alignment += min(cfg.alignment_bias_cap, abs(hedging_bias_score) * cfg.alignment_bias_weight)

    upside_squeeze_risk = round(_clip(
        (cfg.directional_gamma_weight * positive_gamma_pressure)
        + (cfg.directional_flip_weight * flip_proximity_score)
        + (cfg.directional_vol_weight * volatility_transition_score)
        + (cfg.directional_liquidity_weight * liquidity_vacuum_score)
        + (cfg.directional_intraday_weight * intraday_extension_score)
        + (cfg.directional_macro_weight * macro_global_boost)
        + upside_alignment
        - (cfg.directional_dampening_weight * dampening_core),
        0.0,
        1.0,
    ) * feature_confidence, 4)
    downside_airpocket_risk = round(_clip(
        (cfg.directional_gamma_weight * positive_gamma_pressure)
        + (cfg.directional_flip_weight * flip_proximity_score)
        + (cfg.directional_vol_weight * volatility_transition_score)
        + (cfg.directional_liquidity_weight * liquidity_vacuum_score)
        + (cfg.directional_intraday_weight * intraday_extension_score)
        + (cfg.directional_macro_weight * macro_global_boost)
        + downside_alignment
        - (cfg.directional_dampening_weight * dampening_core),
        0.0,
        1.0,
    ) * feature_confidence, 4)
    overnight_convexity_risk = round(_clip(
        (cfg.overnight_acceleration_weight * normalized_acceleration)
        + (cfg.overnight_explosion_weight * _clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0))
        + (cfg.overnight_macro_event_weight * _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0))
        + (cfg.overnight_directional_weight * max(upside_squeeze_risk, downside_airpocket_risk))
        + (cfg.overnight_macro_boost_weight * macro_global_boost)
        + (cfg.overnight_context_weight * (1.0 if holding_context["overnight_relevant"] else 0.0))
        - (cfg.overnight_dampening_weight * dampening_core),
        0.0,
        1.0,
    ), 4)

    warnings = []
    if feature_confidence < cfg.partial_coverage_warning_threshold:
        warnings.append("gamma_vol_partial_input_coverage")
    if gamma_regime is None:
        warnings.append("gamma_regime_missing")
    if gamma_flip_distance_pct is None and spot_vs_flip not in {"AT_FLIP", "ABOVE_FLIP", "BELOW_FLIP"}:
        warnings.append("flip_context_missing")

    return {
        "gamma_regime": gamma_regime,
        "spot_vs_flip": spot_vs_flip,
        "gamma_flip_distance_pct": _safe_float(gamma_flip_distance_pct, None),
        "dealer_hedging_bias": dealer_hedging_bias,
        "liquidity_vacuum_state": liquidity_vacuum_state,
        "intraday_range_pct": _safe_float(intraday_range_pct, None),
        "volatility_compression_score": round(_clip(_safe_float(volatility_compression_score, 0.0), 0.0, 1.0), 4),
        "volatility_shock_score": round(_clip(_safe_float(volatility_shock_score, 0.0), 0.0, 1.0), 4),
        "macro_event_risk_score": _safe_int(macro_event_risk_score, 0),
        "global_risk_state": str(global_state_label or "GLOBAL_NEUTRAL").upper().strip() or "GLOBAL_NEUTRAL",
        "volatility_explosion_probability": round(_clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0), 4),
        "support_wall_available": support_wall is not None,
        "resistance_wall_available": resistance_wall is not None,
        "holding_context": holding_context,
        "input_availability": input_availability,
        "feature_confidence": round(feature_confidence, 4),
        "gamma_regime_score": round(gamma_regime_score, 4),
        "flip_proximity_score": round(flip_proximity_score, 4),
        "volatility_transition_score": round(volatility_transition_score, 4),
        "liquidity_vacuum_score": round(liquidity_vacuum_score, 4),
        "hedging_bias_score": round(hedging_bias_score, 4),
        "pinning_dampener": round(pinning_dampener, 4),
        "intraday_extension_score": round(intraday_extension_score, 4),
        "macro_global_boost": round(macro_global_boost, 4),
        "acceleration_core": round(acceleration_core, 4),
        "dampening_core": round(dampening_core, 4),
        "normalized_acceleration": round(normalized_acceleration, 4),
        "upside_squeeze_risk": upside_squeeze_risk,
        "downside_airpocket_risk": downside_airpocket_risk,
        "overnight_convexity_risk": overnight_convexity_risk,
        "neutral_fallback": feature_confidence == 0.0,
        "warnings": warnings,
    }
