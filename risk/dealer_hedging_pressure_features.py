"""
Deterministic raw feature model for dealer hedging pressure.
"""

from __future__ import annotations

from config.dealer_hedging_pressure_policy import get_dealer_hedging_pressure_policy_config


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


def _gamma_base_scores(gamma_regime, dealer_position):
    cfg = get_dealer_hedging_pressure_policy_config()
    gamma_regime = str(gamma_regime or "").upper().strip()
    dealer_position = str(dealer_position or "").upper().strip()

    acceleration = 0.0
    pinning = 0.0

    if gamma_regime in {"SHORT_GAMMA_ZONE", "NEGATIVE_GAMMA"}:
        acceleration += cfg.gamma_short_acceleration_score
    elif gamma_regime in {"LONG_GAMMA_ZONE", "POSITIVE_GAMMA"}:
        pinning += cfg.gamma_long_pinning_score

    if dealer_position == "SHORT GAMMA":
        acceleration += cfg.dealer_short_gamma_acceleration_bonus
    elif dealer_position == "LONG GAMMA":
        pinning += cfg.dealer_long_gamma_pinning_bonus

    return round(_clip(acceleration, 0.0, 1.0), 4), round(_clip(pinning, 0.0, 1.0), 4)


def _flip_proximity_score(gamma_flip_distance_pct, spot_vs_flip):
    cfg = get_dealer_hedging_pressure_policy_config()
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


def _bias_scores(dealer_hedging_bias):
    cfg = get_dealer_hedging_pressure_policy_config()
    bias = str(dealer_hedging_bias or "").upper().strip()
    upside = 0.0
    downside = 0.0
    pinning = 0.0

    if bias == "UPSIDE_ACCELERATION":
        upside = cfg.bias_acceleration_score
    elif bias == "DOWNSIDE_ACCELERATION":
        downside = cfg.bias_acceleration_score
    elif bias in {"UPSIDE_PINNING", "DOWNSIDE_PINNING"}:
        pinning = cfg.bias_partial_pinning_score
    elif bias == "PINNING":
        pinning = cfg.bias_full_pinning_score

    return upside, downside, pinning


def _hedging_flow_scores(dealer_hedging_flow):
    cfg = get_dealer_hedging_pressure_policy_config()
    flow = str(dealer_hedging_flow or "").upper().strip()
    if flow == "BUY_FUTURES":
        return cfg.hedging_flow_score, 0.0
    if flow == "SELL_FUTURES":
        return 0.0, cfg.hedging_flow_score
    return 0.0, 0.0


def _intraday_gamma_scores(intraday_gamma_state):
    cfg = get_dealer_hedging_pressure_policy_config()
    state = str(intraday_gamma_state or "").upper().strip()
    if state == "VOL_EXPANSION":
        return cfg.intraday_vol_expansion_score, 0.0
    if state == "GAMMA_DECREASE":
        return cfg.intraday_gamma_decrease_score, 0.0
    if state in {"VOL_SUPPRESSION", "GAMMA_INCREASE"}:
        return 0.0, cfg.intraday_pinning_score
    return 0.0, 0.0


def _flow_confirmation_scores(flow_signal, smart_money_flow):
    cfg = get_dealer_hedging_pressure_policy_config()
    bullish = 0.0
    bearish = 0.0

    flow_signal = str(flow_signal or "").upper().strip()
    smart_money_flow = str(smart_money_flow or "").upper().strip()

    if flow_signal == "BULLISH_FLOW":
        bullish += cfg.flow_confirmation_hit_score
    elif flow_signal == "BEARISH_FLOW":
        bearish += cfg.flow_confirmation_hit_score

    if smart_money_flow == "BULLISH_FLOW":
        bullish += cfg.flow_confirmation_hit_score
    elif smart_money_flow == "BEARISH_FLOW":
        bearish += cfg.flow_confirmation_hit_score

    return round(_clip(bullish, 0.0, cfg.flow_confirmation_cap), 4), round(_clip(bearish, 0.0, cfg.flow_confirmation_cap), 4)


def _nearest_level_distance_pct(spot, levels):
    spot_value = _safe_float(spot, None)
    if spot_value in (None, 0):
        return None

    nearest = None
    for level in levels:
        value = _safe_float(level, None)
        if value is None:
            continue
        distance_pct = abs(value - spot_value) / spot_value * 100.0
        if nearest is None or distance_pct < nearest:
            nearest = distance_pct
    return nearest


def _structure_scores(spot, support_wall, resistance_wall, gamma_clusters, liquidity_levels, liquidity_vacuum_state):
    cfg = get_dealer_hedging_pressure_policy_config()
    levels = []
    for level in [support_wall, resistance_wall]:
        if level is not None:
            levels.append(level)
    for bucket in [gamma_clusters or [], liquidity_levels or []]:
        levels.extend(bucket)

    nearest_level_distance_pct = _nearest_level_distance_pct(spot, levels)
    concentration_score = 0.0
    if nearest_level_distance_pct is not None:
        if nearest_level_distance_pct <= cfg.level_distance_near_pct:
            concentration_score = cfg.level_concentration_near_score
        elif nearest_level_distance_pct <= cfg.level_distance_mid_pct:
            concentration_score = cfg.level_concentration_mid_score
        elif nearest_level_distance_pct <= cfg.level_distance_far_pct:
            concentration_score = cfg.level_concentration_far_score
        else:
            concentration_score = cfg.level_concentration_loose_score

    vacuum_state = str(liquidity_vacuum_state or "").upper().strip()
    vacuum_score = {
        "BREAKOUT_ZONE": cfg.vacuum_breakout_score,
        "NEAR_VACUUM": cfg.vacuum_near_score,
        "VACUUM_WATCH": cfg.vacuum_watch_score,
    }.get(vacuum_state, 0.0)

    acceleration_structure = _clip(vacuum_score * (1.0 - (concentration_score * cfg.acceleration_structure_pinning_dampener)), 0.0, 1.0)
    pinning_structure = _clip((cfg.pinning_structure_concentration_weight * concentration_score) + (cfg.pinning_structure_cluster_bonus if len(levels) >= 4 else 0.0), 0.0, 1.0)
    return round(acceleration_structure, 4), round(pinning_structure, 4), nearest_level_distance_pct


def _macro_global_boost(macro_event_risk_score, global_risk_state, volatility_explosion_probability, gamma_vol_acceleration_score):
    cfg = get_dealer_hedging_pressure_policy_config()
    macro_norm = _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0)
    state = (
        global_risk_state.get("global_risk_state")
        if isinstance(global_risk_state, dict)
        else global_risk_state
    )
    state = str(state or "").upper().strip()
    global_boost = {
        "VOL_SHOCK": cfg.macro_global_state_vol_shock,
        "EVENT_LOCKDOWN": cfg.macro_global_state_event_lockdown,
        "RISK_OFF": cfg.macro_global_state_risk_off,
        "GLOBAL_NEUTRAL": cfg.macro_global_state_neutral,
        "RISK_ON": 0.0,
    }.get(state, cfg.macro_global_state_unknown)
    volatility_boost = _clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0)
    gamma_vol_boost = _clip(_safe_float(gamma_vol_acceleration_score, 0.0) / 100.0, 0.0, 1.0)
    return round(_clip(
        (cfg.macro_global_event_weight * macro_norm)
        + (cfg.macro_global_state_weight * global_boost)
        + (cfg.macro_global_explosion_weight * volatility_boost)
        + (cfg.macro_global_gamma_vol_weight * gamma_vol_boost),
        0.0,
        1.0,
    ), 4)


def build_dealer_hedging_pressure_features(
    *,
    spot=None,
    gamma_regime=None,
    spot_vs_flip=None,
    gamma_flip_distance_pct=None,
    dealer_position=None,
    dealer_hedging_bias=None,
    dealer_hedging_flow=None,
    market_gamma=None,
    gamma_clusters=None,
    liquidity_levels=None,
    support_wall=None,
    resistance_wall=None,
    liquidity_vacuum_state=None,
    intraday_gamma_state=None,
    intraday_range_pct=None,
    flow_signal=None,
    smart_money_flow=None,
    macro_event_risk_score=None,
    global_risk_state=None,
    volatility_explosion_probability=None,
    gamma_vol_acceleration_score=None,
    holding_profile="AUTO",
):
    cfg = get_dealer_hedging_pressure_policy_config()
    holding_context = _holding_context(global_risk_state, holding_profile)

    feature_inputs = {
        "gamma_regime": gamma_regime,
        "spot_vs_flip": spot_vs_flip,
        "gamma_flip_distance_pct": gamma_flip_distance_pct,
        "dealer_position": dealer_position,
        "dealer_hedging_bias": dealer_hedging_bias,
        "dealer_hedging_flow": dealer_hedging_flow,
        "liquidity_vacuum_state": liquidity_vacuum_state,
        "intraday_gamma_state": intraday_gamma_state,
        "flow_signal": flow_signal,
        "smart_money_flow": smart_money_flow,
        "macro_event_risk_score": macro_event_risk_score,
        "global_risk_state": (
            global_risk_state.get("global_risk_state")
            if isinstance(global_risk_state, dict)
            else global_risk_state
        ),
    }
    input_availability = {
        key: value not in (None, "", "UNKNOWN")
        for key, value in feature_inputs.items()
    }
    coverage_ratio = round(sum(1 for value in input_availability.values() if value) / max(len(input_availability), 1), 4)
    feature_confidence = _clip(coverage_ratio, 0.0, 1.0)

    gamma_acceleration_base, gamma_pinning_base = _gamma_base_scores(gamma_regime, dealer_position)
    flip_proximity_score = _flip_proximity_score(gamma_flip_distance_pct, spot_vs_flip)
    bias_up, bias_down, bias_pinning = _bias_scores(dealer_hedging_bias)
    flow_up, flow_down = _hedging_flow_scores(dealer_hedging_flow)
    intraday_instability_score, intraday_pinning_score = _intraday_gamma_scores(intraday_gamma_state)
    bullish_flow_confirmation, bearish_flow_confirmation = _flow_confirmation_scores(flow_signal, smart_money_flow)
    acceleration_structure_score, pinning_structure_score, nearest_level_distance_pct = _structure_scores(
        spot,
        support_wall,
        resistance_wall,
        gamma_clusters,
        liquidity_levels,
        liquidity_vacuum_state,
    )
    far_level_dampener = 0.0
    if nearest_level_distance_pct is not None:
        if nearest_level_distance_pct > cfg.far_level_high_pct:
            far_level_dampener = cfg.far_level_high_dampener
        elif nearest_level_distance_pct > cfg.far_level_watch_pct:
            far_level_dampener = cfg.far_level_watch_dampener
    macro_global_boost = _macro_global_boost(
        macro_event_risk_score,
        global_risk_state,
        volatility_explosion_probability,
        gamma_vol_acceleration_score,
    )
    intraday_range_score = _clip(_safe_float(intraday_range_pct, 0.0) / cfg.intraday_range_norm_divisor, 0.0, 1.0)
    gamma_vol_overlay = _clip(_safe_float(gamma_vol_acceleration_score, 0.0) / 100.0, 0.0, 1.0)

    acceleration_base = _clip(
        (cfg.acceleration_base_gamma_weight * gamma_acceleration_base)
        + (cfg.acceleration_base_flip_weight * flip_proximity_score)
        + (cfg.acceleration_base_structure_weight * acceleration_structure_score)
        + (cfg.acceleration_base_intraday_weight * intraday_instability_score)
        + (cfg.acceleration_base_range_weight * intraday_range_score),
        0.0,
        1.0,
    )
    pinning_base = _clip(
        (cfg.pinning_base_gamma_weight * gamma_pinning_base)
        + (cfg.pinning_base_bias_weight * bias_pinning)
        + (cfg.pinning_base_structure_weight * pinning_structure_score)
        + (cfg.pinning_base_intraday_weight * intraday_pinning_score),
        0.0,
        1.0,
    )

    upside_hedging_pressure = round(_clip(
        (cfg.directional_acceleration_weight * acceleration_base)
        + (cfg.directional_bias_weight * bias_up)
        + (cfg.directional_flow_weight * flow_up)
        + (cfg.directional_confirmation_weight * bullish_flow_confirmation)
        + (cfg.directional_macro_weight * macro_global_boost)
        + (cfg.directional_gamma_vol_weight * gamma_vol_overlay)
        + (cfg.directional_intraday_weight * intraday_instability_score)
        - (cfg.directional_pinning_penalty_weight * pinning_base),
        0.0,
        1.0,
    ) * feature_confidence, 4)
    downside_hedging_pressure = round(_clip(
        (cfg.directional_acceleration_weight * acceleration_base)
        + (cfg.directional_bias_weight * bias_down)
        + (cfg.directional_flow_weight * flow_down)
        + (cfg.directional_confirmation_weight * bearish_flow_confirmation)
        + (cfg.directional_macro_weight * macro_global_boost)
        + (cfg.directional_gamma_vol_weight * gamma_vol_overlay)
        + (cfg.directional_intraday_weight * intraday_instability_score)
        - (cfg.directional_pinning_penalty_weight * pinning_base),
        0.0,
        1.0,
    ) * feature_confidence, 4)
    pinning_pressure_score = round(_clip(
        (cfg.pinning_score_base_weight * pinning_base)
        + (cfg.pinning_score_bias_weight * bias_pinning)
        + (cfg.pinning_score_structure_weight * pinning_structure_score)
        + (cfg.pinning_score_flip_inverse_weight * max(0.0, 1.0 - flip_proximity_score))
        + (cfg.pinning_score_gamma_vol_inverse_weight * max(0.0, 1.0 - gamma_vol_overlay))
        + (cfg.pinning_score_intraday_weight * intraday_pinning_score)
        - (cfg.pinning_score_acceleration_penalty_weight * acceleration_base)
        - (cfg.pinning_score_far_level_penalty_weight * far_level_dampener),
        0.0,
        1.0,
    ) * feature_confidence, 4)
    two_sided_instability = round(min(upside_hedging_pressure, downside_hedging_pressure), 4)
    normalized_pressure = round(_clip(
        (cfg.normalized_pressure_directional_weight * max(upside_hedging_pressure, downside_hedging_pressure))
        + (cfg.normalized_pressure_two_sided_weight * two_sided_instability)
        + (cfg.normalized_pressure_pinning_weight * pinning_pressure_score)
        + (cfg.normalized_pressure_acceleration_weight * acceleration_base),
        0.0,
        1.0,
    ), 4)
    overnight_hedging_risk = round(_clip(
        (cfg.overnight_directional_weight * max(upside_hedging_pressure, downside_hedging_pressure))
        + (cfg.overnight_two_sided_weight * two_sided_instability)
        + (cfg.overnight_macro_weight * macro_global_boost)
        + (cfg.overnight_gamma_vol_weight * gamma_vol_overlay)
        + (cfg.overnight_event_weight * _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0))
        + (cfg.overnight_context_weight * (1.0 if holding_context["overnight_relevant"] else 0.0)),
        0.0,
        1.0,
    ), 4)

    warnings = []
    if feature_confidence < cfg.partial_coverage_warning_threshold:
        warnings.append("dealer_pressure_partial_input_coverage")
    if gamma_regime is None:
        warnings.append("gamma_regime_missing")
    if dealer_hedging_bias is None:
        warnings.append("dealer_hedging_bias_missing")

    return {
        "spot": _safe_float(spot, None),
        "gamma_regime": gamma_regime,
        "spot_vs_flip": spot_vs_flip,
        "gamma_flip_distance_pct": _safe_float(gamma_flip_distance_pct, None),
        "dealer_position": dealer_position,
        "dealer_hedging_bias": dealer_hedging_bias,
        "dealer_hedging_flow": dealer_hedging_flow,
        "market_gamma": market_gamma,
        "gamma_clusters": list(gamma_clusters or []),
        "liquidity_levels": list(liquidity_levels or []),
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "liquidity_vacuum_state": liquidity_vacuum_state,
        "intraday_gamma_state": intraday_gamma_state,
        "intraday_range_pct": _safe_float(intraday_range_pct, None),
        "flow_signal": flow_signal,
        "smart_money_flow": smart_money_flow,
        "macro_event_risk_score": _safe_int(macro_event_risk_score, 0),
        "global_risk_state": (
            str(global_risk_state.get("global_risk_state"))
            if isinstance(global_risk_state, dict)
            else str(global_risk_state or "GLOBAL_NEUTRAL")
        ).upper().strip() or "GLOBAL_NEUTRAL",
        "volatility_explosion_probability": round(_clip(_safe_float(volatility_explosion_probability, 0.0), 0.0, 1.0), 4),
        "gamma_vol_acceleration_score": _safe_int(gamma_vol_acceleration_score, 0),
        "holding_context": holding_context,
        "input_availability": input_availability,
        "feature_confidence": round(feature_confidence, 4),
        "gamma_acceleration_base": gamma_acceleration_base,
        "gamma_pinning_base": gamma_pinning_base,
        "flip_proximity_score": round(flip_proximity_score, 4),
        "bias_up_score": round(bias_up, 4),
        "bias_down_score": round(bias_down, 4),
        "bias_pinning_score": round(bias_pinning, 4),
        "flow_up_score": round(flow_up, 4),
        "flow_down_score": round(flow_down, 4),
        "bullish_flow_confirmation": bullish_flow_confirmation,
        "bearish_flow_confirmation": bearish_flow_confirmation,
        "intraday_instability_score": round(intraday_instability_score, 4),
        "intraday_pinning_score": round(intraday_pinning_score, 4),
        "acceleration_structure_score": acceleration_structure_score,
        "pinning_structure_score": pinning_structure_score,
        "nearest_level_distance_pct": nearest_level_distance_pct,
        "far_level_dampener": round(far_level_dampener, 4),
        "macro_global_boost": macro_global_boost,
        "acceleration_base": round(acceleration_base, 4),
        "pinning_base": round(pinning_base, 4),
        "upside_hedging_pressure": upside_hedging_pressure,
        "downside_hedging_pressure": downside_hedging_pressure,
        "pinning_pressure_score": pinning_pressure_score,
        "normalized_pressure": normalized_pressure,
        "overnight_hedging_risk": overnight_hedging_risk,
        "neutral_fallback": feature_confidence == 0.0,
        "warnings": warnings,
    }
