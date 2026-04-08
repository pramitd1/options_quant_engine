"""
Module: trade_strength.py

Purpose:
    Compute additive trade-strength scores from analytics and probability inputs.

Role in the System:
    Part of the strategy layer that converts directional intent into executable option trades.

Key Outputs:
    Strike rankings, trade-construction inputs, and exit or sizing recommendations.

Downstream Usage:
    Consumed by the signal engine and by research tooling that inspects trade construction choices.
"""

from config.signal_policy import (
    get_consensus_score_config,
    get_large_move_scoring_policy_config,
    get_trade_strength_continuous_policy_config,
    get_trade_runtime_thresholds,
    get_trade_strength_weights,
)


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _normalize(value, lo, hi):
    if value is None:
        return 0.0
    if hi <= lo:
        return 0.0
    return _clip((value - lo) / (hi - lo), 0.0, 1.0)


def _distance_factor(spot, level, distance_cap):
    spot_v = _safe_float(spot, None)
    lvl_v = _safe_float(level, None)
    cap_v = abs(_safe_float(distance_cap, 0.0) or 0.0)
    if spot_v is None or lvl_v is None or cap_v <= 0.0:
        return 0.0
    dist = abs(spot_v - lvl_v)
    if dist >= cap_v:
        return 0.0
    return 1.0 - (dist / cap_v)


def _wall_proximity_score(spot, support_wall, resistance_wall, direction, proximity_buffer=None):
    """
    Purpose:
        Process wall proximity score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        spot (Any): Input associated with spot.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        proximity_buffer (Any): Input associated with proximity buffer.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    score = 0
    if proximity_buffer is None:
        rt = get_trade_runtime_thresholds()
        proximity_buffer = abs(float(rt.get("wall_proximity_buffer", 50)))
    else:
        proximity_buffer = abs(float(proximity_buffer or 0))

    if direction == "CALL":
        if support_wall is not None and abs(spot - support_wall) <= proximity_buffer:
            score += weights["wall_support_bonus"]
        if resistance_wall is not None and abs(spot - resistance_wall) <= proximity_buffer:
            score += weights["wall_resistance_penalty"]

    elif direction == "PUT":
        if resistance_wall is not None and abs(spot - resistance_wall) <= proximity_buffer:
            score += weights["wall_support_bonus"]
        if support_wall is not None and abs(spot - support_wall) <= proximity_buffer:
            score += weights["wall_resistance_penalty"]

    return score


def _wall_proximity_score_continuous(spot, support_wall, resistance_wall, direction, proximity_buffer=None):
    """Distance-aware wall scoring that decays linearly with proximity."""
    weights = get_trade_strength_weights()
    cfg = get_trade_strength_continuous_policy_config()

    if proximity_buffer is None:
        rt = get_trade_runtime_thresholds()
        proximity_buffer = abs(float(rt.get("wall_proximity_buffer", 50)))
    else:
        proximity_buffer = abs(float(proximity_buffer or 0))

    cap = max(1.0, proximity_buffer * cfg.wall_distance_cap_multiplier)
    score = 0.0

    support_factor = _distance_factor(spot, support_wall, cap)
    resistance_factor = _distance_factor(spot, resistance_wall, cap)

    if direction == "CALL":
        score += weights["wall_support_bonus"] * support_factor
        score += weights["wall_resistance_penalty"] * resistance_factor
    elif direction == "PUT":
        score += weights["wall_support_bonus"] * resistance_factor
        score += weights["wall_resistance_penalty"] * support_factor

    return int(round(score))


def _hedging_bias_score(hedging_bias, direction):
    """
    Purpose:
        Process hedging bias score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        hedging_bias (Any): Input associated with hedging bias.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    if direction == "CALL":
        if hedging_bias == "UPSIDE_ACCELERATION":
            return weights["hedging_acceleration_support"]
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            return weights["hedging_acceleration_conflict"]

    if direction == "PUT":
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            return weights["hedging_acceleration_support"]
        if hedging_bias == "UPSIDE_ACCELERATION":
            return weights["hedging_acceleration_conflict"]

    return 0


def _gamma_regime_score(gamma_regime):
    """
    Purpose:
        Process gamma regime score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        gamma_regime (Any): Input associated with gamma regime.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    if gamma_regime in ["NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"]:
        return weights["gamma_regime_negative"]

    if gamma_regime in ["POSITIVE_GAMMA", "LONG_GAMMA_ZONE"]:
        return weights["gamma_regime_positive"]

    if gamma_regime == "NEUTRAL_GAMMA":
        return weights["gamma_regime_neutral"]

    return 0


def _spot_vs_flip_score(spot_vs_flip, direction):
    """
    Purpose:
        Process spot vs flip score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        spot_vs_flip (Any): Input associated with spot vs flip.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    if direction == "CALL":
        if spot_vs_flip == "ABOVE_FLIP":
            return weights["spot_flip_primary"]
        if spot_vs_flip == "BELOW_FLIP":
            return weights["spot_flip_secondary"]

    if direction == "PUT":
        if spot_vs_flip == "BELOW_FLIP":
            return weights["spot_flip_primary"]
        if spot_vs_flip == "ABOVE_FLIP":
            return weights["spot_flip_secondary"]

    return 0


def _spot_vs_flip_score_continuous(spot_vs_flip, direction, flip_distance_pct=None):
    """Continuous spot-vs-flip scoring based on directional side and distance."""
    weights = get_trade_strength_weights()
    cfg = get_trade_strength_continuous_policy_config()

    primary = float(weights["spot_flip_primary"])
    secondary = float(weights["spot_flip_secondary"])
    conflict_floor = float(cfg.spot_flip_conflict_floor)

    if spot_vs_flip == "AT_FLIP":
        return int(round(0.5 * secondary))

    d = _normalize(_safe_float(flip_distance_pct, 0.0), 0.0, cfg.flip_distance_cap_pct)

    if direction == "CALL":
        if spot_vs_flip == "ABOVE_FLIP":
            return int(round(secondary + (primary - secondary) * d))
        if spot_vs_flip == "BELOW_FLIP":
            return int(round(secondary + (conflict_floor - secondary) * d))

    if direction == "PUT":
        if spot_vs_flip == "BELOW_FLIP":
            return int(round(secondary + (primary - secondary) * d))
        if spot_vs_flip == "ABOVE_FLIP":
            return int(round(secondary + (conflict_floor - secondary) * d))

    return 0


def _flip_zone_dampener_score(spot_vs_flip, gamma_regime):
    """Apply a trade-strength penalty when spot sits at the gamma flip zone
    and the gamma regime offers no structural advantage.

    Negative or short-gamma at the flip means the dealer-hedging flow is
    unpredictable and the engine has no directional edge.  Neutral gamma is
    slightly better but still weak.  Positive gamma at the flip provides
    mean-reversion support, so no penalty is applied.
    """
    if spot_vs_flip != "AT_FLIP":
        return 0

    weights = get_trade_strength_weights()

    if gamma_regime in ("NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"):
        return weights.get("flip_zone_negative_gamma_penalty", -12)

    if gamma_regime == "NEUTRAL_GAMMA":
        return weights.get("flip_zone_neutral_gamma_penalty", -8)

    return 0


def _flow_score(flow_signal_value, direction):
    """
    Purpose:
        Process flow score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        flow_signal_value (Any): Input associated with flow signal value.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    if direction == "CALL":
        if flow_signal_value == "BULLISH_FLOW":
            return weights["flow_call_bullish"]
        if flow_signal_value == "BEARISH_FLOW":
            return weights["flow_call_bearish"]

    if direction == "PUT":
        if flow_signal_value == "BEARISH_FLOW":
            return weights["flow_put_bearish"]
        if flow_signal_value == "BULLISH_FLOW":
            return weights["flow_put_bullish"]

    return 0


def _smart_money_score(smart_money_signal_value, direction):
    """
    Purpose:
        Process smart money score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        smart_money_signal_value (Any): Input associated with smart money signal value.
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    if direction == "CALL":
        if smart_money_signal_value == "BULLISH_FLOW":
            return weights["smart_call_bullish"]
        if smart_money_signal_value == "BEARISH_FLOW":
            return weights["smart_call_bearish"]

    if direction == "PUT":
        if smart_money_signal_value == "BEARISH_FLOW":
            return weights["smart_put_bearish"]
        if smart_money_signal_value == "BULLISH_FLOW":
            return weights["smart_put_bullish"]

    return 0


def _liquidity_map_score(direction, spot, next_support, next_resistance, squeeze_zone):
    """
    Purpose:
        Process liquidity map score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        spot (Any): Input associated with spot.
        next_support (Any): Input associated with next support.
        next_resistance (Any): Input associated with next resistance.
        squeeze_zone (Any): Input associated with squeeze zone.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    score = 0

    if direction == "CALL":
        if next_resistance is not None and next_resistance > spot:
            score += weights["liquidity_map_path_bonus"]
        if squeeze_zone is not None and squeeze_zone >= spot:
            score += weights["liquidity_map_path_bonus"]

    if direction == "PUT":
        if next_support is not None and next_support < spot:
            score += weights["liquidity_map_path_bonus"]
        if squeeze_zone is not None and squeeze_zone <= spot:
            score += weights["liquidity_map_path_bonus"]

    return score


def _liquidity_map_score_continuous(direction, spot, next_support, next_resistance, squeeze_zone, proximity_buffer):
    """Distance-aware liquidity path scoring with smooth decay."""
    weights = get_trade_strength_weights()
    cfg = get_trade_strength_continuous_policy_config()
    cap = max(1.0, abs(_safe_float(proximity_buffer, 50.0) or 50.0) * cfg.liquidity_path_distance_cap_multiplier)

    score = 0.0
    if direction == "CALL":
        if _safe_float(next_resistance, None) is not None and _safe_float(spot, 0.0) < _safe_float(next_resistance, 0.0):
            score += weights["liquidity_map_path_bonus"] * _distance_factor(spot, next_resistance, cap)
        if _safe_float(squeeze_zone, None) is not None and _safe_float(squeeze_zone, 0.0) >= _safe_float(spot, 0.0):
            score += weights["liquidity_map_path_bonus"] * _distance_factor(spot, squeeze_zone, cap)

    if direction == "PUT":
        if _safe_float(next_support, None) is not None and _safe_float(spot, 0.0) > _safe_float(next_support, 0.0):
            score += weights["liquidity_map_path_bonus"] * _distance_factor(spot, next_support, cap)
        if _safe_float(squeeze_zone, None) is not None and _safe_float(squeeze_zone, 0.0) <= _safe_float(spot, 0.0):
            score += weights["liquidity_map_path_bonus"] * _distance_factor(spot, squeeze_zone, cap)

    return int(round(score))


def _directional_consensus_score(direction, flow_signal_value, smart_money_signal_value, hedging_bias, spot_vs_flip):
    """
    Purpose:
        Process directional consensus score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        flow_signal_value (Any): Input associated with flow signal value.
        smart_money_signal_value (Any): Input associated with smart money signal value.
        hedging_bias (Any): Input associated with hedging bias.
        spot_vs_flip (Any): Input associated with spot vs flip.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    cfg = get_consensus_score_config()
    aligned = 0
    conflicts = 0

    if direction == "CALL":
        aligned += int(flow_signal_value == "BULLISH_FLOW")
        aligned += int(smart_money_signal_value == "BULLISH_FLOW")
        aligned += int(hedging_bias == "UPSIDE_ACCELERATION")
        aligned += int(spot_vs_flip == "ABOVE_FLIP")
        conflicts += int(flow_signal_value == "BEARISH_FLOW")
        conflicts += int(smart_money_signal_value == "BEARISH_FLOW")
        conflicts += int(hedging_bias == "DOWNSIDE_ACCELERATION")
        conflicts += int(spot_vs_flip == "BELOW_FLIP")

    if direction == "PUT":
        aligned += int(flow_signal_value == "BEARISH_FLOW")
        aligned += int(smart_money_signal_value == "BEARISH_FLOW")
        aligned += int(hedging_bias == "DOWNSIDE_ACCELERATION")
        aligned += int(spot_vs_flip == "BELOW_FLIP")
        conflicts += int(flow_signal_value == "BULLISH_FLOW")
        conflicts += int(smart_money_signal_value == "BULLISH_FLOW")
        conflicts += int(hedging_bias == "UPSIDE_ACCELERATION")
        conflicts += int(spot_vs_flip == "ABOVE_FLIP")

    if aligned >= 3 and conflicts == 0:
        return cfg["strong_alignment_bonus"]
    if aligned >= 2 and conflicts <= 1:
        return cfg["moderate_alignment_bonus"]
    if conflicts >= 2 and aligned <= 1:
        return cfg["conflict_penalty"]
    return 0


def _probability_bucket_score(probability, buckets):
    """
    Purpose:
        Process probability bucket score for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        probability (Any): Input associated with probability.
        buckets (Any): Input associated with buckets.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if probability is None:
        return 0

    try:
        p = float(probability)
    except Exception:
        return 0

    for threshold, score in buckets:
        if p >= threshold:
            return score

    return 0


def _large_move_score(large_move_probability, ml_move_probability):
    """
    More gradual scoring than the old step function.

    Hybrid/rule probability is the primary signal.
    ML probability is supportive, with lower standalone weight.
    """
    cfg = get_large_move_scoring_policy_config()
    hybrid_score = _probability_bucket_score(
        large_move_probability,
        [
            (cfg.hybrid_threshold_extreme, cfg.hybrid_score_extreme),
            (cfg.hybrid_threshold_high, cfg.hybrid_score_high),
            (cfg.hybrid_threshold_moderate, cfg.hybrid_score_moderate),
            (cfg.hybrid_threshold_watch, cfg.hybrid_score_watch),
            (cfg.hybrid_threshold_tail, cfg.hybrid_score_tail),
        ],
    )

    ml_score = _probability_bucket_score(
        ml_move_probability,
        [
            (cfg.ml_threshold_extreme, cfg.ml_score_extreme),
            (cfg.ml_threshold_high, cfg.ml_score_high),
            (cfg.ml_threshold_moderate, cfg.ml_score_moderate),
            (cfg.ml_threshold_watch, cfg.ml_score_watch),
            (cfg.ml_threshold_tail, cfg.ml_score_tail),
        ],
    )

    # Avoid double counting when both are saying essentially the same thing.
    if hybrid_score >= cfg.overlap_hybrid_floor and ml_score >= cfg.overlap_ml_floor:
        ml_score -= cfg.overlap_penalty

    total = hybrid_score + max(0, ml_score)
    return min(total, cfg.total_score_cap)


def _large_move_score_continuous(large_move_probability, ml_move_probability):
    """Continuous probability scoring to replace coarse threshold buckets."""
    cfg = get_trade_strength_continuous_policy_config()

    hybrid_prob = _safe_float(large_move_probability, None)
    ml_prob = _safe_float(ml_move_probability, None)

    hybrid_norm = _normalize(hybrid_prob, cfg.hybrid_probability_floor, cfg.hybrid_probability_ceiling)
    ml_norm = _normalize(ml_prob, cfg.ml_probability_floor, cfg.ml_probability_ceiling)

    hybrid_score = cfg.hybrid_max_score * hybrid_norm
    ml_score = cfg.ml_max_score * ml_norm

    if (hybrid_prob is not None and hybrid_prob >= cfg.overlap_hybrid_threshold) and (
        ml_prob is not None and ml_prob >= cfg.overlap_ml_threshold
    ):
        ml_score -= cfg.overlap_penalty

    total = hybrid_score + max(0.0, ml_score)
    return int(round(_clip(total, 0.0, float(cfg.probability_total_score_cap))))


def compute_trade_strength(
    direction,
    flow_signal_value,
    smart_money_signal_value,
    gamma_event,
    dealer_pos,
    vol_regime,
    void_signal,
    vacuum_state,
    spot_vs_flip,
    hedging_bias,
    gamma_regime,
    intraday_gamma_state,
    support_wall,
    resistance_wall,
    spot,
    next_support=None,
    next_resistance=None,
    squeeze_zone=None,
    large_move_probability=None,
    ml_move_probability=None,
    proximity_buffer=50,
    flip_distance_pct=None,
    scoring_mode=None,
    oi_velocity_score=None,
    rr_value=None,
    rr_momentum=None,
    volume_pcr_atm=None,
    gamma_flip_drift=None,
    max_pain_dist=None,
    max_pain_zone=None,
    days_to_expiry=None,
    rr_unit="VOL_POINTS",
):
    """
    Purpose:
        Compute trade strength from the supplied inputs.
    
    Context:
        Public function within the strategy layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        flow_signal_value (Any): Input associated with flow signal value.
        smart_money_signal_value (Any): Input associated with smart money signal value.
        gamma_event (Any): Input associated with gamma event.
        dealer_pos (Any): Input associated with dealer pos.
        vol_regime (Any): Input associated with vol regime.
        void_signal (Any): Input associated with void signal.
        vacuum_state (Any): Structured state payload for vacuum.
        spot_vs_flip (Any): Input associated with spot vs flip.
        hedging_bias (Any): Input associated with hedging bias.
        gamma_regime (Any): Input associated with gamma regime.
        intraday_gamma_state (Any): Structured state payload for intraday gamma.
        support_wall (Any): Input associated with support wall.
        resistance_wall (Any): Input associated with resistance wall.
        spot (Any): Input associated with spot.
        next_support (Any): Input associated with next support.
        next_resistance (Any): Input associated with next resistance.
        squeeze_zone (Any): Input associated with squeeze zone.
        large_move_probability (Any): Input associated with large move probability.
        ml_move_probability (Any): Input associated with ML move probability.
        proximity_buffer (Any): Input associated with proximity buffer.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    weights = get_trade_strength_weights()
    breakdown = {
        "trade_strength_scoring_mode": "continuous",
        "flow_signal_score": 0,
        "smart_money_flow_score": 0,
        "gamma_event_score": 0,
        "dealer_position_score": 0,
        "volatility_regime_score": 0,
        "liquidity_void_score": 0,
        "liquidity_vacuum_score": 0,
        "spot_vs_flip_score": 0,
        "hedging_bias_score": 0,
        "gamma_regime_score": 0,
        "intraday_gamma_shift_score": 0,
        "wall_proximity_score": 0,
        "liquidity_map_score": 0,
        "move_model_score": 0,
        "directional_consensus_score": 0,
        "flip_zone_dampener_score": 0,
        "oi_velocity_score_component": 0,
        "rr_score_component": 0,
        "pcr_score_component": 0,
        "flip_drift_score_component": 0,
        "max_pain_expiry_component": 0,
    }

    if not scoring_mode:
        rt = get_trade_runtime_thresholds()
        scoring_mode = str(rt.get("trade_strength_scoring_mode", "discrete") or "discrete").strip().lower()
    else:
        scoring_mode = str(scoring_mode).strip().lower()

    if scoring_mode not in {"discrete", "continuous"}:
        scoring_mode = "discrete"
    breakdown["trade_strength_scoring_mode"] = scoring_mode

    breakdown["flow_signal_score"] = _flow_score(flow_signal_value, direction)
    breakdown["smart_money_flow_score"] = _smart_money_score(
        smart_money_signal_value,
        direction
    )

    if gamma_event == "GAMMA_SQUEEZE":
        breakdown["gamma_event_score"] = weights["gamma_event_bonus"]

    if dealer_pos == "Short Gamma":
        breakdown["dealer_position_score"] = weights["dealer_short_gamma_bonus"]
    elif dealer_pos == "Long Gamma":
        breakdown["dealer_position_score"] = weights["dealer_long_gamma_bonus"]

    if vol_regime == "VOL_EXPANSION":
        breakdown["volatility_regime_score"] = weights["vol_expansion_bonus"]
    elif vol_regime == "NORMAL_VOL":
        breakdown["volatility_regime_score"] = weights["normal_vol_bonus"]

    if void_signal == "VOID_NEAR":
        breakdown["liquidity_void_score"] = weights["liquidity_void_near_bonus"]
    elif void_signal == "VOID_FAR":
        breakdown["liquidity_void_score"] = weights["liquidity_void_far_bonus"]

    if vacuum_state == "BREAKOUT_ZONE":
        breakdown["liquidity_vacuum_score"] = weights["vacuum_breakout_bonus"]
    elif vacuum_state in ["NEAR_VACUUM", "VACUUM_WATCH"]:
        breakdown["liquidity_vacuum_score"] = weights["vacuum_watch_bonus"]

    if scoring_mode == "continuous":
        breakdown["spot_vs_flip_score"] = _spot_vs_flip_score_continuous(
            spot_vs_flip,
            direction,
            flip_distance_pct=flip_distance_pct,
        )
    else:
        breakdown["spot_vs_flip_score"] = _spot_vs_flip_score(
            spot_vs_flip,
            direction,
        )

    breakdown["hedging_bias_score"] = _hedging_bias_score(
        hedging_bias,
        direction
    )

    breakdown["gamma_regime_score"] = _gamma_regime_score(gamma_regime)

    if intraday_gamma_state == "VOL_EXPANSION":
        breakdown["intraday_gamma_shift_score"] = weights["intraday_vol_expansion_bonus"]
    elif intraday_gamma_state == "GAMMA_DECREASE":
        breakdown["intraday_gamma_shift_score"] = weights["intraday_gamma_decrease_bonus"]

    if scoring_mode == "continuous":
        breakdown["wall_proximity_score"] = _wall_proximity_score_continuous(
            spot,
            support_wall,
            resistance_wall,
            direction,
            proximity_buffer=proximity_buffer,
        )
    else:
        breakdown["wall_proximity_score"] = _wall_proximity_score(
            spot,
            support_wall,
            resistance_wall,
            direction,
            proximity_buffer=proximity_buffer,
        )

    if scoring_mode == "continuous":
        breakdown["liquidity_map_score"] = _liquidity_map_score_continuous(
            direction,
            spot,
            next_support,
            next_resistance,
            squeeze_zone,
            proximity_buffer=proximity_buffer,
        )
    else:
        breakdown["liquidity_map_score"] = _liquidity_map_score(
            direction,
            spot,
            next_support,
            next_resistance,
            squeeze_zone,
        )

    if scoring_mode == "continuous":
        breakdown["move_model_score"] = _large_move_score_continuous(
            large_move_probability,
            ml_move_probability,
        )
    else:
        breakdown["move_model_score"] = _large_move_score(
            large_move_probability,
            ml_move_probability,
        )
    breakdown["directional_consensus_score"] = _directional_consensus_score(
        direction,
        flow_signal_value,
        smart_money_signal_value,
        hedging_bias,
        spot_vs_flip,
    )

    breakdown["flip_zone_dampener_score"] = _flip_zone_dampener_score(
        spot_vs_flip,
        gamma_regime,
    )

    rt = get_trade_runtime_thresholds()
    use_oi_velocity = str(rt.get("use_oi_velocity_in_direction", 1)).strip().lower() not in {"0", "false", "no", "off"}
    use_rr = str(rt.get("use_rr_in_direction", 1)).strip().lower() not in {"0", "false", "no", "off"}
    use_pcr = str(rt.get("use_pcr_in_confirmation", 1)).strip().lower() not in {"0", "false", "no", "off"}
    use_max_pain = str(rt.get("use_max_pain_expiry_overlay", 1)).strip().lower() not in {"0", "false", "no", "off"}

    vel = _safe_float(oi_velocity_score, None)
    vel_on = abs(_safe_float(rt.get("oi_velocity_vote_on"), 0.18) or 0.18)
    if use_oi_velocity and vel is not None and abs(vel) >= vel_on:
        if (direction == "CALL" and vel > 0) or (direction == "PUT" and vel < 0):
            breakdown["oi_velocity_score_component"] = weights.get("oi_velocity_alignment_bonus", 4)
        else:
            breakdown["oi_velocity_score_component"] = weights.get("oi_velocity_conflict_penalty", -3)

    rr = _safe_float(rr_value, None)
    if rr is not None and str(rr_unit or "VOL_POINTS").upper().strip() == "DECIMAL":
        rr *= 100.0
    if use_rr and rr is not None:
        rr_put_dom = _safe_float(rt.get("rr_skew_put_dominant"), 0.75)
        rr_call_dom = _safe_float(rt.get("rr_skew_call_dominant"), -0.75)
        rr_score = 0
        if (direction == "PUT" and rr >= rr_put_dom) or (direction == "CALL" and rr <= rr_call_dom):
            rr_score += weights.get("rr_alignment_bonus", 3)
        elif (direction == "CALL" and rr >= rr_put_dom) or (direction == "PUT" and rr <= rr_call_dom):
            rr_score += weights.get("rr_conflict_penalty", -2)

        rr_m = str(rr_momentum or "").upper().strip()
        if rr_m == "RISING_PUT_SKEW":
            rr_score += 1 if direction == "PUT" else -1
        elif rr_m == "FALLING_PUT_SKEW":
            rr_score += 1 if direction == "CALL" else -1
        rr_component = int(_clip(rr_score, -4, 4))
        breakdown["rr_score_component"] = rr_component
        if rr_component == 0:
            breakdown["rr_score_reason"] = "rr_neutral_under_current_thresholds"

    pcr = _safe_float(volume_pcr_atm, None)
    if use_pcr and pcr is not None:
        pcr_put_dom = _safe_float(rt.get("volume_pcr_atm_put_dominant"), 1.20)
        pcr_call_dom = _safe_float(rt.get("volume_pcr_atm_call_dominant"), 0.80)
        if (direction == "PUT" and pcr >= pcr_put_dom) or (direction == "CALL" and pcr <= pcr_call_dom):
            breakdown["pcr_score_component"] = weights.get("pcr_alignment_bonus", 2)
        elif (direction == "CALL" and pcr >= pcr_put_dom) or (direction == "PUT" and pcr <= pcr_call_dom):
            breakdown["pcr_score_component"] = weights.get("pcr_conflict_penalty", -2)

    drift_pts = None
    if isinstance(gamma_flip_drift, dict):
        drift_pts = _safe_float(gamma_flip_drift.get("drift"), None)
    drift_on = abs(_safe_float(rt.get("gamma_flip_drift_pts_vote_on"), 80.0) or 80.0)
    if drift_pts is not None and abs(drift_pts) >= drift_on:
        if (direction == "CALL" and drift_pts > 0) or (direction == "PUT" and drift_pts < 0):
            breakdown["flip_drift_score_component"] = weights.get("flip_drift_alignment_bonus", 2)
        else:
            breakdown["flip_drift_score_component"] = weights.get("flip_drift_conflict_penalty", -2)

    if use_max_pain:
        dte = _safe_float(days_to_expiry, None)
        max_dte = _safe_float(rt.get("max_pain_overlay_max_dte"), 2.0)
        mp_dist = _safe_float(max_pain_dist, None)
        if dte is not None and dte <= max_dte and mp_dist is not None:
            pin_pts_min = abs(_safe_float(rt.get("max_pain_pin_distance_pts_min"), 80.0) or 80.0)
            pin_base = int(_safe_float(rt.get("max_pain_pin_penalty_base"), -2))
            pin_strong = int(_safe_float(rt.get("max_pain_pin_penalty_strong"), -4))
            if abs(mp_dist) <= pin_pts_min * 0.5:
                breakdown["max_pain_expiry_component"] = pin_strong
            elif abs(mp_dist) <= pin_pts_min:
                breakdown["max_pain_expiry_component"] = pin_base

    total_score = sum(v for v in breakdown.values() if isinstance(v, (int, float)))
    total_score = max(0, min(total_score, 100))

    breakdown["total_score"] = total_score

    return total_score, breakdown
