"""
Trade Strength Model

Direction-aware scoring.
"""

from config.signal_policy import (
    get_consensus_score_config,
    get_large_move_scoring_policy_config,
    get_trade_strength_weights,
)


def _wall_proximity_score(spot, support_wall, resistance_wall, direction, proximity_buffer=50):
    weights = get_trade_strength_weights()
    score = 0
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


def _hedging_bias_score(hedging_bias, direction):
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
    weights = get_trade_strength_weights()
    if gamma_regime in ["NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"]:
        return weights["gamma_regime_negative"]

    if gamma_regime in ["POSITIVE_GAMMA", "LONG_GAMMA_ZONE"]:
        return weights["gamma_regime_positive"]

    if gamma_regime == "NEUTRAL_GAMMA":
        return weights["gamma_regime_neutral"]

    return 0


def _spot_vs_flip_score(spot_vs_flip, direction):
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


def _flow_score(flow_signal_value, direction):
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


def _directional_consensus_score(direction, flow_signal_value, smart_money_signal_value, hedging_bias, spot_vs_flip):
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
):
    weights = get_trade_strength_weights()
    breakdown = {
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
    }

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

    breakdown["spot_vs_flip_score"] = _spot_vs_flip_score(
        spot_vs_flip,
        direction
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

    breakdown["wall_proximity_score"] = _wall_proximity_score(
        spot,
        support_wall,
        resistance_wall,
        direction,
        proximity_buffer=proximity_buffer,
    )

    breakdown["liquidity_map_score"] = _liquidity_map_score(
        direction,
        spot,
        next_support,
        next_resistance,
        squeeze_zone
    )

    breakdown["move_model_score"] = _large_move_score(
        large_move_probability,
        ml_move_probability
    )
    breakdown["directional_consensus_score"] = _directional_consensus_score(
        direction,
        flow_signal_value,
        smart_money_signal_value,
        hedging_bias,
        spot_vs_flip,
    )

    total_score = sum(breakdown.values())
    total_score = max(0, min(total_score, 100))

    breakdown["total_score"] = total_score

    return total_score, breakdown
