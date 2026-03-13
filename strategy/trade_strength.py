"""
Trade Strength Model

Direction-aware scoring.
"""


def _wall_proximity_score(spot, support_wall, resistance_wall, direction):
    score = 0

    if direction == "CALL":
        if support_wall is not None and abs(spot - support_wall) <= 50:
            score += 5
        if resistance_wall is not None and abs(spot - resistance_wall) <= 50:
            score -= 3

    elif direction == "PUT":
        if resistance_wall is not None and abs(spot - resistance_wall) <= 50:
            score += 5
        if support_wall is not None and abs(spot - support_wall) <= 50:
            score -= 3

    return score


def _hedging_bias_score(hedging_bias, direction):
    if direction == "CALL":
        if hedging_bias == "UPSIDE_ACCELERATION":
            return 10
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            return -6

    if direction == "PUT":
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            return 10
        if hedging_bias == "UPSIDE_ACCELERATION":
            return -6

    return 0


def _gamma_regime_score(gamma_regime):
    if gamma_regime in ["NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"]:
        return 10

    if gamma_regime in ["POSITIVE_GAMMA", "LONG_GAMMA_ZONE"]:
        return 2

    if gamma_regime == "NEUTRAL_GAMMA":
        return 5

    return 0


def _spot_vs_flip_score(spot_vs_flip, direction):
    if direction == "CALL":
        if spot_vs_flip == "ABOVE_FLIP":
            return 8
        if spot_vs_flip == "BELOW_FLIP":
            return 2

    if direction == "PUT":
        if spot_vs_flip == "BELOW_FLIP":
            return 8
        if spot_vs_flip == "ABOVE_FLIP":
            return 2

    return 0


def _flow_score(flow_signal_value, direction):
    if direction == "CALL":
        if flow_signal_value == "BULLISH_FLOW":
            return 20
        if flow_signal_value == "BEARISH_FLOW":
            return -10

    if direction == "PUT":
        if flow_signal_value == "BEARISH_FLOW":
            return 20
        if flow_signal_value == "BULLISH_FLOW":
            return -10

    return 0


def _smart_money_score(smart_money_signal_value, direction):
    if direction == "CALL":
        if smart_money_signal_value == "BULLISH_FLOW":
            return 15
        if smart_money_signal_value == "BEARISH_FLOW":
            return -8

    if direction == "PUT":
        if smart_money_signal_value == "BEARISH_FLOW":
            return 15
        if smart_money_signal_value == "BULLISH_FLOW":
            return -8

    return 0


def _liquidity_map_score(direction, spot, next_support, next_resistance, squeeze_zone):
    score = 0

    if direction == "CALL":
        if next_resistance is not None and next_resistance > spot:
            score += 4
        if squeeze_zone is not None and squeeze_zone >= spot:
            score += 4

    if direction == "PUT":
        if next_support is not None and next_support < spot:
            score += 4
        if squeeze_zone is not None and squeeze_zone <= spot:
            score += 4

    return score


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
    hybrid_score = _probability_bucket_score(
        large_move_probability,
        [
            (0.75, 12),
            (0.65, 10),
            (0.55, 8),
            (0.45, 6),
            (0.35, 3),
        ],
    )

    ml_score = _probability_bucket_score(
        ml_move_probability,
        [
            (0.75, 6),
            (0.65, 5),
            (0.55, 4),
            (0.45, 2),
            (0.35, 1),
        ],
    )

    # Avoid double counting when both are saying essentially the same thing.
    if hybrid_score >= 8 and ml_score >= 4:
        ml_score -= 1

    total = hybrid_score + max(0, ml_score)
    return min(total, 14)


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
    ml_move_probability=None
):
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
        "move_model_score": 0
    }

    breakdown["flow_signal_score"] = _flow_score(flow_signal_value, direction)
    breakdown["smart_money_flow_score"] = _smart_money_score(
        smart_money_signal_value,
        direction
    )

    if gamma_event == "GAMMA_SQUEEZE":
        breakdown["gamma_event_score"] = 10

    if dealer_pos == "Short Gamma":
        breakdown["dealer_position_score"] = 10
    elif dealer_pos == "Long Gamma":
        breakdown["dealer_position_score"] = 5

    if vol_regime == "VOL_EXPANSION":
        breakdown["volatility_regime_score"] = 10
    elif vol_regime == "NORMAL_VOL":
        breakdown["volatility_regime_score"] = 5

    if void_signal == "VOID_NEAR":
        breakdown["liquidity_void_score"] = 10
    elif void_signal == "VOID_FAR":
        breakdown["liquidity_void_score"] = 4

    if vacuum_state == "BREAKOUT_ZONE":
        breakdown["liquidity_vacuum_score"] = 10
    elif vacuum_state in ["NEAR_VACUUM", "VACUUM_WATCH"]:
        breakdown["liquidity_vacuum_score"] = 4

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
        breakdown["intraday_gamma_shift_score"] = 5
    elif intraday_gamma_state == "GAMMA_DECREASE":
        breakdown["intraday_gamma_shift_score"] = 3

    breakdown["wall_proximity_score"] = _wall_proximity_score(
        spot,
        support_wall,
        resistance_wall,
        direction
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

    total_score = sum(breakdown.values())
    total_score = max(0, min(total_score, 100))

    breakdown["total_score"] = total_score

    return total_score, breakdown
