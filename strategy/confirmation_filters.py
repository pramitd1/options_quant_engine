"""
Confirmation Filters

These filters do not generate direction on their own.
They only confirm, trim, or veto a direction proposed by the engine.

Design goals:
- keep gamma / flow / hedging as the primary signal layer
- add a lightweight confirmation layer based on live spot behavior
- remain robust even when some inputs are missing
"""

from config.symbol_microstructure import DEFAULT_MICROSTRUCTURE_CONFIG, get_microstructure_config


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sign_label(score):
    if score >= 6:
        return "STRONG_CONFIRMATION"
    if score >= 2:
        return "CONFIRMED"
    if score > -3:
        return "MIXED"
    return "CONFLICT"


def compute_confirmation_filters(
    direction,
    spot,
    symbol=None,
    day_open=None,
    prev_close=None,
    intraday_range_pct=None,
    final_flow_signal=None,
    hedging_bias=None,
    gamma_event=None,
    hybrid_move_probability=None,
    spot_vs_flip=None,
):
    """
    Returns a dict:
    {
        "score_adjustment": int,
        "status": str,
        "veto": bool,
        "reasons": list[str],
        "breakdown": dict
    }
    """

    breakdown = {
        "open_alignment_score": 0,
        "prev_close_alignment_score": 0,
        "range_expansion_score": 0,
        "flow_confirmation_score": 0,
        "hedging_confirmation_score": 0,
        "gamma_event_confirmation_score": 0,
        "move_probability_confirmation_score": 0,
        "flip_alignment_score": 0,
    }

    reasons = []
    veto = False

    spot = _safe_float(spot, None)
    open_px = _safe_float(day_open, None)
    prev_close_px = _safe_float(prev_close, None)
    range_pct = _safe_float(intraday_range_pct, None)
    move_prob = _safe_float(hybrid_move_probability, None)
    micro_cfg = get_microstructure_config(symbol)

    if direction not in ["CALL", "PUT"]:
        return {
            "score_adjustment": 0,
            "status": "NO_DIRECTION",
            "veto": False,
            "reasons": ["no_direction"],
            "breakdown": breakdown,
        }

    bullish = direction == "CALL"
    bearish = direction == "PUT"

    # 1) Spot vs session open
    # Softer than before: open is useful, but not definitive by itself.
    if spot is not None and open_px is not None:
        if bullish:
            if spot >= open_px:
                breakdown["open_alignment_score"] = 2
                reasons.append("spot_above_open_confirms_call")
            else:
                breakdown["open_alignment_score"] = -2
                reasons.append("spot_below_open_conflicts_call")
        elif bearish:
            if spot <= open_px:
                breakdown["open_alignment_score"] = 2
                reasons.append("spot_below_open_confirms_put")
            else:
                breakdown["open_alignment_score"] = -2
                reasons.append("spot_above_open_conflicts_put")

    # 2) Spot vs previous close
    # Even softer: prev close is informative, but gap days can distort it.
    if spot is not None and prev_close_px is not None:
        if bullish:
            if spot >= prev_close_px:
                breakdown["prev_close_alignment_score"] = 1
                reasons.append("spot_above_prev_close_confirms_call")
            else:
                breakdown["prev_close_alignment_score"] = -1
                reasons.append("spot_below_prev_close_conflicts_call")
        elif bearish:
            if spot <= prev_close_px:
                breakdown["prev_close_alignment_score"] = 1
                reasons.append("spot_below_prev_close_confirms_put")
            else:
                breakdown["prev_close_alignment_score"] = -1
                reasons.append("spot_above_prev_close_conflicts_put")

    # 3) Range expansion confirmation
    if range_pct is not None:
        if range_pct >= micro_cfg.get("range_expansion_strong", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_strong"]):
            breakdown["range_expansion_score"] = 3
            reasons.append("strong_intraday_range_expansion")
        elif range_pct >= micro_cfg.get("range_expansion_moderate", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_moderate"]):
            breakdown["range_expansion_score"] = 2
            reasons.append("moderate_intraday_range_expansion")
        elif range_pct >= micro_cfg.get("range_expansion_low", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_low"]):
            breakdown["range_expansion_score"] = 1
            reasons.append("early_intraday_range_expansion")
        elif range_pct < micro_cfg.get("range_expansion_cold", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_cold"]):
            breakdown["range_expansion_score"] = -1
            reasons.append("very_low_intraday_range_expansion")

    # 4) Flow confirmation
    if bullish:
        if final_flow_signal == "BULLISH_FLOW":
            breakdown["flow_confirmation_score"] = 3
            reasons.append("bullish_flow_confirms_call")
        elif final_flow_signal == "BEARISH_FLOW":
            breakdown["flow_confirmation_score"] = -4
            reasons.append("bearish_flow_conflicts_call")
    elif bearish:
        if final_flow_signal == "BEARISH_FLOW":
            breakdown["flow_confirmation_score"] = 3
            reasons.append("bearish_flow_confirms_put")
        elif final_flow_signal == "BULLISH_FLOW":
            breakdown["flow_confirmation_score"] = -4
            reasons.append("bullish_flow_conflicts_put")

    # 5) Hedging confirmation
    # Keep this strong because it is closely tied to your core microstructure thesis.
    if bullish:
        if hedging_bias == "UPSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = 3
            reasons.append("upside_acceleration_confirms_call")
        elif hedging_bias == "DOWNSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = -4
            reasons.append("downside_acceleration_conflicts_call")
    elif bearish:
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = 3
            reasons.append("downside_acceleration_confirms_put")
        elif hedging_bias == "UPSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = -4
            reasons.append("upside_acceleration_conflicts_put")

    # 6) Gamma event confirmation
    if gamma_event == "GAMMA_SQUEEZE":
        breakdown["gamma_event_confirmation_score"] = 2
        reasons.append("gamma_squeeze_supports_directional_move")

    # 7) Hybrid move probability confirmation
    if move_prob is not None:
        if move_prob >= 0.65:
            breakdown["move_probability_confirmation_score"] = 3
            reasons.append("high_hybrid_move_probability")
        elif move_prob >= 0.50:
            breakdown["move_probability_confirmation_score"] = 2
            reasons.append("moderate_hybrid_move_probability")
        elif move_prob >= 0.40:
            breakdown["move_probability_confirmation_score"] = 1
            reasons.append("acceptable_hybrid_move_probability")
        elif move_prob < 0.30:
            breakdown["move_probability_confirmation_score"] = -2
            reasons.append("low_hybrid_move_probability")

    # 8) Spot vs flip alignment
    if bullish:
        if spot_vs_flip == "ABOVE_FLIP":
            breakdown["flip_alignment_score"] = 2
            reasons.append("above_flip_confirms_call")
        elif spot_vs_flip == "BELOW_FLIP":
            breakdown["flip_alignment_score"] = -1
            reasons.append("below_flip_soft_conflict_call")
    elif bearish:
        if spot_vs_flip == "BELOW_FLIP":
            breakdown["flip_alignment_score"] = 2
            reasons.append("below_flip_confirms_put")
        elif spot_vs_flip == "ABOVE_FLIP":
            breakdown["flip_alignment_score"] = -1
            reasons.append("above_flip_soft_conflict_put")

    total = sum(breakdown.values())

    # Veto logic: only for strong directional conflict clusters.
    hard_conflicts = 0
    if breakdown["open_alignment_score"] < 0:
        hard_conflicts += 1
    if breakdown["flow_confirmation_score"] < 0:
        hard_conflicts += 1
    if breakdown["hedging_confirmation_score"] < 0:
        hard_conflicts += 1

    if hard_conflicts >= 3 and (move_prob is None or move_prob < 0.55):
        veto = True
        reasons.append("confirmation_veto_due_to_multi_factor_conflict")

    return {
        "score_adjustment": int(total),
        "status": _sign_label(total),
        "veto": veto,
        "reasons": reasons,
        "breakdown": breakdown,
    }
