"""
Module: confirmation_filters.py

Purpose:
    Evaluate confirmation filters that reinforce or veto candidate trade ideas.

Role in the System:
    Part of the strategy layer that converts directional intent into executable option trades.

Key Outputs:
    Strike rankings, trade-construction inputs, and exit or sizing recommendations.

Downstream Usage:
    Consumed by the signal engine and by research tooling that inspects trade construction choices.
"""

from config.signal_policy import get_confirmation_filter_config
from config.symbol_microstructure import DEFAULT_MICROSTRUCTURE_CONFIG, get_microstructure_config


def _safe_float(x, default=None):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `confirmation filters` module. The module sits in the strategy layer that converts directional intent into executable option trades.

    Inputs:
        x (Any): Raw scalar input supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sign_label(score, cfg):
    """
    Purpose:
        Process sign label for downstream use.
    
    Context:
        Internal helper within the strategy layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        score (Any): Input associated with score.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if score >= cfg["strong_confirmation_threshold"]:
        return "STRONG_CONFIRMATION"
    if score >= cfg["confirmed_threshold"]:
        return "CONFIRMED"
    if score > cfg["mixed_threshold"]:
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
    cfg = get_confirmation_filter_config()

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
                breakdown["open_alignment_score"] = cfg["open_alignment_support"]
                reasons.append("spot_above_open_confirms_call")
            else:
                breakdown["open_alignment_score"] = cfg["open_alignment_conflict"]
                reasons.append("spot_below_open_conflicts_call")
        elif bearish:
            if spot <= open_px:
                breakdown["open_alignment_score"] = cfg["open_alignment_support"]
                reasons.append("spot_below_open_confirms_put")
            else:
                breakdown["open_alignment_score"] = cfg["open_alignment_conflict"]
                reasons.append("spot_above_open_conflicts_put")

    # 2) Spot vs previous close
    # Even softer: prev close is informative, but gap days can distort it.
    if spot is not None and prev_close_px is not None:
        if bullish:
            if spot >= prev_close_px:
                breakdown["prev_close_alignment_score"] = cfg["prev_close_alignment_support"]
                reasons.append("spot_above_prev_close_confirms_call")
            else:
                breakdown["prev_close_alignment_score"] = cfg["prev_close_alignment_conflict"]
                reasons.append("spot_below_prev_close_conflicts_call")
        elif bearish:
            if spot <= prev_close_px:
                breakdown["prev_close_alignment_score"] = cfg["prev_close_alignment_support"]
                reasons.append("spot_below_prev_close_confirms_put")
            else:
                breakdown["prev_close_alignment_score"] = cfg["prev_close_alignment_conflict"]
                reasons.append("spot_above_prev_close_conflicts_put")

    # 3) Range expansion confirmation
    if range_pct is not None:
        if range_pct >= micro_cfg.get("range_expansion_strong", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_strong"]):
            breakdown["range_expansion_score"] = cfg["range_expansion_strong_score"]
            reasons.append("strong_intraday_range_expansion")
        elif range_pct >= micro_cfg.get("range_expansion_moderate", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_moderate"]):
            breakdown["range_expansion_score"] = cfg["range_expansion_moderate_score"]
            reasons.append("moderate_intraday_range_expansion")
        elif range_pct >= micro_cfg.get("range_expansion_low", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_low"]):
            breakdown["range_expansion_score"] = cfg["range_expansion_low_score"]
            reasons.append("early_intraday_range_expansion")
        elif range_pct < micro_cfg.get("range_expansion_cold", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_cold"]):
            breakdown["range_expansion_score"] = cfg["range_expansion_cold_score"]
            reasons.append("very_low_intraday_range_expansion")

    # 4) Flow confirmation
    if bullish:
        if final_flow_signal == "BULLISH_FLOW":
            breakdown["flow_confirmation_score"] = cfg["flow_support"]
            reasons.append("bullish_flow_confirms_call")
        elif final_flow_signal == "BEARISH_FLOW":
            breakdown["flow_confirmation_score"] = cfg["flow_conflict"]
            reasons.append("bearish_flow_conflicts_call")
    elif bearish:
        if final_flow_signal == "BEARISH_FLOW":
            breakdown["flow_confirmation_score"] = cfg["flow_support"]
            reasons.append("bearish_flow_confirms_put")
        elif final_flow_signal == "BULLISH_FLOW":
            breakdown["flow_confirmation_score"] = cfg["flow_conflict"]
            reasons.append("bullish_flow_conflicts_put")

    # 5) Hedging confirmation
    # Keep this strong because it is closely tied to your core microstructure thesis.
    if bullish:
        if hedging_bias == "UPSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = cfg["hedging_support"]
            reasons.append("upside_acceleration_confirms_call")
        elif hedging_bias == "DOWNSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = cfg["hedging_conflict"]
            reasons.append("downside_acceleration_conflicts_call")
    elif bearish:
        if hedging_bias == "DOWNSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = cfg["hedging_support"]
            reasons.append("downside_acceleration_confirms_put")
        elif hedging_bias == "UPSIDE_ACCELERATION":
            breakdown["hedging_confirmation_score"] = cfg["hedging_conflict"]
            reasons.append("upside_acceleration_conflicts_put")

    # 6) Gamma event confirmation
    if gamma_event == "GAMMA_SQUEEZE":
        breakdown["gamma_event_confirmation_score"] = cfg["gamma_event_support"]
        reasons.append("gamma_squeeze_supports_directional_move")

    # 7) Hybrid move probability confirmation
    if move_prob is not None:
        if move_prob >= cfg["move_probability_high_threshold"]:
            breakdown["move_probability_confirmation_score"] = cfg["move_probability_high_score"]
            reasons.append("high_hybrid_move_probability")
        elif move_prob >= cfg["move_probability_moderate_threshold"]:
            breakdown["move_probability_confirmation_score"] = cfg["move_probability_moderate_score"]
            reasons.append("moderate_hybrid_move_probability")
        elif move_prob >= cfg["move_probability_low_support_threshold"]:
            breakdown["move_probability_confirmation_score"] = cfg["move_probability_low_support_score"]
            reasons.append("acceptable_hybrid_move_probability")
        elif move_prob < cfg["move_probability_conflict_threshold"]:
            breakdown["move_probability_confirmation_score"] = cfg["move_probability_conflict_score"]
            reasons.append("low_hybrid_move_probability")

    # 8) Spot vs flip alignment
    if bullish:
        if spot_vs_flip == "ABOVE_FLIP":
            breakdown["flip_alignment_score"] = cfg["flip_alignment_support"]
            reasons.append("above_flip_confirms_call")
        elif spot_vs_flip == "BELOW_FLIP":
            breakdown["flip_alignment_score"] = cfg["flip_alignment_conflict"]
            reasons.append("below_flip_soft_conflict_call")
    elif bearish:
        if spot_vs_flip == "BELOW_FLIP":
            breakdown["flip_alignment_score"] = cfg["flip_alignment_support"]
            reasons.append("below_flip_confirms_put")
        elif spot_vs_flip == "ABOVE_FLIP":
            breakdown["flip_alignment_score"] = cfg["flip_alignment_conflict"]
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

    if hard_conflicts >= cfg["veto_hard_conflicts"] and (
        move_prob is None or move_prob < cfg["veto_move_probability_ceiling"]
    ):
        veto = True
        reasons.append("confirmation_veto_due_to_multi_factor_conflict")

    return {
        "score_adjustment": int(total),
        "status": _sign_label(total, cfg),
        "veto": veto,
        "reasons": reasons,
        "breakdown": breakdown,
    }
