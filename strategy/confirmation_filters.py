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
from config.signal_policy import get_confirmation_filter_config, get_trade_runtime_thresholds
from config.symbol_microstructure import DEFAULT_MICROSTRUCTURE_CONFIG, get_microstructure_config
from utils.numerics import safe_float as _safe_float  # noqa: F401


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        text = str(value).strip().lower()
    except Exception:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _continuous_mode(cfg) -> bool:
    mode = str(cfg.get("confirmation_scoring_mode", "continuous") or "continuous").strip().lower()
    return mode == "continuous"


def _linear_interp(value, lo, hi, out_lo, out_hi):
    span = max(float(hi) - float(lo), 1e-9)
    t = (float(value) - float(lo)) / span
    t = max(0.0, min(1.0, t))
    return float(out_lo) + (float(out_hi) - float(out_lo)) * t


def _bounded_direction_change_penalty(cfg):
    penalty = _safe_float(cfg.get("direction_change_penalty"), 0.0)
    if penalty is None:
        return 0.0
    return max(0.0, min(6.0, float(penalty)))


def _bounded_decay_factor(cfg):
    factor = _safe_float(cfg.get("direction_change_decay_factor"), 0.5)
    if factor is None:
        return 0.5
    return max(0.0, min(1.0, float(factor)))


def _bounded_decay_steps(cfg):
    try:
        steps = int(float(cfg.get("direction_change_decay_steps") or 0))
    except (TypeError, ValueError):
        return 0
    return max(0, min(20, steps))


def _bounded_reversal_veto_steps(cfg):
    try:
        steps = int(float(cfg.get("reversal_veto_steps") or 0))
    except (TypeError, ValueError):
        return 0
    return max(0, min(20, steps))


def _signed_alignment_score(signed_edge, support_score, conflict_score, scale):
    """Map signed directional edge into a smooth support/conflict score."""
    if signed_edge is None:
        return 0.0
    scale = max(float(scale), 1e-9)
    normalized = max(-1.0, min(1.0, float(signed_edge) / scale))
    if normalized >= 0:
        return normalized * float(support_score)
    return (-normalized) * float(conflict_score)


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
    previous_direction=None,
    reversal_age=None,
    day_open=None,
    prev_close=None,
    intraday_range_pct=None,
    final_flow_signal=None,
    hedging_bias=None,
    gamma_event=None,
    hybrid_move_probability=None,
    spot_vs_flip=None,
    gamma_regime=None,
    volume_pcr_atm=None,
    volume_pcr_regime=None,
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
        "open_alignment_score": 0.0,
        "prev_close_alignment_score": 0.0,
        "range_expansion_score": 0.0,
        "flow_confirmation_score": 0.0,
        "hedging_confirmation_score": 0.0,
        "gamma_event_confirmation_score": 0.0,
        "move_probability_confirmation_score": 0.0,
        "flip_alignment_score": 0.0,
        "flip_zone_gamma_score": 0.0,
        "direction_change_penalty": 0.0,
        "direction_change_decay_penalty": 0.0,
        "pcr_alignment": 0.0,
    }

    reasons = []
    veto = False

    spot = _safe_float(spot, None)
    open_px = _safe_float(day_open, None)
    prev_close_px = _safe_float(prev_close, None)
    range_pct = _safe_float(intraday_range_pct, None)
    move_prob = _safe_float(hybrid_move_probability, None)
    micro_cfg = get_microstructure_config(symbol)
    if micro_cfg is None or not isinstance(micro_cfg, dict):
        micro_cfg = {}
    
    cfg = get_confirmation_filter_config()
    if cfg is None or not isinstance(cfg, dict):
        cfg = {}

    rt = get_trade_runtime_thresholds()
    use_pcr = str(rt.get("use_pcr_in_confirmation", 1)).strip().lower() not in {"0", "false", "no", "off"}
    
    continuous_mode = _continuous_mode(cfg)
    cont_open = continuous_mode and _as_bool(cfg.get("continuous_open_alignment", 1), default=True)
    cont_prev_close = continuous_mode and _as_bool(cfg.get("continuous_prev_close_alignment", 1), default=True)
    cont_range = continuous_mode and _as_bool(cfg.get("continuous_range_expansion", 1), default=True)
    cont_move_prob = continuous_mode and _as_bool(cfg.get("continuous_move_probability", 1), default=True)

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
        if cont_open:
            edge = ((spot - open_px) / max(abs(open_px), 1e-6)) if bullish else ((open_px - spot) / max(abs(open_px), 1e-6))
            score = _signed_alignment_score(
                edge,
                support_score=cfg["open_alignment_support"],
                conflict_score=abs(float(cfg["open_alignment_conflict"])),
                scale=0.004,
            )
            breakdown["open_alignment_score"] = score
            if score >= 0:
                reasons.append("spot_open_alignment_support")
            else:
                reasons.append("spot_open_alignment_conflict")
        else:
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
        if cont_prev_close:
            edge = ((spot - prev_close_px) / max(abs(prev_close_px), 1e-6)) if bullish else ((prev_close_px - spot) / max(abs(prev_close_px), 1e-6))
            score = _signed_alignment_score(
                edge,
                support_score=cfg["prev_close_alignment_support"],
                conflict_score=abs(float(cfg["prev_close_alignment_conflict"])),
                scale=0.006,
            )
            breakdown["prev_close_alignment_score"] = score
            if score >= 0:
                reasons.append("prev_close_alignment_support")
            else:
                reasons.append("prev_close_alignment_conflict")
        else:
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
        strong_th = micro_cfg.get("range_expansion_strong", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_strong"])
        moderate_th = micro_cfg.get("range_expansion_moderate", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_moderate"])
        low_th = micro_cfg.get("range_expansion_low", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_low"])
        cold_th = micro_cfg.get("range_expansion_cold", DEFAULT_MICROSTRUCTURE_CONFIG["range_expansion_cold"])

        if cont_range:
            if range_pct >= strong_th:
                breakdown["range_expansion_score"] = float(cfg["range_expansion_strong_score"])
                reasons.append("strong_intraday_range_expansion")
            elif range_pct >= moderate_th:
                breakdown["range_expansion_score"] = _linear_interp(
                    range_pct,
                    moderate_th,
                    strong_th,
                    cfg["range_expansion_moderate_score"],
                    cfg["range_expansion_strong_score"],
                )
                reasons.append("moderate_intraday_range_expansion")
            elif range_pct >= low_th:
                breakdown["range_expansion_score"] = _linear_interp(
                    range_pct,
                    low_th,
                    moderate_th,
                    cfg["range_expansion_low_score"],
                    cfg["range_expansion_moderate_score"],
                )
                reasons.append("early_intraday_range_expansion")
            elif range_pct >= cold_th:
                breakdown["range_expansion_score"] = _linear_interp(
                    range_pct,
                    cold_th,
                    low_th,
                    cfg["range_expansion_cold_score"],
                    cfg["range_expansion_low_score"],
                )
                reasons.append("cold_to_early_range_transition")
            else:
                breakdown["range_expansion_score"] = float(cfg["range_expansion_cold_score"])
                reasons.append("very_low_intraday_range_expansion")
        else:
            if range_pct >= strong_th:
                breakdown["range_expansion_score"] = cfg["range_expansion_strong_score"]
                reasons.append("strong_intraday_range_expansion")
            elif range_pct >= moderate_th:
                breakdown["range_expansion_score"] = cfg["range_expansion_moderate_score"]
                reasons.append("moderate_intraday_range_expansion")
            elif range_pct >= low_th:
                breakdown["range_expansion_score"] = cfg["range_expansion_low_score"]
                reasons.append("early_intraday_range_expansion")
            elif range_pct < cold_th:
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
        high_th = float(cfg["move_probability_high_threshold"])
        moderate_th = float(cfg["move_probability_moderate_threshold"])
        low_th = float(cfg["move_probability_low_support_threshold"])
        conflict_th = float(cfg["move_probability_conflict_threshold"])

        if cont_move_prob:
            if move_prob >= high_th:
                breakdown["move_probability_confirmation_score"] = float(cfg["move_probability_high_score"])
                reasons.append("high_hybrid_move_probability")
            elif move_prob >= moderate_th:
                breakdown["move_probability_confirmation_score"] = _linear_interp(
                    move_prob,
                    moderate_th,
                    high_th,
                    cfg["move_probability_moderate_score"],
                    cfg["move_probability_high_score"],
                )
                reasons.append("moderate_hybrid_move_probability")
            elif move_prob >= low_th:
                breakdown["move_probability_confirmation_score"] = _linear_interp(
                    move_prob,
                    low_th,
                    moderate_th,
                    cfg["move_probability_low_support_score"],
                    cfg["move_probability_moderate_score"],
                )
                reasons.append("acceptable_hybrid_move_probability")
            elif move_prob >= conflict_th:
                breakdown["move_probability_confirmation_score"] = _linear_interp(
                    move_prob,
                    conflict_th,
                    low_th,
                    cfg["move_probability_conflict_score"],
                    cfg["move_probability_low_support_score"],
                )
                reasons.append("borderline_hybrid_move_probability")
            else:
                breakdown["move_probability_confirmation_score"] = float(cfg["move_probability_conflict_score"])
                reasons.append("low_hybrid_move_probability")
        else:
            if move_prob >= high_th:
                breakdown["move_probability_confirmation_score"] = cfg["move_probability_high_score"]
                reasons.append("high_hybrid_move_probability")
            elif move_prob >= moderate_th:
                breakdown["move_probability_confirmation_score"] = cfg["move_probability_moderate_score"]
                reasons.append("moderate_hybrid_move_probability")
            elif move_prob >= low_th:
                breakdown["move_probability_confirmation_score"] = cfg["move_probability_low_support_score"]
                reasons.append("acceptable_hybrid_move_probability")
            elif move_prob < conflict_th:
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

    # 9) Gamma-regime-aware flip-zone penalty
    # When spot is pinned at the flip and gamma structure is unfavorable,
    # the engine has reduced structural edge regardless of direction.
    if spot_vs_flip == "AT_FLIP":
        if gamma_regime in ("NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"):
            breakdown["flip_zone_gamma_score"] = cfg.get("flip_zone_gamma_penalty_negative", -3)
            reasons.append("at_flip_negative_gamma_structural_conflict")
        elif gamma_regime == "NEUTRAL_GAMMA":
            breakdown["flip_zone_gamma_score"] = cfg.get("flip_zone_gamma_penalty_neutral", -2)
            reasons.append("at_flip_neutral_gamma_reduced_edge")

    if use_pcr:
        pcr = _safe_float(volume_pcr_atm, None)
        pcr_put_dom = _safe_float(rt.get("volume_pcr_atm_put_dominant"), 1.20)
        pcr_call_dom = _safe_float(rt.get("volume_pcr_atm_call_dominant"), 0.80)
        if pcr is not None:
            if (bullish and pcr <= pcr_call_dom) or (bearish and pcr >= pcr_put_dom):
                breakdown["pcr_alignment"] = float(cfg.get("pcr_confirmation_support", 1.0))
                reasons.append("pcr_confirms_direction")
            elif (bullish and pcr >= pcr_put_dom) or (bearish and pcr <= pcr_call_dom):
                breakdown["pcr_alignment"] = float(cfg.get("pcr_confirmation_conflict", -1.0))
                reasons.append("pcr_conflicts_direction")
        elif isinstance(volume_pcr_regime, str):
            regime = volume_pcr_regime.strip().upper()
            if (bullish and regime in {"CALL_DOMINANT", "BULLISH"}) or (bearish and regime in {"PUT_DOMINANT", "BEARISH"}):
                breakdown["pcr_alignment"] = float(cfg.get("pcr_confirmation_support", 1.0))
                reasons.append("pcr_regime_confirms_direction")
            elif (bullish and regime in {"PUT_DOMINANT", "BEARISH"}) or (bearish and regime in {"CALL_DOMINANT", "BULLISH"}):
                breakdown["pcr_alignment"] = float(cfg.get("pcr_confirmation_conflict", -1.0))
                reasons.append("pcr_regime_conflicts_direction")

    total = sum(breakdown.values())

    previous_direction_clean = str(previous_direction or "").strip().upper()
    reversal_detected = (
        previous_direction_clean in {"CALL", "PUT"}
        and direction in {"CALL", "PUT"}
        and previous_direction_clean != direction
    )

    # Step-0: immediate reversal penalty (existing behaviour, unchanged)
    if reversal_detected:
        direction_change_penalty = _bounded_direction_change_penalty(cfg)
        if direction_change_penalty > 0:
            breakdown["direction_change_penalty"] = -direction_change_penalty
            total -= direction_change_penalty
            reasons.append("direction_change_penalty_applied")

    # Steps 1..N: post-reversal decay — apply only when the caller supplies
    # reversal_age (an integer counting how many snapshots ago the flip
    # happened).  reversal_age=0 is the flip snapshot itself; the decay
    # window starts at reversal_age=1.
    decay_steps = _bounded_decay_steps(cfg)
    if (
        not reversal_detected
        and reversal_age is not None
        and decay_steps > 0
    ):
        try:
            age = int(reversal_age)
        except (TypeError, ValueError):
            age = None
        if age is not None and 1 <= age <= decay_steps:
            base_penalty = _bounded_direction_change_penalty(cfg)
            if base_penalty > 0:
                decay_factor = _bounded_decay_factor(cfg)
                effective = base_penalty * (decay_factor ** age)
                if effective > 0.0:
                    breakdown["direction_change_decay_penalty"] = -round(effective, 4)
                    total -= effective
                    reasons.append("direction_change_decay_applied")

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

    # Reversal grace period veto: force MIXED status for N snapshots
    # starting from the reversal itself (reversal_age=0), unless breakout
    # evidence is strong enough to justify a faster directional handoff.
    veto_steps = _bounded_reversal_veto_steps(cfg)
    if veto_steps > 0 and reversal_age is not None:
        try:
            age = int(reversal_age)
        except (TypeError, ValueError):
            age = None
        if age is not None and 0 <= age < veto_steps:
            computed_status = _sign_label(total, cfg)
            if computed_status in {"STRONG_CONFIRMATION", "CONFIRMED"}:
                breakout_override_enabled = str(
                    rt.get("reversal_breakout_override_enabled", 0)
                ).strip().lower() in {"1", "true", "yes", "on"}

                override_prob_floor = _safe_float(rt.get("reversal_breakout_override_move_probability_floor"), 0.62)
                override_range_floor = _safe_float(rt.get("reversal_breakout_override_range_pct_floor"), 0.35)
                require_directional_flow = str(rt.get("reversal_breakout_override_requires_flow", 1)).strip().lower() not in {"0", "false", "no", "off"}
                require_hedging_alignment = str(rt.get("reversal_breakout_override_requires_hedging", 0)).strip().lower() not in {"0", "false", "no", "off"}
                try:
                    min_signals = int(float(rt.get("reversal_breakout_override_min_signals", 2)))
                except (TypeError, ValueError):
                    min_signals = 2
                min_signals = max(1, min(4, min_signals))

                directional_flow_ok = final_flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"}
                move_prob_ok = move_prob is not None and move_prob >= override_prob_floor
                range_ok = range_pct is not None and range_pct >= override_range_floor
                hedging_ok = (
                    (bullish and hedging_bias == "UPSIDE_ACCELERATION")
                    or (bearish and hedging_bias == "DOWNSIDE_ACCELERATION")
                )

                evidence_count = int(move_prob_ok) + int(range_ok) + int(directional_flow_ok) + int(hedging_ok)
                override_allowed = evidence_count >= min_signals
                if require_directional_flow:
                    override_allowed = override_allowed and directional_flow_ok
                if require_hedging_alignment:
                    override_allowed = override_allowed and hedging_ok

                # The grace-period veto is the default safety behavior.
                # Breakout bypass must be explicitly enabled by runtime policy.
                if not breakout_override_enabled:
                    override_allowed = False

                if not override_allowed:
                    reasons.append("reversal_grace_period_active")
                    return {
                        "score_adjustment": round(float(total), 2),
                        "status": "MIXED",
                        "veto": veto,
                        "reasons": reasons,
                        "breakdown": breakdown,
                    }
                reasons.append("reversal_grace_bypassed_on_breakout")
                reasons.append(f"reversal_grace_override_evidence:{evidence_count}")

    return {
        "score_adjustment": round(float(total), 2),
        "status": _sign_label(total, cfg),
        "veto": veto,
        "reasons": reasons,
        "breakdown": breakdown,
    }
