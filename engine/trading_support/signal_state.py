from __future__ import annotations

from config.signal_policy import (
    get_data_quality_policy_config,
    get_direction_thresholds,
    get_direction_vote_weights,
    get_execution_regime_policy_config,
    get_trade_runtime_thresholds,
)
from config.symbol_microstructure import get_microstructure_config
from engine.runtime_metadata import empty_confirmation_state, empty_scoring_breakdown
from strategy.confirmation_filters import compute_confirmation_filters
from strategy.trade_strength import compute_trade_strength

from .common import _clip, _normalize_validation_dict, _safe_float


def _compute_data_quality(*, spot_validation, option_chain_validation, analytics_state, probability_state):
    cfg = get_data_quality_policy_config()
    spot_validation = _normalize_validation_dict(spot_validation)
    option_chain_validation = _normalize_validation_dict(option_chain_validation)
    analytics_state = analytics_state if isinstance(analytics_state, dict) else {}
    probability_state = probability_state if isinstance(probability_state, dict) else {}

    score = 100
    reasons = []
    analytics_missing = []

    if not spot_validation.get("is_valid", True):
        score -= cfg.invalid_spot_penalty
        reasons.append("invalid_spot_snapshot")
    elif spot_validation.get("is_stale"):
        score -= cfg.stale_spot_penalty
        reasons.append("stale_spot_snapshot")

    if not option_chain_validation.get("is_valid", True):
        score -= cfg.invalid_option_chain_penalty
        reasons.append("invalid_option_chain")
    elif option_chain_validation.get("is_stale"):
        score -= cfg.stale_option_chain_penalty
        reasons.append("stale_option_chain")

    provider_health = option_chain_validation.get("provider_health") or {}
    provider_summary = provider_health.get("summary_status")
    if provider_summary == "WEAK":
        score -= cfg.provider_health_weak_penalty
        reasons.append("weak_provider_health")
    elif provider_summary == "CAUTION":
        score -= cfg.provider_health_caution_penalty
        reasons.append("provider_health_caution")

    critical_analytics = {
        "flip": analytics_state.get("flip"),
        "gamma_regime": analytics_state.get("gamma_regime"),
        "final_flow_signal": analytics_state.get("final_flow_signal"),
        "dealer_pos": analytics_state.get("dealer_pos"),
        "hedging_bias": analytics_state.get("hedging_bias"),
        "vol_regime": analytics_state.get("vol_regime"),
    }

    for name, value in critical_analytics.items():
        if value in (None, "", "UNKNOWN"):
            analytics_missing.append(name)

    if analytics_missing:
        score -= min(
            len(analytics_missing) * cfg.missing_analytics_penalty_per_field,
            cfg.missing_analytics_penalty_cap,
        )
        reasons.append(f"missing_critical_analytics:{','.join(sorted(analytics_missing))}")

    if probability_state.get("rule_move_probability") is None and probability_state.get("ml_move_probability") is None:
        score -= cfg.missing_all_probabilities_penalty
        reasons.append("missing_all_move_probabilities")
    elif probability_state.get("hybrid_move_probability") is None:
        score -= cfg.missing_hybrid_probability_penalty
        reasons.append("missing_hybrid_move_probability")

    score = int(_clip(score, 0, 100))

    if score >= cfg.status_strong_threshold:
        status = "STRONG"
    elif score >= cfg.status_good_threshold:
        status = "GOOD"
    elif score >= cfg.status_caution_threshold:
        status = "CAUTION"
    else:
        status = "WEAK"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "analytics_quality": {
            "missing_critical": analytics_missing,
            "critical_missing_count": len(analytics_missing),
        },
        "fatal": (not spot_validation.get("is_valid", True)) or (not option_chain_validation.get("is_valid", True)),
    }


def classify_spot_vs_flip(spot, flip):
    return classify_spot_vs_flip_for_symbol(None, spot, flip)


def classify_spot_vs_flip_for_symbol(symbol, spot, flip):
    if flip is None:
        return "UNKNOWN"

    flip_buffer = _safe_float(get_microstructure_config(symbol).get("flip_buffer_points"), 25.0)
    if abs(spot - flip) <= flip_buffer:
        return "AT_FLIP"
    if spot > flip:
        return "ABOVE_FLIP"
    return "BELOW_FLIP"


def classify_signal_quality(trade_strength):
    thresholds = get_trade_runtime_thresholds()
    if trade_strength >= thresholds["strong_signal_threshold"]:
        return "STRONG"
    if trade_strength >= thresholds["medium_signal_threshold"]:
        return "MEDIUM"
    if trade_strength >= thresholds["weak_signal_threshold"]:
        return "WEAK"
    return "VERY_WEAK"


def classify_signal_regime(
    *,
    direction,
    adjusted_trade_strength,
    final_flow_signal,
    gamma_regime,
    confirmation_status,
    event_lockdown_flag,
    data_quality_status,
):
    thresholds = get_trade_runtime_thresholds()
    if event_lockdown_flag:
        return "LOCKDOWN"
    if direction is None:
        return "NEUTRAL"

    directional_flow = final_flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"}
    unstable_gamma = gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}

    if (
        adjusted_trade_strength >= thresholds["expansion_bias_threshold"]
        and directional_flow
        and unstable_gamma
        and confirmation_status in {"STRONG_CONFIRMATION", "CONFIRMED"}
    ):
        return "EXPANSION_BIAS"
    if adjusted_trade_strength >= thresholds["directional_bias_threshold"] and directional_flow:
        return "DIRECTIONAL_BIAS"
    if data_quality_status in {"CAUTION", "WEAK"} or confirmation_status == "CONFLICT":
        return "CONFLICTED"
    return "BALANCED"


def classify_execution_regime(*, trade_status, signal_regime, data_quality_score, macro_position_size_multiplier):
    cfg = get_execution_regime_policy_config()
    if trade_status in {"DATA_INVALID", "EVENT_LOCKDOWN", "NO_TRADE", "BUDGET_FAIL"}:
        return "BLOCKED"
    if trade_status == "TRADE" and macro_position_size_multiplier < cfg.reduced_size_multiplier_threshold:
        return "RISK_REDUCED"
    if trade_status == "TRADE":
        return "ACTIVE"
    if signal_regime == "CONFLICTED" or data_quality_score < cfg.observe_data_quality_threshold:
        return "OBSERVE"
    return "SETUP"


def normalize_flow_signal(flow_signal_value, smart_money_signal_value):
    bullish_votes = 0
    bearish_votes = 0

    if flow_signal_value == "BULLISH_FLOW":
        bullish_votes += 1
    elif flow_signal_value == "BEARISH_FLOW":
        bearish_votes += 1

    if smart_money_signal_value == "BULLISH_FLOW":
        bullish_votes += 1
    elif smart_money_signal_value == "BEARISH_FLOW":
        bearish_votes += 1

    if bullish_votes > bearish_votes:
        return "BULLISH_FLOW"
    if bearish_votes > bullish_votes:
        return "BEARISH_FLOW"
    return "NEUTRAL_FLOW"


def decide_direction(
    final_flow_signal,
    dealer_pos,
    vol_regime,
    spot_vs_flip,
    gamma_regime,
    hedging_bias,
    gamma_event,
    vanna_regime=None,
    charm_regime=None,
    backtest_mode=False,
):
    direction_weights = get_direction_vote_weights()
    direction_thresholds = get_direction_thresholds()
    bullish_votes = []
    bearish_votes = []

    def add_vote(side, reason):
        weight = float(direction_weights.get(reason, 1.0))
        entry = (reason, round(weight, 2))
        if side == "BULLISH":
            bullish_votes.append(entry)
        elif side == "BEARISH":
            bearish_votes.append(entry)

    if final_flow_signal == "BULLISH_FLOW":
        add_vote("BULLISH", "FLOW")
    elif final_flow_signal == "BEARISH_FLOW":
        add_vote("BEARISH", "FLOW")

    if gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}:
        if hedging_bias == "UPSIDE_ACCELERATION":
            add_vote("BULLISH", "HEDGING_BIAS")
        elif hedging_bias == "DOWNSIDE_ACCELERATION":
            add_vote("BEARISH", "HEDGING_BIAS")

    if gamma_event == "GAMMA_SQUEEZE":
        if spot_vs_flip == "ABOVE_FLIP":
            add_vote("BULLISH", "GAMMA_SQUEEZE")
        elif spot_vs_flip == "BELOW_FLIP":
            add_vote("BEARISH", "GAMMA_SQUEEZE")

    if spot_vs_flip == "ABOVE_FLIP":
        if dealer_pos in {"Long Gamma", "Short Gamma"}:
            add_vote("BULLISH", "GAMMA_FLIP")
    elif spot_vs_flip == "BELOW_FLIP" and dealer_pos == "Short Gamma":
        add_vote("BEARISH", "GAMMA_FLIP")

    if dealer_pos == "Short Gamma" and vol_regime == "VOL_EXPANSION":
        if spot_vs_flip == "ABOVE_FLIP":
            add_vote("BULLISH", "DEALER_VOL")
        elif spot_vs_flip == "BELOW_FLIP":
            add_vote("BEARISH", "DEALER_VOL")

    if vanna_regime == "POSITIVE_VANNA" and spot_vs_flip == "ABOVE_FLIP":
        add_vote("BULLISH", "VANNA")
    elif vanna_regime == "NEGATIVE_VANNA" and spot_vs_flip == "BELOW_FLIP":
        add_vote("BEARISH", "VANNA")

    if charm_regime == "POSITIVE_CHARM" and spot_vs_flip == "ABOVE_FLIP":
        add_vote("BULLISH", "CHARM")
    elif charm_regime == "NEGATIVE_CHARM" and spot_vs_flip == "BELOW_FLIP":
        add_vote("BEARISH", "CHARM")

    if backtest_mode and not bullish_votes and not bearish_votes:
        if spot_vs_flip == "ABOVE_FLIP" and dealer_pos == "Long Gamma":
            add_vote("BULLISH", "BACKTEST_FALLBACK")
        elif spot_vs_flip == "BELOW_FLIP" and dealer_pos == "Short Gamma":
            add_vote("BEARISH", "BACKTEST_FALLBACK")

    bullish_score = round(sum(weight for _, weight in bullish_votes), 2)
    bearish_score = round(sum(weight for _, weight in bearish_votes), 2)
    score_margin = round(abs(bullish_score - bearish_score), 2)

    def build_source(votes):
        return "+".join(reason for reason, _ in votes)

    if (
        bullish_score >= direction_thresholds["min_score"]
        and bullish_score > bearish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "CALL", build_source(bullish_votes)

    if (
        bearish_score >= direction_thresholds["min_score"]
        and bearish_score > bullish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "PUT", build_source(bearish_votes)

    return None, None


def _compute_signal_state(
    *,
    spot,
    symbol,
    day_open,
    prev_close,
    intraday_range_pct,
    backtest_mode,
    market_state,
    probability_state,
):
    direction, direction_source = decide_direction(
        final_flow_signal=market_state["final_flow_signal"],
        dealer_pos=market_state["dealer_pos"],
        vol_regime=market_state["vol_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_regime=market_state["gamma_regime"],
        hedging_bias=market_state["hedging_bias"],
        gamma_event=market_state["gamma_event"],
        vanna_regime=market_state["greek_exposures"].get("vanna_regime"),
        charm_regime=market_state["greek_exposures"].get("charm_regime"),
        backtest_mode=backtest_mode,
    )

    if direction is None:
        return {
            "direction": None,
            "direction_source": None,
            "trade_strength": 0,
            "scoring_breakdown": empty_scoring_breakdown(),
            "confirmation": empty_confirmation_state(),
        }

    trade_strength, scoring_breakdown = compute_trade_strength(
        direction=direction,
        flow_signal_value=market_state["flow_signal_value"],
        smart_money_signal_value=market_state["smart_money_signal_value"],
        gamma_event=market_state["gamma_event"],
        dealer_pos=market_state["dealer_pos"],
        vol_regime=market_state["vol_regime"],
        void_signal=market_state["void_signal"],
        vacuum_state=market_state["vacuum_state"],
        spot_vs_flip=market_state["spot_vs_flip"],
        hedging_bias=market_state["hedging_bias"],
        gamma_regime=market_state["gamma_regime"],
        intraday_gamma_state=market_state["intraday_gamma_state"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        spot=spot,
        next_support=market_state["dealer_liquidity_map"].get("next_support"),
        next_resistance=market_state["dealer_liquidity_map"].get("next_resistance"),
        squeeze_zone=market_state["dealer_liquidity_map"].get("gamma_squeeze_zone"),
        large_move_probability=probability_state["hybrid_move_probability"],
        ml_move_probability=probability_state["ml_move_probability"],
        proximity_buffer=get_microstructure_config(symbol).get("wall_proximity_points", 50.0),
    )

    confirmation = compute_confirmation_filters(
        direction=direction,
        spot=spot,
        symbol=symbol,
        day_open=day_open,
        prev_close=prev_close,
        intraday_range_pct=intraday_range_pct,
        final_flow_signal=market_state["final_flow_signal"],
        hedging_bias=market_state["hedging_bias"],
        gamma_event=market_state["gamma_event"],
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        spot_vs_flip=market_state["spot_vs_flip"],
    )

    return {
        "direction": direction,
        "direction_source": direction_source,
        "trade_strength": trade_strength,
        "scoring_breakdown": scoring_breakdown,
        "confirmation": confirmation,
    }
