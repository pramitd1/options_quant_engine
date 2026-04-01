"""
Module: signal_state.py

Purpose:
    Provide signal state helpers used during market-state, probability, or signal assembly.

Role in the System:
    Part of the signal engine that turns analytics, probability estimates, and overlays into final trade decisions.

Key Outputs:
    Trade decisions, intermediate state bundles, and signal diagnostics.

Downstream Usage:
    Consumed by the live runtime loop, backtests, shadow mode, and signal-evaluation logging.
"""
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
    """
    Purpose:
        Compute data quality from the supplied inputs.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        spot_validation (Any): Input associated with spot validation.
        option_chain_validation (Any): Input associated with option chain validation.
        analytics_state (Any): Structured state payload for analytics.
        probability_state (Any): Structured state payload for probability.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    if not isinstance(provider_health, dict):
        provider_health = {}
    provider_summary = str(provider_health.get("summary_status") or "").upper().strip()
    provider_blocking_status = str(provider_health.get("trade_blocking_status") or "").upper().strip()
    if provider_summary == "WEAK":
        score -= cfg.provider_health_weak_penalty
        reasons.append("weak_provider_health")
    elif provider_summary == "CAUTION":
        score -= cfg.provider_health_caution_penalty
        reasons.append("provider_health_caution")

    # If provider health is explicitly blocking trade execution, add an
    # explicit reason and cap optimistic statuses during display/risk scoring.
    if provider_blocking_status == "BLOCK":
        reasons.append("provider_health_trade_block")

    critical_analytics = {
        "flip": analytics_state.get("flip"),
        "gamma_regime": analytics_state.get("gamma_regime"),
        "final_flow_signal": analytics_state.get("final_flow_signal"),
        "dealer_pos": analytics_state.get("dealer_pos"),
        "hedging_bias": analytics_state.get("hedging_bias"),
        "vol_regime": analytics_state.get("vol_regime"),
    }

    # These fields are the minimum structural features the strategy expects to
    # see before it trusts downstream probability and strength calculations.
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

    if provider_blocking_status == "BLOCK" and status in {"STRONG", "GOOD"}:
        status = "CAUTION"

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
    """
    Purpose:
        Classify spot vs flip into the appropriate regime or label.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        spot (Any): Input associated with spot.
        flip (Any): Input associated with flip.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return classify_spot_vs_flip_for_symbol(None, spot, flip)


def classify_spot_vs_flip_for_symbol(symbol, spot, flip):
    """
    Purpose:
        Classify spot vs flip for symbol into the appropriate regime or label.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (Any): Underlying symbol or index identifier.
        spot (Any): Input associated with spot.
        flip (Any): Input associated with flip.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if flip is None:
        return "UNKNOWN"

    flip_buffer = _safe_float(get_microstructure_config(symbol).get("flip_buffer_points"), 25.0)
    if abs(spot - flip) <= flip_buffer:
        return "AT_FLIP"
    if spot > flip:
        return "ABOVE_FLIP"
    return "BELOW_FLIP"


def classify_signal_quality(trade_strength):
    """
    Purpose:
        Classify signal quality into the appropriate regime or label.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade_strength (Any): Input associated with trade strength.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Classify signal regime into the appropriate regime or label.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        adjusted_trade_strength (Any): Input associated with adjusted trade strength.
        final_flow_signal (Any): Input associated with final flow signal.
        gamma_regime (Any): Input associated with gamma regime.
        confirmation_status (Any): Status label for confirmation.
        event_lockdown_flag (Any): Boolean flag indicating whether scheduled-event rules require a hard lockdown.
        data_quality_status (Any): Status label for data quality.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Classify execution regime into the appropriate regime or label.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade_status (Any): Status label for trade.
        signal_regime (Any): Input associated with signal regime.
        data_quality_score (Any): Score value for data quality.
        macro_position_size_multiplier (Any): Input associated with macro position size multiplier.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Normalize flow signal into the repository-standard form.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        flow_signal_value (Any): Input associated with flow signal value.
        smart_money_signal_value (Any): Input associated with smart money signal value.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    volatility_shock_score=None,
    oi_velocity_score=None,
    rr_value=None,
    rr_momentum=None,
    volume_pcr_atm=None,
    gamma_flip_drift=None,
):
    """
    Purpose:
        Process decide direction for downstream use.
    
    Context:
        Public function within the signal-engine layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        final_flow_signal (Any): Input associated with final flow signal.
        dealer_pos (Any): Input associated with dealer pos.
        vol_regime (Any): Input associated with vol regime.
        spot_vs_flip (Any): Input associated with spot vs flip.
        gamma_regime (Any): Input associated with gamma regime.
        hedging_bias (Any): Input associated with hedging bias.
        gamma_event (Any): Input associated with gamma event.
        vanna_regime (Any): Input associated with vanna regime.
        charm_regime (Any): Input associated with charm regime.
        backtest_mode (Any): Input associated with backtest mode.
        volatility_shock_score (Any): Macro-derived volatility shock indicator for regime-aware weighting.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
        Volatility shock score allows dynamic regime weighting: high vol favors puts (bearish direction).
    """
    direction_weights = get_direction_vote_weights()
    direction_thresholds = get_direction_thresholds()
    runtime_thresholds = get_trade_runtime_thresholds()
    bullish_votes = []
    bearish_votes = []

    def _flag_enabled(name, default=True):
        value = runtime_thresholds.get(name, 1 if default else 0)
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def add_vote(side, reason):
        """
        Purpose:
            Append one directional vote to the running bullish or bearish vote ledger.

        Context:
            Internal helper inside direction assembly. Each market mechanism
            contributes a named vote so the final signal remains explainable in
            live diagnostics, replay output, and research captures.

        Inputs:
            side (Any): Direction bucket that should receive the vote, typically `BULLISH` or `BEARISH`.
            reason (Any): Named evidence source that explains why the vote was added.

        Returns:
            None: The helper mutates the surrounding vote collections in place.

        Notes:
            Vote weights come from policy configuration so the engine can rebalance
            how much each heuristic contributes without rewriting this assembly logic.
        """
        weight = float(direction_weights.get(reason, 1.0))
        entry = (reason, round(weight, 2))
        if side == "BULLISH":
            bullish_votes.append(entry)
        elif side == "BEARISH":
            bearish_votes.append(entry)

    # Votes are intentionally additive so operators can inspect which market
    # mechanisms aligned behind the final direction.
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

    if vanna_regime == "POSITIVE_VANNA" and spot_vs_flip in ("ABOVE_FLIP", "AT_FLIP"):
        add_vote("BULLISH", "VANNA")
    elif vanna_regime == "NEGATIVE_VANNA" and spot_vs_flip in ("BELOW_FLIP", "AT_FLIP"):
        add_vote("BEARISH", "VANNA")

    if charm_regime == "POSITIVE_CHARM" and spot_vs_flip in ("ABOVE_FLIP", "AT_FLIP"):
        add_vote("BULLISH", "CHARM")
    elif charm_regime == "NEGATIVE_CHARM" and spot_vs_flip in ("BELOW_FLIP", "AT_FLIP"):
        add_vote("BEARISH", "CHARM")

    if _flag_enabled("use_oi_velocity_in_direction", default=True):
        vel = _safe_float(oi_velocity_score, None)
        vel_on = abs(_safe_float(runtime_thresholds.get("oi_velocity_vote_on"), 0.18) or 0.18)
        if vel is not None and abs(vel) >= vel_on:
            if vel > 0:
                add_vote("BULLISH", "OI_VELOCITY")
            else:
                add_vote("BEARISH", "OI_VELOCITY")

    if _flag_enabled("use_rr_in_direction", default=True):
        rr = _safe_float(rr_value, None)
        rr_put_dom = _safe_float(runtime_thresholds.get("rr_skew_put_dominant"), 0.75)
        rr_call_dom = _safe_float(runtime_thresholds.get("rr_skew_call_dominant"), -0.75)
        if rr is not None:
            if rr >= rr_put_dom:
                add_vote("BEARISH", "RR_SKEW")
            elif rr <= rr_call_dom:
                add_vote("BULLISH", "RR_SKEW")

        rr_m = str(rr_momentum or "").upper().strip()
        if rr_m == "RISING_PUT_SKEW":
            add_vote("BEARISH", "RR_MOMENTUM")
        elif rr_m == "FALLING_PUT_SKEW":
            add_vote("BULLISH", "RR_MOMENTUM")

    pcr = _safe_float(volume_pcr_atm, None)
    pcr_put_dom = _safe_float(runtime_thresholds.get("volume_pcr_atm_put_dominant"), 1.20)
    pcr_call_dom = _safe_float(runtime_thresholds.get("volume_pcr_atm_call_dominant"), 0.80)
    if pcr is not None:
        if pcr >= pcr_put_dom:
            add_vote("BEARISH", "PCR_ATM")
        elif pcr <= pcr_call_dom:
            add_vote("BULLISH", "PCR_ATM")

    drift = None
    if isinstance(gamma_flip_drift, dict):
        drift = _safe_float(gamma_flip_drift.get("drift"), None)
    drift_on = abs(_safe_float(runtime_thresholds.get("gamma_flip_drift_pts_vote_on"), 80.0) or 80.0)
    if drift is not None and abs(drift) >= drift_on:
        if drift > 0:
            add_vote("BULLISH", "FLIP_DRIFT")
        else:
            add_vote("BEARISH", "FLIP_DRIFT")

    bullish_score = round(sum(weight for _, weight in bullish_votes), 2)
    bearish_score = round(sum(weight for _, weight in bearish_votes), 2)
    
    # Apply regime-aware weighting: in elevated volatility, boost bearish (PUT) advantage
    # to prevent extreme call bias in risk-off environments (fix for 6.55:1 call/put ratio)
    volatility_shock = float(volatility_shock_score or 0.0)
    if volatility_shock > 0.3:
        # Regime weight ranges from 1.0 (low vol) to 1.4 (extreme shock)
        regime_multiplier = 1.0 + (min(volatility_shock, 1.0) * 0.4)
        bearish_score = round(bearish_score * regime_multiplier, 2)
    
    score_margin = round(abs(bullish_score - bearish_score), 2)

    def build_source(votes):
        """
        Purpose:
            Collapse the recorded vote reasons into one human-readable source string.

        Context:
            Internal helper used after voting is complete. It preserves the
            evidence trail behind the chosen direction so downstream consumers
            can see which heuristics aligned.

        Inputs:
            votes (Any): Collection of directional votes accumulated during signal assembly.

        Returns:
            str: `+`-delimited list of vote reasons in insertion order.

        Notes:
            Keeping the explanation string deterministic makes signal-review logs
            easier to diff across live, replay, and shadow-mode runs.
        """
        return "+".join(reason for reason, _ in votes)

    # Dual scoring framework: Keep both bull/bear probabilities alive.
    # This allows downstream trade_strength and confidence layers to understand
    # the degree of disagreement, not just the winning direction.
    # Normalized to 0-1 range using min/max scaling for comparability.
    total_score = max(bullish_score + bearish_score, 1e-6)
    bull_probability = round(bullish_score / total_score, 4)
    bear_probability = round(bearish_score / total_score, 4)

    if (
        bullish_score >= direction_thresholds["min_score"]
        and bullish_score > bearish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "CALL", build_source(bullish_votes), bull_probability, bear_probability

    if (
        bearish_score >= direction_thresholds["min_score"]
        and bearish_score > bullish_score
        and score_margin >= direction_thresholds["min_margin"]
    ):
        return "PUT", build_source(bearish_votes), bull_probability, bear_probability

    return None, None, 0.5, 0.5  # Neutral case: equal probabilities


def _compute_signal_state(
    *,
    spot,
    symbol,
    previous_direction,
    reversal_age=None,
    day_open,
    prev_close,
    intraday_range_pct,
    backtest_mode,
    market_state,
    probability_state,
):
    """
    Purpose:
        Compute signal state from the supplied inputs.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        spot (Any): Input associated with spot.
        symbol (Any): Underlying symbol or index identifier.
        day_open (Any): Input associated with day open.
        prev_close (Any): Input associated with prev close.
        intraday_range_pct (Any): Input associated with intraday range percentage.
        backtest_mode (Any): Input associated with backtest mode.
        market_state (Any): Structured state payload for market.
        probability_state (Any): Structured state payload for probability.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    direction, direction_source, bull_probability, bear_probability = decide_direction(
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
        volatility_shock_score=market_state.get("volatility_shock_score", 0.0),
        oi_velocity_score=market_state.get("oi_velocity_score"),
        rr_value=market_state.get("rr_value"),
        rr_momentum=market_state.get("rr_momentum"),
        volume_pcr_atm=market_state.get("volume_pcr_atm"),
        gamma_flip_drift=market_state.get("gamma_flip_drift"),
    )

    if direction is None:
        return {
            "direction": None,
            "direction_source": None,
            "direction_vote_count": 0,
            "bull_probability": 0.5,
            "bear_probability": 0.5,
            "trade_strength": 0,
            "scoring_breakdown": empty_scoring_breakdown(),
            "confirmation": empty_confirmation_state(),
        }

    # Trade strength measures how much structural alignment exists behind the
    # chosen direction before overlays are allowed to resize or veto it.
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
        flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
        proximity_buffer=get_microstructure_config(symbol).get("wall_proximity_points", 50.0),
        oi_velocity_score=market_state.get("oi_velocity_score"),
        rr_value=market_state.get("rr_value"),
        rr_momentum=market_state.get("rr_momentum"),
        volume_pcr_atm=market_state.get("volume_pcr_atm"),
        gamma_flip_drift=market_state.get("gamma_flip_drift"),
        max_pain_dist=market_state.get("max_pain_dist"),
        max_pain_zone=market_state.get("max_pain_zone"),
        days_to_expiry=market_state.get("days_to_expiry"),
    )

    confirmation = compute_confirmation_filters(
        direction=direction,
        spot=spot,
        symbol=symbol,
        previous_direction=previous_direction,
        reversal_age=reversal_age,
        day_open=day_open,
        prev_close=prev_close,
        intraday_range_pct=intraday_range_pct,
        final_flow_signal=market_state["final_flow_signal"],
        hedging_bias=market_state["hedging_bias"],
        gamma_event=market_state["gamma_event"],
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_regime=market_state["gamma_regime"],
        volume_pcr_atm=market_state.get("volume_pcr_atm"),
        volume_pcr_regime=market_state.get("volume_pcr_regime"),
    )

    # Direction vote count measures the breadth of conviction behind the
    # chosen direction.  A thin base (e.g. 2 sources) indicates that fewer
    # independent market mechanisms are aligned, which downstream consumers
    # can use to temper sizing or urgency.
    direction_vote_count = len(direction_source.split("+")) if direction_source else 0

    return {
        "direction": direction,
        "direction_source": direction_source,
        "direction_vote_count": direction_vote_count,
        "bull_probability": bull_probability,
        "bear_probability": bear_probability,
        "trade_strength": trade_strength,
        "scoring_breakdown": scoring_breakdown,
        "confirmation": confirmation,
    }
