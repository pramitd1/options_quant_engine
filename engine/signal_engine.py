"""
Signal assembly module.

This is the canonical layer where analytics features, strategy logic, macro
adjustments, and risk overlays are combined into a final trade decision.
"""

from __future__ import annotations

from config.settings import (
    BACKTEST_MIN_TRADE_STRENGTH,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
    STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT,
)
from config.event_window_policy import get_event_window_policy_config
from config.signal_policy import get_trade_runtime_thresholds
from engine.trading_support import (
    _clip,
    _collect_market_state,
    _compute_data_quality,
    _compute_probability_state,
    _compute_signal_state,
    _safe_float,
    _to_python_number,
    classify_execution_regime,
    classify_signal_quality,
    classify_signal_regime,
    derive_dealer_pressure_trade_modifiers,
    derive_gamma_vol_trade_modifiers,
    derive_global_risk_trade_modifiers,
    derive_option_efficiency_trade_modifiers,
    normalize_option_chain,
)
from macro.engine_adjustments import compute_macro_news_adjustments
from risk import (
    build_dealer_hedging_pressure_state,
    build_gamma_vol_acceleration_state,
    build_option_efficiency_state,
)
from risk.global_risk_layer import evaluate_global_risk_layer
from risk.option_efficiency_layer import score_option_efficiency_candidate
from strategy.budget_optimizer import optimize_lots
from strategy.exit_model import calculate_exit
from strategy.strike_selector import select_best_strike


def generate_trade(
    symbol,
    spot,
    option_chain,
    previous_chain=None,
    day_high=None,
    day_low=None,
    day_open=None,
    prev_close=None,
    lookback_avg_range_pct=None,
    spot_validation=None,
    option_chain_validation=None,
    apply_budget_constraint=False,
    requested_lots=NUMBER_OF_LOTS,
    lot_size=LOT_SIZE,
    max_capital=MAX_CAPITAL_PER_TRADE,
    backtest_mode=False,
    macro_event_state=None,
    macro_news_state=None,
    global_risk_state=None,
    holding_profile="AUTO",
    valuation_time=None,
    target_profit_percent=TARGET_PROFIT_PERCENT,
    stop_loss_percent=STOP_LOSS_PERCENT,
):
    if option_chain is None or option_chain.empty:
        return None

    df = normalize_option_chain(option_chain, spot=spot, valuation_time=valuation_time)
    prev_df = (
        normalize_option_chain(previous_chain, spot=spot, valuation_time=valuation_time)
        if previous_chain is not None else None
    )
    market_state = _collect_market_state(df, spot, symbol=symbol, prev_df=prev_df)
    probability_state = _compute_probability_state(
        df,
        spot=spot,
        symbol=symbol,
        market_state=market_state,
        day_high=day_high,
        day_low=day_low,
        day_open=day_open,
        prev_close=prev_close,
        lookback_avg_range_pct=lookback_avg_range_pct,
    )
    intraday_range_pct = probability_state["components"]["intraday_range_pct"]

    analytics_state = {
        "flip": market_state["flip"],
        "dealer_pos": market_state["dealer_pos"],
        "vol_regime": market_state["vol_regime"],
        "gamma_regime": market_state["gamma_regime"],
        "vanna_regime": market_state["greek_exposures"].get("vanna_regime"),
        "charm_regime": market_state["greek_exposures"].get("charm_regime"),
        "final_flow_signal": market_state["final_flow_signal"],
        "hedging_bias": market_state["hedging_bias"],
        "atm_iv": market_state["atm_iv"],
        "support_wall": market_state["support_wall"],
        "resistance_wall": market_state["resistance_wall"],
        "market_gamma_summary": market_state["market_gamma_summary"],
        "provider_health": option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else None,
    }

    data_quality = _compute_data_quality(
        spot_validation=spot_validation,
        option_chain_validation=option_chain_validation,
        analytics_state=analytics_state,
        probability_state=probability_state,
    )

    macro_event_state = macro_event_state if isinstance(macro_event_state, dict) else {}
    macro_event_risk_score = int(_safe_float(macro_event_state.get("macro_event_risk_score"), 0))
    event_window_status = macro_event_state.get("event_window_status", "NO_EVENT_DATA")
    event_lockdown_flag = bool(macro_event_state.get("event_lockdown_flag", False))
    minutes_to_next_event = macro_event_state.get("minutes_to_next_event")
    next_event_name = macro_event_state.get("next_event_name")
    active_event_name = macro_event_state.get("active_event_name")
    signal_state = _compute_signal_state(
        spot=spot,
        symbol=symbol,
        day_open=day_open,
        prev_close=prev_close,
        intraday_range_pct=intraday_range_pct,
        backtest_mode=backtest_mode,
        market_state=market_state,
        probability_state=probability_state,
    )
    direction = signal_state["direction"]
    direction_source = signal_state["direction_source"]
    trade_strength = signal_state["trade_strength"]
    scoring_breakdown = signal_state["scoring_breakdown"]
    confirmation = signal_state["confirmation"]

    macro_news_adjustments = compute_macro_news_adjustments(
        direction=direction,
        macro_news_state=macro_news_state,
    )
    event_cfg = get_event_window_policy_config()

    macro_event_score_adjustment = 0
    if event_window_status == "PRE_EVENT_WATCH":
        macro_event_score_adjustment = (
            event_cfg.pre_event_watch_penalty_high
            if macro_event_risk_score >= event_cfg.watch_risk_threshold
            else event_cfg.pre_event_watch_penalty_normal
        )
    elif event_window_status == "POST_EVENT_COOLDOWN":
        macro_event_score_adjustment = (
            event_cfg.post_event_cooldown_penalty_high
            if macro_event_risk_score >= event_cfg.watch_risk_threshold
            else event_cfg.post_event_cooldown_penalty_normal
        )
    elif event_window_status in {"PRE_EVENT_LOCKDOWN", "LIVE_EVENT"}:
        macro_event_score_adjustment = event_cfg.lockdown_penalty

    confirmation["score_adjustment"] += macro_news_adjustments["macro_confirmation_adjustment"]

    scoring_breakdown["confirmation_filter_score"] = confirmation["score_adjustment"]
    scoring_breakdown["macro_event_score"] = macro_event_score_adjustment
    scoring_breakdown["macro_news_score"] = macro_news_adjustments["macro_adjustment_score"]
    global_risk_trade_modifiers = derive_global_risk_trade_modifiers(global_risk_state)
    global_risk_adjustment_score = global_risk_trade_modifiers["effective_adjustment_score"]
    scoring_breakdown["global_risk_base_adjustment_score"] = global_risk_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["global_risk_feature_adjustment_score"] = global_risk_trade_modifiers["feature_adjustment_score"]
    scoring_breakdown["global_risk_adjustment_score"] = global_risk_adjustment_score
    gamma_vol_state = build_gamma_vol_acceleration_state(
        gamma_regime=market_state["gamma_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
        dealer_hedging_bias=market_state["hedging_bias"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        intraday_range_pct=intraday_range_pct,
        volatility_compression_score=(
            global_risk_state.get("global_risk_features", {}).get("volatility_compression_score")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        volatility_shock_score=(
            global_risk_state.get("global_risk_features", {}).get("volatility_shock_score")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=(
            global_risk_state.get("global_risk_features", {}).get("volatility_explosion_probability")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        holding_profile=holding_profile,
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
    )
    gamma_vol_trade_modifiers = derive_gamma_vol_trade_modifiers(gamma_vol_state, direction=direction)
    gamma_vol_adjustment_score = gamma_vol_trade_modifiers["effective_adjustment_score"]
    scoring_breakdown["gamma_vol_base_adjustment_score"] = gamma_vol_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["gamma_vol_alignment_adjustment_score"] = gamma_vol_trade_modifiers["alignment_adjustment_score"]
    scoring_breakdown["gamma_vol_adjustment_score"] = gamma_vol_adjustment_score
    dealer_pressure_state = build_dealer_hedging_pressure_state(
        spot=spot,
        gamma_regime=market_state["gamma_regime"],
        spot_vs_flip=market_state["spot_vs_flip"],
        gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
        dealer_position=market_state["dealer_pos"],
        dealer_hedging_bias=market_state["hedging_bias"],
        dealer_hedging_flow=market_state["hedging_flow"],
        market_gamma=market_state["market_gamma_summary"],
        gamma_clusters=market_state["gamma_clusters"],
        liquidity_levels=market_state["liquidity_levels"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        intraday_gamma_state=market_state["intraday_gamma_state"],
        intraday_range_pct=intraday_range_pct,
        flow_signal=market_state["flow_signal_value"],
        smart_money_flow=market_state["smart_money_signal_value"],
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=global_risk_state,
        volatility_explosion_probability=(
            global_risk_state.get("global_risk_features", {}).get("volatility_explosion_probability")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        holding_profile=holding_profile,
    )
    dealer_pressure_trade_modifiers = derive_dealer_pressure_trade_modifiers(dealer_pressure_state, direction=direction)
    dealer_pressure_adjustment_score = dealer_pressure_trade_modifiers["effective_adjustment_score"]
    scoring_breakdown["dealer_pressure_base_adjustment_score"] = dealer_pressure_trade_modifiers["base_adjustment_score"]
    scoring_breakdown["dealer_pressure_alignment_adjustment_score"] = dealer_pressure_trade_modifiers["alignment_adjustment_score"]
    scoring_breakdown["dealer_pressure_adjustment_score"] = dealer_pressure_adjustment_score
    global_risk_features = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    option_efficiency_state = {}
    option_efficiency_trade_modifiers = derive_option_efficiency_trade_modifiers(option_efficiency_state)
    option_efficiency_adjustment_score = option_efficiency_trade_modifiers["option_efficiency_adjustment_score"]
    scoring_breakdown["option_efficiency_adjustment_score"] = option_efficiency_adjustment_score
    adjusted_trade_strength = int(
        _clip(
            trade_strength
            + confirmation["score_adjustment"]
            + macro_event_score_adjustment
            + macro_news_adjustments["macro_adjustment_score"],
            0,
            100,
        )
    )
    adjusted_trade_strength = int(
        _clip(
            adjusted_trade_strength + global_risk_adjustment_score + gamma_vol_adjustment_score + dealer_pressure_adjustment_score,
            0,
            100,
        )
    )
    scoring_breakdown["base_trade_strength"] = trade_strength
    scoring_breakdown["total_score"] = adjusted_trade_strength
    signal_regime = classify_signal_regime(
        direction=direction,
        adjusted_trade_strength=adjusted_trade_strength,
        final_flow_signal=market_state["final_flow_signal"],
        gamma_regime=market_state["gamma_regime"],
        confirmation_status=confirmation["status"],
        event_lockdown_flag=event_lockdown_flag or macro_news_adjustments["event_lockdown_flag"],
        data_quality_status=data_quality["status"],
    )

    runtime_thresholds = get_trade_runtime_thresholds()
    min_trade_strength = (
        BACKTEST_MIN_TRADE_STRENGTH
        if backtest_mode
        else runtime_thresholds["min_trade_strength"]
    )

    base_payload = {
        "symbol": symbol,
        "spot": round(spot, 2),
        "gamma_exposure": round(market_state["gamma"], 2) if market_state["gamma"] is not None else None,
        "market_gamma": market_state["market_gamma_summary"],
        "gamma_flip": _to_python_number(market_state["flip"]),
        "spot_vs_flip": market_state["spot_vs_flip"],
        "gamma_regime": market_state["gamma_regime"],
        "gamma_clusters": market_state["gamma_clusters"],
        "delta_exposure": market_state["greek_exposures"].get("delta_exposure"),
        "gamma_exposure_greeks": market_state["greek_exposures"].get("gamma_exposure_greeks"),
        "theta_exposure": market_state["greek_exposures"].get("theta_exposure"),
        "vega_exposure": market_state["greek_exposures"].get("vega_exposure"),
        "rho_exposure": market_state["greek_exposures"].get("rho_exposure"),
        "vanna_exposure": market_state["greek_exposures"].get("vanna_exposure"),
        "charm_exposure": market_state["greek_exposures"].get("charm_exposure"),
        "vanna_regime": market_state["greek_exposures"].get("vanna_regime"),
        "charm_regime": market_state["greek_exposures"].get("charm_regime"),
        "dealer_position": market_state["dealer_pos"],
        "dealer_inventory_basis": market_state["dealer_metrics"].get("basis"),
        "call_oi_change": market_state["dealer_metrics"].get("call_oi_change"),
        "put_oi_change": market_state["dealer_metrics"].get("put_oi_change"),
        "net_oi_change_bias": market_state["dealer_metrics"].get("net_oi_change_bias"),
        "dealer_hedging_flow": market_state["hedging_flow"],
        "dealer_hedging_bias": market_state["hedging_bias"],
        "intraday_gamma_state": market_state["intraday_gamma_state"],
        "volatility_regime": market_state["vol_regime"],
        "vol_surface_regime": market_state["surface_regime"],
        "atm_iv": round(float(market_state["atm_iv"]), 2) if market_state["atm_iv"] is not None else None,
        "flow_signal": market_state["flow_signal_value"],
        "smart_money_flow": market_state["smart_money_signal_value"],
        "final_flow_signal": market_state["final_flow_signal"],
        "gamma_event": market_state["gamma_event"],
        "support_wall": market_state["support_wall"],
        "resistance_wall": market_state["resistance_wall"],
        "liquidity_levels": market_state["liquidity_levels"],
        "liquidity_voids": market_state["voids"],
        "liquidity_void_signal": market_state["void_signal"],
        "liquidity_vacuum_zones": market_state["vacuum_zones"],
        "liquidity_vacuum_state": market_state["vacuum_state"],
        "dealer_liquidity_map": market_state["dealer_liquidity_map"],
        "rule_move_probability": probability_state["rule_move_probability"],
        "ml_move_probability": probability_state["ml_move_probability"],
        "hybrid_move_probability": probability_state["hybrid_move_probability"],
        "large_move_probability": probability_state["hybrid_move_probability"],
        "move_probability_components": probability_state["components"],
        "spot_validation": spot_validation,
        "option_chain_validation": option_chain_validation,
        "provider_health": option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else None,
        "data_quality_score": data_quality["score"],
        "data_quality_status": data_quality["status"],
        "data_quality_reasons": data_quality["reasons"],
        "analytics_quality": data_quality["analytics_quality"],
        "confirmation_status": confirmation["status"],
        "confirmation_veto": confirmation["veto"],
        "confirmation_reasons": confirmation["reasons"],
        "confirmation_breakdown": confirmation["breakdown"],
        "direction_source": direction_source,
        "trade_strength": adjusted_trade_strength,
        "signal_quality": classify_signal_quality(adjusted_trade_strength),
        "signal_regime": signal_regime,
        "scoring_breakdown": scoring_breakdown,
        "macro_event_risk_score": macro_event_risk_score,
        "event_window_status": event_window_status,
        "event_lockdown_flag": event_lockdown_flag,
        "minutes_to_next_event": minutes_to_next_event,
        "next_event_name": next_event_name,
        "active_event_name": active_event_name,
        "macro_regime": macro_news_adjustments["macro_regime"],
        "macro_sentiment_score": macro_news_adjustments["macro_sentiment_score"],
        "volatility_shock_score": macro_news_adjustments["volatility_shock_score"],
        "news_confidence_score": macro_news_adjustments["news_confidence_score"],
        "macro_adjustment_score": macro_news_adjustments["macro_adjustment_score"],
        "macro_confirmation_adjustment": macro_news_adjustments["macro_confirmation_adjustment"],
        "macro_position_size_multiplier": macro_news_adjustments["macro_position_size_multiplier"],
        "macro_adjustment_reasons": macro_news_adjustments["macro_adjustment_reasons"],
        "global_risk_state": global_risk_state.get("global_risk_state") if isinstance(global_risk_state, dict) else "GLOBAL_NEUTRAL",
        "global_risk_score": global_risk_state.get("global_risk_score") if isinstance(global_risk_state, dict) else 0,
        "overnight_gap_risk_score": global_risk_state.get("overnight_gap_risk_score") if isinstance(global_risk_state, dict) else 0,
        "volatility_expansion_risk_score": global_risk_state.get("volatility_expansion_risk_score") if isinstance(global_risk_state, dict) else 0,
        "overnight_hold_allowed": global_risk_trade_modifiers["overnight_hold_allowed"],
        "overnight_hold_reason": global_risk_trade_modifiers["overnight_hold_reason"],
        "overnight_risk_penalty": global_risk_trade_modifiers["overnight_risk_penalty"],
        "overnight_trade_block": global_risk_trade_modifiers["overnight_trade_block"],
        "global_risk_adjustment_score": global_risk_adjustment_score,
        "gamma_vol_acceleration_score": gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        "squeeze_risk_state": gamma_vol_trade_modifiers["squeeze_risk_state"],
        "directional_convexity_state": gamma_vol_trade_modifiers["directional_convexity_state"],
        "upside_squeeze_risk": gamma_vol_trade_modifiers["upside_squeeze_risk"],
        "downside_airpocket_risk": gamma_vol_trade_modifiers["downside_airpocket_risk"],
        "overnight_convexity_risk": gamma_vol_trade_modifiers["overnight_convexity_risk"],
        "overnight_convexity_penalty": gamma_vol_trade_modifiers["overnight_convexity_penalty"],
        "overnight_convexity_boost": gamma_vol_trade_modifiers["overnight_convexity_boost"],
        "gamma_vol_adjustment_score": gamma_vol_adjustment_score,
        "dealer_hedging_pressure_score": dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
        "dealer_flow_state": dealer_pressure_trade_modifiers["dealer_flow_state"],
        "upside_hedging_pressure": dealer_pressure_trade_modifiers["upside_hedging_pressure"],
        "downside_hedging_pressure": dealer_pressure_trade_modifiers["downside_hedging_pressure"],
        "pinning_pressure_score": dealer_pressure_trade_modifiers["pinning_pressure_score"],
        "overnight_hedging_risk": dealer_pressure_trade_modifiers["overnight_hedging_risk"],
        "overnight_dealer_pressure_penalty": dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"],
        "overnight_dealer_pressure_boost": dealer_pressure_trade_modifiers["overnight_dealer_pressure_boost"],
        "dealer_pressure_adjustment_score": dealer_pressure_adjustment_score,
        "oil_shock_score": global_risk_trade_modifiers["oil_shock_score"],
        "market_volatility_shock_score": global_risk_trade_modifiers["volatility_shock_score"],
        "commodity_risk_score": global_risk_trade_modifiers["commodity_risk_score"],
        "risk_off_intensity": (
            global_risk_state.get("global_risk_features", {}).get("risk_off_intensity")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        "volatility_compression_score": (
            global_risk_state.get("global_risk_features", {}).get("volatility_compression_score")
            if isinstance(global_risk_state, dict)
            else 0.0
        ),
        "volatility_explosion_probability": global_risk_trade_modifiers["volatility_explosion_probability"],
        "global_risk_features": global_risk_state.get("global_risk_features") if isinstance(global_risk_state, dict) else {},
        "global_risk_diagnostics": global_risk_state.get("global_risk_diagnostics") if isinstance(global_risk_state, dict) else {},
        "gamma_vol_features": gamma_vol_state.get("gamma_vol_features") if isinstance(gamma_vol_state, dict) else {},
        "gamma_vol_diagnostics": gamma_vol_state.get("gamma_vol_diagnostics") if isinstance(gamma_vol_state, dict) else {},
        "dealer_pressure_features": dealer_pressure_state.get("dealer_pressure_features") if isinstance(dealer_pressure_state, dict) else {},
        "dealer_pressure_diagnostics": dealer_pressure_state.get("dealer_pressure_diagnostics") if isinstance(dealer_pressure_state, dict) else {},
        "budget_constraint_applied": apply_budget_constraint,
        "lot_size": lot_size,
        "requested_lots": requested_lots,
        "max_capital_per_trade": max_capital,
        "backtest_mode": backtest_mode,
    }

    global_risk = evaluate_global_risk_layer(
        data_quality=data_quality,
        confirmation=confirmation,
        adjusted_trade_strength=adjusted_trade_strength,
        min_trade_strength=min_trade_strength,
        event_window_status=event_window_status,
        macro_event_risk_score=macro_event_risk_score,
        event_lockdown_flag=event_lockdown_flag,
        next_event_name=next_event_name,
        active_event_name=active_event_name,
        macro_news_adjustments=macro_news_adjustments,
        global_risk_state=global_risk_state,
        holding_profile=holding_profile,
    )
    base_payload.update(
        {
            "global_risk_state": global_risk["global_risk_state"],
            "global_risk_score": global_risk["global_risk_score"],
            "overnight_gap_risk_score": global_risk["overnight_gap_risk_score"],
            "volatility_expansion_risk_score": global_risk["volatility_expansion_risk_score"],
            "overnight_hold_allowed": (
                global_risk_trade_modifiers["overnight_hold_allowed"]
                and gamma_vol_trade_modifiers["overnight_hold_allowed"]
                and dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                and option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
            "overnight_hold_reason": (
                global_risk_trade_modifiers["overnight_hold_reason"]
                if not global_risk_trade_modifiers["overnight_hold_allowed"]
                else (
                    gamma_vol_trade_modifiers["overnight_hold_reason"]
                    if not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                    else (
                        dealer_pressure_trade_modifiers["overnight_hold_reason"]
                        if not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                        else option_efficiency_trade_modifiers["overnight_hold_reason"]
                    )
                )
            ),
            "overnight_risk_penalty": (
                global_risk_trade_modifiers["overnight_risk_penalty"]
                + gamma_vol_trade_modifiers["overnight_convexity_penalty"]
                + dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"]
                + option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"]
            ),
            "overnight_trade_block": (
                global_risk_trade_modifiers["overnight_trade_block"]
                or not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                or not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                or not option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
            "global_risk_adjustment_score": global_risk_adjustment_score,
            "gamma_vol_acceleration_score": gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
            "squeeze_risk_state": gamma_vol_trade_modifiers["squeeze_risk_state"],
            "directional_convexity_state": gamma_vol_trade_modifiers["directional_convexity_state"],
            "upside_squeeze_risk": gamma_vol_trade_modifiers["upside_squeeze_risk"],
            "downside_airpocket_risk": gamma_vol_trade_modifiers["downside_airpocket_risk"],
            "overnight_convexity_risk": gamma_vol_trade_modifiers["overnight_convexity_risk"],
            "overnight_convexity_penalty": gamma_vol_trade_modifiers["overnight_convexity_penalty"],
            "overnight_convexity_boost": gamma_vol_trade_modifiers["overnight_convexity_boost"],
            "gamma_vol_adjustment_score": gamma_vol_adjustment_score,
            "dealer_hedging_pressure_score": dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
            "dealer_flow_state": dealer_pressure_trade_modifiers["dealer_flow_state"],
            "upside_hedging_pressure": dealer_pressure_trade_modifiers["upside_hedging_pressure"],
            "downside_hedging_pressure": dealer_pressure_trade_modifiers["downside_hedging_pressure"],
            "pinning_pressure_score": dealer_pressure_trade_modifiers["pinning_pressure_score"],
            "overnight_hedging_risk": dealer_pressure_trade_modifiers["overnight_hedging_risk"],
            "overnight_dealer_pressure_penalty": dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"],
            "overnight_dealer_pressure_boost": dealer_pressure_trade_modifiers["overnight_dealer_pressure_boost"],
            "dealer_pressure_adjustment_score": dealer_pressure_adjustment_score,
            "expected_move_points": option_efficiency_trade_modifiers["expected_move_points"],
            "expected_move_pct": option_efficiency_trade_modifiers["expected_move_pct"],
            "expected_move_quality": option_efficiency_trade_modifiers["expected_move_quality"],
            "target_reachability_score": option_efficiency_trade_modifiers["target_reachability_score"],
            "premium_efficiency_score": option_efficiency_trade_modifiers["premium_efficiency_score"],
            "strike_efficiency_score": option_efficiency_trade_modifiers["strike_efficiency_score"],
            "option_efficiency_score": option_efficiency_trade_modifiers["option_efficiency_score"],
            "option_efficiency_adjustment_score": option_efficiency_adjustment_score,
            "overnight_option_efficiency_penalty": option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"],
            "strike_moneyness_bucket": option_efficiency_trade_modifiers["strike_moneyness_bucket"],
            "strike_distance_from_spot": option_efficiency_trade_modifiers["strike_distance_from_spot"],
            "payoff_efficiency_hint": option_efficiency_trade_modifiers["payoff_efficiency_hint"],
            "oil_shock_score": global_risk_trade_modifiers["oil_shock_score"],
            "market_volatility_shock_score": global_risk_trade_modifiers["volatility_shock_score"],
            "commodity_risk_score": global_risk_trade_modifiers["commodity_risk_score"],
            "risk_off_intensity": global_risk["global_risk_features"].get("risk_off_intensity", 0.0),
            "volatility_compression_score": global_risk["global_risk_features"].get("volatility_compression_score", 0.0),
            "volatility_explosion_probability": global_risk_trade_modifiers["volatility_explosion_probability"],
            "global_risk_level": global_risk["global_risk_level"],
            "global_risk_action": global_risk["global_risk_action"],
            "global_risk_size_cap": global_risk["global_risk_size_cap"],
            "global_risk_reasons": global_risk["global_risk_reasons"],
            "global_risk_features": global_risk["global_risk_features"],
            "global_risk_diagnostics": global_risk["global_risk_diagnostics"],
            "gamma_vol_reasons": gamma_vol_state.get("gamma_vol_reasons", []) if isinstance(gamma_vol_state, dict) else [],
            "gamma_vol_features": gamma_vol_state.get("gamma_vol_features", {}) if isinstance(gamma_vol_state, dict) else {},
            "gamma_vol_diagnostics": gamma_vol_state.get("gamma_vol_diagnostics", {}) if isinstance(gamma_vol_state, dict) else {},
            "dealer_pressure_reasons": dealer_pressure_state.get("dealer_pressure_reasons", []) if isinstance(dealer_pressure_state, dict) else [],
            "dealer_pressure_features": dealer_pressure_state.get("dealer_pressure_features", {}) if isinstance(dealer_pressure_state, dict) else {},
            "dealer_pressure_diagnostics": dealer_pressure_state.get("dealer_pressure_diagnostics", {}) if isinstance(dealer_pressure_state, dict) else {},
            "option_efficiency_reasons": option_efficiency_state.get("option_efficiency_reasons", []) if isinstance(option_efficiency_state, dict) else [],
            "option_efficiency_features": option_efficiency_state.get("option_efficiency_features", {}) if isinstance(option_efficiency_state, dict) else {},
            "option_efficiency_diagnostics": option_efficiency_state.get("option_efficiency_diagnostics", {}) if isinstance(option_efficiency_state, dict) else {},
        }
    )

    def _finalize(payload, trade_status, message):
        payload["message"] = message
        payload["trade_status"] = trade_status
        execution_size_multiplier = min(
            _safe_float(macro_news_adjustments.get("macro_position_size_multiplier"), 1.0),
            _safe_float(global_risk.get("global_risk_size_cap"), 1.0),
        )
        payload["execution_regime"] = classify_execution_regime(
            trade_status=trade_status,
            signal_regime=signal_regime,
            data_quality_score=data_quality["score"],
            macro_position_size_multiplier=execution_size_multiplier,
        )
        return payload

    if global_risk["risk_trade_status"] == "DATA_INVALID":
        return _finalize(base_payload, "DATA_INVALID", global_risk["risk_message"])

    if global_risk_trade_modifiers["force_no_trade"] or global_risk["risk_trade_status"] == "EVENT_LOCKDOWN":
        return _finalize(base_payload, "NO_TRADE", global_risk["risk_message"] or "Trade blocked due to global event lockdown")

    if direction is None:
        return _finalize(base_payload, "NO_SIGNAL", "No trade signal")

    if (
        market_state["final_flow_signal"] == "NEUTRAL_FLOW"
        and probability_state["hybrid_move_probability"] is not None
        and probability_state["hybrid_move_probability"] < runtime_thresholds["neutral_flow_probability_floor"]
    ):
        return _finalize(base_payload, "NO_SIGNAL", "No trade signal: neutral flow and insufficient directional edge")

    ranked_strikes = []
    strike = None

    if direction is not None:
        def option_efficiency_candidate_hook(row):
            return score_option_efficiency_candidate(
                row,
                spot=spot,
                direction=direction,
                atm_iv=market_state["atm_iv"],
                selected_expiry=(
                    option_chain_validation.get("selected_expiry")
                    if isinstance(option_chain_validation, dict)
                    else None
                ),
                valuation_time=valuation_time,
                hybrid_move_probability=probability_state["hybrid_move_probability"],
                gamma_regime=market_state["gamma_regime"],
                volatility_regime=market_state["vol_regime"],
                volatility_shock_score=global_risk_features.get("volatility_shock_score"),
                volatility_compression_score=global_risk_features.get("volatility_compression_score"),
                macro_event_risk_score=macro_event_risk_score,
                global_risk_state=base_payload.get("global_risk_state"),
                gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
                dealer_hedging_pressure_score=dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
                liquidity_vacuum_state=market_state["vacuum_state"],
                support_wall=market_state["support_wall"],
                resistance_wall=market_state["resistance_wall"],
            )

        strike, ranked_strikes = select_best_strike(
            option_chain=df,
            direction=direction,
            spot=spot,
            support_wall=market_state["support_wall"],
            resistance_wall=market_state["resistance_wall"],
            gamma_clusters=market_state["gamma_clusters"],
            lot_size=lot_size,
            max_capital=max_capital if apply_budget_constraint else None,
            candidate_score_hook=option_efficiency_candidate_hook,
        )

    base_payload["ranked_strike_candidates"] = ranked_strikes

    if strike is None:
        base_payload["direction"] = direction
        return _finalize(base_payload, "NO_SIGNAL", "No valid strike found")

    option_type = "CE" if direction == "CALL" else "PE"

    option_row = df[
        (df["strikePrice"] == strike) &
        (df["OPTION_TYP"] == option_type)
    ]

    if option_row.empty:
        base_payload["direction"] = direction
        return _finalize(base_payload, "NO_SIGNAL", "Selected strike/option type not available")

    entry_price = float(option_row.iloc[0]["lastPrice"])
    target, stop_loss = calculate_exit(
        entry_price,
        target_profit_percent=target_profit_percent,
        stop_loss_percent=stop_loss_percent,
    )
    option_row_dict = option_row.iloc[0].to_dict()
    option_efficiency_state = build_option_efficiency_state(
        spot=spot,
        atm_iv=market_state["atm_iv"],
        fallback_iv=option_row_dict.get("impliedVolatility", option_row_dict.get("IV")),
        expiry_value=option_row_dict.get(
            "EXPIRY_DT",
            option_chain_validation.get("selected_expiry") if isinstance(option_chain_validation, dict) else None,
        ),
        valuation_time=valuation_time,
        time_to_expiry_years=option_row_dict.get("TTE"),
        direction=direction,
        strike=strike,
        option_type=option_type,
        entry_price=entry_price,
        target=target,
        stop_loss=stop_loss,
        trade_strength=adjusted_trade_strength,
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        rule_move_probability=probability_state["rule_move_probability"],
        ml_move_probability=probability_state["ml_move_probability"],
        gamma_regime=market_state["gamma_regime"],
        volatility_regime=market_state["vol_regime"],
        volatility_shock_score=global_risk_features.get("volatility_shock_score"),
        volatility_compression_score=global_risk_features.get("volatility_compression_score"),
        macro_event_risk_score=macro_event_risk_score,
        global_risk_state=base_payload.get("global_risk_state"),
        gamma_vol_acceleration_score=gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        dealer_hedging_pressure_score=dealer_pressure_trade_modifiers["dealer_hedging_pressure_score"],
        liquidity_vacuum_state=market_state["vacuum_state"],
        support_wall=market_state["support_wall"],
        resistance_wall=market_state["resistance_wall"],
        delta=option_row_dict.get("DELTA"),
        holding_profile=holding_profile,
    )
    option_efficiency_trade_modifiers = derive_option_efficiency_trade_modifiers(option_efficiency_state)
    option_efficiency_adjustment_score = option_efficiency_trade_modifiers["option_efficiency_adjustment_score"]
    adjusted_trade_strength = int(_clip(adjusted_trade_strength + option_efficiency_adjustment_score, 0, 100))
    scoring_breakdown["option_efficiency_adjustment_score"] = option_efficiency_adjustment_score
    scoring_breakdown["total_score"] = adjusted_trade_strength
    base_payload["trade_strength"] = adjusted_trade_strength
    base_payload["signal_quality"] = classify_signal_quality(adjusted_trade_strength)
    signal_regime = classify_signal_regime(
        direction=direction,
        adjusted_trade_strength=adjusted_trade_strength,
        final_flow_signal=market_state["final_flow_signal"],
        gamma_regime=market_state["gamma_regime"],
        confirmation_status=confirmation["status"],
        event_lockdown_flag=event_lockdown_flag or macro_news_adjustments["event_lockdown_flag"],
        data_quality_status=data_quality["status"],
    )
    base_payload["signal_regime"] = signal_regime

    base_payload.update({
        "direction": direction,
        "strike": _to_python_number(strike),
        "option_type": option_type,
        "entry_price": round(entry_price, 2),
        "target": round(target, 2),
        "stop_loss": round(stop_loss, 2),
        "expected_move_points": option_efficiency_trade_modifiers["expected_move_points"],
        "expected_move_pct": option_efficiency_trade_modifiers["expected_move_pct"],
        "expected_move_quality": option_efficiency_trade_modifiers["expected_move_quality"],
        "target_reachability_score": option_efficiency_trade_modifiers["target_reachability_score"],
        "premium_efficiency_score": option_efficiency_trade_modifiers["premium_efficiency_score"],
        "strike_efficiency_score": option_efficiency_trade_modifiers["strike_efficiency_score"],
        "option_efficiency_score": option_efficiency_trade_modifiers["option_efficiency_score"],
        "option_efficiency_adjustment_score": option_efficiency_trade_modifiers["option_efficiency_adjustment_score"],
        "overnight_option_efficiency_penalty": option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"],
        "strike_moneyness_bucket": option_efficiency_trade_modifiers["strike_moneyness_bucket"],
        "strike_distance_from_spot": option_efficiency_trade_modifiers["strike_distance_from_spot"],
        "payoff_efficiency_hint": option_efficiency_trade_modifiers["payoff_efficiency_hint"],
        "option_efficiency_reasons": option_efficiency_state.get("option_efficiency_reasons", []),
        "option_efficiency_features": option_efficiency_state.get("option_efficiency_features", {}),
        "option_efficiency_diagnostics": option_efficiency_state.get("option_efficiency_diagnostics", {}),
    })
    base_payload.update(
        {
            "overnight_hold_allowed": (
                global_risk_trade_modifiers["overnight_hold_allowed"]
                and gamma_vol_trade_modifiers["overnight_hold_allowed"]
                and dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                and option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
            "overnight_hold_reason": (
                global_risk_trade_modifiers["overnight_hold_reason"]
                if not global_risk_trade_modifiers["overnight_hold_allowed"]
                else (
                    gamma_vol_trade_modifiers["overnight_hold_reason"]
                    if not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                    else (
                        dealer_pressure_trade_modifiers["overnight_hold_reason"]
                        if not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                        else option_efficiency_trade_modifiers["overnight_hold_reason"]
                    )
                )
            ),
            "overnight_risk_penalty": (
                global_risk_trade_modifiers["overnight_risk_penalty"]
                + gamma_vol_trade_modifiers["overnight_convexity_penalty"]
                + dealer_pressure_trade_modifiers["overnight_dealer_pressure_penalty"]
                + option_efficiency_trade_modifiers["overnight_option_efficiency_penalty"]
            ),
            "overnight_trade_block": (
                global_risk_trade_modifiers["overnight_trade_block"]
                or not gamma_vol_trade_modifiers["overnight_hold_allowed"]
                or not dealer_pressure_trade_modifiers["overnight_hold_allowed"]
                or not option_efficiency_trade_modifiers["overnight_hold_allowed"]
            ),
        }
    )

    if apply_budget_constraint:
        budget_info = optimize_lots(
            entry_price=entry_price,
            lot_size=lot_size,
            max_capital=max_capital,
            requested_lots=requested_lots,
        )

        base_payload.update(budget_info)

        if not budget_info["budget_ok"]:
            return _finalize(base_payload, "BUDGET_FAIL", "Trade filtered out due to budget constraint")

        base_payload["number_of_lots"] = budget_info.get("optimized_lots", requested_lots)
    else:
        base_payload["number_of_lots"] = requested_lots
        base_payload["capital_per_lot"] = round(entry_price * lot_size, 2)
        base_payload["capital_required"] = round(entry_price * lot_size * requested_lots, 2)

    risk_size_cap = global_risk["global_risk_size_cap"]
    suggested_lots = max(0, int(base_payload["number_of_lots"] * risk_size_cap))
    if base_payload["number_of_lots"] > 0 and suggested_lots == 0 and risk_size_cap > 0:
        suggested_lots = 1
    base_payload["macro_suggested_lots"] = suggested_lots
    base_payload["macro_size_applied"] = suggested_lots > 0 and suggested_lots < base_payload["number_of_lots"]
    if suggested_lots > 0:
        base_payload["number_of_lots"] = min(base_payload["number_of_lots"], suggested_lots)

    base_payload["optimized_lots"] = base_payload["number_of_lots"]

    if "entry_price" in base_payload:
        base_payload["capital_per_lot"] = round(entry_price * lot_size, 2)
        base_payload["capital_required"] = round(entry_price * lot_size * base_payload["number_of_lots"], 2)

    if global_risk["risk_trade_status"] == "WATCHLIST":
        return _finalize(base_payload, "WATCHLIST", global_risk["risk_message"])

    if apply_budget_constraint:
        base_payload["message"] = "Tradable signal generated with budget optimization"
    else:
        base_payload["message"] = "Tradable signal generated"

    return _finalize(
        base_payload,
        "TRADE",
        "Tradable signal generated with budget optimization" if apply_budget_constraint else "Tradable signal generated",
    )
