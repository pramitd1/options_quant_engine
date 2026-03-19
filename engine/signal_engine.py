"""
Module: signal_engine.py

Purpose:
    Assemble the final trade decision from normalized market data, analytics features, macro context, and risk controls.

Role in the System:
    Part of the signal engine layer that assembles analytics, strategy logic, and overlays into trade decisions.

Key Outputs:
    A fully explained trade or no-trade payload, including diagnostics, overlay scores, strike selection, and sizing fields.

Downstream Usage:
    Consumed by the live runtime loop, replay tooling, shadow-mode comparisons, and signal-evaluation logging.
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
from config.signal_policy import get_activation_score_policy_config, get_trade_runtime_thresholds
from engine.runtime_metadata import attach_trade_views
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
from analytics.signal_confidence import compute_signal_confidence
from macro.engine_adjustments import compute_macro_news_adjustments
from risk import (
    build_dealer_hedging_pressure_state,
    build_gamma_vol_acceleration_state,
    build_option_efficiency_state,
)
from risk.global_risk_layer import evaluate_global_risk_layer
from risk.option_efficiency_layer import score_option_efficiency_candidate
from strategy.budget_optimizer import optimize_lots
from strategy.exit_model import calculate_exit, compute_exit_timing
from strategy.strike_selector import select_best_strike


def _as_upper(value):
    return str(value or "").upper().strip()


def _is_directional_flow(flow_label):
    return _as_upper(flow_label) in {"BULLISH_FLOW", "BEARISH_FLOW"}


def _is_convexity_active(convexity_state):
    return _as_upper(convexity_state) in {
        "UPSIDE_SQUEEZE_RISK",
        "DOWNSIDE_AIRPOCKET_RISK",
        "TWO_SIDED_VOLATILITY_RISK",
    }


def _is_dealer_structure_active(dealer_flow_state):
    return _as_upper(dealer_flow_state) in {
        "UPSIDE_HEDGING_ACCELERATION",
        "DOWNSIDE_HEDGING_ACCELERATION",
        "PINNING_DOMINANT",
        "TWO_SIDED_INSTABILITY",
    }


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in (None, "", []):
            continue
        normalized = str(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _collect_neutralization_states(payload):
    option_efficiency_features = payload.get("option_efficiency_features")
    option_efficiency_diagnostics = payload.get("option_efficiency_diagnostics")
    option_efficiency_reasons = payload.get("option_efficiency_reasons")
    macro_adjustment_reasons = payload.get("macro_adjustment_reasons")
    global_risk_features = payload.get("global_risk_features")
    global_risk_diagnostics = payload.get("global_risk_diagnostics")

    option_efficiency_features = option_efficiency_features if isinstance(option_efficiency_features, dict) else {}
    option_efficiency_diagnostics = option_efficiency_diagnostics if isinstance(option_efficiency_diagnostics, dict) else {}
    option_efficiency_reasons = option_efficiency_reasons if isinstance(option_efficiency_reasons, list) else []
    macro_adjustment_reasons = macro_adjustment_reasons if isinstance(macro_adjustment_reasons, list) else []
    global_risk_features = global_risk_features if isinstance(global_risk_features, dict) else {}
    global_risk_diagnostics = global_risk_diagnostics if isinstance(global_risk_diagnostics, dict) else {}

    option_efficiency_status = "AVAILABLE"
    option_efficiency_reason = "features_available"
    if (
        option_efficiency_features.get("neutral_fallback")
        or "option_efficiency_neutral_fallback" in option_efficiency_reasons
        or payload.get("expected_move_points") is None
    ):
        option_efficiency_status = "UNAVAILABLE_NEUTRALIZED"
        warnings = option_efficiency_diagnostics.get("warnings")
        if isinstance(warnings, list) and warnings:
            option_efficiency_reason = str(warnings[0])
        else:
            option_efficiency_reason = "expected_move_not_computable"

    global_risk_status = "ACTIVE"
    global_risk_reason = "global_risk_features_available"
    if global_risk_features.get("market_features_neutralized"):
        global_risk_status = "LOW_CONFIDENCE_NEUTRALIZED"
        global_risk_reason = "market_features_neutralized"
    elif global_risk_diagnostics.get("fallback"):
        global_risk_status = "LOW_CONFIDENCE_NEUTRALIZED"
        global_risk_reason = "fallback_global_risk_state"

    macro_news_status = "ACTIVE"
    macro_news_reason = "headline_adjustments_available"
    if "macro_news_neutral_fallback" in macro_adjustment_reasons:
        macro_news_status = "STALE_NEUTRALIZED"
        macro_news_reason = "macro_news_neutral_fallback"
    elif _as_upper(payload.get("macro_regime")) == "MACRO_NEUTRAL":
        macro_news_status = "NEUTRAL"
        macro_news_reason = "macro_regime_neutral"

    return {
        "option_efficiency_status": option_efficiency_status,
        "option_efficiency_reason": option_efficiency_reason,
        "global_risk_status": global_risk_status,
        "global_risk_reason": global_risk_reason,
        "macro_news_status": macro_news_status,
        "macro_news_reason": macro_news_reason,
    }


def _normalize_gamma_vol_score(raw_score, normalization_scale):
    scale = max(_safe_float(normalization_scale, 100.0), 1.0)
    raw = _safe_float(raw_score, 0.0)
    return int(_clip((raw / scale) * 100.0, 0, 100))


def _compute_runtime_composite_score(
    *,
    trade_strength,
    hybrid_move_probability,
    confirmation_status,
    data_quality_status,
    gamma_vol_acceleration_score_normalized,
):
    confirmation_map = {
        "STRONG_CONFIRMATION": 100,
        "CONFIRMED": 85,
        "MIXED": 55,
        "CONFLICT": 25,
        "NO_DIRECTION": 10,
    }
    data_quality_map = {
        "STRONG": 100,
        "GOOD": 85,
        "CAUTION": 60,
        "WEAK": 35,
    }

    trade_strength_score = _clip(_safe_float(trade_strength, 0.0), 0, 100)
    move_probability_score = _clip(_safe_float(hybrid_move_probability, 0.0) * 100.0, 0, 100)
    confirmation_score = confirmation_map.get(_as_upper(confirmation_status), 45)
    data_quality_score = data_quality_map.get(_as_upper(data_quality_status), 50)
    gamma_stability_score = 100.0 - _clip(_safe_float(gamma_vol_acceleration_score_normalized, 0.0), 0, 100)

    composite = (
        0.50 * trade_strength_score
        + 0.20 * move_probability_score
        + 0.15 * confirmation_score
        + 0.10 * data_quality_score
        + 0.05 * gamma_stability_score
    )
    return int(_clip(round(composite), 0, 100))


def _resolve_regime_thresholds(*, runtime_thresholds, base_min_trade_strength, base_min_composite_score, market_state):
    adjustments = []
    effective_trade_strength = int(base_min_trade_strength)
    effective_composite = int(base_min_composite_score)

    spot_vs_flip = _as_upper(market_state.get("spot_vs_flip"))
    gamma_regime = _as_upper(market_state.get("gamma_regime"))
    dealer_position = _as_upper(market_state.get("dealer_pos"))
    toxic_gamma = gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}
    dealer_short_gamma = ("SHORT" in dealer_position) and ("GAMMA" in dealer_position)

    if spot_vs_flip == "AT_FLIP":
        add_strength = int(_safe_float(runtime_thresholds.get("regime_strength_add_at_flip"), 4.0))
        add_composite = int(_safe_float(runtime_thresholds.get("regime_composite_add_at_flip"), 3.0))
        effective_trade_strength += add_strength
        effective_composite += add_composite
        adjustments.append("at_flip_threshold_tightening")

    if toxic_gamma or dealer_short_gamma:
        add_strength = int(_safe_float(runtime_thresholds.get("regime_strength_add_toxic"), 8.0))
        add_composite = int(_safe_float(runtime_thresholds.get("regime_composite_add_toxic"), 6.0))
        effective_trade_strength += add_strength
        effective_composite += add_composite
        adjustments.append("toxic_regime_threshold_tightening")

    return {
        "effective_min_trade_strength": int(_clip(effective_trade_strength, 0, 100)),
        "effective_min_composite_score": int(_clip(effective_composite, 0, 100)),
        "adjustments": adjustments,
        "toxic_context": bool(toxic_gamma or dealer_short_gamma),
    }


def _compute_structural_imbalance_audit(*, market_state, direction):
    dealer_metrics = market_state.get("dealer_metrics") if isinstance(market_state, dict) else {}
    dealer_metrics = dealer_metrics if isinstance(dealer_metrics, dict) else {}
    call_oi_change = _safe_float(dealer_metrics.get("call_oi_change"), 0.0)
    put_oi_change = _safe_float(dealer_metrics.get("put_oi_change"), 0.0)
    directional_imbalance = call_oi_change - put_oi_change

    direction = _as_upper(direction)
    alignment = "NEUTRAL"
    severity = "LOW"
    if direction == "CALL":
        alignment = "ALIGNED" if directional_imbalance >= 0 else "CONFLICT"
    elif direction == "PUT":
        alignment = "ALIGNED" if directional_imbalance <= 0 else "CONFLICT"

    abs_imbalance = abs(directional_imbalance)
    if abs_imbalance >= 200000:
        severity = "HIGH"
    elif abs_imbalance >= 80000:
        severity = "MEDIUM"

    return {
        "call_put_imbalance_score": round(directional_imbalance, 2),
        "call_put_imbalance_abs": round(abs_imbalance, 2),
        "call_put_alignment": alignment,
        "call_put_imbalance_severity": severity,
    }


def _build_decision_explainability(payload, *, trade_status, min_trade_strength):
    direction = payload.get("direction")
    flow_signal = _as_upper(payload.get("final_flow_signal") or payload.get("flow_signal"))
    smart_money_flow = _as_upper(payload.get("smart_money_flow"))
    confirmation_status = _as_upper(payload.get("confirmation_status"))
    signal_quality = _as_upper(payload.get("signal_quality"))
    directional_convexity_state = _as_upper(payload.get("directional_convexity_state"))
    dealer_flow_state = _as_upper(payload.get("dealer_flow_state"))
    dealer_hedging_bias = _as_upper(payload.get("dealer_hedging_bias"))
    global_risk_action = _as_upper(payload.get("global_risk_action"))
    data_quality_status = _as_upper(payload.get("data_quality_status"))
    trade_strength = int(_safe_float(payload.get("trade_strength"), 0.0))
    hybrid_move_probability = _safe_float(payload.get("hybrid_move_probability"), 0.0)
    spot = _safe_float(payload.get("spot"), 0.0)
    support_wall = payload.get("support_wall")
    resistance_wall = payload.get("resistance_wall")
    gamma_flip = payload.get("gamma_flip")

    activation_score = 0
    acfg = get_activation_score_policy_config()
    if _is_directional_flow(flow_signal):
        activation_score += acfg.flow_bonus
    if _is_directional_flow(smart_money_flow):
        activation_score += acfg.smart_money_bonus
    if _is_convexity_active(directional_convexity_state):
        activation_score += acfg.convexity_bonus
    if _is_dealer_structure_active(dealer_flow_state):
        activation_score += acfg.dealer_structure_bonus
    if trade_strength >= max(12, int(min_trade_strength * acfg.trade_strength_min_ratio)):
        activation_score += acfg.trade_strength_bonus
    if hybrid_move_probability >= acfg.move_probability_floor:
        activation_score += acfg.move_probability_bonus
    activation_score = int(_clip(activation_score, 0, acfg.activation_cap))

    confirmation_score = 0
    if confirmation_status in {"STRONG_CONFIRMATION", "CONFIRMED"}:
        confirmation_score = acfg.confirmation_score_strong
    elif confirmation_status == "MIXED":
        confirmation_score = acfg.confirmation_score_mixed
    elif confirmation_status == "CONFLICT":
        confirmation_score = acfg.confirmation_score_conflict
    elif confirmation_status == "NO_DIRECTION":
        confirmation_score = acfg.confirmation_score_no_direction

    data_ready_score = acfg.data_ready_strong
    if data_quality_status == "GOOD":
        data_ready_score = acfg.data_ready_good
    elif data_quality_status == "CAUTION":
        data_ready_score = acfg.data_ready_caution
    elif data_quality_status == "WEAK":
        data_ready_score = acfg.data_ready_weak

    maturity_score = int(
        _clip(
            (acfg.maturity_weight_trade_strength * trade_strength)
            + (acfg.maturity_weight_confirmation * confirmation_score)
            + (acfg.maturity_weight_data_ready * data_ready_score),
            0,
            100,
        )
    )

    explainability_confidence = "LOW"
    if data_ready_score >= acfg.high_confidence_data_ready_floor and confirmation_score >= acfg.high_confidence_confirmation_floor:
        explainability_confidence = "HIGH"
    elif data_ready_score >= acfg.medium_confidence_data_ready_floor:
        explainability_confidence = "MEDIUM"

    missing_requirements = []
    missing_confirmations = []
    blocked_by = []
    promotion_requirements = []
    setup_upgrade_conditions = []
    reason_details = []

    no_trade_reason_code = None
    no_trade_reason = None
    watchlist_flag = False
    watchlist_reason = None
    setup_state = "NONE"
    setup_quality = "NONE"
    directional_resolution_needed = False
    likely_next_trigger = None

    if trade_status == "TRADE":
        decision_classification = "TRADE_READY"
        setup_state = "NONE"
        setup_quality = "READY"
    else:
        setup_quality = signal_quality or "VERY_WEAK"
        if trade_status == "DATA_INVALID":
            decision_classification = "DATA_BLOCKED"
            setup_state = "DATA_BLOCKED"
            blocked_by.append("data_quality")
            no_trade_reason_code = "DATA_QUALITY_INSUFFICIENT"
            no_trade_reason = "Trade blocked due to invalid or stale market data"
        elif trade_status == "NO_TRADE" or global_risk_action == "BLOCK":
            decision_classification = "RISK_BLOCKED"
            setup_state = "RISK_BLOCKED"
            blocked_by.append("global_risk")
            if bool(payload.get("event_lockdown_flag")):
                no_trade_reason_code = "EVENT_LOCKDOWN_BLOCK"
                no_trade_reason = "Trade blocked due to event lockdown window"
                blocked_by.append("event_lockdown")
            else:
                no_trade_reason_code = "GLOBAL_RISK_BLOCK"
                no_trade_reason = "Trade blocked by global risk overlay"
            reason_details.extend(payload.get("global_risk_reasons") or [])
        elif trade_status == "BUDGET_FAIL":
            decision_classification = "RISK_BLOCKED"
            setup_state = "RISK_BLOCKED"
            blocked_by.append("budget")
            no_trade_reason_code = "BUDGET_CONSTRAINT_BLOCK"
            no_trade_reason = "Signal passed but budget constraint blocked execution"
        elif direction is None:
            directional_resolution_needed = True
            missing_requirements.append("missing_directional_consensus")
            missing_confirmations.append("direction")

            if activation_score < acfg.dead_inactive_threshold:
                decision_classification = "DEAD_INACTIVE"
                setup_state = "NONE"
                no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
                no_trade_reason = "Market activity is below watchlist threshold"
                watchlist_flag = False
                watchlist_reason = None
            elif directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK":
                decision_classification = "DIRECTIONALLY_AMBIGUOUS"
                setup_state = "DIRECTION_PENDING"
                no_trade_reason_code = "TWO_SIDED_VOLATILITY_WITHOUT_EDGE"
                no_trade_reason = "Two-sided convexity risk without directional edge"
                watchlist_flag = True
                watchlist_reason = "Convexity active but directional asymmetry is unresolved"
                setup_upgrade_conditions.append("move away from gamma flip with aligned flow + dealer bias")
            elif flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"} or dealer_flow_state in {
                "UPSIDE_HEDGING_ACCELERATION",
                "DOWNSIDE_HEDGING_ACCELERATION",
                "PINNING_DOMINANT",
            }:
                decision_classification = "WATCHLIST_SETUP"
                setup_state = "DIRECTION_PENDING"
                no_trade_reason_code = "DIRECTIONAL_CONVICTION_INSUFFICIENT"
                no_trade_reason = "Directional signals are present but conviction threshold is not met"
                watchlist_flag = True
                watchlist_reason = "Setup has structure but direction is not confirmed"
            else:
                decision_classification = "DEAD_INACTIVE"
                setup_state = "NONE"
                no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
                no_trade_reason = "Market is currently inactive with no directional edge"

            if confirmation_status in {"NO_DIRECTION", "CONFLICT"}:
                missing_requirements.append("confirmation_filter_not_met")
                missing_confirmations.append("confirmation")

            if flow_signal == "NEUTRAL_FLOW" and smart_money_flow == "NEUTRAL_FLOW":
                missing_requirements.append("missing_flow_confirmation")

            if dealer_hedging_bias in {"PINNING", "DOWNSIDE_PINNING", "UPSIDE_PINNING"} or dealer_flow_state == "PINNING_DOMINANT":
                missing_requirements.append("pinning_structure_dampens_signal")
                promotion_requirements.append("dealer hedging bias shifts from pinning to acceleration")

            if trade_strength < min_trade_strength:
                missing_requirements.append("insufficient_trade_strength")
                promotion_requirements.append(f"trade_strength >= {int(min_trade_strength)}")
        elif trade_status == "WATCHLIST":
            watchlist_flag = True
            if data_quality_status in {"CAUTION", "WEAK"}:
                decision_classification = "WATCHLIST_CONFIRMATION_PENDING"
                setup_state = "CONFIRMATION_PENDING"
                watchlist_reason = "Data quality and confirmation filters require more clarity"
            elif global_risk_action in {"WATCHLIST", "REDUCE"}:
                decision_classification = "BLOCKED_SETUP"
                setup_state = "RISK_BLOCKED"
                watchlist_reason = "Signal is structurally valid but downgraded by risk overlays"
                blocked_by.append("risk_overlay")
            else:
                decision_classification = "WATCHLIST_SETUP"
                setup_state = "CONFIRMATION_PENDING"
                watchlist_reason = "Directional thesis exists but confirmations are incomplete"

            if confirmation_status in {"CONFLICT", "NO_DIRECTION"}:
                missing_requirements.append("confirmation_filter_not_met")
                missing_confirmations.append("confirmation")

            if trade_strength < min_trade_strength:
                missing_requirements.append("insufficient_trade_strength")
                promotion_requirements.append(f"trade_strength >= {int(min_trade_strength)}")

            if flow_signal == "NEUTRAL_FLOW":
                missing_requirements.append("missing_flow_confirmation")

            no_trade_reason_code = no_trade_reason_code or "FLOW_NOT_CONFIRMED"
            no_trade_reason = no_trade_reason or "Setup is on watchlist pending stronger confirmation"
        else:
            decision_classification = "WATCHLIST_SETUP" if signal_quality in {"WEAK", "MEDIUM"} else "DEAD_INACTIVE"
            setup_state = "CONFIRMATION_PENDING" if decision_classification == "WATCHLIST_SETUP" else "NONE"
            if decision_classification == "WATCHLIST_SETUP":
                watchlist_flag = True
                watchlist_reason = "Setup exists but does not yet meet execution thresholds"
            no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
            no_trade_reason = "Signal did not reach execution threshold"

    if flow_signal == "NEUTRAL_FLOW" and smart_money_flow == "NEUTRAL_FLOW":
        promotion_requirements.append("flow turns directional and aligns with smart-money flow")
        setup_upgrade_conditions.append("directional flow confirmation on both flow lenses")
    elif flow_signal in {"BULLISH_FLOW", "BEARISH_FLOW"} and smart_money_flow not in {"BULLISH_FLOW", "BEARISH_FLOW"}:
        missing_requirements.append("missing_flow_confirmation")
        promotion_requirements.append("smart-money flow confirms directional flow")

    if directional_convexity_state == "TWO_SIDED_VOLATILITY_RISK":
        setup_upgrade_conditions.append("resolve two-sided convexity into one-sided acceleration")

    if dealer_flow_state == "PINNING_DOMINANT":
        setup_upgrade_conditions.append("pinning pressure eases and hedging acceleration emerges")

    if support_wall is not None and resistance_wall is not None:
        if flow_signal == "BEARISH_FLOW":
            likely_next_trigger = f"break below support wall {support_wall}"
            setup_upgrade_conditions.append(f"decisive move below support wall {support_wall}")
        elif flow_signal == "BULLISH_FLOW":
            likely_next_trigger = f"break above resistance wall {resistance_wall}"
            setup_upgrade_conditions.append(f"decisive move above resistance wall {resistance_wall}")

    if likely_next_trigger is None and gamma_flip is not None:
        likely_next_trigger = f"clean move away from gamma flip {gamma_flip} with confirmation"

    if payload.get("expected_move_points") is None:
        missing_requirements.append("option_efficiency_unavailable")
        promotion_requirements.append("option efficiency features become available and supportive")

    if hybrid_move_probability > 0:
        prob_value = hybrid_move_probability
        if prob_value < 0.55:
            missing_requirements.append("move_probability_not_high_enough")
            promotion_requirements.append("hybrid move probability rises above conviction floor")

    if not no_trade_reason_code and trade_status != "TRADE":
        no_trade_reason_code = "SIGNAL_SCORE_BELOW_THRESHOLD"
        no_trade_reason = "Setup has not met the minimum execution bar"

    neutralization = _collect_neutralization_states(payload)

    explainability = {
        "decision_classification": decision_classification,
        "setup_state": setup_state,
        "setup_quality": setup_quality,
        "setup_activation_score": activation_score,
        "setup_maturity_score": maturity_score,
        "explainability_confidence": explainability_confidence,
        "watchlist_flag": bool(watchlist_flag),
        "watchlist_reason": watchlist_reason,
        "no_trade_reason_code": no_trade_reason_code,
        "no_trade_reason": no_trade_reason,
        "no_trade_reason_details": _dedupe_keep_order(reason_details),
        "blocked_by": _dedupe_keep_order(blocked_by),
        "missing_confirmations": _dedupe_keep_order(missing_confirmations),
        "missing_signal_requirements": _dedupe_keep_order(missing_requirements),
        "signal_promotion_requirements": _dedupe_keep_order(promotion_requirements),
        "setup_upgrade_conditions": _dedupe_keep_order(setup_upgrade_conditions),
        "setup_upgrade_path": _dedupe_keep_order(setup_upgrade_conditions + promotion_requirements),
        "likely_next_trigger": likely_next_trigger,
        "watchlist_trigger_levels": {
            "spot": spot,
            "support_wall": support_wall,
            "resistance_wall": resistance_wall,
            "gamma_flip": gamma_flip,
        },
        "directional_resolution_needed": bool(directional_resolution_needed),
        **neutralization,
    }
    return explainability


def _safe_weekday(valuation_time):
    """Return weekday int (0=Mon) from valuation_time, coercing strings via pd.Timestamp."""
    if valuation_time is None:
        import datetime as _dt
        return _dt.datetime.now().weekday()
    if hasattr(valuation_time, "weekday"):
        return valuation_time.weekday()
    try:
        import pandas as _pd
        return _pd.Timestamp(valuation_time).weekday()
    except Exception:
        import datetime as _dt
        return _dt.datetime.now().weekday()


def _estimate_days_to_expiry(option_chain_validation, valuation_time):
    """Estimate calendar days to expiry from chain validation and valuation time."""
    import datetime as _dt

    selected = (
        option_chain_validation.get("selected_expiry")
        if isinstance(option_chain_validation, dict)
        else None
    )
    if selected is None or valuation_time is None:
        return None
    try:
        import pandas as _pd
        expiry_ts = _pd.Timestamp(selected)
        val_ts = _pd.Timestamp(valuation_time)
        delta = (expiry_ts - val_ts).total_seconds() / 86400.0
        return max(delta, 0.0)
    except Exception:
        return None


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
    """
    Purpose:
        Assemble the final trade or no-trade payload for one market snapshot.
    
    Context:
        This is the engine's top-level orchestration entry point. It sits after data normalization and analytics extraction, then layers probability estimates, macro context, risk overlays, strike selection, and position sizing into a single payload that live runtime, replay tools, and research logging can all consume.
    
    Inputs:
        symbol (Any): Underlying symbol or index identifier.
        spot (Any): Current underlying spot price.
        option_chain (Any): Current option-chain snapshot.
        previous_chain (Any): Previous option-chain snapshot used for change-sensitive features such as flow and open-interest shifts.
        day_high (Any): Session high used for intraday range context.
        day_low (Any): Session low used for intraday range context.
        day_open (Any): Session open used for intraday context and early-session fallback logic.
        prev_close (Any): Previous session close used as a reference anchor.
        lookback_avg_range_pct (Any): Historical average range percentage used to normalize today's move.
        spot_validation (Any): Validation summary for the spot snapshot.
        option_chain_validation (Any): Validation summary for the option-chain snapshot.
        apply_budget_constraint (Any): Whether capital-budget rules should be enforced during trade construction.
        requested_lots (Any): Requested lot count before any optimizer or budget cap adjusts size.
        lot_size (Any): Contract lot size used when translating premium into capital required.
        max_capital (Any): Maximum capital budget allowed for the trade.
        backtest_mode (Any): Whether the snapshot is being evaluated in a backtest or replay context.
        macro_event_state (Any): Scheduled-event state produced by the macro layer.
        macro_news_state (Any): Headline-driven macro state produced by the news layer.
        global_risk_state (Any): Precomputed cross-asset risk state, when already available.
        holding_profile (Any): Holding intent used by overnight-sensitive overlays.
        valuation_time (Any): Timestamp used when normalizing expiries and Greeks.
        target_profit_percent (Any): Target-profit percentage passed into the exit model.
        stop_loss_percent (Any): Stop-loss percentage passed into the exit model.
    
    Returns:
        dict | None: Final trade or no-trade payload. Returns `None` only when the option chain is unusable at the very first gate.
    
    Notes:
        The returned payload doubles as the live engine contract and the structured record captured by evaluation and tuning workflows, so the function keeps diagnostics and decision-state fields explicit.
    """
    if option_chain is None or option_chain.empty:
        return None

    # Normalize provider-specific column names and enrich missing Greeks once so
    # every downstream model works off a consistent option-chain schema.
    df = normalize_option_chain(option_chain, spot=spot, valuation_time=valuation_time)
    prev_df = (
        normalize_option_chain(previous_chain, spot=spot, valuation_time=valuation_time)
        if previous_chain is not None else None
    )
    market_state = _collect_market_state(df, spot, symbol=symbol, prev_df=prev_df)

    # Build global context for v2 ML model features (available before probability).
    _grs = global_risk_state if isinstance(global_risk_state, dict) else {}
    _grf = _grs.get("global_risk_features", {}) if isinstance(_grs.get("global_risk_features"), dict) else {}
    _mes = macro_event_state if isinstance(macro_event_state, dict) else {}
    _global_ctx = {
        "india_vix_level": _grf.get("india_vix_level"),
        "india_vix_change_24h": _grf.get("india_vix_change_24h"),
        "oil_shock_score": _grf.get("oil_shock_score"),
        "commodity_risk_score": _grf.get("commodity_risk_score"),
        "volatility_shock_score": _grf.get("volatility_shock_score"),
        "macro_event_risk_score": _mes.get("macro_event_risk_score", 0.0),
        "macro_regime": _mes.get("macro_regime", "NO_EVENT"),
        "days_to_expiry": _estimate_days_to_expiry(option_chain_validation, valuation_time),
        "weekday": _safe_weekday(valuation_time),
    }
    
    # Inject volatility shock score into market_state for regime-aware direction weighting
    # This allows the direction decision to be sensitive to elevated volatility environments
    market_state["volatility_shock_score"] = _grf.get("volatility_shock_score", 0.0)

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
        global_context=_global_ctx,
    )
    intraday_range_pct = probability_state["components"]["intraday_range_pct"]

    # Keep the analytics subset explicit because these values feed both
    # confidence checks and the final audit payload consumed by research tools.
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

    # Scheduled events and headline overlays are scored separately so operators
    # can see whether a downgrade came from the calendar, the news tape, or both.
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

    # Each overlay returns both diagnostics and a score contribution so the
    # engine can explain not just the decision, but why the decision changed.
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
    india_vix_level = global_risk_features.get("india_vix_level")
    india_vix_change_24h = global_risk_features.get("india_vix_change_24h")
    option_efficiency_state = {}
    option_efficiency_trade_modifiers = derive_option_efficiency_trade_modifiers(option_efficiency_state)
    option_efficiency_adjustment_score = option_efficiency_trade_modifiers["option_efficiency_adjustment_score"]
    scoring_breakdown["option_efficiency_adjustment_score"] = option_efficiency_adjustment_score

    # Trade strength is accumulated in layers: base directional edge, then
    # confirmation/macro adjustments, then stateful risk overlays.
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
    runtime_thresholds = get_trade_runtime_thresholds()
    min_trade_strength = (
        BACKTEST_MIN_TRADE_STRENGTH
        if backtest_mode
        else runtime_thresholds["min_trade_strength"]
    )
    min_composite_score = int(_safe_float(runtime_thresholds.get("min_composite_score"), 55.0))

    regime_thresholds = _resolve_regime_thresholds(
        runtime_thresholds=runtime_thresholds,
        base_min_trade_strength=min_trade_strength,
        base_min_composite_score=min_composite_score,
        market_state=market_state,
    )
    min_trade_strength = regime_thresholds["effective_min_trade_strength"]
    min_composite_score = regime_thresholds["effective_min_composite_score"]

    provider_health = option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else {}
    provider_health = provider_health if isinstance(provider_health, dict) else {}
    provider_health_summary = _as_upper(provider_health.get("summary_status"))

    at_flip_penalty_applied = 0
    at_flip_size_cap = 1.0
    at_flip_toxic_context = False
    if _as_upper(market_state["spot_vs_flip"]) == "AT_FLIP":
        at_flip_penalty_applied = int(_safe_float(runtime_thresholds.get("at_flip_trade_strength_penalty"), 8.0))
        adjusted_trade_strength = int(_clip(adjusted_trade_strength - at_flip_penalty_applied, 0, 100))
        dealer_position_upper = _as_upper(market_state.get("dealer_pos"))
        at_flip_toxic_context = (
            _as_upper(market_state.get("gamma_regime")) == "POSITIVE_GAMMA"
            and ("SHORT" in dealer_position_upper)
            and ("GAMMA" in dealer_position_upper)
        )
        at_flip_size_cap = _safe_float(
            runtime_thresholds.get("at_flip_toxic_size_cap" if at_flip_toxic_context else "at_flip_size_cap"),
            0.50 if at_flip_toxic_context else 0.75,
        )

    gamma_vol_acceleration_score_normalized = _normalize_gamma_vol_score(
        gamma_vol_trade_modifiers["gamma_vol_acceleration_score"],
        runtime_thresholds.get("gamma_vol_normalization_scale"),
    )
    structural_imbalance_audit = _compute_structural_imbalance_audit(
        market_state=market_state,
        direction=direction,
    )

    scoring_breakdown["base_trade_strength"] = trade_strength
    scoring_breakdown["at_flip_trade_strength_penalty"] = -at_flip_penalty_applied
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

    # This payload is intentionally verbose because it serves three audiences:
    # the live trader, the risk overlays, and the offline evaluation dataset.
    base_payload = {
        "symbol": symbol,
        "spot": round(spot, 2),
        "ranked_strike_candidates": [],
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
        "large_move_probability": probability_state["hybrid_move_probability"],  # legacy alias — use hybrid_move_probability
        "move_probability_components": probability_state["components"],
        "spot_validation": spot_validation,
        "option_chain_validation": option_chain_validation,
        "provider_health": option_chain_validation.get("provider_health") if isinstance(option_chain_validation, dict) else None,
        "provider_health_summary": provider_health_summary,
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
        "macro_news_volatility_shock_score": macro_news_adjustments["volatility_shock_score"],
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
        "gamma_vol_acceleration_score_normalized": gamma_vol_acceleration_score_normalized,
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
        "call_put_imbalance_score": structural_imbalance_audit["call_put_imbalance_score"],
        "call_put_imbalance_abs": structural_imbalance_audit["call_put_imbalance_abs"],
        "call_put_alignment": structural_imbalance_audit["call_put_alignment"],
        "call_put_imbalance_severity": structural_imbalance_audit["call_put_imbalance_severity"],
        "gamma_vol_features": gamma_vol_state.get("gamma_vol_features") if isinstance(gamma_vol_state, dict) else {},
        "gamma_vol_diagnostics": gamma_vol_state.get("gamma_vol_diagnostics") if isinstance(gamma_vol_state, dict) else {},
        "dealer_pressure_features": dealer_pressure_state.get("dealer_pressure_features") if isinstance(dealer_pressure_state, dict) else {},
        "dealer_pressure_diagnostics": dealer_pressure_state.get("dealer_pressure_diagnostics") if isinstance(dealer_pressure_state, dict) else {},
        "budget_constraint_applied": apply_budget_constraint,
        "lot_size": lot_size,
        "requested_lots": requested_lots,
        "max_capital_per_trade": max_capital,
        "at_flip_trade_strength_penalty": at_flip_penalty_applied,
        "at_flip_size_cap": round(at_flip_size_cap, 2),
        "at_flip_toxic_context": at_flip_toxic_context,
        "regime_toxic_context": regime_thresholds["toxic_context"],
        "regime_threshold_adjustments": regime_thresholds["adjustments"],
        "min_trade_strength_threshold": min_trade_strength,
        "min_composite_score_threshold": min_composite_score,
        "runtime_composite_score": None,
        "backtest_mode": backtest_mode,
    }

    # The global risk layer is the final pre-trade gate. It can block, downgrade
    # to watchlist, or cap size even when the analytics stack is directionally strong.
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
            "india_vix_level": india_vix_level,
            "india_vix_change_24h": india_vix_change_24h,
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
        """
        Purpose:
            Finalize the response payload with execution status and regime
            metadata.

        Context:
            Used at every exit path in `generate_trade` so blocked trades,
            watchlist outcomes, and executable trades all share the same output
            contract.

        Inputs:
            payload (Any): Base response payload that already contains shared diagnostics for the current snapshot.
            trade_status (Any): Final trade-status label such as `OK`, `NO_TRADE`, or a validation-specific code.
            message (Any): Human-readable explanation attached to the final payload.

        Returns:
            dict: Final response payload ready for runtime consumption and
            signal-evaluation logging.

        Notes:
            Centralizing this bookkeeping keeps decision branches focused on why
            the trade changed state rather than how the payload is shaped.
        """
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
        explainability = _build_decision_explainability(
            payload,
            trade_status=trade_status,
            min_trade_strength=min_trade_strength,
        )
        payload.update(explainability)
        payload["explainability"] = explainability

        confidence = compute_signal_confidence(payload)
        payload["signal_confidence_score"] = confidence["confidence_score"]
        payload["signal_confidence_level"] = confidence["confidence_level"]

        return attach_trade_views(payload)

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

    # Strike ranking only happens after the directional thesis survives macro
    # and risk gating. That keeps expensive contract-specific work off the
    # path for obvious no-trade scenarios.
    if direction is not None:
        def option_efficiency_candidate_hook(row):
            """
            Purpose:
                Score a strike candidate with contract-level option-efficiency
                heuristics.

            Context:
                Used by strike ranking after the engine has already chosen a
                direction. This lets contract selection account for payoff
                geometry, expected move, and overlay state without mutating the
                base strike-ranking model.

            Inputs:
                row (Any): Candidate option row under evaluation.

            Returns:
                dict: Optional score adjustment plus option-efficiency
                diagnostics for the candidate strike.

            Notes:
                The hook is nested because it depends on the fully assembled
                signal state for the current snapshot.
            """
            return score_option_efficiency_candidate(
                row,
                spot=spot,
                direction=direction,
                atm_iv=market_state["atm_iv"],
                india_vix_level=india_vix_level,
                india_vix_change_24h=india_vix_change_24h,
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
            gamma_regime=market_state["gamma_regime"],
            spot_vs_flip=market_state["spot_vs_flip"],
            dealer_hedging_bias=market_state["hedging_bias"],
            gamma_flip_distance_pct=probability_state["components"].get("gamma_flip_distance_pct"),
            atm_iv=market_state["atm_iv"],
            days_to_expiry=_estimate_days_to_expiry(option_chain_validation, valuation_time),
            vol_surface_regime=market_state["surface_regime"],
            volatility_shock_score=market_state.get("volatility_shock_score", 0.0),
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

    # --- Time-based exit recommendation ------------------------------------
    _gr_features = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    _mtc = _safe_float(_gr_features.get("minutes_to_close"), None)
    _mso = round(375.0 - _mtc, 2) if _mtc is not None else None
    exit_timing = compute_exit_timing(
        trade_strength=adjusted_trade_strength,
        gamma_regime=market_state["gamma_regime"],
        vol_regime=market_state["vol_regime"],
        minutes_since_open=_mso,
        minutes_to_close=_mtc,
    )
    hold_cap_minutes = int(_safe_float(runtime_thresholds.get("max_intraday_hold_minutes"), 90.0))
    if regime_thresholds["toxic_context"] or at_flip_toxic_context:
        hold_cap_minutes = min(
            hold_cap_minutes,
            int(_safe_float(runtime_thresholds.get("toxic_regime_hold_cap_minutes"), 60.0)),
        )
    recommended_hold_minutes = min(int(exit_timing["recommended_hold_minutes"]), hold_cap_minutes)
    max_hold_minutes = min(int(exit_timing["max_hold_minutes"]), hold_cap_minutes)
    exit_timing_reasons = list(exit_timing.get("exit_timing_reasons") or [])
    if recommended_hold_minutes < int(exit_timing["recommended_hold_minutes"]) or max_hold_minutes < int(exit_timing["max_hold_minutes"]):
        exit_timing_reasons.append(f"hard_hold_cap_applied_{hold_cap_minutes}m")

    option_row_dict = option_row.iloc[0].to_dict()

    # Once a specific contract is chosen, recompute option-efficiency metrics
    # with contract-level Greeks, expiry, and payoff geometry.
    option_efficiency_state = build_option_efficiency_state(
        spot=spot,
        atm_iv=market_state["atm_iv"],
        india_vix_level=india_vix_level,
        india_vix_change_24h=india_vix_change_24h,
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
        "recommended_hold_minutes": recommended_hold_minutes,
        "max_hold_minutes": max_hold_minutes,
        "exit_urgency": exit_timing["exit_urgency"],
        "exit_timing_reasons": exit_timing_reasons,
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
    # --- Overnight hold consolidation -----------------------------------------
    _overnight_layers = [
        ("global_risk", global_risk_trade_modifiers),
        ("gamma_vol", gamma_vol_trade_modifiers),
        ("dealer_pressure", dealer_pressure_trade_modifiers),
        ("option_efficiency", option_efficiency_trade_modifiers),
    ]
    _overnight_allowed = True
    _overnight_reason = option_efficiency_trade_modifiers["overnight_hold_reason"]
    for _layer_name, _layer_mods in _overnight_layers:
        if not _layer_mods["overnight_hold_allowed"]:
            _overnight_allowed = False
            _overnight_reason = _layer_mods["overnight_hold_reason"]
            break

    _penalty_keys = {
        "global_risk": "overnight_risk_penalty",
        "gamma_vol": "overnight_convexity_penalty",
        "dealer_pressure": "overnight_dealer_pressure_penalty",
        "option_efficiency": "overnight_option_efficiency_penalty",
    }
    _overnight_penalty = sum(
        int(_safe_float(_layer_mods.get(_penalty_keys[_layer_name]), 0.0))
        for _layer_name, _layer_mods in _overnight_layers
    )

    base_payload.update(
        {
            "overnight_hold_allowed": _overnight_allowed,
            "overnight_hold_reason": _overnight_reason,
            "overnight_risk_penalty": _overnight_penalty,
            "overnight_trade_block": not _overnight_allowed,
        }
    )

    # Budget controls are applied after the signal is fully validated so they
    # affect position size, not the informational content of the signal itself.
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

    # Macro and global-risk size caps can reduce exposure without vetoing the
    # idea entirely, which is useful for elevated-risk but still actionable setups.
    risk_size_cap = min(_safe_float(global_risk["global_risk_size_cap"], 1.0), _safe_float(at_flip_size_cap, 1.0))
    base_payload["effective_size_cap"] = round(risk_size_cap, 2)
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

    runtime_composite_score = _compute_runtime_composite_score(
        trade_strength=adjusted_trade_strength,
        hybrid_move_probability=probability_state["hybrid_move_probability"],
        confirmation_status=confirmation["status"],
        data_quality_status=data_quality["status"],
        gamma_vol_acceleration_score_normalized=gamma_vol_acceleration_score_normalized,
    )
    base_payload["runtime_composite_score"] = runtime_composite_score

    if runtime_composite_score < min_composite_score:
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Runtime composite score {runtime_composite_score} below threshold {min_composite_score}",
        )

    caution_blocks_trade = bool(int(_safe_float(runtime_thresholds.get("provider_health_caution_blocks_trade"), 1.0)))
    if caution_blocks_trade and provider_health_summary in {"CAUTION", "WEAK"}:
        return _finalize(
            base_payload,
            "WATCHLIST",
            f"Provider health {provider_health_summary} blocks TRADE and routes to WATCHLIST",
        )

    if apply_budget_constraint:
        base_payload["message"] = "Tradable signal generated with budget optimization"
    else:
        base_payload["message"] = "Tradable signal generated"

    return _finalize(
        base_payload,
        "TRADE",
        "Tradable signal generated with budget optimization" if apply_budget_constraint else "Tradable signal generated",
    )
