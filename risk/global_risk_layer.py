"""
Module: global_risk_layer.py

Purpose:
    Score cross-asset and volatility stress signals before they are allowed to modify a trade decision.

Role in the System:
    Part of the risk-overlay layer that can downgrade, suppress, or annotate otherwise valid trades.

Key Outputs:
    A structured global-risk assessment containing feature diagnostics, overlay scores, and gating flags.

Downstream Usage:
    Consumed by the signal engine, shadow evaluations, and research diagnostics.
"""

from __future__ import annotations

from config.global_risk_policy import get_global_risk_policy_config
from risk.global_risk_features import build_global_risk_features
from risk.global_risk_regime import classify_global_risk_state
from utils.numerics import clip as _clip, safe_float as _safe_float  # noqa: F401


def _event_name(active_event_name=None, next_event_name=None):
    """
    Purpose:
        Resolve the event name used in global-risk fallback messaging.
    
    Context:
        Internal helper in the risk overlay. When the global-risk layer blocks or downgrades a trade because of scheduled events, the message path needs a single human-readable event label even if only partial metadata is available.
    
    Inputs:
        active_event_name (Any): Currently active scheduled event, when one exists.
        next_event_name (Any): Next scheduled event, when one exists.
    
    Returns:
        str: Event label used in operator-facing or research-facing diagnostics.
    
    Notes:
        This keeps fallback messaging deterministic even when upstream event metadata is incomplete.
    """
    return active_event_name or next_event_name or "scheduled macro event"


def build_global_risk_state(
    *,
    macro_event_state=None,
    macro_news_state=None,
    global_market_snapshot=None,
    holding_profile: str = "AUTO",
    as_of=None,
):
    """
    Purpose:
        Assemble the global-risk state from macro-event, news, and cross-asset inputs.
    
    Context:
        This function belongs to the risk-overlay layer and runs before the final trade decision is emitted. It converts macro-event state, news adjustments, and global-market features into a normalized risk-state payload that later overlays can consume.
    
    Inputs:
        macro_event_state (Any): Scheduled-event state produced by the macro layer.
        macro_news_state (Any): Headline-driven macro state produced by the news layer.
        global_market_snapshot (Any): Cross-asset market snapshot used by global-risk feature builders.
        holding_profile (str): Holding intent that determines whether overnight logic should matter.
        as_of (Any): Timestamp representing when the global snapshot should be interpreted.
    
    Returns:
        dict: Serialized global-risk state containing feature diagnostics, regime labels, and overnight-risk fields.
    
    Notes:
        The output is normalized into a dictionary so the same state contract can flow through live runtime, replay, shadow evaluation, and research logging.
    """
    features = build_global_risk_features(
        macro_event_state=macro_event_state,
        macro_news_state=macro_news_state,
        global_market_snapshot=global_market_snapshot,
        holding_profile=holding_profile,
        as_of=as_of,
    )
    return classify_global_risk_state(features).to_dict()


def _fallback_global_risk_state(
    *,
    event_window_status,
    macro_event_risk_score,
    event_lockdown_flag,
    next_event_name=None,
    active_event_name=None,
    holding_profile="AUTO",
):
    """
    Purpose:
        Construct a conservative fallback global-risk state when full market data is unavailable.
    
    Context:
        Internal helper used when the engine has event-window information but does not have a complete cross-asset risk snapshot. It prevents missing market data from silently bypassing obvious scheduled-event risk.
    
    Inputs:
        event_window_status (Any): Scheduled-event window label such as pre-event watch or live event.
        macro_event_risk_score (Any): Macro-event risk score available from the event layer.
        event_lockdown_flag (Any): Whether scheduled-event policy requires a hard lockdown.
        next_event_name (Any): Next scheduled event, when one exists.
        active_event_name (Any): Currently active scheduled event, when one exists.
        holding_profile (Any): Holding intent used to populate overnight context fields.
    
    Returns:
        dict: Conservative fallback state that preserves the same schema as the full global-risk path.
    
    Notes:
        The fallback intentionally biases toward caution so event-driven risk is still surfaced even if cross-asset feeds are delayed or unavailable.
    """
    holding_profile_norm = str(holding_profile or "AUTO").upper().strip() or "AUTO"
    overnight_relevant = holding_profile_norm in {"AUTO", "OVERNIGHT", "SWING", "POSITIONAL"}

    state = "EVENT_LOCKDOWN" if event_lockdown_flag else "GLOBAL_NEUTRAL"
    score = 100 if event_lockdown_flag else int(_clip(_safe_float(macro_event_risk_score, 0.0) * 0.5, 0.0, 100.0))
    reasons = ["event_lockdown"] if event_lockdown_flag else ["global_risk_neutral_fallback"]

    return {
        "global_risk_state": state,
        "global_risk_score": score,
        "overnight_gap_risk_score": score if event_window_status in {"PRE_EVENT_WATCH", "PRE_EVENT_LOCKDOWN", "LIVE_EVENT"} else 0,
        "volatility_expansion_risk_score": 0,
        "overnight_hold_allowed": not event_lockdown_flag,
        "overnight_hold_reason": "event_lockdown_block" if event_lockdown_flag else "overnight_risk_contained",
        "overnight_risk_penalty": 10 if event_lockdown_flag else 0,
        "global_risk_adjustment_score": -6 if event_lockdown_flag else 0,
        "global_risk_veto": bool(event_lockdown_flag),
        "global_risk_position_size_multiplier": 0.0 if event_lockdown_flag else 1.0,
        "neutral_fallback": True,
        "holding_context": {
            "holding_profile": holding_profile_norm,
            "overnight_relevant": overnight_relevant,
            "market_session": "UNKNOWN",
            "minutes_to_close": None,
        },
        "global_risk_reasons": reasons,
        "global_risk_features": {
            "event_window_status": event_window_status,
            "macro_event_risk_score": int(_safe_float(macro_event_risk_score, 0.0)),
            "event_lockdown_flag": bool(event_lockdown_flag),
            "next_event_name": next_event_name,
            "active_event_name": active_event_name,
            "oil_shock_score": 0.0,
            "gold_risk_score": 0.0,
            "copper_growth_signal": 0.0,
            "commodity_risk_score": 0.0,
            "volatility_shock_score": 0.0,
            "us_equity_risk_score": 0.0,
            "rates_shock_score": 0.0,
            "currency_shock_score": 0.0,
            "risk_off_intensity": 0.0,
            "volatility_compression_score": 0.0,
            "volatility_explosion_probability": 0.0,
        },
        "global_risk_diagnostics": {
            "event_window_status": event_window_status,
            "fallback": True,
        },
    }


def _result(*, state, score, level, action, size_cap, reasons, trade_status=None, message=None, portfolio_guard=None):
    """
    Purpose:
        Assemble the normalized global-risk result payload returned to the signal engine.
    
    Context:
        Internal helper in the risk overlay. Multiple decision branches can block, watchlist, or pass a trade, and this helper keeps the resulting payload shape identical across those branches.
    
    Inputs:
        state (Any): Base global-risk state dictionary.
        score (Any): Composite overlay risk score on a 0-100 style scale.
        level (Any): High-level severity label such as low, medium, or high.
        action (Any): Overlay action label such as pass, watchlist, or block.
        size_cap (Any): Position-size multiplier cap allowed by the overlay.
        reasons (Any): Ordered list of reasons explaining the overlay decision.
        trade_status (Any): Trade-status label to expose back to the signal engine, when one is needed.
        message (Any): Human-readable explanation attached to the risk decision.
    
    Returns:
        dict: Normalized risk-layer payload containing scores, action flags, diagnostics, and trade-status metadata.
    
    Notes:
        Centralizing this payload assembly keeps downstream code focused on the reason for the decision instead of the mechanics of shaping the response.
    """
    state = state if isinstance(state, dict) else {}
    portfolio_guard = portfolio_guard if isinstance(portfolio_guard, dict) else {
        "enabled": False,
        "verdict": "UNAVAILABLE",
        "reason": "portfolio_context_unavailable",
        "recent_signal_count": 0,
        "same_direction_count": 0,
        "same_direction_share": 0.0,
        "heat_score": 0,
        "heat_label": "COOL",
        "regime_flags": [],
    }
    return {
        "global_risk_state": state.get("global_risk_state", "GLOBAL_NEUTRAL"),
        "global_risk_score": int(_clip(score, 0, 100)),
        "overnight_gap_risk_score": int(_clip(_safe_float(state.get("overnight_gap_risk_score"), 0.0), 0, 100)),
        "volatility_expansion_risk_score": int(_clip(_safe_float(state.get("volatility_expansion_risk_score"), 0.0), 0, 100)),
        "overnight_hold_allowed": bool(state.get("overnight_hold_allowed", True)),
        "overnight_hold_reason": str(state.get("overnight_hold_reason", "overnight_risk_contained")),
        "overnight_risk_penalty": int(_clip(_safe_float(state.get("overnight_risk_penalty"), 0.0), 0, 10)),
        "global_risk_adjustment_score": int(_safe_float(state.get("global_risk_adjustment_score"), 0.0)),
        "global_risk_level": level,
        "global_risk_action": action,
        "global_risk_size_cap": round(_clip(size_cap, 0.0, 1.0), 2),
        "global_risk_reasons": reasons,
        "global_risk_features": state.get("global_risk_features", {}),
        "global_risk_diagnostics": state.get("global_risk_diagnostics", {}),
        "portfolio_concentration_guard": portfolio_guard,
        "risk_trade_status": trade_status,
        "risk_message": message,
    }


def _evaluate_portfolio_concentration_guard(*, portfolio_context, cfg):
    """Assess whether the recent options idea book is too concentrated in one direction."""
    enabled = bool(int(_safe_float(getattr(cfg, "enable_portfolio_concentration_guard", 1), 1.0)))
    context = portfolio_context if isinstance(portfolio_context, dict) else {}
    direction = str(context.get("direction") or "").upper().strip()
    recent_signal_count = max(int(_safe_float(context.get("recent_signal_count"), 0.0)), 0)
    same_direction_count = max(int(_safe_float(context.get("same_direction_count"), 0.0)), 0)
    same_direction_share = float(_clip(_safe_float(context.get("same_direction_share"), 0.0), 0.0, 1.0))
    avg_close_bps = _safe_float(context.get("same_direction_avg_close_bps"), None)
    avg_tradeability = _safe_float(context.get("same_direction_avg_tradeability_score"), None)
    gamma_regime = str(context.get("gamma_regime") or "").upper().strip()
    vol_regime = str(context.get("vol_regime") or context.get("volatility_regime") or "").upper().strip()
    macro_regime = str(context.get("macro_regime") or "").upper().strip()
    provider_health_summary = str(context.get("provider_health_summary") or "").upper().strip()
    data_quality_status = str(context.get("data_quality_status") or "").upper().strip()

    guard = {
        "enabled": enabled,
        "verdict": "PASS",
        "action": "ALLOW",
        "reason": "portfolio_concentration_within_limits",
        "recent_signal_count": recent_signal_count,
        "same_direction_count": same_direction_count,
        "same_direction_share": round(same_direction_share, 4),
        "same_direction_avg_close_bps": round(float(avg_close_bps), 2) if avg_close_bps is not None else None,
        "same_direction_avg_tradeability_score": round(float(avg_tradeability), 2) if avg_tradeability is not None else None,
        "direction": direction,
        "size_cap": 1.0,
        "reasons": [],
        "heat_score": 0,
        "heat_label": "COOL",
        "regime_flags": [],
        "regime_penalty": 0,
    }

    if not enabled:
        guard.update({"verdict": "DISABLED", "reason": "guard_disabled"})
        return guard

    min_recent = max(int(_safe_float(getattr(cfg, "portfolio_concentration_min_recent_signals", 4), 4.0)), 2)
    if direction not in {"CALL", "PUT"}:
        guard.update({"verdict": "UNAVAILABLE", "reason": "direction_unavailable"})
        return guard
    if recent_signal_count < min_recent or same_direction_count < 2:
        guard.update({"verdict": "UNAVAILABLE", "reason": "insufficient_recent_book_context"})
        return guard

    soft_count = max(int(_safe_float(getattr(cfg, "portfolio_concentration_soft_same_direction_count", 3), 3.0)), 2)
    hard_count = max(int(_safe_float(getattr(cfg, "portfolio_concentration_hard_same_direction_count", 4), 4.0)), soft_count)
    soft_share = float(_clip(_safe_float(getattr(cfg, "portfolio_concentration_soft_same_direction_share", 0.70), 0.70), 0.0, 1.0))
    hard_share = float(_clip(_safe_float(getattr(cfg, "portfolio_concentration_hard_same_direction_share", 0.80), 0.80), soft_share, 1.0))
    weak_close_threshold = _safe_float(getattr(cfg, "portfolio_concentration_weak_close_bps_threshold", 0.0), 0.0)
    weak_tradeability_threshold = _safe_float(getattr(cfg, "portfolio_concentration_weak_tradeability_threshold", 55.0), 55.0)

    weak_recent_edge = (
        (avg_close_bps is not None and avg_close_bps <= weak_close_threshold)
        or (avg_tradeability is not None and avg_tradeability < weak_tradeability_threshold)
    )

    regime_flags = []
    regime_penalty = 0
    if gamma_regime in {"NEGATIVE_GAMMA", "SHORT_GAMMA"}:
        regime_flags.append("negative_gamma")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_negative_gamma_penalty", 12), 12.0))
    if vol_regime in {"VOL_EXPANSION", "HIGH_VOL", "VOL_SHOCK", "PANIC_VOL", "EXTREME_VOL"}:
        regime_flags.append("vol_expansion")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_vol_expansion_penalty", 10), 10.0))
    if macro_regime in {"RISK_OFF", "EVENT_RISK", "STRESS"}:
        regime_flags.append("risk_off")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_risk_off_penalty", 8), 8.0))
    if provider_health_summary == "CAUTION":
        regime_flags.append("provider_caution")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_provider_caution_penalty", 5), 5.0))
    elif provider_health_summary in {"WEAK", "BLOCK"}:
        regime_flags.append("provider_weak")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_provider_weak_penalty", 10), 10.0))
    if data_quality_status == "CAUTION":
        regime_flags.append("data_caution")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_data_caution_penalty", 5), 5.0))
    elif data_quality_status in {"WEAK", "BAD"}:
        regime_flags.append("data_weak")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_data_weak_penalty", 10), 10.0))
    if weak_recent_edge:
        regime_flags.append("recent_edge_weak")
        regime_penalty += int(_safe_float(getattr(cfg, "portfolio_heat_edge_weak_penalty", 12), 12.0))

    base_heat = (same_direction_share * 55.0) + (min(same_direction_count, 5) * 6.0) + (min(recent_signal_count, 6) * 2.0)
    heat_score = int(round(_clip(base_heat + regime_penalty, 0.0, 100.0)))
    warm_threshold = int(_safe_float(getattr(cfg, "portfolio_heat_warm_threshold", 40), 40.0))
    hot_threshold = int(_safe_float(getattr(cfg, "portfolio_heat_hot_threshold", 60), 60.0))
    critical_threshold = int(_safe_float(getattr(cfg, "portfolio_heat_critical_threshold", 80), 80.0))

    if heat_score >= critical_threshold:
        heat_label = "CRITICAL"
    elif heat_score >= hot_threshold:
        heat_label = "HOT"
    elif heat_score >= warm_threshold:
        heat_label = "WARM"
    else:
        heat_label = "COOL"

    guard.update(
        {
            "heat_score": heat_score,
            "heat_label": heat_label,
            "regime_flags": regime_flags,
            "regime_penalty": regime_penalty,
        }
    )

    reasons = []
    if same_direction_count >= soft_count or same_direction_share >= soft_share:
        reasons.append("portfolio_same_way_concentration")
    if regime_penalty > 0:
        reasons.append("portfolio_regime_heat")
    if weak_recent_edge:
        reasons.append("portfolio_recent_edge_weak")
    reasons = list(dict.fromkeys(reasons))

    if same_direction_count >= hard_count and (same_direction_share >= hard_share or weak_recent_edge or heat_score >= critical_threshold):
        critical_size_cap = float(_clip(_safe_float(getattr(cfg, "portfolio_heat_critical_size_cap", 0.35), 0.35), 0.0, 1.0))
        watchlist_size_cap = float(_clip(_safe_float(getattr(cfg, "portfolio_concentration_watchlist_size_cap", 0.40), 0.40), 0.0, 1.0))
        guard.update(
            {
                "verdict": "WATCHLIST",
                "action": "WATCHLIST",
                "reason": "Trade downgraded due to a concentrated same-way options book",
                "size_cap": min(watchlist_size_cap, critical_size_cap if heat_score >= critical_threshold else watchlist_size_cap),
                "reasons": reasons or ["portfolio_same_way_concentration"],
            }
        )
        return guard

    hot_size_cap = float(_clip(_safe_float(getattr(cfg, "portfolio_heat_hot_size_cap", 0.55), 0.55), 0.0, 1.0))
    reduce_size_cap = float(_clip(_safe_float(getattr(cfg, "portfolio_concentration_reduce_size_cap", 0.65), 0.65), 0.0, 1.0))
    reduce_due_to_heat = same_direction_count >= max(2, soft_count - 1) and heat_score >= hot_threshold
    reduce_due_to_crowding = same_direction_count >= soft_count and same_direction_share >= soft_share
    if reduce_due_to_crowding or reduce_due_to_heat:
        guard.update(
            {
                "verdict": "REDUCE",
                "action": "REDUCE",
                "reason": "Position size reduced due to elevated same-way options concentration",
                "size_cap": min(reduce_size_cap, hot_size_cap if heat_score >= hot_threshold else reduce_size_cap),
                "reasons": reasons or ["portfolio_same_way_concentration"],
            }
        )

    return guard


def evaluate_global_risk_layer(
    *,
    data_quality,
    confirmation,
    adjusted_trade_strength,
    min_trade_strength,
    event_window_status,
    macro_event_risk_score,
    event_lockdown_flag,
    next_event_name=None,
    active_event_name=None,
    macro_news_adjustments=None,
    global_risk_state=None,
    holding_profile="AUTO",
    portfolio_context=None,
):
    """
    Purpose:
        Apply the global-risk overlay to the current trade candidate and return the resulting action.
    
    Context:
        This is the final decision layer for cross-asset and event-driven risk. It combines data quality, confirmation state, scheduled-event risk, news adjustments, and the prebuilt global-risk state to decide whether the trade should pass, shrink, move to watchlist, or be blocked.
    
    Inputs:
        data_quality (Any): Data-quality payload produced earlier in signal assembly.
        confirmation (Any): Confirmation-filter payload produced by the signal engine.
        adjusted_trade_strength (Any): Trade-strength score after earlier overlay adjustments.
        min_trade_strength (Any): Minimum trade-strength threshold required for execution.
        event_window_status (Any): Scheduled-event window label from the macro layer.
        macro_event_risk_score (Any): Macro-event risk score from the macro layer.
        event_lockdown_flag (Any): Whether scheduled-event policy requires a hard lockdown.
        next_event_name (Any): Next scheduled event, when one exists.
        active_event_name (Any): Currently active scheduled event, when one exists.
        macro_news_adjustments (Any): News-derived macro adjustment payload already computed by the engine.
        global_risk_state (Any): Precomputed global-risk state, when available.
        holding_profile (Any): Holding intent used for overnight-sensitive risk decisions.
    
    Returns:
        dict: Overlay decision payload containing risk scores, action labels, size caps, and any trade-status downgrade or veto.
    
    Notes:
        The score is intentionally composite: weak market-data quality, scheduled-event risk, cross-asset stress, and reduced size capacity all contribute to the final overlay action.
    """
    cfg = get_global_risk_policy_config()
    data_quality = data_quality if isinstance(data_quality, dict) else {}
    confirmation = confirmation if isinstance(confirmation, dict) else {}
    macro_news_adjustments = macro_news_adjustments if isinstance(macro_news_adjustments, dict) else {}
    global_risk_state = global_risk_state if isinstance(global_risk_state, dict) else None

    if global_risk_state is None:
        global_risk_state = _fallback_global_risk_state(
            event_window_status=event_window_status,
            macro_event_risk_score=macro_event_risk_score,
            event_lockdown_flag=event_lockdown_flag,
            next_event_name=next_event_name,
            active_event_name=active_event_name,
            holding_profile=holding_profile,
        )

    reasons = []
    portfolio_guard = _evaluate_portfolio_concentration_guard(
        portfolio_context=portfolio_context,
        cfg=cfg,
    )
    size_cap = min(
        _clip(_safe_float(macro_news_adjustments.get("macro_position_size_multiplier"), 1.0), 0.0, 1.0),
        _clip(_safe_float(global_risk_state.get("global_risk_position_size_multiplier"), 1.0), 0.0, 1.0),
    )
    # The risk score is intentionally composite: weak data, event risk, broad
    # global stress, and size constraints all contribute to the final action.
    score = int(round((100 - _safe_float(data_quality.get("score"), 100.0)) * cfg.layer_data_quality_weight))
    score += int(round(_safe_float(macro_event_risk_score, 0.0) * cfg.layer_macro_event_weight))
    score += int(round(_safe_float(global_risk_state.get("global_risk_score"), 0.0) * cfg.layer_global_risk_weight))
    score += int(round((1.0 - size_cap) * cfg.layer_size_cap_penalty_scale))

    if data_quality.get("fatal"):
        return _result(
            state=global_risk_state,
            score=100,
            level="BLOCKED",
            action="BLOCK",
            size_cap=0.0,
            reasons=["invalid_market_data"],
            trade_status="DATA_INVALID",
            message="Trade blocked due to invalid market data",
            portfolio_guard=portfolio_guard,
        )

    if event_lockdown_flag or macro_news_adjustments.get("event_lockdown_flag", False):
        return _result(
            state=global_risk_state,
            score=100,
            level="BLOCKED",
            action="BLOCK",
            size_cap=0.0,
            reasons=["event_lockdown"],
            trade_status="EVENT_LOCKDOWN",
            message=f"Trade blocked due to scheduled macro event lockdown: {_event_name(active_event_name, next_event_name)}",
            portfolio_guard=portfolio_guard,
        )

    if global_risk_state.get("global_risk_veto"):
        reasons.extend(global_risk_state.get("global_risk_reasons", []))
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_veto_block_score_floor),
            level="HIGH",
            action="BLOCK",
            size_cap=0.0,
            reasons=reasons,
            trade_status="GLOBAL_RISK_BLOCKED",
            message="Trade blocked due to elevated global risk conditions",
            portfolio_guard=portfolio_guard,
        )

    # Overnight handling is evaluated explicitly because some setups are
    # acceptable intraday but not safe to carry through the close.
    holding_context = global_risk_state.get("holding_context", {})
    overnight_relevant = bool(holding_context.get("overnight_relevant", False))
    if overnight_relevant and not global_risk_state.get("overnight_hold_allowed", True):
        reasons.extend(global_risk_state.get("global_risk_reasons", []))
        overnight_hold_reason = global_risk_state.get("overnight_hold_reason")
        if overnight_hold_reason:
            reasons.append(str(overnight_hold_reason))
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_overnight_watch_score_floor),
            level="HIGH",
            action="WATCHLIST",
            size_cap=min(size_cap, cfg.layer_overnight_watch_size_cap),
            reasons=reasons or ["overnight_hold_not_allowed"],
            trade_status="WATCHLIST",
            message=f"Trade downgraded due to elevated overnight risk: {global_risk_state.get('overnight_hold_reason', 'overnight_hold_not_allowed')}",
            portfolio_guard=portfolio_guard,
        )

    portfolio_action = str(portfolio_guard.get("action") or "ALLOW").upper().strip()
    if portfolio_action == "WATCHLIST":
        reasons.extend(portfolio_guard.get("reasons", []))
        return _result(
            state=global_risk_state,
            score=max(score, int(_safe_float(getattr(cfg, "portfolio_concentration_watch_score_floor", 68), 68.0))),
            level="HIGH",
            action="WATCHLIST",
            size_cap=min(size_cap, _clip(_safe_float(portfolio_guard.get("size_cap"), 1.0), 0.0, 1.0)),
            reasons=reasons,
            trade_status="WATCHLIST",
            message=portfolio_guard.get("reason") or "Trade downgraded due to a concentrated same-way options book",
            portfolio_guard=portfolio_guard,
        )
    if portfolio_action == "REDUCE":
        reasons.extend(portfolio_guard.get("reasons", []))
        size_cap = min(size_cap, _clip(_safe_float(portfolio_guard.get("size_cap"), 1.0), 0.0, 1.0))
        score = max(score, int(_safe_float(getattr(cfg, "portfolio_concentration_reduce_score_floor", 56), 56.0)))

    if confirmation.get("veto"):
        reasons.append("confirmation_veto")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_confirmation_watch_score_floor),
            level="HIGH",
            action="WATCHLIST",
            size_cap=size_cap,
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to confirmation conflict",
            portfolio_guard=portfolio_guard,
        )

    global_risk_features = global_risk_state.get("global_risk_features", {}) if isinstance(global_risk_state, dict) else {}
    macro_uncertainty_score = _safe_float(global_risk_features.get("macro_uncertainty_score"), 0.0)
    if macro_uncertainty_score >= cfg.macro_uncertainty_watch_threshold:
        reasons.append("macro_uncertainty_window")
        if global_risk_features.get("headline_data_stale"):
            reasons.append("headline_data_stale")
        if global_risk_features.get("global_macro_data_stale"):
            reasons.append("global_macro_data_stale")
        if _safe_float(global_risk_features.get("event_uncertainty_score"), 0.0) > 0:
            reasons.append("event_uncertainty_elevated")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.macro_uncertainty_watch_score_floor),
            level="HIGH",
            action="WATCHLIST",
            size_cap=min(size_cap, cfg.macro_uncertainty_watch_size_cap),
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to elevated macro uncertainty",
            portfolio_guard=portfolio_guard,
        )

    if adjusted_trade_strength < min_trade_strength:
        reasons.append("insufficient_trade_strength")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_low_strength_watch_score_floor),
            level="MEDIUM",
            action="WATCHLIST",
            size_cap=size_cap,
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade filtered out due to low strength",
            portfolio_guard=portfolio_guard,
        )

    if _safe_float(data_quality.get("score"), 0.0) < cfg.layer_weak_data_quality_score_threshold:
        reasons.append("weak_data_quality")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_weak_data_quality_watch_score_floor),
            level="HIGH",
            action="WATCHLIST",
            size_cap=min(size_cap, cfg.layer_weak_data_quality_size_cap),
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to weak data quality",
            portfolio_guard=portfolio_guard,
        )

    if data_quality.get("status") == "CAUTION" and adjusted_trade_strength < (min_trade_strength + cfg.layer_caution_strength_buffer):
        reasons.append("cautionary_data_quality")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_caution_watch_score_floor),
            level="MEDIUM",
            action="WATCHLIST",
            size_cap=min(size_cap, cfg.layer_caution_watch_size_cap),
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to cautionary data quality",
            portfolio_guard=portfolio_guard,
        )

    if confirmation.get("status") == "CONFLICT" and adjusted_trade_strength < (min_trade_strength + cfg.layer_confirmation_conflict_strength_buffer):
        reasons.append("live_confirmation_conflict")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_confirmation_conflict_watch_score_floor),
            level="MEDIUM",
            action="WATCHLIST",
            size_cap=min(size_cap, cfg.layer_caution_watch_size_cap),
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to weak live confirmation",
            portfolio_guard=portfolio_guard,
        )

    if size_cap < cfg.layer_caution_watch_size_cap and adjusted_trade_strength < (min_trade_strength + cfg.layer_size_reduction_strength_buffer):
        reasons.append("global_macro_size_reduction")
        return _result(
            state=global_risk_state,
            score=max(score, cfg.layer_size_reduction_watch_score_floor),
            level="MEDIUM",
            action="WATCHLIST",
            size_cap=size_cap,
            reasons=reasons,
            trade_status="WATCHLIST",
            message="Trade downgraded to watchlist due to global risk reduction",
            portfolio_guard=portfolio_guard,
        )

    reasons.extend(global_risk_state.get("global_risk_reasons", []))
    if size_cap < 1.0:
        reasons.append("size_reduced")

    if not reasons:
        reasons.append("risk_checks_passed")

    level = "LOW"
    if score >= cfg.layer_high_level_threshold:
        level = "HIGH"
    elif score >= cfg.layer_medium_level_threshold:
        level = "MEDIUM"

    return _result(
        state=global_risk_state,
        score=score,
        level=level,
        action="REDUCE" if size_cap < 1.0 else "ALLOW",
        size_cap=size_cap,
        reasons=reasons,
        portfolio_guard=portfolio_guard,
    )
