"""
Module: engine_adjustments.py

Purpose:
    Implement engine adjustments logic used to score scheduled events and macro catalysts.

Role in the System:
    Part of the macro context layer that scores scheduled events and broad market catalysts.

Key Outputs:
    Macro-event state, catalyst scores, and gating diagnostics.

Downstream Usage:
    Consumed by the signal engine, risk overlays, and research diagnostics.
"""

from __future__ import annotations

from macro.macro_news_config import get_macro_news_adjustment_config


def _clip(value, lo, hi):
    """
    Purpose:
        Clamp a numeric value to the configured bounds.

    Context:
        Function inside the `engine adjustments` module. The module sits in the macro overlay layer that models scheduled events and headline-driven context.

    Inputs:
        value (Any): Raw value supplied by the caller.
        lo (Any): Inclusive lower bound for the returned value.
        hi (Any): Inclusive upper bound for the returned value.

    Returns:
        float | int: Bounded value returned by the helper.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `engine adjustments` module. The module sits in the macro overlay layer that models scheduled events and headline-driven context.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_macro_news_adjustments(*, direction, macro_news_state=None):
    """
    Purpose:
        Compute macro news adjustments from the supplied inputs.
    
    Context:
        Public function within the macro context layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        direction (Any): Trade direction label associated with the current signal, typically `CALL` or `PUT`.
        macro_news_state (Any): Headline-driven macro state produced by the news layer.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    cfg = get_macro_news_adjustment_config()
    macro_news_state = macro_news_state if isinstance(macro_news_state, dict) else {}

    macro_regime = macro_news_state.get("macro_regime", "MACRO_NEUTRAL")
    macro_sentiment_score = _safe_float(macro_news_state.get("macro_sentiment_score"), 0.0)
    volatility_shock_score = _safe_float(macro_news_state.get("volatility_shock_score"), 0.0)
    news_confidence_score = _safe_float(macro_news_state.get("news_confidence_score"), 0.0)
    event_lockdown_flag = bool(macro_news_state.get("event_lockdown_flag", False))
    neutral_fallback = bool(macro_news_state.get("neutral_fallback", True))

    adjustment_score = 0
    confirmation_adjustment = 0
    size_multiplier = 1.0
    reasons = []

    if neutral_fallback:
        reasons.append("macro_news_neutral_fallback")
        return {
            "macro_regime": macro_regime,
            "macro_sentiment_score": macro_sentiment_score,
            "volatility_shock_score": volatility_shock_score,
            "news_confidence_score": news_confidence_score,
            "event_lockdown_flag": event_lockdown_flag,
            "macro_adjustment_score": 0,
            "macro_confirmation_adjustment": 0,
            "macro_position_size_multiplier": 1.0,
            "macro_adjustment_reasons": reasons,
        }

    if macro_regime == "EVENT_LOCKDOWN" or event_lockdown_flag:
        reasons.append("macro_event_lockdown")
        return {
            "macro_regime": "EVENT_LOCKDOWN",
            "macro_sentiment_score": macro_sentiment_score,
            "volatility_shock_score": volatility_shock_score,
            "news_confidence_score": news_confidence_score,
            "event_lockdown_flag": True,
            "macro_adjustment_score": cfg.lockdown_adjustment_score,
            "macro_confirmation_adjustment": cfg.lockdown_confirmation_adjustment,
            "macro_position_size_multiplier": 0.0,
            "macro_adjustment_reasons": reasons,
        }

    strong_confidence = news_confidence_score >= cfg.strong_confidence_threshold
    moderate_confidence = news_confidence_score >= cfg.moderate_confidence_threshold
    high_vol_shock = volatility_shock_score >= cfg.high_vol_shock_threshold
    medium_vol_shock = volatility_shock_score >= cfg.medium_vol_shock_threshold

    if macro_regime == "RISK_OFF":
        reasons.append("macro_regime_risk_off")
        if direction == "CALL":
            adjustment_score += cfg.risk_off_call_score_strong if strong_confidence else cfg.risk_off_call_score_moderate
            confirmation_adjustment += (
                cfg.risk_off_call_confirmation_strong
                if strong_confidence else cfg.risk_off_call_confirmation_moderate
            )
            size_multiplier = cfg.risk_off_call_size_high_vol if high_vol_shock else cfg.risk_off_call_size_normal
            reasons.append("risk_off_conflicts_with_call")
        elif direction == "PUT":
            adjustment_score += cfg.risk_off_put_score_support if moderate_confidence else cfg.risk_off_put_score_soft
            confirmation_adjustment += cfg.risk_off_put_confirmation_support if moderate_confidence else 0
            size_multiplier = cfg.risk_off_put_size_medium_vol if medium_vol_shock else 1.0
            reasons.append("risk_off_supports_put")

    elif macro_regime == "RISK_ON":
        reasons.append("macro_regime_risk_on")
        if direction == "CALL":
            adjustment_score += cfg.risk_on_call_score_support if moderate_confidence else cfg.risk_on_call_score_soft
            confirmation_adjustment += cfg.risk_on_call_confirmation_support if moderate_confidence else 0
            size_multiplier = 1.0 if not medium_vol_shock else cfg.risk_on_call_size_medium_vol
            reasons.append("risk_on_supports_call")
        elif direction == "PUT":
            adjustment_score += cfg.risk_on_put_score_strong if strong_confidence else cfg.risk_on_put_score_moderate
            confirmation_adjustment += (
                cfg.risk_on_put_confirmation_strong
                if strong_confidence else cfg.risk_on_put_confirmation_moderate
            )
            size_multiplier = cfg.risk_on_put_size_high_vol if high_vol_shock else cfg.risk_on_put_size_normal
            reasons.append("risk_on_conflicts_with_put")

    if high_vol_shock:
        adjustment_score += cfg.high_vol_extra_score
        confirmation_adjustment += cfg.high_vol_extra_confirmation
        size_multiplier = min(size_multiplier, cfg.generic_high_vol_size_cap)
        reasons.append("high_volatility_shock")
    elif medium_vol_shock:
        adjustment_score += cfg.medium_vol_extra_score
        size_multiplier = min(size_multiplier, cfg.generic_medium_vol_size_cap)
        reasons.append("medium_volatility_shock")

    if abs(macro_sentiment_score) < 8 and macro_regime == "MACRO_NEUTRAL":
        reasons.append("macro_news_neutral")

    return {
        "macro_regime": macro_regime,
        "macro_sentiment_score": round(macro_sentiment_score, 2),
        "volatility_shock_score": round(volatility_shock_score, 2),
        "news_confidence_score": round(news_confidence_score, 2),
        "event_lockdown_flag": event_lockdown_flag,
        "macro_adjustment_score": int(_clip(adjustment_score, -12, 6)),
        "macro_confirmation_adjustment": int(_clip(confirmation_adjustment, -4, 2)),
        "macro_position_size_multiplier": round(_clip(size_multiplier, 0.0, 1.0), 2),
        "macro_adjustment_reasons": reasons,
    }
