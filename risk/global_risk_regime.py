"""
Interpretable regime classifier for the global risk layer.
"""

from __future__ import annotations

from config.global_risk_policy import get_global_risk_policy_config
from risk.global_risk_models import GlobalRiskState, HoldingContext


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _scaled_score(value, full_scale):
    value = _safe_float(value, 0.0)
    if full_scale <= 0:
        return 0.0
    return _clip((value / full_scale) * 100.0, 0.0, 100.0)


def _legacy_state(global_risk_score, volatility_expansion_risk_score, event_window_status):
    cfg = get_global_risk_policy_config()
    if event_window_status in {"PRE_EVENT_WATCH", "POST_EVENT_COOLDOWN"} and _safe_float(global_risk_score, 0.0) >= cfg.event_risk_state_threshold:
        return "EVENT_RISK"
    if _safe_float(global_risk_score, 0.0) >= cfg.risk_off_threshold:
        return "RISK_OFF"
    if (
        _safe_float(global_risk_score, 0.0) >= cfg.caution_threshold
        or _safe_float(volatility_expansion_risk_score, 0.0) >= cfg.volatility_expansion_medium_threshold
    ):
        return "CAUTION"
    return "NEUTRAL"


def _evaluate_overnight_risk(
    *,
    global_risk_state,
    volatility_explosion_probability,
    macro_event_risk_score,
    oil_shock_score,
    us_equity_risk_score,
    overnight_relevant,
):
    cfg = get_global_risk_policy_config()
    penalty = 0
    reasons = []

    if global_risk_state == "VOL_SHOCK":
        return False, "vol_shock_block", 10

    if global_risk_state == "EVENT_LOCKDOWN":
        return False, "event_lockdown_block", 10

    if _safe_float(volatility_explosion_probability, 0.0) > cfg.overnight_vol_explosion_high_threshold:
        penalty += cfg.overnight_vol_explosion_high_penalty
        reasons.append("volatility_explosion_risk")
    elif _safe_float(volatility_explosion_probability, 0.0) > cfg.overnight_vol_explosion_watch_threshold:
        penalty += cfg.overnight_vol_explosion_watch_penalty
        reasons.append("volatility_explosion_watch")

    if _safe_float(macro_event_risk_score, 0.0) >= cfg.overnight_macro_event_high_threshold:
        penalty += cfg.overnight_macro_event_high_penalty
        reasons.append("macro_event_risk_high")
    elif _safe_float(macro_event_risk_score, 0.0) >= cfg.overnight_macro_event_watch_threshold:
        penalty += cfg.overnight_macro_event_watch_penalty
        reasons.append("macro_event_risk_elevated")

    if _safe_float(oil_shock_score, 0.0) >= cfg.overnight_oil_shock_threshold:
        penalty += cfg.overnight_oil_shock_penalty
        reasons.append("oil_shock_elevated")

    if _safe_float(us_equity_risk_score, 0.0) >= cfg.overnight_us_equity_high_threshold:
        penalty += cfg.overnight_us_equity_high_penalty
        reasons.append("us_equity_risk_elevated")
    elif _safe_float(us_equity_risk_score, 0.0) >= cfg.overnight_us_equity_watch_threshold:
        penalty += cfg.overnight_us_equity_watch_penalty
        reasons.append("us_equity_risk_watch")

    if global_risk_state == "RISK_OFF":
        penalty += cfg.overnight_risk_off_regime_penalty
        reasons.append("global_risk_off_regime")

    penalty = int(_clip(penalty, 0.0, 10.0))
    if overnight_relevant and penalty >= 7:
        return False, reasons[0] if reasons else "overnight_risk_block", penalty

    if reasons:
        return True, reasons[0], penalty

    return True, "overnight_risk_contained", 0


def classify_global_risk_state(features: dict | None) -> GlobalRiskState:
    cfg = get_global_risk_policy_config()
    features = features if isinstance(features, dict) else {}

    macro_event_risk_score = int(_clip(_safe_float(features.get("macro_event_risk_score"), 0.0), 0.0, 100.0))
    event_window_status = features.get("event_window_status", "NO_EVENT_DATA")
    event_lockdown_flag = bool(features.get("event_lockdown_flag", False))
    macro_regime = str(features.get("macro_regime", "MACRO_NEUTRAL")).upper().strip()
    neutral_fallback = bool(features.get("neutral_fallback", True))
    overnight_relevant = bool(features.get("overnight_relevant", False))
    market_data_available = bool(features.get("market_data_available", False))
    market_feature_confidence = _clip(_safe_float(features.get("market_feature_confidence"), 0.0), 0.0, 1.0)
    market_feature_multiplier = market_feature_confidence if market_data_available else 0.0

    news_confidence_score = _clip(_safe_float(features.get("news_confidence_score"), 0.0), 0.0, 100.0)
    confidence_multiplier = 0.0 if neutral_fallback else _clip(news_confidence_score / 100.0, 0.25, 1.0)
    headline_volatility_score = _clip(
        _safe_float(features.get("headline_volatility_shock_score"), 0.0) * confidence_multiplier,
        0.0,
        100.0,
    )
    market_volatility_score = _clip(
        _safe_float(features.get("volatility_shock_score"), 0.0) * 100.0 * market_feature_multiplier,
        0.0,
        100.0,
    )
    risk_off_intensity = _clip(_safe_float(features.get("risk_off_intensity"), 0.0), 0.0, 1.0)
    risk_off_intensity_score = risk_off_intensity * 100.0 * market_feature_multiplier
    volatility_explosion_probability = _clip(
        _safe_float(features.get("volatility_explosion_probability"), 0.0),
        0.0,
        1.0,
    )
    volatility_explosion_score = volatility_explosion_probability * 100.0 * market_feature_multiplier
    volatility_compression_score = _clip(
        _safe_float(features.get("volatility_compression_score"), 0.0),
        0.0,
        1.0,
    ) * 100.0 * market_feature_multiplier
    volatility_shock_score = _clip(_safe_float(features.get("volatility_shock_score"), 0.0), 0.0, 1.0)
    us_equity_risk_score = _clip(_safe_float(features.get("us_equity_risk_score"), 0.0), 0.0, 1.0)
    rates_shock_score_raw = _clip(_safe_float(features.get("rates_shock_score"), 0.0), 0.0, 1.0)
    currency_shock_score_raw = _clip(_safe_float(features.get("currency_shock_score"), 0.0), 0.0, 1.0)
    currency_shock_score = _clip(
        _safe_float(features.get("currency_shock_score"), 0.0) * 100.0 * market_feature_multiplier,
        0.0,
        100.0,
    )
    commodity_risk_score = _safe_float(features.get("commodity_risk_score"), 0.0)
    commodity_stress_score = _clip(
        (
            max(_safe_float(features.get("oil_shock_score"), 0.0), 0.0) * 0.55
            + _safe_float(features.get("gold_risk_score"), 0.0) * 0.20
            + max(-_safe_float(features.get("copper_growth_signal"), 0.0), 0.0) * 0.25
        )
        * 100.0
        * market_feature_multiplier,
        0.0,
        100.0,
    )
    macro_event_risk_norm = _clip(macro_event_risk_score / 100.0, 0.0, 1.0)
    positive_commodity_support = max(-commodity_risk_score, 0.0)
    positive_global_bias_norm = _clip(
        max(_safe_float(features.get("global_risk_bias"), 0.0), 0.0) / cfg.global_bias_risk_full_scale,
        0.0,
        1.0,
    )
    positive_macro_sentiment_norm = _clip(
        max(_safe_float(features.get("macro_sentiment_score"), 0.0), 0.0) / cfg.positive_macro_sentiment_full_scale,
        0.0,
        1.0,
    )
    risk_off_pressure = _clip(
        (cfg.risk_off_pressure_vol_weight * volatility_shock_score)
        + (cfg.risk_off_pressure_us_equity_weight * us_equity_risk_score)
        + (cfg.risk_off_pressure_rates_weight * rates_shock_score_raw)
        + (cfg.risk_off_pressure_currency_weight * currency_shock_score_raw)
        + (cfg.risk_off_pressure_macro_event_weight * macro_event_risk_norm)
        + (cfg.risk_off_pressure_vol_explosion_weight * volatility_explosion_probability)
        + (cfg.risk_off_pressure_commodity_weight * max(commodity_risk_score, 0.0)),
        0.0,
        1.0,
    )
    risk_on_support = _clip(
        (cfg.risk_on_support_commodity_weight * positive_commodity_support)
        + (cfg.risk_on_support_global_bias_weight * positive_global_bias_norm)
        + (cfg.risk_on_support_macro_sentiment_weight * positive_macro_sentiment_norm),
        0.0,
        1.0,
    )
    regime_score = round(_clip(risk_off_pressure - risk_on_support, -1.0, 1.0), 4)

    volatility_expansion_risk_score = int(round(_clip(
        market_volatility_score * cfg.volatility_expansion_market_vol_weight
        + volatility_explosion_score * cfg.volatility_expansion_explosion_weight
        + headline_volatility_score * cfg.volatility_expansion_headline_weight,
        0.0,
        100.0,
    )))

    downside_global_bias = max(0.0, -_safe_float(features.get("global_risk_bias"), 0.0))
    global_bias_risk_score = _scaled_score(downside_global_bias, cfg.global_bias_risk_full_scale)
    headline_velocity_score = _scaled_score(
        _safe_float(features.get("headline_velocity"), 0.0),
        cfg.headline_velocity_full_scale,
    ) * confidence_multiplier

    global_risk_score = int(round(_clip(
        risk_off_pressure * 100.0 * cfg.global_risk_score_risk_off_pressure_weight
        + macro_event_risk_score * cfg.global_risk_score_macro_event_weight
        + volatility_expansion_risk_score * cfg.global_risk_score_volatility_expansion_weight
        + risk_off_intensity_score * cfg.global_risk_score_risk_off_intensity_weight
        + headline_velocity_score * cfg.global_risk_score_headline_velocity_weight
        + global_bias_risk_score * cfg.global_risk_score_global_bias_weight
        + currency_shock_score * cfg.global_risk_score_currency_weight
        + (cfg.global_risk_score_macro_regime_risk_off_bonus if macro_regime == "RISK_OFF" and not neutral_fallback else 0.0),
        0.0,
        100.0,
    )))

    overnight_gap_risk_score = int(round(_clip(
        macro_event_risk_score * cfg.overnight_gap_macro_event_weight
        + volatility_expansion_risk_score * cfg.overnight_gap_volatility_expansion_weight
        + currency_shock_score * cfg.overnight_gap_currency_weight
        + risk_off_intensity_score * cfg.overnight_gap_risk_off_intensity_weight
        + headline_velocity_score * cfg.overnight_gap_headline_velocity_weight
        + max(global_risk_score - cfg.overnight_gap_global_score_excess_floor, 0.0) * cfg.overnight_gap_global_score_excess_weight
        + (cfg.overnight_gap_overnight_context_bonus if overnight_relevant else 0.0),
        0.0,
        100.0,
    )))

    reasons = []
    if neutral_fallback:
        reasons.append("global_risk_neutral_fallback")
    if event_lockdown_flag:
        reasons.append("event_lockdown")
    if macro_event_risk_score >= cfg.event_risk_state_threshold:
        reasons.append("elevated_macro_event_risk")
    if volatility_expansion_risk_score >= cfg.volatility_expansion_high_threshold:
        reasons.append("volatility_expansion_high")
    elif volatility_expansion_risk_score >= cfg.volatility_expansion_medium_threshold:
        reasons.append("volatility_expansion_elevated")
    if risk_off_intensity_score >= 65:
        reasons.append("cross_asset_risk_off_intense")
    elif risk_off_intensity_score >= 40:
        reasons.append("cross_asset_risk_off_elevated")
    if volatility_explosion_score >= 50:
        reasons.append("volatility_explosion_risk")
    if currency_shock_score >= 50:
        reasons.append("currency_shock_active")
    if global_bias_risk_score >= 45:
        reasons.append("global_risk_bias_negative")
    if headline_velocity_score >= 45:
        reasons.append("headline_velocity_elevated")
    if commodity_stress_score >= 45:
        reasons.append("commodity_stress_elevated")
    if regime_score <= -0.3:
        reasons.append("cross_asset_risk_on_support")
    if overnight_relevant:
        reasons.append("overnight_context_active")

    if volatility_explosion_probability > cfg.state_vol_shock_probability_threshold:
        state = "VOL_SHOCK"
        adjustment_score = cfg.risk_adjustment_extreme
        size_multiplier = cfg.size_cap_extreme
    elif event_lockdown_flag or macro_event_risk_norm > cfg.state_event_lockdown_probability_threshold:
        state = "EVENT_LOCKDOWN"
        adjustment_score = cfg.risk_adjustment_risk_off
        size_multiplier = 0.0 if event_lockdown_flag else cfg.size_cap_risk_off
    elif regime_score > cfg.state_risk_off_regime_score_threshold or macro_regime == "RISK_OFF":
        state = "RISK_OFF"
        adjustment_score = cfg.risk_adjustment_risk_off
        size_multiplier = cfg.size_cap_risk_off
    elif regime_score < cfg.state_risk_on_regime_score_threshold and not neutral_fallback:
        state = "RISK_ON"
        adjustment_score = 0
        size_multiplier = 1.0
    else:
        state = "GLOBAL_NEUTRAL"
        adjustment_score = 0
        size_multiplier = 1.0

    global_risk_veto = (
        event_lockdown_flag
        or state == "VOL_SHOCK"
        or global_risk_score >= cfg.extreme_veto_threshold
        or (overnight_relevant and overnight_gap_risk_score >= cfg.overnight_gap_veto_threshold)
    )
    overnight_hold_allowed, overnight_hold_reason, overnight_risk_penalty = _evaluate_overnight_risk(
        global_risk_state=state,
        volatility_explosion_probability=volatility_explosion_probability,
        macro_event_risk_score=macro_event_risk_score,
        oil_shock_score=features.get("oil_shock_score"),
        us_equity_risk_score=features.get("us_equity_risk_score"),
        overnight_relevant=overnight_relevant,
    )

    if not reasons:
        reasons.append("global_risk_contained")

    holding_context = HoldingContext(
        holding_profile=features.get("holding_profile", "AUTO"),
        overnight_relevant=overnight_relevant,
        market_session=features.get("market_session", "UNKNOWN"),
        minutes_to_close=features.get("minutes_to_close"),
    )

    diagnostics = {
        "macro_regime": macro_regime,
        "confidence_multiplier": round(confidence_multiplier, 3),
        "market_data_available": market_data_available,
        "market_feature_confidence": round(market_feature_confidence, 4),
        "market_features_neutralized": bool(features.get("market_features_neutralized", False)),
        "market_neutralization_reason": features.get("market_neutralization_reason"),
        "market_input_availability": dict(features.get("market_input_availability", {})),
        "regime_score": regime_score,
        "risk_off_pressure": round(risk_off_pressure, 4),
        "risk_on_support": round(risk_on_support, 4),
        "legacy_global_risk_state": _legacy_state(
            global_risk_score,
            volatility_expansion_risk_score,
            event_window_status,
        ),
        "global_bias_risk_score": round(global_bias_risk_score, 2),
        "headline_velocity_risk_score": round(headline_velocity_score, 2),
        "risk_off_intensity_score": round(risk_off_intensity_score, 2),
        "volatility_explosion_score": round(volatility_explosion_score, 2),
        "volatility_compression_score": round(volatility_compression_score, 2),
        "commodity_stress_score": round(commodity_stress_score, 2),
        "commodity_risk_score": round(commodity_risk_score, 4),
        "market_volatility_shock_score": round(market_volatility_score, 2),
        "headline_volatility_shock_score": round(headline_volatility_score, 2),
        "currency_shock_score": round(currency_shock_score, 2),
        "component_contributions": {
            "risk_off_pressure": round(risk_off_pressure * 100.0 * cfg.global_risk_score_risk_off_pressure_weight, 2),
            "macro_event_risk": round(macro_event_risk_score * cfg.global_risk_score_macro_event_weight, 2),
            "volatility_expansion_risk": round(volatility_expansion_risk_score * cfg.global_risk_score_volatility_expansion_weight, 2),
            "risk_off_intensity": round(risk_off_intensity_score * cfg.global_risk_score_risk_off_intensity_weight, 2),
            "headline_velocity": round(headline_velocity_score * cfg.global_risk_score_headline_velocity_weight, 2),
            "global_bias_risk": round(global_bias_risk_score * cfg.global_risk_score_global_bias_weight, 2),
            "currency_shock": round(currency_shock_score * cfg.global_risk_score_currency_weight, 2),
        },
        "dominant_risk_driver": max(
            {
                "risk_off_pressure": round(risk_off_pressure * 100.0 * cfg.global_risk_score_risk_off_pressure_weight, 2),
                "macro_event_risk": round(macro_event_risk_score * cfg.global_risk_score_macro_event_weight, 2),
                "volatility_expansion_risk": round(volatility_expansion_risk_score * cfg.global_risk_score_volatility_expansion_weight, 2),
                "risk_off_intensity": round(risk_off_intensity_score * cfg.global_risk_score_risk_off_intensity_weight, 2),
                "headline_velocity": round(headline_velocity_score * cfg.global_risk_score_headline_velocity_weight, 2),
                "global_bias_risk": round(global_bias_risk_score * cfg.global_risk_score_global_bias_weight, 2),
                "currency_shock": round(currency_shock_score * cfg.global_risk_score_currency_weight, 2),
            }.items(),
            key=lambda item: item[1],
        )[0],
        "event_window_status": event_window_status,
        "overnight_hold_reason": overnight_hold_reason,
        "overnight_risk_penalty": overnight_risk_penalty,
        "news_data_available": bool(features.get("news_data_available", False)),
        "event_data_available": bool(features.get("event_data_available", False)),
        "issues": list(features.get("issues", [])),
        "warnings": list(features.get("warnings", [])),
    }

    return GlobalRiskState(
        global_risk_state=state,
        global_risk_score=global_risk_score,
        overnight_gap_risk_score=overnight_gap_risk_score,
        volatility_expansion_risk_score=volatility_expansion_risk_score,
        overnight_hold_allowed=overnight_hold_allowed,
        overnight_hold_reason=overnight_hold_reason,
        overnight_risk_penalty=overnight_risk_penalty,
        global_risk_adjustment_score=adjustment_score,
        global_risk_veto=global_risk_veto,
        global_risk_position_size_multiplier=round(_clip(size_multiplier, 0.0, 1.0), 2),
        neutral_fallback=neutral_fallback,
        holding_context=holding_context.to_dict(),
        global_risk_reasons=reasons,
        global_risk_features=features,
        global_risk_diagnostics=diagnostics,
    )
