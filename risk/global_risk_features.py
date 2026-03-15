"""
Cross-asset feature builder for the global risk layer.
"""

from __future__ import annotations

import pandas as pd

from config.global_risk_policy import get_global_risk_policy_config


IST_TIMEZONE = "Asia/Kolkata"
OVERNIGHT_HOLDING_PROFILES = {"OVERNIGHT", "SWING", "POSITIONAL"}
MARKET_INPUT_KEYS = [
    "oil_change_24h",
    "gold_change_24h",
    "copper_change_24h",
    "vix_change_24h",
    "sp500_change_24h",
    "nasdaq_change_24h",
    "us10y_change_bp",
    "usdinr_change_24h",
    "realized_vol_5d",
    "realized_vol_30d",
]


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _coerce_timestamp(value):
    if value is None or value == "":
        return None

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return None

    try:
        return parsed.tz_convert(IST_TIMEZONE)
    except Exception:
        return None


def _market_session_context(*, cfg, as_of=None):
    ts = _coerce_timestamp(as_of)
    if ts is None:
        ts = pd.Timestamp.now(tz=IST_TIMEZONE)

    market_open = ts.replace(
        hour=cfg.market_open_hour,
        minute=cfg.market_open_minute,
        second=0,
        microsecond=0,
    )
    market_close = ts.replace(
        hour=cfg.market_close_hour,
        minute=cfg.market_close_minute,
        second=0,
        microsecond=0,
    )

    if ts < market_open:
        return {
            "as_of": ts.isoformat(),
            "market_session": "PRE_OPEN",
            "minutes_to_close": None,
            "overnight_session": True,
            "near_close": False,
        }

    if ts > market_close:
        return {
            "as_of": ts.isoformat(),
            "market_session": "POST_CLOSE",
            "minutes_to_close": None,
            "overnight_session": True,
            "near_close": False,
        }

    minutes_to_close = max((market_close - ts).total_seconds() / 60.0, 0.0)
    return {
        "as_of": ts.isoformat(),
        "market_session": "INTRADAY",
        "minutes_to_close": round(minutes_to_close, 2),
        "overnight_session": False,
        "near_close": minutes_to_close <= cfg.near_close_overnight_minutes,
    }


def _normalize_holding_profile(value) -> str:
    normalized = str(value or "AUTO").upper().strip()
    return normalized or "AUTO"


def _oil_shock_score(change_24h, *, cfg):
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.oil_shock_extreme_change_pct:
        return cfg.oil_shock_extreme_score
    if change_24h > cfg.oil_shock_medium_change_pct:
        return cfg.oil_shock_medium_score
    if change_24h < cfg.oil_shock_relief_change_pct:
        return cfg.oil_shock_relief_score
    return 0.0


def _gold_risk_score(change_24h, *, cfg):
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.gold_risk_extreme_change_pct:
        return cfg.gold_risk_extreme_score
    if change_24h > cfg.gold_risk_medium_change_pct:
        return cfg.gold_risk_medium_score
    return 0.0


def _copper_growth_signal(change_24h, *, cfg):
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h < cfg.copper_growth_severe_drop_pct:
        return cfg.copper_growth_severe_score
    if change_24h < cfg.copper_growth_moderate_drop_pct:
        return cfg.copper_growth_moderate_score
    return 0.0


def _volatility_shock_score(vix_change_24h, *, cfg):
    vix_change_24h = _safe_float(vix_change_24h, 0.0)
    if vix_change_24h > cfg.vix_shock_extreme_change_pct:
        return cfg.vix_shock_extreme_score
    if vix_change_24h > cfg.vix_shock_medium_change_pct:
        return cfg.vix_shock_medium_score
    if vix_change_24h > cfg.vix_shock_low_change_pct:
        return cfg.vix_shock_low_score
    return 0.0


def _us_equity_risk_score(sp500_change_24h, nasdaq_change_24h, *, cfg):
    worst_move = min(
        _safe_float(sp500_change_24h, 0.0),
        _safe_float(nasdaq_change_24h, 0.0),
    )
    if worst_move < cfg.us_equity_risk_extreme_move_pct:
        return cfg.us_equity_risk_extreme_score
    if worst_move < cfg.us_equity_risk_moderate_move_pct:
        return cfg.us_equity_risk_moderate_score
    return 0.0


def _rates_shock_score(us10y_change_bp, *, cfg):
    return cfg.rates_shock_score if _safe_float(us10y_change_bp, 0.0) > cfg.rates_shock_threshold_bp else 0.0


def _currency_shock_score(usdinr_change_24h, *, cfg):
    return cfg.currency_shock_score_base if _safe_float(usdinr_change_24h, 0.0) > cfg.currency_shock_threshold_pct else 0.0


def _volatility_compression_score(realized_vol_5d, realized_vol_30d, *, cfg):
    realized_vol_5d = _safe_float(realized_vol_5d, 0.0)
    realized_vol_30d = _safe_float(realized_vol_30d, 0.0)
    if realized_vol_30d <= 0:
        return 0.0

    compression_ratio = realized_vol_5d / realized_vol_30d
    if compression_ratio < cfg.vol_compression_extreme_ratio:
        return cfg.vol_compression_extreme_score
    if compression_ratio < cfg.vol_compression_medium_ratio:
        return cfg.vol_compression_medium_score
    if compression_ratio < cfg.vol_compression_low_ratio:
        return cfg.vol_compression_low_score
    return 0.0


def _market_snapshot_details(global_market_snapshot):
    snapshot = global_market_snapshot if isinstance(global_market_snapshot, dict) else {}
    market_inputs = snapshot.get("market_inputs", {}) if isinstance(snapshot.get("market_inputs", {}), dict) else {}
    data_available = bool(snapshot.get("data_available", False))
    stale = bool(snapshot.get("stale", False))
    neutral_fallback = bool(snapshot.get("neutral_fallback", not data_available))

    return snapshot, market_inputs, data_available and not stale, stale, neutral_fallback


def _market_input_state(market_inputs, *, market_data_usable, market_data_stale):
    raw_market_inputs = {
        key: _safe_float(market_inputs.get(key), None)
        for key in MARKET_INPUT_KEYS
    }
    availability = {
        key: raw_market_inputs.get(key) is not None
        for key in MARKET_INPUT_KEYS
    }
    available_count = sum(1 for value in availability.values() if value)
    coverage_ratio = round(_safe_float(available_count / max(len(MARKET_INPUT_KEYS), 1), 0.0), 4)

    if market_data_usable:
        effective_market_inputs = dict(raw_market_inputs)
        neutralized = False
        neutralization_reason = None
    else:
        effective_market_inputs = {key: None for key in MARKET_INPUT_KEYS}
        neutralized = available_count > 0
        neutralization_reason = "market_data_stale" if market_data_stale else "market_data_unavailable"

    return {
        "raw_market_inputs": raw_market_inputs,
        "effective_market_inputs": effective_market_inputs,
        "market_input_availability": availability,
        "market_input_available_count": available_count,
        "market_input_coverage_ratio": coverage_ratio,
        "market_feature_confidence": coverage_ratio if market_data_usable else 0.0,
        "market_features_neutralized": neutralized,
        "market_neutralization_reason": neutralization_reason,
    }


def build_global_risk_features(
    *,
    macro_event_state=None,
    macro_news_state=None,
    global_market_snapshot=None,
    holding_profile: str = "AUTO",
    as_of=None,
):
    cfg = get_global_risk_policy_config()
    macro_event_state = macro_event_state if isinstance(macro_event_state, dict) else {}
    macro_news_state = macro_news_state if isinstance(macro_news_state, dict) else {}
    global_market_snapshot, market_inputs, market_data_available, market_data_stale, market_neutral_fallback = (
        _market_snapshot_details(global_market_snapshot)
    )
    market_state = _market_input_state(
        market_inputs,
        market_data_usable=market_data_available,
        market_data_stale=market_data_stale,
    )

    holding_profile = _normalize_holding_profile(holding_profile)
    session_context = _market_session_context(cfg=cfg, as_of=as_of)
    overnight_relevant = (
        holding_profile in OVERNIGHT_HOLDING_PROFILES
        or session_context["overnight_session"]
        or (holding_profile == "AUTO" and session_context["near_close"])
    )

    headline_neutral_fallback = bool(macro_news_state.get("neutral_fallback", True))
    event_data_available = bool(macro_event_state.get("event_data_available", False))
    news_data_available = (not headline_neutral_fallback) and not bool(macro_news_state.get("issues"))

    effective_market_inputs = market_state["effective_market_inputs"]
    raw_market_inputs = market_state["raw_market_inputs"]
    oil_change_24h = _safe_float(effective_market_inputs.get("oil_change_24h"), None)
    gold_change_24h = _safe_float(effective_market_inputs.get("gold_change_24h"), None)
    copper_change_24h = _safe_float(effective_market_inputs.get("copper_change_24h"), None)
    vix_change_24h = _safe_float(effective_market_inputs.get("vix_change_24h"), None)
    sp500_change_24h = _safe_float(effective_market_inputs.get("sp500_change_24h"), None)
    nasdaq_change_24h = _safe_float(effective_market_inputs.get("nasdaq_change_24h"), None)
    us10y_change_bp = _safe_float(effective_market_inputs.get("us10y_change_bp"), None)
    usdinr_change_24h = _safe_float(effective_market_inputs.get("usdinr_change_24h"), None)
    realized_vol_5d = _safe_float(effective_market_inputs.get("realized_vol_5d"), None)
    realized_vol_30d = _safe_float(effective_market_inputs.get("realized_vol_30d"), None)

    oil_shock_score = _oil_shock_score(oil_change_24h, cfg=cfg)
    gold_risk_score = _gold_risk_score(gold_change_24h, cfg=cfg)
    copper_growth_signal = _copper_growth_signal(copper_change_24h, cfg=cfg)
    commodity_risk_score = round(
        (cfg.commodity_risk_oil_weight * oil_shock_score)
        + (cfg.commodity_risk_gold_weight * gold_risk_score)
        + (cfg.commodity_risk_copper_weight * copper_growth_signal),
        4,
    )
    volatility_shock_score = _volatility_shock_score(vix_change_24h, cfg=cfg)
    us_equity_risk_score = _us_equity_risk_score(sp500_change_24h, nasdaq_change_24h, cfg=cfg)
    rates_shock_score = _rates_shock_score(us10y_change_bp, cfg=cfg)
    currency_shock_score = _currency_shock_score(usdinr_change_24h, cfg=cfg)
    macro_event_risk_score = _safe_int(macro_event_state.get("macro_event_risk_score"), 0)
    macro_event_risk_norm = _clip(_safe_float(macro_event_risk_score, 0.0) / 100.0, 0.0, 1.0)
    volatility_compression_score = _volatility_compression_score(realized_vol_5d, realized_vol_30d, cfg=cfg)
    commodity_stress_component = _clip(
        (cfg.commodity_stress_oil_weight * max(oil_shock_score, 0.0))
        + (cfg.commodity_stress_gold_weight * gold_risk_score)
        + (cfg.commodity_stress_copper_weight * max(-copper_growth_signal, 0.0)),
        0.0,
        1.0,
    )
    risk_off_intensity = round(
        _clip(
            (cfg.risk_off_intensity_vol_weight * volatility_shock_score)
            + (cfg.risk_off_intensity_us_equity_weight * us_equity_risk_score)
            + (cfg.risk_off_intensity_rates_weight * rates_shock_score)
            + (cfg.risk_off_intensity_currency_weight * currency_shock_score)
            + (cfg.risk_off_intensity_commodity_weight * commodity_stress_component)
            + (cfg.risk_off_intensity_macro_event_weight * macro_event_risk_norm),
            0.0,
            1.0,
        ),
        4,
    )
    volatility_explosion_probability = round(
        _clip(
            volatility_compression_score
            * (volatility_shock_score + macro_event_risk_norm),
            0.0,
            1.0,
        ),
        4,
    )
    neutral_fallback = not market_data_available and headline_neutral_fallback and not event_data_available
    warnings = list(macro_news_state.get("warnings", [])) + list(global_market_snapshot.get("warnings", []))
    if market_state["market_features_neutralized"]:
        warnings.append(f"market_features_neutralized:{market_state['market_neutralization_reason']}")

    return {
        "holding_profile": holding_profile,
        "overnight_relevant": overnight_relevant,
        "market_session": session_context["market_session"],
        "minutes_to_close": session_context["minutes_to_close"],
        "overnight_session": session_context["overnight_session"],
        "macro_event_risk_score": macro_event_risk_score,
        "event_window_status": macro_event_state.get("event_window_status", "NO_EVENT_DATA"),
        "event_lockdown_flag": bool(
            macro_event_state.get("event_lockdown_flag", False)
            or macro_news_state.get("event_lockdown_flag", False)
        ),
        "next_event_name": macro_event_state.get("next_event_name") or macro_news_state.get("next_event_name"),
        "macro_regime": macro_news_state.get("macro_regime", "MACRO_NEUTRAL"),
        "macro_sentiment_score": _safe_float(macro_news_state.get("macro_sentiment_score"), 0.0),
        "headline_volatility_shock_score": _safe_float(macro_news_state.get("volatility_shock_score"), 0.0),
        "news_confidence_score": _safe_float(macro_news_state.get("news_confidence_score"), 0.0),
        "headline_velocity": _safe_float(macro_news_state.get("headline_velocity"), 0.0),
        "headline_impact_score": _safe_float(macro_news_state.get("headline_impact_score"), 0.0),
        "global_risk_bias": _safe_float(macro_news_state.get("global_risk_bias"), 0.0),
        "india_macro_bias": _safe_float(macro_news_state.get("india_macro_bias"), 0.0),
        "headline_count": _safe_int(macro_news_state.get("headline_count"), 0),
        "classified_headline_count": _safe_int(macro_news_state.get("classified_headline_count"), 0),
        "raw_market_inputs": raw_market_inputs,
        "market_input_availability": market_state["market_input_availability"],
        "market_input_available_count": market_state["market_input_available_count"],
        "market_input_coverage_ratio": market_state["market_input_coverage_ratio"],
        "market_feature_confidence": market_state["market_feature_confidence"],
        "market_features_neutralized": market_state["market_features_neutralized"],
        "market_neutralization_reason": market_state["market_neutralization_reason"],
        "oil_change_24h": oil_change_24h,
        "gold_change_24h": gold_change_24h,
        "copper_change_24h": copper_change_24h,
        "vix_change_24h": vix_change_24h,
        "sp500_change_24h": sp500_change_24h,
        "nasdaq_change_24h": nasdaq_change_24h,
        "us10y_change_bp": us10y_change_bp,
        "usdinr_change_24h": usdinr_change_24h,
        "realized_vol_5d": realized_vol_5d,
        "realized_vol_30d": realized_vol_30d,
        "oil_shock_score": oil_shock_score,
        "gold_risk_score": gold_risk_score,
        "copper_growth_signal": copper_growth_signal,
        "commodity_risk_score": commodity_risk_score,
        "volatility_shock_score": volatility_shock_score,
        "us_equity_risk_score": us_equity_risk_score,
        "rates_shock_score": rates_shock_score,
        "currency_shock_score": currency_shock_score,
        "risk_off_intensity": risk_off_intensity,
        "volatility_compression_score": volatility_compression_score,
        "volatility_explosion_probability": volatility_explosion_probability,
        "market_data_available": market_data_available,
        "market_data_stale": market_data_stale,
        "market_data_provider": global_market_snapshot.get("provider"),
        "market_data_as_of": global_market_snapshot.get("as_of"),
        "market_data_latest_timestamp": global_market_snapshot.get("latest_market_timestamp"),
        "market_data_neutral_fallback": market_neutral_fallback,
        "neutral_fallback": neutral_fallback,
        "news_data_available": news_data_available,
        "event_data_available": event_data_available,
        "issues": list(macro_news_state.get("issues", [])) + list(global_market_snapshot.get("issues", [])),
        "warnings": warnings,
        "as_of": session_context["as_of"],
    }
