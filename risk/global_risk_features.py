"""
Module: global_risk_features.py

Purpose:
    Build global risk features used by the risk overlay.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

import pandas as pd

from config.global_risk_policy import get_global_risk_policy_config
from utils.numerics import clip as _clip, safe_float as _safe_float  # noqa: F401


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
    "dxy_change_24h",
    "gift_nifty_change_24h",
    "realized_vol_5d",
    "realized_vol_30d",
]


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _coerce_timestamp(value):
    """
    Purpose:
        Parse flexible timestamp inputs into timezone-aware timestamps.

    Context:
        Used within the global risk features workflow. The module sits in the risk-overlay layer that can resize, downgrade, or block trade ideas.

    Inputs:
        value (Any): Raw value supplied by the caller.

    Returns:
        pd.Timestamp | None: Parsed timestamp or `None` when parsing fails.

    Notes:
        Internal helper that keeps the surrounding trading logic compact and readable.
    """
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
    """
    Purpose:
        Process market session context for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        cfg (Any): Input associated with cfg.
        as_of (Any): Input associated with as of.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Normalize holding profile into the repository-standard representation.
    
    Context:
        Internal helper in the `global risk features` module. It isolates one overlay heuristic so risk adjustments remain auditable.
    
    Inputs:
        value (Any): Raw value supplied by the caller.
    
    Returns:
        str: Value returned by the current workflow step.
    
    Notes:
        The helper intentionally produces bounded, serializable values so overlays can be inspected alongside the final trade decision.
    """
    normalized = str(value or "AUTO").upper().strip()
    return normalized or "AUTO"


def _oil_shock_score(change_24h, *, cfg):
    """
    Purpose:
        Process oil shock score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        change_24h (Any): Input associated with change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.oil_shock_extreme_change_pct:
        return cfg.oil_shock_extreme_score
    if change_24h > cfg.oil_shock_medium_change_pct:
        return cfg.oil_shock_medium_score
    if change_24h > cfg.oil_shock_notable_change_pct:
        return cfg.oil_shock_notable_score
    if change_24h < cfg.oil_shock_relief_change_pct:
        return cfg.oil_shock_relief_score
    return 0.0


def _gold_risk_score(change_24h, *, cfg):
    """
    Purpose:
        Process gold risk score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        change_24h (Any): Input associated with change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h > cfg.gold_risk_extreme_change_pct:
        return cfg.gold_risk_extreme_score
    if change_24h > cfg.gold_risk_medium_change_pct:
        return cfg.gold_risk_medium_score
    return 0.0


def _copper_growth_signal(change_24h, *, cfg):
    """
    Purpose:
        Process copper growth signal for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        change_24h (Any): Input associated with change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    change_24h = _safe_float(change_24h, 0.0)
    if change_24h < cfg.copper_growth_severe_drop_pct:
        return cfg.copper_growth_severe_score
    if change_24h < cfg.copper_growth_moderate_drop_pct:
        return cfg.copper_growth_moderate_score
    return 0.0


def _volatility_shock_score(vix_change_24h, *, cfg):
    """
    Purpose:
        Process volatility shock score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        vix_change_24h (Any): Input associated with vix change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    vix_change_24h = _safe_float(vix_change_24h, 0.0)
    if vix_change_24h > cfg.vix_shock_extreme_change_pct:
        return cfg.vix_shock_extreme_score
    if vix_change_24h > cfg.vix_shock_medium_change_pct:
        return cfg.vix_shock_medium_score
    if vix_change_24h > cfg.vix_shock_low_change_pct:
        return cfg.vix_shock_low_score
    return 0.0


def _us_equity_risk_score(sp500_change_24h, nasdaq_change_24h, *, cfg):
    """
    Purpose:
        Process us equity risk score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        sp500_change_24h (Any): Input associated with sp500 change 24h.
        nasdaq_change_24h (Any): Input associated with nasdaq change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Process rates shock score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        us10y_change_bp (Any): Input associated with us10y change basis points.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return cfg.rates_shock_score if _safe_float(us10y_change_bp, 0.0) > cfg.rates_shock_threshold_bp else 0.0


def _currency_shock_score(usdinr_change_24h, *, cfg):
    """
    Purpose:
        Process currency shock score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        usdinr_change_24h (Any): Input associated with usdinr change 24h.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return cfg.currency_shock_score_base if _safe_float(usdinr_change_24h, 0.0) > cfg.currency_shock_threshold_pct else 0.0


def _dxy_shock_score(dxy_change_24h, *, cfg):
    return cfg.dxy_shock_score_base if _safe_float(dxy_change_24h, 0.0) > cfg.dxy_shock_threshold_pct else 0.0


def _gift_nifty_lead_score(gift_nifty_change_24h, *, cfg):
    move = _safe_float(gift_nifty_change_24h, 0.0)
    if move >= cfg.gift_nifty_positive_threshold_pct:
        return cfg.gift_nifty_lead_score_base
    if move <= cfg.gift_nifty_negative_threshold_pct:
        return -cfg.gift_nifty_lead_score_base
    return 0.0


def _volatility_compression_score(realized_vol_5d, realized_vol_30d, *, cfg):
    """
    Purpose:
        Process volatility compression score for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        realized_vol_5d (Any): Input associated with realized vol 5d.
        realized_vol_30d (Any): Input associated with realized vol 30d.
        cfg (Any): Input associated with cfg.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Process market snapshot details for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        global_market_snapshot (Any): Cross-asset market snapshot used by the global-risk overlay.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    snapshot = global_market_snapshot if isinstance(global_market_snapshot, dict) else {}
    market_inputs = snapshot.get("market_inputs", {}) if isinstance(snapshot.get("market_inputs", {}), dict) else {}
    data_available = bool(snapshot.get("data_available", False))
    stale = bool(snapshot.get("stale", False))
    neutral_fallback = bool(snapshot.get("neutral_fallback", not data_available))

    return snapshot, market_inputs, data_available and not stale, stale, neutral_fallback


def _market_input_state(market_inputs, *, market_data_usable, market_data_stale):
    """
    Purpose:
        Process market input state for downstream use.
    
    Context:
        Internal helper within the risk-overlay layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        market_inputs (Any): Input associated with market inputs.
        market_data_usable (Any): Input associated with market data usable.
        market_data_stale (Any): Input associated with market data stale.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
        neutralized = True
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
    """
    Purpose:
        Build the global risk features used by downstream components.
    
    Context:
        Public function within the risk-overlay layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        macro_event_state (Any): Scheduled-event state produced by the macro-event layer.
        macro_news_state (Any): Headline-driven macro state produced by the news layer.
        global_market_snapshot (Any): Cross-asset market snapshot used by the global-risk overlay.
        holding_profile (str): Holding intent that determines whether overnight rules should be considered.
        as_of (Any): Input associated with as of.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    india_vix_change_24h = (
        _safe_float(market_inputs.get("india_vix_change_24h"), None)
        if market_data_available
        else None
    )
    india_vix_level = (
        _safe_float(market_inputs.get("india_vix_level"), None)
        if market_data_available
        else None
    )
    sp500_change_24h = _safe_float(effective_market_inputs.get("sp500_change_24h"), None)
    nasdaq_change_24h = _safe_float(effective_market_inputs.get("nasdaq_change_24h"), None)
    us10y_change_bp = _safe_float(effective_market_inputs.get("us10y_change_bp"), None)
    usdinr_change_24h = _safe_float(effective_market_inputs.get("usdinr_change_24h"), None)
    dxy_change_24h = _safe_float(effective_market_inputs.get("dxy_change_24h"), None)
    gift_nifty_change_24h = _safe_float(effective_market_inputs.get("gift_nifty_change_24h"), None)
    realized_vol_5d = _safe_float(effective_market_inputs.get("realized_vol_5d"), None)
    realized_vol_30d = _safe_float(effective_market_inputs.get("realized_vol_30d"), None)

    # These components capture different ways global conditions can leak into
    # local option behavior: commodities, volatility, rates, FX, and events.
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
    dxy_shock_score = _dxy_shock_score(dxy_change_24h, cfg=cfg)
    gift_nifty_lead_score = _gift_nifty_lead_score(gift_nifty_change_24h, cfg=cfg)
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
    # `risk_off_intensity` is the broad stress score, while
    # `volatility_explosion_probability` focuses on compression plus shock.
    risk_off_intensity = round(
        _clip(
            (cfg.risk_off_intensity_vol_weight * volatility_shock_score)
            + (cfg.risk_off_intensity_us_equity_weight * us_equity_risk_score)
            + (cfg.risk_off_intensity_rates_weight * rates_shock_score)
            + (cfg.risk_off_intensity_currency_weight * currency_shock_score)
            + (cfg.risk_off_intensity_dxy_weight * dxy_shock_score)
            + (cfg.risk_off_intensity_gift_nifty_weight * max(-gift_nifty_lead_score, 0.0))
            + (cfg.risk_off_intensity_commodity_weight * commodity_stress_component)
            + (cfg.risk_off_intensity_macro_event_weight * macro_event_risk_norm),
            0.0,
            1.0,
        ),
        4,
    )
    headline_data_stale = bool(macro_news_state.get("neutral_fallback", True)) and any(
        "stale" in str(item).lower() for item in (macro_news_state.get("warnings") or [])
    )
    global_macro_data_stale = bool(market_data_stale or market_state["market_features_neutralized"])
    event_uncertainty_score = _clip(
        _safe_float(((macro_news_state.get("event_features") or {}).get("event_uncertainty_score")), 0.0) / 100.0,
        0.0,
        1.0,
    )
    macro_uncertainty_score = round(
        _clip(
            (cfg.macro_uncertainty_event_weight * event_uncertainty_score)
            + (cfg.macro_uncertainty_headline_velocity_weight * _clip(_safe_float(macro_news_state.get("headline_velocity"), 0.0), 0.0, 1.0))
            + (cfg.macro_uncertainty_headline_stale_weight * float(headline_data_stale))
            + (cfg.macro_uncertainty_market_stale_weight * float(global_macro_data_stale)),
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
    gift_nifty_proxy_in_use = any("gift_nifty_proxy_in_use" in str(item) for item in warnings)

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
        "us_vix_change_24h": vix_change_24h,
        "vix_change_24h": vix_change_24h,
        "india_vix_change_24h": india_vix_change_24h,
        "india_vix_level": india_vix_level,
        "sp500_change_24h": sp500_change_24h,
        "nasdaq_change_24h": nasdaq_change_24h,
        "us10y_change_bp": us10y_change_bp,
        "usdinr_change_24h": usdinr_change_24h,
        "dxy_change_24h": dxy_change_24h,
        "gift_nifty_change_24h": gift_nifty_change_24h,
        "gift_nifty_proxy_in_use": gift_nifty_proxy_in_use,
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
        "dxy_shock_score": dxy_shock_score,
        "gift_nifty_lead_score": gift_nifty_lead_score,
        "risk_off_intensity": risk_off_intensity,
        "volatility_compression_score": volatility_compression_score,
        "volatility_explosion_probability": volatility_explosion_probability,
        "headline_data_stale": headline_data_stale,
        "global_macro_data_stale": global_macro_data_stale,
        "event_uncertainty_score": event_uncertainty_score,
        "macro_uncertainty_score": macro_uncertainty_score,
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
