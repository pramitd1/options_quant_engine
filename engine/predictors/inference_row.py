"""
Helpers to build ML-inference input rows from runtime engine context.
"""
from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_inference_row(market_ctx: dict[str, Any], raw_probability_state: dict[str, Any]) -> dict[str, Any]:
    """
    Build a signal-row-like payload for research.ml_models.ml_inference.infer_single.

    The research inference stack expects dict-style signal rows with named
    fields (not a numpy feature vector). This helper maps runtime state into
    that schema so all research predictors call infer_single consistently.
    """
    market_state = market_ctx.get("market_state") if isinstance(market_ctx, dict) else {}
    market_state = market_state if isinstance(market_state, dict) else {}

    components = raw_probability_state.get("components") if isinstance(raw_probability_state, dict) else {}
    components = components if isinstance(components, dict) else {}

    global_context = market_ctx.get("global_context") if isinstance(market_ctx, dict) else {}
    global_context = global_context if isinstance(global_context, dict) else {}

    day_high = components.get("day_high")
    day_low = components.get("day_low")
    day_open = components.get("day_open")
    prev_close = components.get("prev_close")
    spot = market_ctx.get("spot")

    gap_pct = 0.0
    close_vs_prev_close_pct = 0.0
    if prev_close not in (None, 0):
        gap_pct = ((_safe_float(day_open) - _safe_float(prev_close)) / _safe_float(prev_close)) * 100.0
        close_vs_prev_close_pct = ((_safe_float(spot) - _safe_float(prev_close)) / _safe_float(prev_close)) * 100.0

    spot_in_day_range = 0.5
    if day_high is not None and day_low is not None and _safe_float(day_high) > _safe_float(day_low):
        spot_in_day_range = (
            (_safe_float(spot) - _safe_float(day_low))
            / (_safe_float(day_high) - _safe_float(day_low))
        )

    greek_exposures = market_state.get("greek_exposures")
    greek_exposures = greek_exposures if isinstance(greek_exposures, dict) else {}

    return {
        "gamma_regime": market_state.get("gamma_regime"),
        "final_flow_signal": market_state.get("final_flow_signal"),
        "volatility_regime": market_state.get("vol_regime"),
        "dealer_hedging_bias": market_state.get("hedging_bias"),
        "spot_vs_flip": market_state.get("spot_vs_flip"),
        "liquidity_vacuum_state": market_state.get("vacuum_state"),
        "move_probability": raw_probability_state.get("hybrid_move_probability"),
        "gamma_flip_distance_pct": components.get("gamma_flip_distance_pct"),
        "vacuum_strength": components.get("vacuum_strength"),
        "hedging_flow_ratio": components.get("hedging_flow_ratio"),
        "smart_money_flow_score": components.get("smart_money_flow_score"),
        "atm_iv_percentile": components.get("atm_iv_percentile"),
        "intraday_range_pct": components.get("intraday_range_pct"),
        "lookback_avg_range_pct": components.get("lookback_avg_range_pct"),
        "gap_pct": gap_pct,
        "close_vs_prev_close_pct": close_vs_prev_close_pct,
        "spot_in_day_range": spot_in_day_range,
        "dealer_position": market_state.get("dealer_pos"),
        "vanna_regime": greek_exposures.get("vanna_regime"),
        "charm_regime": greek_exposures.get("charm_regime"),
        "macro_event_risk_score": global_context.get("macro_event_risk_score", 0.0),
        "macro_regime": global_context.get("macro_regime", "NO_EVENT"),
        "india_vix_level": global_context.get("india_vix_level"),
        "india_vix_change_24h": global_context.get("india_vix_change_24h"),
        "oil_shock_score": global_context.get("oil_shock_score"),
        "commodity_risk_score": global_context.get("commodity_risk_score"),
        "volatility_shock_score": global_context.get("volatility_shock_score"),
        "days_to_expiry": global_context.get("days_to_expiry"),
    }
