"""
Backward-compatible trading engine support facade.

The implementation now lives in `engine.trading_support` so helper domains can
evolve independently without changing existing engine imports.
"""

from __future__ import annotations

from engine.trading_support import (
    _blend_move_probability,
    _call_first,
    _categorical_flow_score,
    _clean_zone_list,
    _clip,
    _collect_market_state,
    _compute_atm_iv_percentile,
    _compute_data_quality,
    _compute_gamma_flip_distance_pct,
    _compute_intraday_range_pct,
    _compute_probability_state,
    _compute_signal_state,
    _extract_hedge_flow_value,
    _extract_nearest_vacuum_gap_pct,
    _extract_probability,
    _get_move_predictor,
    _map_hedging_flow_ratio,
    _map_smart_money_flow_score,
    _map_vacuum_strength,
    _normalize_validation_dict,
    _safe_div,
    _safe_float,
    _summarize_market_gamma,
    _to_python_number,
    classify_execution_regime,
    classify_signal_quality,
    classify_signal_regime,
    classify_spot_vs_flip,
    classify_spot_vs_flip_for_symbol,
    decide_direction,
    derive_dealer_pressure_trade_modifiers,
    derive_gamma_vol_trade_modifiers,
    derive_global_risk_trade_modifiers,
    derive_option_efficiency_trade_modifiers,
    normalize_flow_signal,
    normalize_option_chain,
)
