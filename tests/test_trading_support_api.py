from __future__ import annotations

import engine.trading_support as trading_support
import engine.trading_engine_support as support_facade


EXPECTED_EXPORTS = {
    "_clip",
    "_safe_float",
    "_safe_div",
    "_call_first",
    "_to_python_number",
    "_clean_zone_list",
    "_normalize_validation_dict",
    "normalize_option_chain",
    "_collect_market_state",
    "_summarize_market_gamma",
    "_compute_probability_state",
    "_compute_gamma_flip_distance_pct",
    "_compute_intraday_range_pct",
    "_compute_atm_iv_percentile",
    "_blend_move_probability",
    "_get_move_predictor",
    "_extract_nearest_vacuum_gap_pct",
    "_extract_hedge_flow_value",
    "_categorical_flow_score",
    "_extract_probability",
    "_compute_data_quality",
    "classify_spot_vs_flip",
    "classify_spot_vs_flip_for_symbol",
    "classify_signal_quality",
    "classify_signal_regime",
    "classify_execution_regime",
    "normalize_flow_signal",
    "decide_direction",
    "_compute_signal_state",
    "derive_global_risk_trade_modifiers",
    "derive_gamma_vol_trade_modifiers",
    "derive_dealer_pressure_trade_modifiers",
    "derive_option_efficiency_trade_modifiers",
}


def test_trading_support_exports_expected_api():
    missing = sorted(name for name in EXPECTED_EXPORTS if not hasattr(trading_support, name))
    assert missing == []


def test_trading_engine_support_facade_matches_package_exports():
    mismatched = sorted(
        name
        for name in EXPECTED_EXPORTS
        if getattr(support_facade, name) is not getattr(trading_support, name)
    )
    assert mismatched == []
