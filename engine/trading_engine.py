"""
Backward-compatible trading engine facade.

Signal assembly now lives in `engine.signal_engine`, while helper logic lives in
`engine.trading_support`.
"""

from __future__ import annotations

from engine.signal_engine import generate_trade
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
    classify_spot_vs_flip,
    classify_spot_vs_flip_for_symbol,
    decide_direction,
    derive_dealer_pressure_trade_modifiers,
    derive_gamma_vol_trade_modifiers,
    derive_global_risk_trade_modifiers,
    derive_option_efficiency_trade_modifiers,
    normalize_option_chain,
)
