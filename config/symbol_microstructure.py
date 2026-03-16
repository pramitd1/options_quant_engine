"""
Module: symbol_microstructure.py

Purpose:
    Define symbol-specific microstructure assumptions used by analytics and probability models.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations

DEFAULT_MICROSTRUCTURE_CONFIG = {
    "flip_buffer_points": 25.0,
    "wall_proximity_points": 50.0,
    "range_baseline_floor_pct": 0.9,
    "range_expansion_low": 0.20,
    "range_expansion_moderate": 0.40,
    "range_expansion_strong": 0.70,
    "range_expansion_cold": 0.08,
}

SYMBOL_MICROSTRUCTURE_CONFIG = {
    "NIFTY": {
        "flip_buffer_points": 25.0,
        "wall_proximity_points": 50.0,
        "range_baseline_floor_pct": 0.9,
    },
    "BANKNIFTY": {
        "flip_buffer_points": 60.0,
        "wall_proximity_points": 100.0,
        "range_baseline_floor_pct": 1.1,
    },
    "FINNIFTY": {
        "flip_buffer_points": 35.0,
        "wall_proximity_points": 60.0,
        "range_baseline_floor_pct": 0.95,
    },
}


def get_microstructure_config(symbol: str | None) -> dict:
    """
    Purpose:
        Return the configuration bundle for microstructure.
    
    Context:
        Public function within the configuration layer that centralizes policy defaults and thresholds. It exposes a reusable workflow step to other parts of the repository.
    
    Inputs:
        symbol (str | None): Value supplied for symbol.
    
    Returns:
        dict: Value resolved or fetched for downstream use.
    
    Notes:
        Centralizing this policy contract keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    normalized = str(symbol or "").upper().strip()

    config = dict(DEFAULT_MICROSTRUCTURE_CONFIG)
    if normalized in SYMBOL_MICROSTRUCTURE_CONFIG:
        config.update(SYMBOL_MICROSTRUCTURE_CONFIG[normalized])
        return config

    # Default stock behavior: smaller absolute thresholds than index options.
    if normalized and normalized not in SYMBOL_MICROSTRUCTURE_CONFIG:
        config.update({
            "flip_buffer_points": 8.0,
            "wall_proximity_points": 12.0,
            "range_baseline_floor_pct": 1.4,
        })

    return config
