"""
Symbol-aware live trading thresholds for intraday microstructure logic.
"""

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
