"""
Signal policy calibration for live engine scoring and direction selection.
"""

DIRECTION_VOTE_WEIGHTS = {
    "FLOW": 1.2,
    "HEDGING_BIAS": 1.1,
    "GAMMA_SQUEEZE": 0.9,
    "GAMMA_FLIP": 0.85,
    "DEALER_VOL": 0.8,
    "VANNA": 0.45,
    "CHARM": 0.45,
    "BACKTEST_FALLBACK": 0.6,
}

DIRECTION_MIN_SCORE = 1.75
DIRECTION_MIN_MARGIN = 0.7

TRADE_STRENGTH_WEIGHTS = {
    "flow_call_bullish": 20,
    "flow_call_bearish": -10,
    "flow_put_bearish": 20,
    "flow_put_bullish": -10,
    "smart_call_bullish": 15,
    "smart_call_bearish": -8,
    "smart_put_bearish": 15,
    "smart_put_bullish": -8,
    "hedging_acceleration_support": 10,
    "hedging_acceleration_conflict": -6,
    "gamma_regime_negative": 10,
    "gamma_regime_positive": 2,
    "gamma_regime_neutral": 5,
    "spot_flip_primary": 8,
    "spot_flip_secondary": 2,
    "wall_support_bonus": 5,
    "wall_resistance_penalty": -3,
    "liquidity_map_path_bonus": 4,
    "gamma_event_bonus": 10,
    "dealer_short_gamma_bonus": 10,
    "dealer_long_gamma_bonus": 5,
    "vol_expansion_bonus": 10,
    "normal_vol_bonus": 5,
    "liquidity_void_near_bonus": 10,
    "liquidity_void_far_bonus": 4,
    "vacuum_breakout_bonus": 10,
    "vacuum_watch_bonus": 4,
    "intraday_vol_expansion_bonus": 5,
    "intraday_gamma_decrease_bonus": 3,
}

CONSENSUS_SCORE_CONFIG = {
    "strong_alignment_bonus": 8,
    "moderate_alignment_bonus": 4,
    "conflict_penalty": -6,
}
