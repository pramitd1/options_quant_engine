"""
Configurable scoring policy for realized signal evaluation.
"""

SIGNAL_EVALUATION_SCORE_WEIGHTS = {
    "direction_score": 0.30,
    "magnitude_score": 0.25,
    "timing_score": 0.20,
    "tradeability_score": 0.25,
}

SIGNAL_EVALUATION_DIRECTION_WEIGHTS = {
    "correct_5m": 1.0,
    "correct_15m": 1.2,
    "correct_30m": 1.1,
    "correct_60m": 1.0,
    "correct_session_close": 1.0,
}

SIGNAL_EVALUATION_TIMING_WEIGHTS = {
    "realized_return_5m": 1.4,
    "realized_return_15m": 1.2,
    "realized_return_30m": 1.0,
    "realized_return_60m": 0.8,
}

SIGNAL_EVALUATION_THRESHOLDS = {
    "magnitude_vs_range_weak": 0.20,
    "magnitude_vs_range_good": 0.50,
    "magnitude_vs_range_strong": 1.00,
    "timing_positive_return_floor": 0.0005,
    "tradeability_ratio_floor": 0.75,
    "tradeability_ratio_good": 1.50,
    "tradeability_ratio_strong": 2.50,
}
