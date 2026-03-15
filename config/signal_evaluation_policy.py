from __future__ import annotations


SIGNAL_EVALUATION_WINDOW_MINUTES = 120
SIGNAL_EVALUATION_HORIZON_MINUTES = (5, 15, 30, 60)
TRADE_STRENGTH_BUCKETS = (
    (80.0, "80_100"),
    (65.0, "65_79"),
    (50.0, "50_64"),
    (35.0, "35_49"),
)
MOVE_PROBABILITY_BUCKETS = (
    (0.80, "0.80_1.00"),
    (0.65, "0.65_0.79"),
    (0.50, "0.50_0.64"),
    (0.35, "0.35_0.49"),
)


def bucket_from_thresholds(value, thresholds, default_label: str):
    if value is None:
        return None

    for minimum, label in thresholds:
        if value >= minimum:
            return label
    return default_label
