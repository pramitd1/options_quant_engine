"""
Module: signal_evaluation_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by signal evaluation.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""
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
    """
    Purpose:
        Bucket from thresholds into configured ranges.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        value (Any): Input associated with value.
        thresholds (Any): Input associated with thresholds.
        default_label (str): Label associated with default.
    
    Returns:
        Any: Bucket or regime label returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    if value is None:
        return None

    for minimum, label in thresholds:
        if value >= minimum:
            return label
    return default_label
