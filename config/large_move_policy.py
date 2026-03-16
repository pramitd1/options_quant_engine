"""
Module: large_move_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by large move.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations


LARGE_MOVE_PROBABILITY_CONFIG = {
    "base_probability": 0.22,
    "short_gamma_bonus": 0.14,
    "long_gamma_penalty": -0.08,
    "breakout_zone_bonus": 0.16,
    "near_vacuum_bonus": 0.07,
    "acceleration_bias_bonus": 0.14,
    "pinning_bias_penalty": -0.06,
    "directional_flow_bonus": 0.08,
    "neutral_flow_penalty": -0.02,
    "gamma_flip_distance_weight": 0.12,
    "vacuum_strength_weight": 0.12,
    "hedging_flow_ratio_weight": 0.10,
    "smart_money_flow_weight": 0.08,
    "atm_iv_percentile_weight": 0.07,
    "intraday_range_weight": 0.08,
    "directional_conflict_penalty": -0.10,
    "positive_gamma_breakout_penalty": -0.05,
    "neutral_gamma_breakout_bonus": 0.02,
    "probability_floor": 0.05,
    "probability_ceiling": 0.95,
}


def get_large_move_probability_config():
    """
    Purpose:
        Return the large-move probability policy bundle used by heuristic models.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("large_move_probability.core", LARGE_MOVE_PROBABILITY_CONFIG)
