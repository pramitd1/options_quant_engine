"""
Module: strike_selection_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by strike selection.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations


STRIKE_SELECTION_SCORE_CONFIG = {
    "strike_scoring_mode": "continuous",
    "atm_distance_pct": 0.20,
    "near_distance_pct": 0.40,
    "mid_distance_pct": 0.70,
    "far_distance_pct": 1.20,
    "moneyness_atm_score": 10,
    "moneyness_near_score": 8,
    "moneyness_mid_score": 5,
    "moneyness_far_score": 2,
    "moneyness_deep_penalty": -2,
    "call_above_spot_score": 2,
    "call_below_spot_score": 1,
    "put_below_spot_score": 2,
    "put_above_spot_score": 1,
    "premium_optimal_min": 80.0,
    "premium_optimal_max": 250.0,
    "premium_secondary_min": 40.0,
    "premium_secondary_max": 400.0,
    "premium_lower_tail_min": 20.0,
    "premium_optimal_score": 8,
    "premium_secondary_score": 6,
    "premium_upper_mid_score": 4,
    "premium_lower_tail_score": 3,
    "premium_default_score": 1,
    "premium_invalid_penalty": -10,
    "premium_over_budget_penalty": -5,
    "premium_near_budget_penalty": -2,
    "premium_near_budget_ratio": 0.85,
    "volume_high_threshold": 5000.0,
    "volume_medium_threshold": 2000.0,
    "volume_low_threshold": 500.0,
    "volume_high_score": 6,
    "volume_medium_score": 4,
    "volume_low_score": 2,
    "oi_high_threshold": 100000.0,
    "oi_medium_threshold": 50000.0,
    "oi_low_threshold": 10000.0,
    "oi_high_score": 6,
    "oi_medium_score": 4,
    "oi_low_score": 2,
    "wall_near_distance_points": 50.0,
    "wall_medium_distance_points": 100.0,
    "wall_near_penalty": -4,
    "wall_medium_penalty": -2,
    "gamma_cluster_near_distance_points": 50.0,
    "gamma_cluster_medium_distance_points": 100.0,
    "gamma_cluster_near_penalty": -2,
    "gamma_cluster_medium_penalty": -1,
    "gamma_cluster_far_bonus": 1,
    "iv_low_min": 10.0,
    "iv_low_max": 22.0,
    "iv_mid_max": 30.0,
    "iv_high_threshold": 40.0,
    "iv_low_score": 3,
    "iv_mid_score": 1,
    "iv_high_penalty": -2,
    "strike_window_steps": 8,
    # Bid-ask spread quality: penalise wide spreads relative to mid-price.
    # spread_ratio = (ask - bid) / mid.  Above the threshold, apply the penalty.
    "ba_spread_ratio_threshold": 0.04,
    "ba_spread_ratio_wide": 0.10,
    "ba_spread_narrow_bonus": 1,
    "ba_spread_wide_penalty": -3,
}


def get_strike_selection_score_config():
    """
    Purpose:
        Return the configuration bundle for strike selection score.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import resolve_mapping

    return resolve_mapping("strike_selection.core", STRIKE_SELECTION_SCORE_CONFIG)
