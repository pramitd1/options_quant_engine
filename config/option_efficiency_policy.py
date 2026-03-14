"""
Centralized configuration for the expected move / option efficiency overlay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionEfficiencyPolicyConfig:
    neutral_score: int = 50
    high_efficiency_threshold: int = 75
    good_efficiency_threshold: int = 62
    weak_efficiency_threshold: int = 40
    poor_efficiency_threshold: int = 28
    overnight_block_threshold: int = 5
    overnight_watch_threshold: int = 3
    iv_percent_unit_threshold: float = 1.5
    minimum_time_to_expiry_years: float = 0.000114155
    min_effective_delta: float = 0.25
    max_effective_delta: float = 0.85
    fallback_delta: float = 0.35
    target_delta_floor: float = 0.25
    target_intrinsic_hurdle_multiplier: float = 0.75
    strike_moneyness_atm_distance_pct: float = 0.20
    payoff_far_otm_distance_ratio: float = 1.0
    payoff_deep_itm_premium_ratio: float = 0.65
    trade_probability_floor: float = 0.05
    trade_probability_ceiling: float = 0.95
    convexity_base: float = 1.0
    convexity_gamma_vol_weight: float = 0.22
    convexity_dealer_pressure_weight: float = 0.16
    convexity_liquidity_vacuum_bonus: float = 0.08
    option_move_probability_base: float = 0.75
    option_move_probability_weight: float = 0.50
    target_reachability_boost: int = 3
    target_reachability_moderate_boost: int = 1
    premium_penalty: int = -3
    strike_penalty: int = -2
    poor_efficiency_penalty: int = -4


OPTION_EFFICIENCY_POLICY_CONFIG = OptionEfficiencyPolicyConfig()


def get_option_efficiency_policy_config() -> OptionEfficiencyPolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("option_efficiency.core", OptionEfficiencyPolicyConfig())
