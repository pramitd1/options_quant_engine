"""
Module: option_efficiency_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by option efficiency.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionEfficiencyPolicyConfig:
    """
    Purpose:
        Dataclass representing OptionEfficiencyPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        neutral_score (int): Score value for neutral.
        high_efficiency_threshold (int): Threshold used to classify or trigger high efficiency.
        good_efficiency_threshold (int): Threshold used to classify or trigger good efficiency.
        weak_efficiency_threshold (int): Threshold used to classify or trigger weak efficiency.
        poor_efficiency_threshold (int): Threshold used to classify or trigger poor efficiency.
        overnight_block_threshold (int): Threshold used to classify or trigger overnight block.
        overnight_watch_threshold (int): Threshold used to classify or trigger overnight watch.
        iv_percent_unit_threshold (float): Threshold used to classify or trigger IV percent unit.
        minimum_time_to_expiry_years (float): Value supplied for minimum time to expiry years.
        min_effective_delta (float): Value supplied for min effective delta.
        max_effective_delta (float): Value supplied for max effective delta.
        fallback_delta (float): Value supplied for fallback delta.
        target_delta_floor (float): Floor value used for target delta.
        target_intrinsic_hurdle_multiplier (float): Multiplier applied to target intrinsic hurdle.
        strike_moneyness_atm_distance_pct (float): Value supplied for strike moneyness ATM distance percentage.
        payoff_far_otm_distance_ratio (float): Ratio used for payoff far otm distance.
        payoff_deep_itm_premium_ratio (float): Ratio used for payoff deep itm premium.
        trade_probability_floor (float): Floor value used for trade probability.
        trade_probability_ceiling (float): Value supplied for trade probability ceiling.
        convexity_base (float): Value supplied for convexity base.
        convexity_gamma_vol_weight (float): Weight applied to convexity gamma vol.
        convexity_dealer_pressure_weight (float): Weight applied to convexity dealer pressure.
        convexity_liquidity_vacuum_bonus (float): Bonus applied when convexity liquidity vacuum is active.
        option_move_probability_base (float): Value supplied for option move probability base.
        option_move_probability_weight (float): Weight applied to option move probability.
        target_reachability_boost (int): Value supplied for target reachability boost.
        target_reachability_moderate_boost (int): Value supplied for target reachability moderate boost.
        premium_penalty (int): Penalty applied when premium is active.
        strike_penalty (int): Penalty applied when strike is active.
        poor_efficiency_penalty (int): Penalty applied when poor efficiency is active.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
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
    """
    Purpose:
        Return the option-efficiency policy bundle used by contract scoring.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        OptionEfficiencyPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("option_efficiency.core", OptionEfficiencyPolicyConfig())
