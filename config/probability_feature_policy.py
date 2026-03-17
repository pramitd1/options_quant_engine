"""
Module: probability_feature_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by probability feature.

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
class ProbabilityFeaturePolicyConfig:
    """
    Purpose:
        Dataclass representing ProbabilityFeaturePolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        vacuum_breakout_strength (float): Value supplied for vacuum breakout strength.
        vacuum_near_strength (float): Value supplied for vacuum near strength.
        vacuum_watch_strength (float): Value supplied for vacuum watch strength.
        vacuum_default_strength (float): Value supplied for vacuum default strength.
        vacuum_gap_pct_cap (float): Cap applied to vacuum gap percentage.
        vacuum_gap_base_weight (float): Weight applied to vacuum gap base.
        vacuum_gap_proximity_weight (float): Weight applied to vacuum gap proximity.
        vacuum_void_count_cap (int): Cap applied to vacuum void count.
        vacuum_void_increment (float): Value supplied for vacuum void increment.
        hedging_bias_upside_acceleration_score (float): Score value for hedging bias upside acceleration.
        hedging_bias_downside_acceleration_score (float): Score value for hedging bias downside acceleration.
        hedging_bias_upside_pinning_score (float): Score value for hedging bias upside pinning.
        hedging_bias_downside_pinning_score (float): Score value for hedging bias downside pinning.
        hedging_bias_pinning_score (float): Score value for hedging bias pinning.
        smart_money_bullish_score (float): Score value for smart money bullish.
        smart_money_bearish_score (float): Score value for smart money bearish.
        smart_money_neutral_score (float): Score value for smart money neutral.
        smart_money_categorical_weight (float): Weight applied to smart money categorical.
        smart_money_flow_imbalance_weight (float): Weight applied to smart money flow imbalance.
        intraday_range_anchor_multiplier (float): Multiplier applied to intraday range anchor.
        intraday_range_baseline_floor_pct (float): Value supplied for intraday range baseline floor percentage.
        intraday_range_denominator_floor_pct (float): Value supplied for intraday range denominator floor percentage.
        intraday_range_clip_cap (float): Cap applied to intraday range clip.
        atm_iv_low (float): Value supplied for ATM IV low.
        atm_iv_high (float): Value supplied for ATM IV high.
        probability_default_rule (float): Value supplied for probability default rule.
        probability_floor (float): Floor value used for probability.
        probability_ceiling (float): Value supplied for probability ceiling.
        probability_rule_weight (float): Weight applied to probability rule.
        probability_ml_weight (float): Weight applied to probability ML.
        probability_intercept (float): Value supplied for probability intercept.
        probability_scale (float): Value supplied for probability scale.
        categorical_flow_weight (float): Weight applied to categorical flow.
        smart_money_flow_weight (float): Weight applied to smart money flow.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    vacuum_breakout_strength: float = 0.85
    vacuum_near_strength: float = 0.60
    vacuum_watch_strength: float = 0.40
    vacuum_default_strength: float = 0.15
    vacuum_gap_pct_cap: float = 1.5
    vacuum_gap_base_weight: float = 0.6
    vacuum_gap_proximity_weight: float = 0.4
    vacuum_void_count_cap: int = 5
    vacuum_void_increment: float = 0.03
    hedging_bias_upside_acceleration_score: float = 0.75
    hedging_bias_downside_acceleration_score: float = -0.75
    hedging_bias_upside_pinning_score: float = 0.20
    hedging_bias_downside_pinning_score: float = -0.20
    hedging_bias_pinning_score: float = 0.0
    smart_money_bullish_score: float = 0.70
    smart_money_bearish_score: float = -0.70
    smart_money_neutral_score: float = 0.0
    smart_money_categorical_weight: float = 0.5
    smart_money_flow_imbalance_weight: float = 0.5
    intraday_range_anchor_multiplier: float = 1.5
    intraday_range_baseline_floor_pct: float = 0.9
    intraday_range_denominator_floor_pct: float = 0.25
    intraday_range_clip_cap: float = 1.5
    atm_iv_low: float = 8.0
    atm_iv_high: float = 28.0
    probability_default_rule: float = 0.22
    probability_floor: float = 0.05
    probability_ceiling: float = 0.95
    probability_rule_weight: float = 0.35
    probability_ml_weight: float = 0.65
    probability_intercept: float = 0.10
    probability_scale: float = 0.80
    categorical_flow_weight: float = 0.5
    smart_money_flow_weight: float = 0.5

    # Post-blend logistic recalibration — stretches the compressed hybrid
    # probability distribution so that confident setups reach higher values
    # and weak setups are pushed lower, improving discrimination.
    calibration_enabled: bool = True
    calibration_midpoint: float = 0.40
    calibration_steepness: float = 5.0


def get_probability_feature_policy_config() -> ProbabilityFeaturePolicyConfig:
    """
    Purpose:
        Return the probability-feature policy bundle used by the move-probability pipeline.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        ProbabilityFeaturePolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.probability", ProbabilityFeaturePolicyConfig())
