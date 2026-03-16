"""
Module: dealer_hedging_pressure_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by dealer hedging pressure.

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
class DealerHedgingPressurePolicyConfig:
    """
    Purpose:
        Dataclass representing DealerHedgingPressurePolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        gamma_short_acceleration_score (float): Score value for gamma short acceleration.
        gamma_long_pinning_score (float): Score value for gamma long pinning.
        dealer_short_gamma_acceleration_bonus (float): Bonus applied when dealer short gamma acceleration is active.
        dealer_long_gamma_pinning_bonus (float): Bonus applied when dealer long gamma pinning is active.
        flip_unknown_context_score (float): Score value for flip unknown context.
        flip_at_score (float): Score value for flip at.
        flip_distance_1_pct (float): Value supplied for flip distance 1 percentage.
        flip_distance_1_score (float): Score value for flip distance 1.
        flip_distance_2_pct (float): Value supplied for flip distance 2 percentage.
        flip_distance_2_score (float): Score value for flip distance 2.
        flip_distance_3_pct (float): Value supplied for flip distance 3 percentage.
        flip_distance_3_score (float): Score value for flip distance 3.
        flip_distance_4_pct (float): Value supplied for flip distance 4 percentage.
        flip_distance_4_score (float): Score value for flip distance 4.
        flip_distance_far_score (float): Score value for flip distance far.
        bias_acceleration_score (float): Score value for bias acceleration.
        bias_partial_pinning_score (float): Score value for bias partial pinning.
        bias_full_pinning_score (float): Score value for bias full pinning.
        hedging_flow_score (float): Score value for hedging flow.
        intraday_vol_expansion_score (float): Score value for intraday vol expansion.
        intraday_gamma_decrease_score (float): Score value for intraday gamma decrease.
        intraday_pinning_score (float): Score value for intraday pinning.
        flow_confirmation_hit_score (float): Score value for flow confirmation hit.
        flow_confirmation_cap (float): Cap applied to flow confirmation.
        level_distance_near_pct (float): Value supplied for level distance near percentage.
        level_distance_mid_pct (float): Value supplied for level distance mid percentage.
        level_distance_far_pct (float): Value supplied for level distance far percentage.
        level_concentration_near_score (float): Score value for level concentration near.
        level_concentration_mid_score (float): Score value for level concentration mid.
        level_concentration_far_score (float): Score value for level concentration far.
        level_concentration_loose_score (float): Score value for level concentration loose.
        vacuum_breakout_score (float): Score value for vacuum breakout.
        vacuum_near_score (float): Score value for vacuum near.
        vacuum_watch_score (float): Score value for vacuum watch.
        acceleration_structure_pinning_dampener (float): Value supplied for acceleration structure pinning dampener.
        pinning_structure_concentration_weight (float): Weight applied to pinning structure concentration.
        pinning_structure_cluster_bonus (float): Bonus applied when pinning structure cluster is active.
        macro_global_event_weight (float): Weight applied to macro global event.
        macro_global_state_weight (float): Weight applied to macro global state.
        macro_global_explosion_weight (float): Weight applied to macro global explosion.
        macro_global_gamma_vol_weight (float): Weight applied to macro global gamma vol.
        macro_global_state_vol_shock (float): Value supplied for macro global state vol shock.
        macro_global_state_event_lockdown (float): Value supplied for macro global state event lockdown.
        macro_global_state_risk_off (float): Value supplied for macro global state risk off.
        macro_global_state_neutral (float): Value supplied for macro global state neutral.
        macro_global_state_unknown (float): Value supplied for macro global state unknown.
        far_level_high_pct (float): Value supplied for far level high percentage.
        far_level_watch_pct (float): Value supplied for far level watch percentage.
        far_level_high_dampener (float): Value supplied for far level high dampener.
        far_level_watch_dampener (float): Value supplied for far level watch dampener.
        intraday_range_norm_divisor (float): Value supplied for intraday range norm divisor.
        acceleration_base_gamma_weight (float): Weight applied to acceleration base gamma.
        acceleration_base_flip_weight (float): Weight applied to acceleration base flip.
        acceleration_base_structure_weight (float): Weight applied to acceleration base structure.
        acceleration_base_intraday_weight (float): Weight applied to acceleration base intraday.
        acceleration_base_range_weight (float): Weight applied to acceleration base range.
        pinning_base_gamma_weight (float): Weight applied to pinning base gamma.
        pinning_base_bias_weight (float): Weight applied to pinning base bias.
        pinning_base_structure_weight (float): Weight applied to pinning base structure.
        pinning_base_intraday_weight (float): Weight applied to pinning base intraday.
        directional_acceleration_weight (float): Weight applied to directional acceleration.
        directional_bias_weight (float): Weight applied to directional bias.
        directional_flow_weight (float): Weight applied to directional flow.
        directional_confirmation_weight (float): Weight applied to directional confirmation.
        directional_macro_weight (float): Weight applied to directional macro.
        directional_gamma_vol_weight (float): Weight applied to directional gamma vol.
        directional_intraday_weight (float): Weight applied to directional intraday.
        directional_pinning_penalty_weight (float): Weight applied to directional pinning penalty.
        pinning_score_base_weight (float): Weight applied to pinning score base.
        pinning_score_bias_weight (float): Weight applied to pinning score bias.
        pinning_score_structure_weight (float): Weight applied to pinning score structure.
        pinning_score_flip_inverse_weight (float): Weight applied to pinning score flip inverse.
        pinning_score_gamma_vol_inverse_weight (float): Weight applied to pinning score gamma vol inverse.
        pinning_score_intraday_weight (float): Weight applied to pinning score intraday.
        pinning_score_acceleration_penalty_weight (float): Weight applied to pinning score acceleration penalty.
        pinning_score_far_level_penalty_weight (float): Weight applied to pinning score far level penalty.
        normalized_pressure_directional_weight (float): Weight applied to normalized pressure directional.
        normalized_pressure_two_sided_weight (float): Weight applied to normalized pressure two sided.
        normalized_pressure_pinning_weight (float): Weight applied to normalized pressure pinning.
        normalized_pressure_acceleration_weight (float): Weight applied to normalized pressure acceleration.
        overnight_directional_weight (float): Weight applied to overnight directional.
        overnight_two_sided_weight (float): Weight applied to overnight two sided.
        overnight_macro_weight (float): Weight applied to overnight macro.
        overnight_gamma_vol_weight (float): Weight applied to overnight gamma vol.
        overnight_event_weight (float): Weight applied to overnight event.
        overnight_context_weight (float): Weight applied to overnight context.
        partial_coverage_warning_threshold (float): Threshold used to classify or trigger partial coverage warning.
        upside_pressure_threshold (float): Threshold used to classify or trigger upside pressure.
        downside_pressure_threshold (float): Threshold used to classify or trigger downside pressure.
        pinning_pressure_threshold (float): Threshold used to classify or trigger pinning pressure.
        two_sided_threshold (float): Threshold used to classify or trigger two sided.
        state_balance_tolerance (float): Value supplied for state balance tolerance.
        moderate_pressure_threshold (int): Threshold used to classify or trigger moderate pressure.
        high_pressure_threshold (int): Threshold used to classify or trigger high pressure.
        extreme_pressure_threshold (int): Threshold used to classify or trigger extreme pressure.
        overnight_block_threshold (int): Threshold used to classify or trigger overnight block.
        overnight_watch_threshold (int): Threshold used to classify or trigger overnight watch.
        acceleration_adjustment_high (int): Value supplied for acceleration adjustment high.
        acceleration_adjustment_extreme (int): Value supplied for acceleration adjustment extreme.
        pinning_adjustment (int): Value supplied for pinning adjustment.
        two_sided_adjustment (int): Value supplied for two sided adjustment.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    gamma_short_acceleration_score: float = 0.7
    gamma_long_pinning_score: float = 0.65
    dealer_short_gamma_acceleration_bonus: float = 0.25
    dealer_long_gamma_pinning_bonus: float = 0.25
    flip_unknown_context_score: float = 0.22
    flip_at_score: float = 1.0
    flip_distance_1_pct: float = 0.10
    flip_distance_1_score: float = 1.0
    flip_distance_2_pct: float = 0.25
    flip_distance_2_score: float = 0.82
    flip_distance_3_pct: float = 0.50
    flip_distance_3_score: float = 0.60
    flip_distance_4_pct: float = 0.90
    flip_distance_4_score: float = 0.36
    flip_distance_far_score: float = 0.10
    bias_acceleration_score: float = 0.95
    bias_partial_pinning_score: float = 0.65
    bias_full_pinning_score: float = 0.85
    hedging_flow_score: float = 0.55
    intraday_vol_expansion_score: float = 0.7
    intraday_gamma_decrease_score: float = 0.55
    intraday_pinning_score: float = 0.5
    flow_confirmation_hit_score: float = 0.18
    flow_confirmation_cap: float = 0.4
    level_distance_near_pct: float = 0.12
    level_distance_mid_pct: float = 0.25
    level_distance_far_pct: float = 0.50
    level_concentration_near_score: float = 0.9
    level_concentration_mid_score: float = 0.72
    level_concentration_far_score: float = 0.45
    level_concentration_loose_score: float = 0.15
    vacuum_breakout_score: float = 0.9
    vacuum_near_score: float = 0.7
    vacuum_watch_score: float = 0.45
    acceleration_structure_pinning_dampener: float = 0.55
    pinning_structure_concentration_weight: float = 0.65
    pinning_structure_cluster_bonus: float = 0.15
    macro_global_event_weight: float = 0.40
    macro_global_state_weight: float = 0.25
    macro_global_explosion_weight: float = 0.20
    macro_global_gamma_vol_weight: float = 0.15
    macro_global_state_vol_shock: float = 1.0
    macro_global_state_event_lockdown: float = 0.9
    macro_global_state_risk_off: float = 0.55
    macro_global_state_neutral: float = 0.12
    macro_global_state_unknown: float = 0.08
    far_level_high_pct: float = 0.60
    far_level_watch_pct: float = 0.35
    far_level_high_dampener: float = 0.35
    far_level_watch_dampener: float = 0.18
    intraday_range_norm_divisor: float = 1.2
    acceleration_base_gamma_weight: float = 0.42
    acceleration_base_flip_weight: float = 0.24
    acceleration_base_structure_weight: float = 0.18
    acceleration_base_intraday_weight: float = 0.10
    acceleration_base_range_weight: float = 0.06
    pinning_base_gamma_weight: float = 0.48
    pinning_base_bias_weight: float = 0.22
    pinning_base_structure_weight: float = 0.22
    pinning_base_intraday_weight: float = 0.08
    directional_acceleration_weight: float = 0.38
    directional_bias_weight: float = 0.20
    directional_flow_weight: float = 0.10
    directional_confirmation_weight: float = 0.10
    directional_macro_weight: float = 0.08
    directional_gamma_vol_weight: float = 0.08
    directional_intraday_weight: float = 0.06
    directional_pinning_penalty_weight: float = 0.24
    pinning_score_base_weight: float = 0.52
    pinning_score_bias_weight: float = 0.12
    pinning_score_structure_weight: float = 0.12
    pinning_score_flip_inverse_weight: float = 0.10
    pinning_score_gamma_vol_inverse_weight: float = 0.08
    pinning_score_intraday_weight: float = 0.06
    pinning_score_acceleration_penalty_weight: float = 0.24
    pinning_score_far_level_penalty_weight: float = 0.25
    normalized_pressure_directional_weight: float = 0.48
    normalized_pressure_two_sided_weight: float = 0.20
    normalized_pressure_pinning_weight: float = 0.14
    normalized_pressure_acceleration_weight: float = 0.18
    overnight_directional_weight: float = 0.34
    overnight_two_sided_weight: float = 0.20
    overnight_macro_weight: float = 0.16
    overnight_gamma_vol_weight: float = 0.15
    overnight_event_weight: float = 0.10
    overnight_context_weight: float = 0.05
    partial_coverage_warning_threshold: float = 0.55
    upside_pressure_threshold: float = 0.60
    downside_pressure_threshold: float = 0.60
    pinning_pressure_threshold: float = 0.62
    two_sided_threshold: float = 0.48
    state_balance_tolerance: float = 0.12
    moderate_pressure_threshold: int = 38
    high_pressure_threshold: int = 60
    extreme_pressure_threshold: int = 78
    overnight_block_threshold: int = 7
    overnight_watch_threshold: int = 4
    acceleration_adjustment_high: int = 2
    acceleration_adjustment_extreme: int = 3
    pinning_adjustment: int = -3
    two_sided_adjustment: int = -1


DEALER_HEDGING_PRESSURE_POLICY_CONFIG = DealerHedgingPressurePolicyConfig()


def get_dealer_hedging_pressure_policy_config() -> DealerHedgingPressurePolicyConfig:
    """
    Purpose:
        Return the dealer-hedging-pressure policy bundle used by the overlay layer.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        DealerHedgingPressurePolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("dealer_pressure.core", DealerHedgingPressurePolicyConfig())
