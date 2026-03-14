"""
Centralized configuration for the dealer hedging pressure overlay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DealerHedgingPressurePolicyConfig:
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
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("dealer_pressure.core", DealerHedgingPressurePolicyConfig())
