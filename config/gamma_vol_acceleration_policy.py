"""
Centralized configuration for the gamma-vol acceleration overlay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GammaVolAccelerationPolicyConfig:
    gamma_regime_short_score: float = 0.85
    gamma_regime_long_score: float = -0.55
    flip_unknown_context_score: float = 0.25
    flip_at_score: float = 1.0
    flip_distance_1_pct: float = 0.10
    flip_distance_1_score: float = 1.0
    flip_distance_2_pct: float = 0.25
    flip_distance_2_score: float = 0.82
    flip_distance_3_pct: float = 0.50
    flip_distance_3_score: float = 0.62
    flip_distance_4_pct: float = 0.90
    flip_distance_4_score: float = 0.38
    flip_distance_far_score: float = 0.12
    vol_transition_compression_weight: float = 0.45
    vol_transition_shock_weight: float = 0.30
    vol_transition_explosion_weight: float = 0.25
    liquidity_breakout_score: float = 0.9
    liquidity_near_vacuum_score: float = 0.7
    liquidity_watch_score: float = 0.45
    liquidity_contained_score: float = 0.2
    hedging_upside_acceleration_score: float = 0.9
    hedging_downside_acceleration_score: float = -0.9
    hedging_upside_pinning_score: float = 0.15
    hedging_downside_pinning_score: float = -0.15
    pinning_bias_full_dampener: float = 0.45
    pinning_bias_partial_dampener: float = 0.25
    intraday_extension_low_threshold: float = 0.35
    intraday_extension_mid_threshold: float = 0.70
    intraday_extension_high_threshold: float = 1.00
    intraday_extension_mid_score: float = 0.25
    intraday_extension_high_score: float = 0.45
    intraday_extension_extreme_score: float = 0.65
    macro_global_event_weight: float = 0.45
    macro_global_state_weight: float = 0.30
    macro_global_explosion_weight: float = 0.25
    macro_global_state_vol_shock: float = 1.0
    macro_global_state_event_lockdown: float = 0.95
    macro_global_state_risk_off: float = 0.60
    macro_global_state_neutral: float = 0.15
    macro_global_state_unknown: float = 0.10
    acceleration_gamma_weight: float = 0.34
    acceleration_flip_weight: float = 0.18
    acceleration_vol_weight: float = 0.16
    acceleration_liquidity_weight: float = 0.12
    acceleration_hedging_weight: float = 0.10
    acceleration_intraday_weight: float = 0.05
    acceleration_macro_weight: float = 0.05
    dampening_gamma_weight: float = 0.60
    dampening_pinning_weight: float = 0.40
    acceleration_dampening_weight: float = 0.50
    alignment_above_flip_boost: float = 0.18
    alignment_below_flip_boost: float = 0.18
    alignment_at_flip_boost: float = 0.08
    alignment_bias_weight: float = 0.24
    alignment_bias_cap: float = 0.24
    directional_gamma_weight: float = 0.40
    directional_flip_weight: float = 0.18
    directional_vol_weight: float = 0.16
    directional_liquidity_weight: float = 0.10
    directional_intraday_weight: float = 0.06
    directional_macro_weight: float = 0.10
    directional_dampening_weight: float = 0.28
    overnight_acceleration_weight: float = 0.28
    overnight_explosion_weight: float = 0.20
    overnight_macro_event_weight: float = 0.18
    overnight_directional_weight: float = 0.14
    overnight_macro_boost_weight: float = 0.10
    overnight_context_weight: float = 0.10
    overnight_dampening_weight: float = 0.22
    partial_coverage_warning_threshold: float = 0.55
    low_risk_threshold: int = 25
    moderate_risk_threshold: int = 40
    high_risk_threshold: int = 60
    extreme_risk_threshold: int = 78
    directional_edge_threshold: float = 0.58
    two_sided_edge_threshold: float = 0.48
    two_sided_balance_tolerance: float = 0.12
    overnight_block_threshold: int = 7
    overnight_watch_threshold: int = 4
    score_boost_extreme: int = 4
    score_boost_high: int = 2
    score_boost_moderate: int = 1
    score_dampen_long_gamma: int = -3
    score_direction_mismatch_penalty: int = -2
    score_two_sided_bonus: int = 1


GAMMA_VOL_ACCELERATION_POLICY_CONFIG = GammaVolAccelerationPolicyConfig()


def get_gamma_vol_acceleration_policy_config() -> GammaVolAccelerationPolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("gamma_vol_acceleration.core", GammaVolAccelerationPolicyConfig())
