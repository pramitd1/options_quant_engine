"""
Module: gamma_vol_acceleration_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by gamma vol acceleration.

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
class GammaVolAccelerationPolicyConfig:
    """
    Purpose:
        Dataclass representing GammaVolAccelerationPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        gamma_regime_short_score (float): Score value for gamma regime short.
        gamma_regime_long_score (float): Score value for gamma regime long.
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
        vol_transition_compression_weight (float): Weight applied to vol transition compression.
        vol_transition_shock_weight (float): Weight applied to vol transition shock.
        vol_transition_explosion_weight (float): Weight applied to vol transition explosion.
        liquidity_breakout_score (float): Score value for liquidity breakout.
        liquidity_near_vacuum_score (float): Score value for liquidity near vacuum.
        liquidity_watch_score (float): Score value for liquidity watch.
        liquidity_contained_score (float): Score value for liquidity contained.
        hedging_upside_acceleration_score (float): Score value for hedging upside acceleration.
        hedging_downside_acceleration_score (float): Score value for hedging downside acceleration.
        hedging_upside_pinning_score (float): Score value for hedging upside pinning.
        hedging_downside_pinning_score (float): Score value for hedging downside pinning.
        pinning_bias_full_dampener (float): Value supplied for pinning bias full dampener.
        pinning_bias_partial_dampener (float): Value supplied for pinning bias partial dampener.
        intraday_extension_low_threshold (float): Threshold used to classify or trigger intraday extension low.
        intraday_extension_mid_threshold (float): Threshold used to classify or trigger intraday extension mid.
        intraday_extension_high_threshold (float): Threshold used to classify or trigger intraday extension high.
        intraday_extension_mid_score (float): Score value for intraday extension mid.
        intraday_extension_high_score (float): Score value for intraday extension high.
        intraday_extension_extreme_score (float): Score value for intraday extension extreme.
        macro_global_event_weight (float): Weight applied to macro global event.
        macro_global_state_weight (float): Weight applied to macro global state.
        macro_global_explosion_weight (float): Weight applied to macro global explosion.
        macro_global_state_vol_shock (float): Value supplied for macro global state vol shock.
        macro_global_state_event_lockdown (float): Value supplied for macro global state event lockdown.
        macro_global_state_risk_off (float): Value supplied for macro global state risk off.
        macro_global_state_neutral (float): Value supplied for macro global state neutral.
        macro_global_state_unknown (float): Value supplied for macro global state unknown.
        acceleration_gamma_weight (float): Weight applied to acceleration gamma.
        acceleration_flip_weight (float): Weight applied to acceleration flip.
        acceleration_vol_weight (float): Weight applied to acceleration vol.
        acceleration_liquidity_weight (float): Weight applied to acceleration liquidity.
        acceleration_hedging_weight (float): Weight applied to acceleration hedging.
        acceleration_intraday_weight (float): Weight applied to acceleration intraday.
        acceleration_macro_weight (float): Weight applied to acceleration macro.
        dampening_gamma_weight (float): Weight applied to dampening gamma.
        dampening_pinning_weight (float): Weight applied to dampening pinning.
        acceleration_dampening_weight (float): Weight applied to acceleration dampening.
        alignment_above_flip_boost (float): Value supplied for alignment above flip boost.
        alignment_below_flip_boost (float): Value supplied for alignment below flip boost.
        alignment_at_flip_boost (float): Value supplied for alignment at flip boost.
        alignment_bias_weight (float): Weight applied to alignment bias.
        alignment_bias_cap (float): Cap applied to alignment bias.
        directional_gamma_weight (float): Weight applied to directional gamma.
        directional_flip_weight (float): Weight applied to directional flip.
        directional_vol_weight (float): Weight applied to directional vol.
        directional_liquidity_weight (float): Weight applied to directional liquidity.
        directional_intraday_weight (float): Weight applied to directional intraday.
        directional_macro_weight (float): Weight applied to directional macro.
        directional_dampening_weight (float): Weight applied to directional dampening.
        overnight_acceleration_weight (float): Weight applied to overnight acceleration.
        overnight_explosion_weight (float): Weight applied to overnight explosion.
        overnight_macro_event_weight (float): Weight applied to overnight macro event.
        overnight_directional_weight (float): Weight applied to overnight directional.
        overnight_macro_boost_weight (float): Weight applied to overnight macro boost.
        overnight_context_weight (float): Weight applied to overnight context.
        overnight_dampening_weight (float): Weight applied to overnight dampening.
        partial_coverage_warning_threshold (float): Threshold used to classify or trigger partial coverage warning.
        low_risk_threshold (int): Threshold used to classify or trigger low risk.
        moderate_risk_threshold (int): Threshold used to classify or trigger moderate risk.
        high_risk_threshold (int): Threshold used to classify or trigger high risk.
        extreme_risk_threshold (int): Threshold used to classify or trigger extreme risk.
        directional_edge_threshold (float): Threshold used to classify or trigger directional edge.
        two_sided_edge_threshold (float): Threshold used to classify or trigger two sided edge.
        two_sided_balance_tolerance (float): Value supplied for two sided balance tolerance.
        overnight_block_threshold (int): Threshold used to classify or trigger overnight block.
        overnight_watch_threshold (int): Threshold used to classify or trigger overnight watch.
        score_boost_extreme (int): Value supplied for score boost extreme.
        score_boost_high (int): Value supplied for score boost high.
        score_boost_moderate (int): Value supplied for score boost moderate.
        score_dampen_long_gamma (int): Value supplied for score dampen long gamma.
        score_direction_mismatch_penalty (int): Penalty applied when score direction mismatch is active.
        score_two_sided_bonus (int): Bonus applied when score two sided is active.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
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
    """
    Purpose:
        Return the gamma-volatility-acceleration policy bundle used by the risk layer.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        GammaVolAccelerationPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("gamma_vol_acceleration.core", GammaVolAccelerationPolicyConfig())
