"""
Module: signal_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by signal.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

DIRECTION_VOTE_WEIGHTS = {
    "FLOW": 1.2,
    "HEDGING_BIAS": 1.1,
    "GAMMA_SQUEEZE": 0.9,
    "GAMMA_FLIP": 0.85,
    "DEALER_VOL": 0.8,
    "VANNA": 0.55,
    "CHARM": 0.55,
    "OI_VELOCITY": 0.45,
    "RR_SKEW": 0.40,
    "RR_MOMENTUM": 0.30,
    "PCR_ATM": 0.25,
    "FLIP_DRIFT": 0.30,
    "BREAKOUT_STRUCTURE": 0.95,
    "RANGE_EXPANSION": 0.55,
    "MACRO_PRESSURE": 0.45,
    "FX_PRESSURE": 0.45,
    "DXY_PRESSURE": 0.40,
    "GIFT_LEAD": 0.45,
}

DIRECTION_MIN_SCORE = 1.50
DIRECTION_MIN_MARGIN = 0.7

TRADE_STRENGTH_WEIGHTS = {
    "flow_call_bullish": 20,
    "flow_call_bearish": -10,
    "flow_put_bearish": 20,
    "flow_put_bullish": -10,
    "smart_call_bullish": 15,
    "smart_call_bearish": -8,
    "smart_put_bearish": 15,
    "smart_put_bullish": -8,
    "hedging_acceleration_support": 10,
    "hedging_acceleration_conflict": -6,
    "gamma_regime_negative": 10,
    "gamma_regime_positive": 2,
    "gamma_regime_neutral": 5,
    "spot_flip_primary": 8,
    "spot_flip_secondary": 2,
    "wall_support_bonus": 5,
    "wall_resistance_penalty": -3,
    "liquidity_map_path_bonus": 4,
    "gamma_event_bonus": 10,
    "dealer_short_gamma_bonus": 10,
    "dealer_long_gamma_bonus": 5,
    "vol_expansion_bonus": 10,
    "normal_vol_bonus": 5,
    "liquidity_void_near_bonus": 10,
    "liquidity_void_far_bonus": 4,
    "vacuum_breakout_bonus": 10,
    "vacuum_watch_bonus": 4,
    "intraday_vol_expansion_bonus": 5,
    "intraday_gamma_decrease_bonus": 3,
    "flip_zone_negative_gamma_penalty": -12,
    "flip_zone_neutral_gamma_penalty": -8,
    "oi_velocity_alignment_bonus": 4,
    "oi_velocity_conflict_penalty": -3,
    "rr_alignment_bonus": 3,
    "rr_conflict_penalty": -2,
    "pcr_alignment_bonus": 2,
    "pcr_conflict_penalty": -2,
    "flip_drift_alignment_bonus": 2,
    "flip_drift_conflict_penalty": -2,
}

CONSENSUS_SCORE_CONFIG = {
    "strong_alignment_bonus": 8,
    "moderate_alignment_bonus": 4,
    "conflict_penalty": -6,
}

TRADE_RUNTIME_THRESHOLDS = {
    "min_trade_strength": 62,
    "min_composite_score": 58,
    # Temporary guardrail while probability calibration is improved.
    "move_probability_score_cap": 75,
    "composite_weight_trade_strength": 0.50,
    "composite_weight_move_probability": 0.20,
    "composite_weight_confirmation": 0.15,
    "composite_weight_data_quality": 0.10,
    "composite_weight_gamma_stability": 0.05,
    "strong_signal_threshold": 75,
    "medium_signal_threshold": 60,
    "weak_signal_threshold": 40,
    "expansion_bias_threshold": 75,
    "directional_bias_threshold": 55,
    "neutral_flow_probability_floor": 0.58,
    "wall_proximity_buffer": 50,
    "max_intraday_hold_minutes": 90,
    "toxic_regime_hold_cap_minutes": 60,
    "provider_health_caution_blocks_trade": 1,
    # Degraded provider-health override is disabled by default while live
    # calibration and weak-data pollution are being tightened. Re-enable only
    # for explicitly governed experiments.
    "enable_provider_health_degraded_override": 0,
    "provider_health_override_dte_max": 1.0,
    "provider_health_override_min_strength_buffer": 18,
    "provider_health_override_min_composite_buffer": 10,
    "provider_health_override_min_effective_priced_ratio": 0.70,
    "provider_health_override_max_proxy_ratio": 0.25,
    "provider_health_override_size_cap": 0.20,
    "provider_health_override_hold_cap_minutes": 20,
    "provider_health_override_require_strong_confirmation": 1,
    "provider_health_override_one_sided_quote_ratio_max": 0.20,
    "provider_health_override_allow_block_status": 0,
    "provider_health_override_allowed_summary_statuses": ["CAUTION"],
    "provider_health_override_allowed_data_quality_statuses": ["GOOD", "STRONG"],
    "provider_health_override_allowed_block_reasons": ["core_iv_weak"],
    # Weak-data circuit breaker: degrade fragile setups to WATCHLIST when
    # data quality is weak and multiple fragility signals co-occur.
    "enable_weak_data_circuit_breaker": 1,
    "weak_data_circuit_breaker_data_quality_statuses": ["WEAK", "CAUTION"],
    "weak_data_circuit_breaker_provider_statuses": ["WEAK", "CAUTION"],
    "weak_data_circuit_breaker_require_strong_confirmation": 1,
    "weak_data_circuit_breaker_min_trade_strength": 74,
    "weak_data_circuit_breaker_min_runtime_composite_score": 70,
    "weak_data_circuit_breaker_max_proxy_ratio": 0.35,
    "weak_data_circuit_breaker_min_trigger_count": 2,
    # Bearish drift guard: tighten PUT qualification in toxic/high-vol regimes.
    "enable_bearish_bias_guard": 1,
    "bearish_bias_guard_composite_add": 3,
    "bearish_bias_guard_strength_add": 2,
    "bearish_bias_guard_size_cap": 0.70,
    "at_flip_trade_strength_penalty": 8,
    "at_flip_size_cap": 0.75,
    "at_flip_toxic_size_cap": 0.50,
    "regime_strength_add_at_flip": 4,
    "regime_strength_add_toxic": 8,
    "regime_composite_add_at_flip": 3,
    "regime_composite_add_toxic": 6,
    # Positive gamma underperformed in recent evaluations; tighten gates there.
    "regime_strength_add_positive_gamma": 5,
    "regime_composite_add_positive_gamma": 3,
    # Negative gamma performed strongly; modestly relax gates there.
    "regime_strength_relief_negative_gamma": 2,
    "regime_composite_relief_negative_gamma": 1,
    # Regime-aware sizing multipliers.
    "positive_gamma_size_multiplier": 0.85,
    "negative_gamma_size_multiplier": 1.15,
    "gamma_vol_normalization_scale": 100,
    "gamma_vol_winsor_lower": 12,
    "gamma_vol_winsor_upper": 88,
    "trade_strength_scoring_mode": "continuous",
    # Confidence-weighted threshold adjustment.
    # When data quality is GOOD and confirmation is STRONG, the engine
    # can require a slightly lower trade strength (relief).  Conversely,
    # a WEAK/CONFLICTED environment demands a higher bar (surcharge).
    "high_confidence_strength_relief": 5,
    "low_confidence_strength_surcharge": 8,
    # Feature toggles for optional directional/score enrichments.
    "use_oi_velocity_in_direction": 1,
    "use_rr_in_direction": 1,
    "use_pcr_in_confirmation": 1,
    "use_flip_drift_in_overlays": 1,
    "use_max_pain_expiry_overlay": 1,
    # OI velocity thresholds.
    "oi_velocity_vote_on": 0.18,
    "oi_velocity_vote_strong": 0.40,
    # Risk-reversal thresholds (vol points).
    "rr_skew_put_dominant": 0.75,
    "rr_skew_call_dominant": -0.75,
    "rr_momentum_vote_threshold": 0.20,
    # PCR confirmation thresholds.
    "volume_pcr_atm_put_dominant": 1.20,
    "volume_pcr_atm_call_dominant": 0.80,
    # Flip drift thresholds and overlay weighting.
    "gamma_flip_drift_pts_vote_on": 80,
    "gamma_flip_drift_pts_strong": 180,
    "gamma_flip_drift_gamma_vol_weight": 0.08,
    # Breakout sensitivity controls for sudden directional expansions.
    "direction_breakout_buffer_points": 20,
    "direction_breakout_range_pct_floor": 0.35,
    "direction_breakout_evidence_threshold": 2.0,
    "direction_breakout_margin_relief": 0.25,
    "direction_breakout_score_relief": 0.15,
    "reversal_fast_handoff_evidence_threshold": 1.2,
    "reversal_fast_handoff_margin_relief": 0.20,
    "reversal_fast_handoff_score_relief": 0.10,
    "expansion_mode_breakout_evidence_threshold": 1.25,
    "expansion_mode_move_probability_floor": 0.56,
    "expansion_mode_range_pct_floor": 0.25,
    "expansion_mode_margin_relief": 0.30,
    "expansion_mode_score_relief": 0.20,
    "expansion_mode_strength_relief": 3,
    "expansion_mode_size_mult": 1.10,
    "expansion_mode_hold_mult": 0.75,
    "asymmetric_flipback_guard_steps": 3,
    "asymmetric_flipback_margin_surcharge": 0.35,
    "asymmetric_flipback_score_surcharge": 0.20,
    "macro_direction_confidence_floor": 55.0,
    "macro_direction_bias_floor": 10.0,
    "macro_regime_vote_bonus": 5.0,
    "fx_usdinr_put_threshold_pct": 0.35,
    "fx_usdinr_call_threshold_pct": -0.35,
    "dxy_put_threshold_pct": 0.30,
    "dxy_call_threshold_pct": -0.30,
    "gift_nifty_call_threshold_pct": 0.35,
    "gift_nifty_put_threshold_pct": -0.35,
    # Execution microstructure friction penalties applied at direction stage.
    "direction_microstructure_penalty_provider_caution": 0.15,
    "direction_microstructure_penalty_provider_weak": 0.30,
    "direction_microstructure_penalty_provider_block": 0.40,
    "direction_microstructure_penalty_quote_integrity_weak": 0.25,
    "direction_microstructure_one_sided_soft": 0.20,
    "direction_microstructure_one_sided_hard": 0.45,
    "direction_microstructure_penalty_one_sided_soft": 0.12,
    "direction_microstructure_penalty_one_sided_hard": 0.25,
    "direction_microstructure_priced_ratio_floor": 0.55,
    "direction_microstructure_penalty_priced_ratio_max": 0.30,
    # Probabilistic direction head (calibrated P(up|X)) integration controls.
    "enable_probabilistic_direction_head": 1,
    "direction_probability_calibrator_path": "models_store/direction_probability_calibrator.json",
    "direction_head_call_threshold": 0.53,
    "direction_head_put_threshold": 0.47,
    "direction_head_min_confidence": 0.57,
    "direction_head_allow_vote_override": 1,
    "direction_head_override_min_confidence": 0.66,
    "headline_staleness_score_penalty": 4,
    "headline_staleness_size_cap": 0.75,
    "global_macro_staleness_score_penalty": 5,
    "global_macro_staleness_size_cap": 0.70,
    # Two-stage reversal state controls (EARLY candidate -> CONFIRMED reversal)
    "reversal_stage_min_vote_count": 3,
    "reversal_stage_min_breakout_votes": 1,
    "reversal_stage_strength_relief": 4,
    "reversal_stage_early_size_mult": 0.60,
    "reversal_stage_early_hold_mult": 0.60,
    "reversal_stage_confirmed_size_mult": 1.00,
    # Reversal grace can be bypassed only on high-conviction breakout evidence.
    "reversal_breakout_override_move_probability_floor": 0.55,
    "reversal_breakout_override_range_pct_floor": 0.25,
    "reversal_breakout_override_requires_flow": 0,
    "reversal_breakout_override_requires_hedging": 0,
    "reversal_breakout_override_min_signals": 2,
    # Expiry-conditioned max pain pinning settings.
    "max_pain_overlay_max_dte": 2,
    "max_pain_pin_distance_pct": 0.35,
    "max_pain_pin_distance_pts_min": 80,
    "max_pain_pin_penalty_base": -2,
    "max_pain_pin_penalty_strong": -4,
    "max_pain_overlay_weight": 0.12,
    # ============================================================================
    # IMPROVEMENTS LAYER: Score Calibration, Time-Decay, Path Filtering, Regime Thresholds
    # ============================================================================
    # Score Calibration (Isotonic Regression + Temperature Scaling)
    "enable_score_calibration": 1,
    "calibration_backend": "isotonic",  # or "temperature"
    "runtime_score_calibrator_path": "models_store/runtime_score_calibrator.json",
    # Time-Decay Model (Regime-Aware Half-Lives)
    "enable_time_decay_model": 1,
    "time_decay_lambda": 1.5,
    # Optional fixed elapsed minutes for replay experiments; None means auto signal-age tracking.
    "time_decay_elapsed_minutes": None,
    # Fallback conversion when reversal_age is available (steps -> minutes).
    "time_decay_minutes_per_snapshot": 5,
    "time_decay_positive_gamma_half_life_m": 240,
    "time_decay_negative_gamma_half_life_m": 240,
    "time_decay_neutral_gamma_half_life_m": 230,
    # Path-Aware Entry Filtering
    "enable_path_aware_filtering": 1,
    "path_filtering_mae_zscore_threshold": 1.5,
    "path_filtering_mfe_mae_ratio_threshold": 0.87,
    "path_filtering_hostile_score_penalty": 15,
    "path_filtering_delay_entry_on_hostile": 1,
    "path_filtering_entry_confirmation_window_m": 5,
    # Regime-Conditional Thresholds
    "enable_regime_conditional_thresholds": 1,
    "regime_positive_gamma_composite_delta": -3,
    "regime_positive_gamma_strength_delta": -2,
    "regime_positive_gamma_position_size_mult": 1.15,
    "regime_positive_gamma_holding_delta_m": 60,
    # Bias/stickiness audit (2026-04-06): tighten toxic regime filters and size.
    "regime_negative_gamma_composite_delta": 7,
    "regime_negative_gamma_strength_delta": 5,
    "regime_negative_gamma_position_size_mult": 0.75,
    "regime_negative_gamma_holding_delta_m": -60,
    "regime_neutral_gamma_composite_delta": 0,
    "regime_neutral_gamma_strength_delta": 0,
    "regime_neutral_gamma_position_size_mult": 1.0,
}

CONFIRMATION_FILTER_CONFIG = {
    "confirmation_scoring_mode": "continuous",
    "continuous_open_alignment": 1,
    "continuous_prev_close_alignment": 1,
    "continuous_range_expansion": 1,
    "continuous_move_probability": 1,
    # Direction reversal control — manages confirmation status stickiness across direction changes.
    # See documentation/system_docs/reference/reversal_stickiness_control.md for the full guide.
    #
    # Three mechanisms (all tunable, independent):
    # 1. revers_veto_steps (RECOMMENDED): forces MIXED status for N snapshots after reversal
    # 2. direction_change_penalty: fixed score deduction on reversal snapshot (0-6 points)
    # 3. post-reversal decay: extend penalty across N steps with geometric decay
    #
    # Sweep findings (live NIFTY data, 13 reversals):
    #   reversal_veto_steps=0: flip_persist_ratio=1.0  (baseline problem)
    #   reversal_veto_steps=1: flip_persist_ratio=0.0  (optimum: 100% stickiness eliminated)
    #   reversal_veto_steps=2+: flip_persist_ratio=0.0 (no additional benefit)
    #
    "direction_change_penalty": 0.0,  # Bounded to [0.0, 6.0]
    "direction_change_decay_steps": 0,  # Post-reversal decay window (0 disables)
    "direction_change_decay_factor": 0.5,  # Decay multiplier per step (0.0-1.0)
    "reversal_veto_steps": 0,  # 0-step grace: reduce stickiness on fast trend reversals
    "strong_confirmation_threshold": 6,
    "confirmed_threshold": 2,
    "mixed_threshold": -3,
    "open_alignment_support": 2,
    "open_alignment_conflict": -2,
    "prev_close_alignment_support": 1,
    "prev_close_alignment_conflict": -1,
    "range_expansion_strong_score": 3,
    "range_expansion_moderate_score": 2,
    "range_expansion_low_score": 1,
    "range_expansion_cold_score": -1,
    "flow_support": 3,
    "flow_conflict": -4,
    "hedging_support": 3,
    "hedging_conflict": -4,
    "gamma_event_support": 2,
    "move_probability_high_threshold": 0.65,
    "move_probability_high_score": 3,
    "move_probability_moderate_threshold": 0.50,
    "move_probability_moderate_score": 2,
    "move_probability_low_support_threshold": 0.40,
    "move_probability_low_support_score": 1,
    "move_probability_conflict_threshold": 0.30,
    "move_probability_conflict_score": -2,
    "flip_alignment_support": 2,
    "flip_alignment_conflict": -1,
    "flip_zone_gamma_penalty_negative": -3,
    "flip_zone_gamma_penalty_neutral": -2,
    # Optional PCR alignment check.
    "pcr_confirmation_support": 1,
    "pcr_confirmation_conflict": -2,
    "veto_hard_conflicts": 3,
    "veto_move_probability_ceiling": 0.55,
}


@dataclass(frozen=True)
class DataQualityPolicyConfig:
    """
    Purpose:
        Dataclass representing DataQualityPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        invalid_spot_penalty (int): Penalty applied when invalid spot is active.
        stale_spot_penalty (int): Penalty applied when stale spot is active.
        invalid_option_chain_penalty (int): Penalty applied when invalid option chain is active.
        stale_option_chain_penalty (int): Penalty applied when stale option chain is active.
        provider_health_weak_penalty (int): Penalty applied when provider health weak is active.
        provider_health_caution_penalty (int): Penalty applied when provider health caution is active.
        missing_analytics_penalty_per_field (int): Value supplied for missing analytics penalty per field.
        missing_analytics_penalty_cap (int): Cap applied to missing analytics penalty.
        missing_all_probabilities_penalty (int): Penalty applied when missing all probabilities is active.
        missing_hybrid_probability_penalty (int): Penalty applied when missing hybrid probability is active.
        status_strong_threshold (int): Threshold used to classify or trigger status strong.
        status_good_threshold (int): Threshold used to classify or trigger status good.
        status_caution_threshold (int): Threshold used to classify or trigger status caution.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    invalid_spot_penalty: int = 45
    stale_spot_penalty: int = 10
    invalid_option_chain_penalty: int = 45
    stale_option_chain_penalty: int = 10
    provider_health_weak_penalty: int = 18
    provider_health_caution_penalty: int = 8
    missing_analytics_penalty_per_field: int = 6
    missing_analytics_penalty_cap: int = 24
    missing_all_probabilities_penalty: int = 10
    missing_hybrid_probability_penalty: int = 5
    status_strong_threshold: int = 85
    status_good_threshold: int = 70
    status_caution_threshold: int = 55


@dataclass(frozen=True)
class ExecutionRegimePolicyConfig:
    """
    Purpose:
        Dataclass representing ExecutionRegimePolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        reduced_size_multiplier_threshold (float): Threshold used to classify or trigger reduced size multiplier.
        observe_data_quality_threshold (int): Threshold used to classify or trigger observe data quality.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    reduced_size_multiplier_threshold: float = 1.0
    observe_data_quality_threshold: int = 70


@dataclass(frozen=True)
class LargeMoveScoringPolicyConfig:
    """
    Purpose:
        Dataclass representing LargeMoveScoringPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        hybrid_threshold_extreme (float): Value supplied for hybrid threshold extreme.
        hybrid_score_extreme (int): Value supplied for hybrid score extreme.
        hybrid_threshold_high (float): Value supplied for hybrid threshold high.
        hybrid_score_high (int): Value supplied for hybrid score high.
        hybrid_threshold_moderate (float): Value supplied for hybrid threshold moderate.
        hybrid_score_moderate (int): Value supplied for hybrid score moderate.
        hybrid_threshold_watch (float): Value supplied for hybrid threshold watch.
        hybrid_score_watch (int): Value supplied for hybrid score watch.
        hybrid_threshold_tail (float): Value supplied for hybrid threshold tail.
        hybrid_score_tail (int): Value supplied for hybrid score tail.
        ml_threshold_extreme (float): Value supplied for ML threshold extreme.
        ml_score_extreme (int): Value supplied for ML score extreme.
        ml_threshold_high (float): Value supplied for ML threshold high.
        ml_score_high (int): Value supplied for ML score high.
        ml_threshold_moderate (float): Value supplied for ML threshold moderate.
        ml_score_moderate (int): Value supplied for ML score moderate.
        ml_threshold_watch (float): Value supplied for ML threshold watch.
        ml_score_watch (int): Value supplied for ML score watch.
        ml_threshold_tail (float): Value supplied for ML threshold tail.
        ml_score_tail (int): Value supplied for ML score tail.
        overlap_hybrid_floor (int): Floor value used for overlap hybrid.
        overlap_ml_floor (int): Floor value used for overlap ML.
        overlap_penalty (int): Penalty applied when overlap is active.
        total_score_cap (int): Cap applied to total score.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    hybrid_threshold_extreme: float = 0.75
    hybrid_score_extreme: int = 12
    hybrid_threshold_high: float = 0.65
    hybrid_score_high: int = 10
    hybrid_threshold_moderate: float = 0.55
    hybrid_score_moderate: int = 8
    hybrid_threshold_watch: float = 0.45
    hybrid_score_watch: int = 6
    hybrid_threshold_tail: float = 0.35
    hybrid_score_tail: int = 3
    ml_threshold_extreme: float = 0.75
    ml_score_extreme: int = 6
    ml_threshold_high: float = 0.65
    ml_score_high: int = 5
    ml_threshold_moderate: float = 0.55
    ml_score_moderate: int = 4
    ml_threshold_watch: float = 0.45
    ml_score_watch: int = 2
    ml_threshold_tail: float = 0.35
    ml_score_tail: int = 1
    overlap_hybrid_floor: int = 8
    overlap_ml_floor: int = 4
    overlap_penalty: int = 1
    total_score_cap: int = 14


@dataclass(frozen=True)
class TradeModifierPolicyConfig:
    """
    Purpose:
        Dataclass representing TradeModifierPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        gamma_alignment_score (int): Score value for gamma alignment.
        gamma_conflict_penalty (int): Penalty applied when gamma conflict is active.
        gamma_two_sided_score (int): Score value for gamma two sided.
        dealer_pinning_penalty (int): Penalty applied when dealer pinning is active.
        dealer_instability_penalty (int): Penalty applied when dealer instability is active.
        dealer_alignment_score (int): Score value for dealer alignment.
        dealer_conflict_penalty (int): Penalty applied when dealer conflict is active.
        alignment_score_floor (int): Floor value used for alignment score.
        alignment_score_cap (int): Cap applied to alignment score.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    gamma_alignment_score: int = 2
    gamma_conflict_penalty: int = -6
    gamma_two_sided_score: int = 1
    dealer_pinning_penalty: int = -2
    dealer_instability_penalty: int = -1
    dealer_alignment_score: int = 2
    dealer_conflict_penalty: int = -3
    alignment_score_floor: int = -6
    alignment_score_cap: int = 8


@dataclass(frozen=True)
class TradeStrengthContinuousPolicyConfig:
    """Configuration for continuous trade-strength component scoring."""

    hybrid_probability_floor: float = 0.30
    hybrid_probability_ceiling: float = 0.80
    hybrid_max_score: int = 12
    ml_probability_floor: float = 0.30
    ml_probability_ceiling: float = 0.80
    ml_max_score: int = 6
    overlap_hybrid_threshold: float = 0.65
    overlap_ml_threshold: float = 0.65
    overlap_penalty: int = 1
    probability_total_score_cap: int = 14
    wall_distance_cap_multiplier: float = 1.0
    liquidity_path_distance_cap_multiplier: float = 3.0
    flip_distance_cap_pct: float = 0.8
    spot_flip_conflict_floor: float = -2.0


def get_direction_vote_weights():
    """
    Purpose:
        Return direction vote weights for downstream use.
    
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

    return resolve_mapping("trade_strength.direction_vote", DIRECTION_VOTE_WEIGHTS)


def get_direction_thresholds():
    """
    Purpose:
        Return direction thresholds for downstream use.
    
    Context:
        Public function within the configuration layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Centralizing this contract keeps runtime, replay, and research workflows aligned on the same configuration semantics.
    """
    from config.policy_resolver import get_parameter_value

    return {
        "min_score": float(get_parameter_value("trade_strength.direction_thresholds.min_score", DIRECTION_MIN_SCORE)),
        "min_margin": float(get_parameter_value("trade_strength.direction_thresholds.min_margin", DIRECTION_MIN_MARGIN)),
    }


def get_trade_strength_weights():
    """
    Purpose:
        Return trade strength weights for downstream use.
    
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

    return resolve_mapping("trade_strength.scoring", TRADE_STRENGTH_WEIGHTS)


def get_consensus_score_config():
    """
    Purpose:
        Return the configuration bundle for consensus score.
    
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

    return resolve_mapping("trade_strength.consensus", CONSENSUS_SCORE_CONFIG)


def get_trade_runtime_thresholds():
    """
    Purpose:
        Return trade runtime thresholds for downstream use.
    
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

    return resolve_mapping("trade_strength.runtime_thresholds", TRADE_RUNTIME_THRESHOLDS)


def get_confirmation_filter_config():
    """
    Purpose:
        Return the confirmation-filter policy bundle used by signal assembly.
    
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

    return resolve_mapping("confirmation_filter.core", CONFIRMATION_FILTER_CONFIG)


def get_data_quality_policy_config() -> DataQualityPolicyConfig:
    """
    Purpose:
        Return the data-quality policy bundle used when validating signal inputs.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        DataQualityPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.data_quality", DataQualityPolicyConfig())


def get_execution_regime_policy_config() -> ExecutionRegimePolicyConfig:
    """
    Purpose:
        Return the execution-regime policy bundle used by trade classification.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        ExecutionRegimePolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.execution_regime", ExecutionRegimePolicyConfig())


def get_large_move_scoring_policy_config() -> LargeMoveScoringPolicyConfig:
    """
    Purpose:
        Return the large-move scoring policy bundle used by the signal engine.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        LargeMoveScoringPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.large_move_scoring", LargeMoveScoringPolicyConfig())


def get_trade_modifier_policy_config() -> TradeModifierPolicyConfig:
    """
    Purpose:
        Return the trade-modifier policy bundle used by overlay scoring.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        TradeModifierPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.trade_modifiers", TradeModifierPolicyConfig())


def get_trade_strength_continuous_policy_config() -> TradeStrengthContinuousPolicyConfig:
    """Return the continuous trade-strength scoring policy bundle."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config(
        "signal_engine.trade_strength_continuous",
        TradeStrengthContinuousPolicyConfig(),
    )


# ---------------------------------------------------------------------------
# Exit timing policy — controls recommended holding period and time-based
# exit guidance based on empirical alpha-decay observations.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitTimingPolicyConfig:
    """Weights and thresholds for time-based exit recommendations."""

    # Peak alpha window (minutes from entry)
    peak_alpha_minutes: int = 120
    # Maximum recommended holding period before forced exit consideration
    max_hold_minutes: int = 180
    # Early session trades get longer runway
    early_session_cutoff_minutes_from_open: int = 60
    early_session_peak_alpha_minutes: int = 150
    # Late session trades get shorter runway
    late_session_cutoff_minutes_to_close: int = 90
    late_session_max_hold_minutes: int = 45
    # Strong signals can hold longer
    strong_signal_hold_extension_minutes: int = 30
    strong_signal_threshold: int = 75
    # High volatility regime shortens holding
    vol_expansion_hold_reduction_minutes: int = 30
    # Negative gamma environments favor faster exits
    negative_gamma_hold_reduction_minutes: int = 20
    # Exit urgency thresholds (minutes remaining)
    urgency_critical_minutes: int = 15
    urgency_high_minutes: int = 30
    urgency_moderate_minutes: int = 60


def get_exit_timing_policy_config() -> ExitTimingPolicyConfig:
    """Return the exit-timing policy bundle."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config(
        "signal_engine.exit_timing", ExitTimingPolicyConfig()
    )


# ---------------------------------------------------------------------------
# Activation score policy — controls the setup-readiness scoring that decides
# whether a no-direction snapshot is DEAD_INACTIVE, WATCHLIST, or active.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationScorePolicyConfig:
    """Weights and thresholds for the activation-score subsystem in
    ``_build_decision_explainability``."""

    flow_bonus: int = 24
    smart_money_bonus: int = 16
    convexity_bonus: int = 20
    dealer_structure_bonus: int = 14
    trade_strength_bonus: int = 14
    move_probability_bonus: int = 12
    move_probability_floor: float = 0.60
    trade_strength_min_ratio: float = 0.5
    activation_cap: int = 100
    dead_inactive_threshold: int = 35

    # Confirmation-status → numeric score mapping
    confirmation_score_strong: int = 100
    confirmation_score_mixed: int = 55
    confirmation_score_conflict: int = 20
    confirmation_score_no_direction: int = 10

    # Data-ready numeric mapping
    data_ready_strong: int = 100
    data_ready_good: int = 80
    data_ready_caution: int = 55
    data_ready_weak: int = 30

    # Maturity score blend weights
    maturity_weight_trade_strength: float = 0.45
    maturity_weight_confirmation: float = 0.30
    maturity_weight_data_ready: float = 0.25

    # Explainability confidence thresholds
    high_confidence_data_ready_floor: int = 75
    high_confidence_confirmation_floor: int = 55
    medium_confidence_data_ready_floor: int = 55


def get_activation_score_policy_config() -> ActivationScorePolicyConfig:
    """Return the activation-score policy bundle."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config(
        "signal_engine.activation_score", ActivationScorePolicyConfig()
    )
