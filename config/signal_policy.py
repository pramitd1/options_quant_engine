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
    "VANNA": 0.45,
    "CHARM": 0.45,
    "BACKTEST_FALLBACK": 0.6,
}

DIRECTION_MIN_SCORE = 1.75
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
}

CONSENSUS_SCORE_CONFIG = {
    "strong_alignment_bonus": 8,
    "moderate_alignment_bonus": 4,
    "conflict_penalty": -6,
}

TRADE_RUNTIME_THRESHOLDS = {
    "min_trade_strength": 45,
    "strong_signal_threshold": 75,
    "medium_signal_threshold": 60,
    "weak_signal_threshold": 40,
    "expansion_bias_threshold": 75,
    "directional_bias_threshold": 55,
    "neutral_flow_probability_floor": 0.55,
}

CONFIRMATION_FILTER_CONFIG = {
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
