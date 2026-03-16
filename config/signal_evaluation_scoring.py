"""
Module: signal_evaluation_scoring.py

Purpose:
    Define configuration values used by signal evaluation scoring.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""

SIGNAL_EVALUATION_SCORE_WEIGHTS = {
    "direction_score": 0.30,
    "magnitude_score": 0.25,
    "timing_score": 0.20,
    "tradeability_score": 0.25,
}

SIGNAL_EVALUATION_DIRECTION_WEIGHTS = {
    "correct_5m": 1.0,
    "correct_15m": 1.2,
    "correct_30m": 1.1,
    "correct_60m": 1.0,
    "correct_session_close": 1.0,
}

SIGNAL_EVALUATION_TIMING_WEIGHTS = {
    "realized_return_5m": 1.4,
    "realized_return_15m": 1.2,
    "realized_return_30m": 1.0,
    "realized_return_60m": 0.8,
}

SIGNAL_EVALUATION_THRESHOLDS = {
    "magnitude_vs_range_weak": 0.20,
    "magnitude_vs_range_good": 0.50,
    "magnitude_vs_range_strong": 1.00,
    "timing_positive_return_floor": 0.0005,
    "tradeability_ratio_floor": 0.75,
    "tradeability_ratio_good": 1.50,
    "tradeability_ratio_strong": 2.50,
}

SIGNAL_EVALUATION_SELECTION_POLICY = {
    "trade_strength_floor": 45.0,
    "composite_signal_score_floor": 55.0,
    "tradeability_score_floor": 50.0,
    "move_probability_floor": 0.40,
    "option_efficiency_score_floor": 35.0,
    "global_risk_score_cap": 85.0,
    "require_overnight_hold_allowed": False,
}


def get_signal_evaluation_score_weights():
    """
    Purpose:
        Return signal evaluation score weights for downstream use.
    
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

    return resolve_mapping("evaluation_thresholds.score_weights", SIGNAL_EVALUATION_SCORE_WEIGHTS)


def get_signal_evaluation_direction_weights():
    """
    Purpose:
        Return signal evaluation direction weights for downstream use.
    
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

    return resolve_mapping("evaluation_thresholds.direction_weights", SIGNAL_EVALUATION_DIRECTION_WEIGHTS)


def get_signal_evaluation_timing_weights():
    """
    Purpose:
        Return signal evaluation timing weights for downstream use.
    
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

    return resolve_mapping("evaluation_thresholds.timing_weights", SIGNAL_EVALUATION_TIMING_WEIGHTS)


def get_signal_evaluation_thresholds():
    """
    Purpose:
        Return signal evaluation thresholds for downstream use.
    
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

    return resolve_mapping("evaluation_thresholds.core", SIGNAL_EVALUATION_THRESHOLDS)


def get_signal_evaluation_selection_policy():
    """
    Purpose:
        Return signal evaluation selection policy for downstream use.
    
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

    return resolve_mapping("evaluation_thresholds.selection", SIGNAL_EVALUATION_SELECTION_POLICY)
