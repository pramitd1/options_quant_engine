"""
Module: macro_news_config.py

Purpose:
    Implement macro news config logic used to score scheduled events and macro catalysts.

Role in the System:
    Part of the macro context layer that scores scheduled events and broad market catalysts.

Key Outputs:
    Macro-event state, catalyst scores, and gating diagnostics.

Downstream Usage:
    Consumed by the signal engine, risk overlays, and research diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import (
    HEADLINE_BURST_LOOKBACK_MINUTES,
    HEADLINE_VELOCITY_BASE_COUNT,
    MACRO_NEWS_RISK_BIAS_THRESHOLD,
    MACRO_NEWS_SENTIMENT_OFF_THRESHOLD,
    MACRO_NEWS_SENTIMENT_ON_THRESHOLD,
)


@dataclass(frozen=True)
class HeadlineClassificationConfig:
    """
    Purpose:
        Dataclass representing HeadlineClassificationConfig within the repository.
    
    Context:
        Used within the macro context layer that scores scheduled events and broad catalysts. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        positive_hit_weight (float): Weight applied to positive hit.
        negative_hit_weight (float): Weight applied to negative hit.
        volatility_keyword_bonus (float): Bonus applied when volatility keyword is active.
        india_bias_hit_weight (float): Weight applied to india bias hit.
        global_bias_positive_hit_weight (float): Weight applied to global bias positive hit.
        global_bias_negative_hit_weight (float): Weight applied to global bias negative hit.
        uncategorized_keyword_impact (float): Value supplied for uncategorized keyword impact.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    positive_hit_weight: float = 0.18
    negative_hit_weight: float = 0.22
    volatility_keyword_bonus: float = 0.08
    india_bias_hit_weight: float = 0.12
    global_bias_positive_hit_weight: float = 0.15
    global_bias_negative_hit_weight: float = 0.18
    uncategorized_keyword_impact: float = 10.0


@dataclass(frozen=True)
class MacroNewsAggregationConfig:
    """
    Purpose:
        Dataclass representing MacroNewsAggregationConfig within the repository.
    
    Context:
        Used within the macro context layer that scores scheduled events and broad catalysts. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        headline_burst_lookback_minutes (int): Number of minutes used for headline burst lookback.
        headline_velocity_base_count (int): Count recorded for headline velocity base.
        decay_half_life_minutes (float): Number of minutes used for decay half life.
        confidence_count_divisor (float): Value supplied for confidence count divisor.
        confidence_count_weight (float): Weight applied to confidence count.
        confidence_velocity_weight (float): Weight applied to confidence velocity.
        confidence_total_weight_weight (float): Weight applied to confidence total weight.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    headline_burst_lookback_minutes: int = HEADLINE_BURST_LOOKBACK_MINUTES
    headline_velocity_base_count: int = HEADLINE_VELOCITY_BASE_COUNT
    decay_half_life_minutes: float = 120.0
    confidence_count_divisor: float = 6.0
    confidence_count_weight: float = 0.5
    confidence_velocity_weight: float = 0.3
    confidence_total_weight_weight: float = 0.2


@dataclass(frozen=True)
class MacroNewsRegimeConfig:
    """
    Purpose:
        Dataclass representing MacroNewsRegimeConfig within the repository.
    
    Context:
        Used within the macro context layer that scores scheduled events and broad catalysts. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        sentiment_on_threshold (float): Threshold used to classify or trigger sentiment on.
        sentiment_off_threshold (float): Threshold used to classify or trigger sentiment off.
        risk_bias_threshold (float): Threshold used to classify or trigger risk bias.
        risk_on_vol_shock_max (float): Value supplied for risk on vol shock max.
        risk_off_vol_shock_min (float): Value supplied for risk off vol shock min.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    sentiment_on_threshold: float = MACRO_NEWS_SENTIMENT_ON_THRESHOLD
    sentiment_off_threshold: float = MACRO_NEWS_SENTIMENT_OFF_THRESHOLD
    risk_bias_threshold: float = MACRO_NEWS_RISK_BIAS_THRESHOLD
    risk_on_vol_shock_max: float = 55.0
    risk_off_vol_shock_min: float = 65.0


@dataclass(frozen=True)
class MacroNewsAdjustmentConfig:
    """
    Purpose:
        Dataclass representing MacroNewsAdjustmentConfig within the repository.
    
    Context:
        Used within the macro context layer that scores scheduled events and broad catalysts. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        strong_confidence_threshold (float): Threshold used to classify or trigger strong confidence.
        moderate_confidence_threshold (float): Threshold used to classify or trigger moderate confidence.
        high_vol_shock_threshold (float): Threshold used to classify or trigger high vol shock.
        medium_vol_shock_threshold (float): Threshold used to classify or trigger medium vol shock.
        risk_off_call_score_strong (int): Value supplied for risk off call score strong.
        risk_off_call_score_moderate (int): Value supplied for risk off call score moderate.
        risk_off_call_confirmation_strong (int): Value supplied for risk off call confirmation strong.
        risk_off_call_confirmation_moderate (int): Value supplied for risk off call confirmation moderate.
        risk_off_put_score_support (int): Value supplied for risk off put score support.
        risk_off_put_score_soft (int): Value supplied for risk off put score soft.
        risk_off_put_confirmation_support (int): Value supplied for risk off put confirmation support.
        risk_on_call_score_support (int): Value supplied for risk on call score support.
        risk_on_call_score_soft (int): Value supplied for risk on call score soft.
        risk_on_call_confirmation_support (int): Value supplied for risk on call confirmation support.
        risk_on_put_score_strong (int): Value supplied for risk on put score strong.
        risk_on_put_score_moderate (int): Value supplied for risk on put score moderate.
        risk_on_put_confirmation_strong (int): Value supplied for risk on put confirmation strong.
        risk_on_put_confirmation_moderate (int): Value supplied for risk on put confirmation moderate.
        lockdown_adjustment_score (int): Score value for lockdown adjustment.
        lockdown_confirmation_adjustment (int): Value supplied for lockdown confirmation adjustment.
        high_vol_extra_score (int): Score value for high vol extra.
        high_vol_extra_confirmation (int): Value supplied for high vol extra confirmation.
        medium_vol_extra_score (int): Score value for medium vol extra.
        risk_off_call_size_high_vol (float): Value supplied for risk off call size high vol.
        risk_off_call_size_normal (float): Value supplied for risk off call size normal.
        risk_off_put_size_medium_vol (float): Value supplied for risk off put size medium vol.
        risk_on_call_size_medium_vol (float): Value supplied for risk on call size medium vol.
        risk_on_put_size_high_vol (float): Value supplied for risk on put size high vol.
        risk_on_put_size_normal (float): Value supplied for risk on put size normal.
        generic_high_vol_size_cap (float): Cap applied to generic high vol size.
        generic_medium_vol_size_cap (float): Cap applied to generic medium vol size.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    strong_confidence_threshold: float = 55.0
    moderate_confidence_threshold: float = 30.0
    high_vol_shock_threshold: float = 70.0
    medium_vol_shock_threshold: float = 55.0
    risk_off_call_score_strong: int = -8
    risk_off_call_score_moderate: int = -5
    risk_off_call_confirmation_strong: int = -3
    risk_off_call_confirmation_moderate: int = -2
    risk_off_put_score_support: int = 3
    risk_off_put_score_soft: int = 1
    risk_off_put_confirmation_support: int = 1
    risk_on_call_score_support: int = 3
    risk_on_call_score_soft: int = 1
    risk_on_call_confirmation_support: int = 1
    risk_on_put_score_strong: int = -6
    risk_on_put_score_moderate: int = -4
    risk_on_put_confirmation_strong: int = -2
    risk_on_put_confirmation_moderate: int = -1
    lockdown_adjustment_score: int = -12
    lockdown_confirmation_adjustment: int = -3
    high_vol_extra_score: int = -2
    high_vol_extra_confirmation: int = -1
    medium_vol_extra_score: int = -1
    risk_off_call_size_high_vol: float = 0.5
    risk_off_call_size_normal: float = 0.7
    risk_off_put_size_medium_vol: float = 0.85
    risk_on_call_size_medium_vol: float = 0.9
    risk_on_put_size_high_vol: float = 0.6
    risk_on_put_size_normal: float = 0.75
    generic_high_vol_size_cap: float = 0.6
    generic_medium_vol_size_cap: float = 0.8


HEADLINE_CLASSIFICATION_CONFIG = HeadlineClassificationConfig()
MACRO_NEWS_AGGREGATION_CONFIG = MacroNewsAggregationConfig()
MACRO_NEWS_REGIME_CONFIG = MacroNewsRegimeConfig()
MACRO_NEWS_ADJUSTMENT_CONFIG = MacroNewsAdjustmentConfig()


def get_headline_classification_config() -> HeadlineClassificationConfig:
    """
    Purpose:
        Return the headline-classification policy bundle used by the news layer.
    
    Context:
        Public function in the macro/news context layer. It standardizes how macro-event policies and headline signals are consumed.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        HeadlineClassificationConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        The output is designed to stay serializable and easy to inspect in downstream workflows.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("macro_news.headline_classification", HeadlineClassificationConfig())


def get_macro_news_aggregation_config() -> MacroNewsAggregationConfig:
    """
    Purpose:
        Return the macro-news aggregation policy bundle used when combining headline signals.
    
    Context:
        Public function in the macro/news context layer. It standardizes how macro-event policies and headline signals are consumed.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        MacroNewsAggregationConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        The output is designed to stay serializable and easy to inspect in downstream workflows.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("macro_news.aggregation", MacroNewsAggregationConfig())


def get_macro_news_regime_config() -> MacroNewsRegimeConfig:
    """
    Purpose:
        Return the macro-news regime policy bundle used by the signal engine.
    
    Context:
        Public function in the macro/news context layer. It standardizes how macro-event policies and headline signals are consumed.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        MacroNewsRegimeConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        The output is designed to stay serializable and easy to inspect in downstream workflows.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("macro_news.regime", MacroNewsRegimeConfig())


def get_macro_news_adjustment_config() -> MacroNewsAdjustmentConfig:
    """
    Purpose:
        Return the macro-news adjustment policy bundle used by overlay scoring.
    
    Context:
        Public function in the macro/news context layer. It standardizes how macro-event policies and headline signals are consumed.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        MacroNewsAdjustmentConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        The output is designed to stay serializable and easy to inspect in downstream workflows.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("macro_news.adjustment", MacroNewsAdjustmentConfig())
