"""
Centralized macro/news configuration and tuning assumptions.

These are intentionally grouped away from trading_engine.py so the
macro/news layer can evolve without leaking low-level thresholds across
the engine.
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
    positive_hit_weight: float = 0.18
    negative_hit_weight: float = 0.22
    volatility_keyword_bonus: float = 0.08
    india_bias_hit_weight: float = 0.12
    global_bias_positive_hit_weight: float = 0.15
    global_bias_negative_hit_weight: float = 0.18
    uncategorized_keyword_impact: float = 10.0


@dataclass(frozen=True)
class MacroNewsAggregationConfig:
    headline_burst_lookback_minutes: int = HEADLINE_BURST_LOOKBACK_MINUTES
    headline_velocity_base_count: int = HEADLINE_VELOCITY_BASE_COUNT
    decay_half_life_minutes: float = 120.0
    confidence_count_divisor: float = 6.0
    confidence_count_weight: float = 0.5
    confidence_velocity_weight: float = 0.3
    confidence_total_weight_weight: float = 0.2


@dataclass(frozen=True)
class MacroNewsRegimeConfig:
    sentiment_on_threshold: float = MACRO_NEWS_SENTIMENT_ON_THRESHOLD
    sentiment_off_threshold: float = MACRO_NEWS_SENTIMENT_OFF_THRESHOLD
    risk_bias_threshold: float = MACRO_NEWS_RISK_BIAS_THRESHOLD
    risk_on_vol_shock_max: float = 55.0
    risk_off_vol_shock_min: float = 65.0


@dataclass(frozen=True)
class MacroNewsAdjustmentConfig:
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
