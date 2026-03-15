"""
Centralized signal-engine probability feature calibration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityFeaturePolicyConfig:
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


def get_probability_feature_policy_config() -> ProbabilityFeaturePolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("signal_engine.probability", ProbabilityFeaturePolicyConfig())
