"""
Centralized configuration for the stage-1 global risk layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalRiskPolicyConfig:
    oil_shock_extreme_change_pct: float = 7.0
    oil_shock_medium_change_pct: float = 4.0
    oil_shock_relief_change_pct: float = -5.0
    oil_shock_extreme_score: float = 1.0
    oil_shock_medium_score: float = 0.7
    oil_shock_relief_score: float = -0.5
    gold_risk_extreme_change_pct: float = 3.0
    gold_risk_medium_change_pct: float = 2.0
    gold_risk_extreme_score: float = 0.8
    gold_risk_medium_score: float = 0.5
    copper_growth_severe_drop_pct: float = -5.0
    copper_growth_moderate_drop_pct: float = -3.0
    copper_growth_severe_score: float = -1.0
    copper_growth_moderate_score: float = -0.6
    commodity_risk_oil_weight: float = 0.5
    commodity_risk_gold_weight: float = 0.3
    commodity_risk_copper_weight: float = 0.2
    vix_shock_extreme_change_pct: float = 15.0
    vix_shock_medium_change_pct: float = 10.0
    vix_shock_low_change_pct: float = 5.0
    vix_shock_extreme_score: float = 1.0
    vix_shock_medium_score: float = 0.7
    vix_shock_low_score: float = 0.4
    us_equity_risk_extreme_move_pct: float = -2.0
    us_equity_risk_moderate_move_pct: float = -1.0
    us_equity_risk_extreme_score: float = 0.7
    us_equity_risk_moderate_score: float = 0.4
    rates_shock_threshold_bp: float = 10.0
    rates_shock_score: float = 0.6
    currency_shock_threshold_pct: float = 0.7
    currency_shock_score_base: float = 0.5
    vol_compression_extreme_ratio: float = 0.45
    vol_compression_medium_ratio: float = 0.60
    vol_compression_low_ratio: float = 0.75
    vol_compression_extreme_score: float = 1.0
    vol_compression_medium_score: float = 0.7
    vol_compression_low_score: float = 0.4
    commodity_stress_oil_weight: float = 0.55
    commodity_stress_gold_weight: float = 0.20
    commodity_stress_copper_weight: float = 0.25
    risk_off_intensity_vol_weight: float = 0.28
    risk_off_intensity_us_equity_weight: float = 0.20
    risk_off_intensity_rates_weight: float = 0.12
    risk_off_intensity_currency_weight: float = 0.12
    risk_off_intensity_commodity_weight: float = 0.16
    risk_off_intensity_macro_event_weight: float = 0.12
    risk_off_pressure_vol_weight: float = 0.24
    risk_off_pressure_us_equity_weight: float = 0.18
    risk_off_pressure_rates_weight: float = 0.10
    risk_off_pressure_currency_weight: float = 0.10
    risk_off_pressure_macro_event_weight: float = 0.14
    risk_off_pressure_vol_explosion_weight: float = 0.12
    risk_off_pressure_commodity_weight: float = 0.12
    risk_on_support_commodity_weight: float = 0.45
    risk_on_support_global_bias_weight: float = 0.35
    risk_on_support_macro_sentiment_weight: float = 0.20
    positive_macro_sentiment_full_scale: float = 30.0
    volatility_expansion_market_vol_weight: float = 0.50
    volatility_expansion_explosion_weight: float = 0.30
    volatility_expansion_headline_weight: float = 0.20
    global_risk_score_risk_off_pressure_weight: float = 0.52
    global_risk_score_macro_event_weight: float = 0.16
    global_risk_score_volatility_expansion_weight: float = 0.14
    global_risk_score_risk_off_intensity_weight: float = 0.08
    global_risk_score_headline_velocity_weight: float = 0.05
    global_risk_score_global_bias_weight: float = 0.03
    global_risk_score_currency_weight: float = 0.02
    global_risk_score_macro_regime_risk_off_bonus: float = 10.0
    overnight_gap_macro_event_weight: float = 0.32
    overnight_gap_volatility_expansion_weight: float = 0.22
    overnight_gap_currency_weight: float = 0.14
    overnight_gap_risk_off_intensity_weight: float = 0.14
    overnight_gap_headline_velocity_weight: float = 0.08
    overnight_gap_global_score_excess_weight: float = 0.10
    overnight_gap_global_score_excess_floor: float = 45.0
    overnight_gap_overnight_context_bonus: float = 10.0
    state_vol_shock_probability_threshold: float = 0.7
    state_event_lockdown_probability_threshold: float = 0.7
    state_risk_off_regime_score_threshold: float = 0.6
    state_risk_on_regime_score_threshold: float = -0.3
    caution_threshold: int = 35
    risk_off_threshold: int = 55
    extreme_threshold: int = 75
    event_risk_state_threshold: int = 60
    extreme_veto_threshold: int = 85
    overnight_gap_block_threshold: int = 68
    overnight_gap_veto_threshold: int = 82
    volatility_expansion_high_threshold: float = 65.0
    volatility_expansion_medium_threshold: float = 45.0
    global_bias_risk_full_scale: float = 0.85
    news_confidence_floor: float = 20.0
    headline_velocity_full_scale: float = 1.0
    risk_adjustment_caution: int = -2
    risk_adjustment_risk_off: int = -4
    risk_adjustment_extreme: int = -6
    overnight_vol_explosion_high_threshold: float = 0.7
    overnight_vol_explosion_watch_threshold: float = 0.4
    overnight_vol_explosion_high_penalty: int = 6
    overnight_vol_explosion_watch_penalty: int = 3
    overnight_macro_event_high_threshold: float = 70.0
    overnight_macro_event_watch_threshold: float = 45.0
    overnight_macro_event_high_penalty: int = 4
    overnight_macro_event_watch_penalty: int = 2
    overnight_oil_shock_threshold: float = 0.7
    overnight_oil_shock_penalty: int = 3
    overnight_us_equity_high_threshold: float = 0.7
    overnight_us_equity_watch_threshold: float = 0.4
    overnight_us_equity_high_penalty: int = 3
    overnight_us_equity_watch_penalty: int = 2
    overnight_risk_off_regime_penalty: int = 2
    volatility_explosion_penalty_threshold: float = 0.7
    volatility_explosion_penalty_score: int = -6
    oil_shock_penalty_threshold: float = 0.7
    oil_shock_penalty_score: int = -4
    size_cap_caution: float = 0.85
    size_cap_risk_off: float = 0.65
    size_cap_extreme: float = 0.35
    near_close_overnight_minutes: int = 45
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30


GLOBAL_RISK_POLICY_CONFIG = GlobalRiskPolicyConfig()


def get_global_risk_policy_config() -> GlobalRiskPolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("global_risk.core", GlobalRiskPolicyConfig())
