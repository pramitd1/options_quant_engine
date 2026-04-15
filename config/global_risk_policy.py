"""
Module: global_risk_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by global risk.

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
class GlobalRiskPolicyConfig:
    """
    Purpose:
        Dataclass representing GlobalRiskPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        oil_shock_extreme_change_pct (float): Value supplied for oil shock extreme change percentage.
        oil_shock_medium_change_pct (float): Value supplied for oil shock medium change percentage.
        oil_shock_notable_change_pct (float): Value supplied for oil shock notable change percentage (early warning tier).
        oil_shock_relief_change_pct (float): Value supplied for oil shock relief change percentage.
        oil_shock_extreme_score (float): Score value for oil shock extreme.
        oil_shock_medium_score (float): Score value for oil shock medium.
        oil_shock_notable_score (float): Score value for oil shock notable (early warning).
        oil_shock_relief_score (float): Score value for oil shock relief.
        gold_risk_extreme_change_pct (float): Value supplied for gold risk extreme change percentage.
        gold_risk_medium_change_pct (float): Value supplied for gold risk medium change percentage.
        gold_risk_extreme_score (float): Score value for gold risk extreme.
        gold_risk_medium_score (float): Score value for gold risk medium.
        copper_growth_severe_drop_pct (float): Value supplied for copper growth severe drop percentage.
        copper_growth_moderate_drop_pct (float): Value supplied for copper growth moderate drop percentage.
        copper_growth_severe_score (float): Score value for copper growth severe.
        copper_growth_moderate_score (float): Score value for copper growth moderate.
        commodity_risk_oil_weight (float): Weight applied to commodity risk oil.
        commodity_risk_gold_weight (float): Weight applied to commodity risk gold.
        commodity_risk_copper_weight (float): Weight applied to commodity risk copper.
        vix_shock_extreme_change_pct (float): Value supplied for vix shock extreme change percentage.
        vix_shock_medium_change_pct (float): Value supplied for vix shock medium change percentage.
        vix_shock_low_change_pct (float): Value supplied for vix shock low change percentage.
        vix_shock_extreme_score (float): Score value for vix shock extreme.
        vix_shock_medium_score (float): Score value for vix shock medium.
        vix_shock_low_score (float): Score value for vix shock low.
        us_equity_risk_extreme_move_pct (float): Value supplied for us equity risk extreme move percentage.
        us_equity_risk_moderate_move_pct (float): Value supplied for us equity risk moderate move percentage.
        us_equity_risk_extreme_score (float): Score value for us equity risk extreme.
        us_equity_risk_moderate_score (float): Score value for us equity risk moderate.
        rates_shock_threshold_bp (float): Value supplied for rates shock threshold basis points.
        rates_shock_score (float): Score value for rates shock.
        currency_shock_threshold_pct (float): Value supplied for currency shock threshold percentage.
        currency_shock_score_base (float): Value supplied for currency shock score base.
        vol_compression_extreme_ratio (float): Ratio used for vol compression extreme.
        vol_compression_medium_ratio (float): Ratio used for vol compression medium.
        vol_compression_low_ratio (float): Ratio used for vol compression low.
        vol_compression_extreme_score (float): Score value for vol compression extreme.
        vol_compression_medium_score (float): Score value for vol compression medium.
        vol_compression_low_score (float): Score value for vol compression low.
        commodity_stress_oil_weight (float): Weight applied to commodity stress oil.
        commodity_stress_gold_weight (float): Weight applied to commodity stress gold.
        commodity_stress_copper_weight (float): Weight applied to commodity stress copper.
        risk_off_intensity_vol_weight (float): Weight applied to risk off intensity vol.
        risk_off_intensity_us_equity_weight (float): Weight applied to risk off intensity us equity.
        risk_off_intensity_rates_weight (float): Weight applied to risk off intensity rates.
        risk_off_intensity_currency_weight (float): Weight applied to risk off intensity currency.
        risk_off_intensity_commodity_weight (float): Weight applied to risk off intensity commodity.
        risk_off_intensity_macro_event_weight (float): Weight applied to risk off intensity macro event.
        risk_off_pressure_vol_weight (float): Weight applied to risk off pressure vol.
        risk_off_pressure_us_equity_weight (float): Weight applied to risk off pressure us equity.
        risk_off_pressure_rates_weight (float): Weight applied to risk off pressure rates.
        risk_off_pressure_currency_weight (float): Weight applied to risk off pressure currency.
        risk_off_pressure_macro_event_weight (float): Weight applied to risk off pressure macro event.
        risk_off_pressure_vol_explosion_weight (float): Weight applied to risk off pressure vol explosion.
        risk_off_pressure_commodity_weight (float): Weight applied to risk off pressure commodity.
        risk_on_support_commodity_weight (float): Weight applied to risk on support commodity.
        risk_on_support_global_bias_weight (float): Weight applied to risk on support global bias.
        risk_on_support_macro_sentiment_weight (float): Weight applied to risk on support macro sentiment.
        positive_macro_sentiment_full_scale (float): Value supplied for positive macro sentiment full scale.
        volatility_expansion_market_vol_weight (float): Weight applied to volatility expansion market vol.
        volatility_expansion_explosion_weight (float): Weight applied to volatility expansion explosion.
        volatility_expansion_headline_weight (float): Weight applied to volatility expansion headline.
        global_risk_score_risk_off_pressure_weight (float): Weight applied to global risk score risk off pressure.
        global_risk_score_macro_event_weight (float): Weight applied to global risk score macro event.
        global_risk_score_volatility_expansion_weight (float): Weight applied to global risk score volatility expansion.
        global_risk_score_risk_off_intensity_weight (float): Weight applied to global risk score risk off intensity.
        global_risk_score_headline_velocity_weight (float): Weight applied to global risk score headline velocity.
        global_risk_score_global_bias_weight (float): Weight applied to global risk score global bias.
        global_risk_score_currency_weight (float): Weight applied to global risk score currency.
        global_risk_score_macro_regime_risk_off_bonus (float): Bonus applied when global risk score macro regime risk off is active.
        overnight_gap_macro_event_weight (float): Weight applied to overnight gap macro event.
        overnight_gap_volatility_expansion_weight (float): Weight applied to overnight gap volatility expansion.
        overnight_gap_currency_weight (float): Weight applied to overnight gap currency.
        overnight_gap_risk_off_intensity_weight (float): Weight applied to overnight gap risk off intensity.
        overnight_gap_headline_velocity_weight (float): Weight applied to overnight gap headline velocity.
        overnight_gap_global_score_excess_weight (float): Weight applied to overnight gap global score excess.
        overnight_gap_global_score_excess_floor (float): Floor value used for overnight gap global score excess.
        overnight_gap_overnight_context_bonus (float): Bonus applied when overnight gap overnight context is active.
        state_vol_shock_probability_threshold (float): Threshold used to classify or trigger state vol shock probability.
        state_event_lockdown_probability_threshold (float): Threshold used to classify or trigger state event lockdown probability.
        state_risk_off_regime_score_threshold (float): Threshold used to classify or trigger state risk off regime score.
        state_risk_on_regime_score_threshold (float): Threshold used to classify or trigger state risk on regime score.
        caution_threshold (int): Threshold used to classify or trigger caution.
        risk_off_threshold (int): Threshold used to classify or trigger risk off.
        extreme_threshold (int): Threshold used to classify or trigger extreme.
        event_risk_state_threshold (int): Threshold used to classify or trigger event risk state.
        extreme_veto_threshold (int): Threshold used to classify or trigger extreme veto.
        overnight_gap_block_threshold (int): Threshold used to classify or trigger overnight gap block.
        overnight_gap_veto_threshold (int): Threshold used to classify or trigger overnight gap veto.
        volatility_expansion_high_threshold (float): Threshold used to classify or trigger volatility expansion high.
        volatility_expansion_medium_threshold (float): Threshold used to classify or trigger volatility expansion medium.
        global_bias_risk_full_scale (float): Value supplied for global bias risk full scale.
        news_confidence_floor (float): Floor value used for news confidence.
        headline_velocity_full_scale (float): Value supplied for headline velocity full scale.
        risk_adjustment_caution (int): Value supplied for risk adjustment caution.
        risk_adjustment_risk_off (int): Value supplied for risk adjustment risk off.
        risk_adjustment_extreme (int): Value supplied for risk adjustment extreme.
        overnight_vol_explosion_high_threshold (float): Threshold used to classify or trigger overnight vol explosion high.
        overnight_vol_explosion_watch_threshold (float): Threshold used to classify or trigger overnight vol explosion watch.
        overnight_vol_explosion_high_penalty (int): Penalty applied when overnight vol explosion high is active.
        overnight_vol_explosion_watch_penalty (int): Penalty applied when overnight vol explosion watch is active.
        overnight_macro_event_high_threshold (float): Threshold used to classify or trigger overnight macro event high.
        overnight_macro_event_watch_threshold (float): Threshold used to classify or trigger overnight macro event watch.
        overnight_macro_event_high_penalty (int): Penalty applied when overnight macro event high is active.
        overnight_macro_event_watch_penalty (int): Penalty applied when overnight macro event watch is active.
        overnight_oil_shock_threshold (float): Threshold used to classify or trigger overnight oil shock.
        overnight_oil_shock_penalty (int): Penalty applied when overnight oil shock is active.
        overnight_us_equity_high_threshold (float): Threshold used to classify or trigger overnight us equity high.
        overnight_us_equity_watch_threshold (float): Threshold used to classify or trigger overnight us equity watch.
        overnight_us_equity_high_penalty (int): Penalty applied when overnight us equity high is active.
        overnight_us_equity_watch_penalty (int): Penalty applied when overnight us equity watch is active.
        overnight_risk_off_regime_penalty (int): Penalty applied when overnight risk off regime is active.
        volatility_explosion_penalty_threshold (float): Threshold used to classify or trigger volatility explosion penalty.
        volatility_explosion_penalty_score (int): Score value for volatility explosion penalty.
        oil_shock_penalty_threshold (float): Threshold used to classify or trigger oil shock penalty.
        oil_shock_penalty_score (int): Score value for oil shock penalty.
        size_cap_caution (float): Value supplied for size cap caution.
        size_cap_risk_off (float): Value supplied for size cap risk off.
        size_cap_extreme (float): Value supplied for size cap extreme.
        near_close_overnight_minutes (int): Number of minutes used for near close overnight.
        market_open_hour (int): Hour component used for market open.
        market_open_minute (int): Minute component used for market open.
        market_close_hour (int): Hour component used for market close.
        market_close_minute (int): Minute component used for market close.
        layer_data_quality_weight (float): Weight applied to layer data quality.
        layer_macro_event_weight (float): Weight applied to layer macro event.
        layer_global_risk_weight (float): Weight applied to layer global risk.
        layer_size_cap_penalty_scale (float): Value supplied for layer size cap penalty scale.
        layer_veto_block_score_floor (int): Floor value used for layer veto block score.
        layer_overnight_watch_score_floor (int): Floor value used for layer overnight watch score.
        layer_confirmation_watch_score_floor (int): Floor value used for layer confirmation watch score.
        layer_low_strength_watch_score_floor (int): Floor value used for layer low strength watch score.
        layer_weak_data_quality_score_threshold (float): Threshold used to classify or trigger layer weak data quality score.
        layer_weak_data_quality_watch_score_floor (int): Floor value used for layer weak data quality watch score.
        layer_caution_strength_buffer (int): Buffer applied around layer caution strength.
        layer_caution_watch_score_floor (int): Floor value used for layer caution watch score.
        layer_confirmation_conflict_strength_buffer (int): Buffer applied around layer confirmation conflict strength.
        layer_confirmation_conflict_watch_score_floor (int): Floor value used for layer confirmation conflict watch score.
        layer_size_reduction_strength_buffer (int): Buffer applied around layer size reduction strength.
        layer_size_reduction_watch_score_floor (int): Floor value used for layer size reduction watch score.
        layer_medium_level_threshold (int): Threshold used to classify or trigger layer medium level.
        layer_high_level_threshold (int): Threshold used to classify or trigger layer high level.
        layer_caution_watch_size_cap (float): Cap applied to layer caution watch size.
        layer_weak_data_quality_size_cap (float): Cap applied to layer weak data quality size.
        layer_overnight_watch_size_cap (float): Cap applied to layer overnight watch size.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    oil_shock_extreme_change_pct: float = 7.0
    oil_shock_medium_change_pct: float = 2.5
    oil_shock_notable_change_pct: float = 1.5
    oil_shock_relief_change_pct: float = -5.0
    oil_shock_extreme_score: float = 1.0
    oil_shock_medium_score: float = 0.7
    oil_shock_notable_score: float = 0.3
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
    vix_shock_extreme_change_pct: float = 12.0
    vix_shock_medium_change_pct: float = 7.0
    vix_shock_low_change_pct: float = 3.0
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
    layer_data_quality_weight: float = 0.35
    layer_macro_event_weight: float = 0.20
    layer_global_risk_weight: float = 0.45
    layer_size_cap_penalty_scale: float = 20.0
    layer_veto_block_score_floor: int = 82
    layer_overnight_watch_score_floor: int = 70
    layer_confirmation_watch_score_floor: int = 72
    layer_low_strength_watch_score_floor: int = 58
    layer_weak_data_quality_score_threshold: float = 55.0
    layer_weak_data_quality_watch_score_floor: int = 66
    layer_caution_strength_buffer: int = 8
    layer_caution_watch_score_floor: int = 54
    layer_confirmation_conflict_strength_buffer: int = 10
    layer_confirmation_conflict_watch_score_floor: int = 57
    layer_size_reduction_strength_buffer: int = 12
    layer_size_reduction_watch_score_floor: int = 60
    layer_medium_level_threshold: int = 35
    layer_high_level_threshold: int = 60
    layer_caution_watch_size_cap: float = 0.75
    layer_weak_data_quality_size_cap: float = 0.50
    layer_overnight_watch_size_cap: float = 0.50
    enable_portfolio_concentration_guard: int = 1
    portfolio_concentration_min_recent_signals: int = 4
    portfolio_concentration_soft_same_direction_count: int = 3
    portfolio_concentration_hard_same_direction_count: int = 4
    portfolio_concentration_soft_same_direction_share: float = 0.70
    portfolio_concentration_hard_same_direction_share: float = 0.80
    portfolio_concentration_weak_close_bps_threshold: float = 0.0
    portfolio_concentration_weak_tradeability_threshold: float = 55.0
    portfolio_concentration_reduce_size_cap: float = 0.65
    portfolio_concentration_watchlist_size_cap: float = 0.40
    portfolio_concentration_reduce_score_floor: int = 56
    portfolio_concentration_watch_score_floor: int = 68
    portfolio_heat_negative_gamma_penalty: int = 12
    portfolio_heat_vol_expansion_penalty: int = 10
    portfolio_heat_risk_off_penalty: int = 8
    portfolio_heat_provider_caution_penalty: int = 5
    portfolio_heat_provider_weak_penalty: int = 10
    portfolio_heat_data_caution_penalty: int = 5
    portfolio_heat_data_weak_penalty: int = 10
    portfolio_heat_edge_weak_penalty: int = 12
    portfolio_heat_warm_threshold: int = 40
    portfolio_heat_hot_threshold: int = 60
    portfolio_heat_critical_threshold: int = 80
    portfolio_heat_hot_size_cap: float = 0.55
    portfolio_heat_critical_size_cap: float = 0.35
    dxy_shock_threshold_pct: float = 0.35
    dxy_shock_score_base: float = 0.45
    gift_nifty_positive_threshold_pct: float = 0.45
    gift_nifty_negative_threshold_pct: float = -0.45
    gift_nifty_lead_score_base: float = 0.35
    risk_off_intensity_dxy_weight: float = 0.10
    risk_off_intensity_gift_nifty_weight: float = 0.08
    macro_uncertainty_event_weight: float = 0.45
    macro_uncertainty_headline_velocity_weight: float = 0.15
    macro_uncertainty_headline_stale_weight: float = 0.20
    macro_uncertainty_market_stale_weight: float = 0.20
    macro_uncertainty_watch_threshold: float = 0.60
    macro_uncertainty_watch_score_floor: int = 64
    macro_uncertainty_watch_size_cap: float = 0.50


GLOBAL_RISK_POLICY_CONFIG = GlobalRiskPolicyConfig()


def get_global_risk_policy_config() -> GlobalRiskPolicyConfig:
    """
    Purpose:
        Return the global-risk policy bundle used by the overlay layer.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        GlobalRiskPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("global_risk.core", GlobalRiskPolicyConfig())
