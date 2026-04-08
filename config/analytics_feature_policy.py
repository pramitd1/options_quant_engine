"""
Module: analytics_feature_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by analytics feature.

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
class FlowImbalancePolicyConfig:
    """
    Purpose:
        Dataclass representing FlowImbalancePolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        neutral_default_imbalance (float): Value supplied for neutral default imbalance.
        no_put_flow_fallback_imbalance (float): Value supplied for no put flow fallback imbalance.
        bullish_threshold (float): Threshold used to classify or trigger bullish.
        bearish_threshold (float): Threshold used to classify or trigger bearish.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    neutral_default_imbalance: float = 1.0
    no_put_flow_fallback_imbalance: float = 2.0
    bullish_threshold: float = 1.20
    bearish_threshold: float = 0.83


@dataclass(frozen=True)
class SmartMoneyFlowPolicyConfig:
    """
    Purpose:
        Dataclass representing SmartMoneyFlowPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        unusual_volume_ratio_threshold (float): Threshold used to classify or trigger unusual volume ratio.
        opening_activity_threshold (float): Threshold used to classify or trigger opening activity.
        bullish_ratio_threshold (float): Threshold used to classify or trigger bullish ratio.
        bearish_ratio_threshold (float): Threshold used to classify or trigger bearish ratio.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    unusual_volume_ratio_threshold: float = 1.0
    opening_activity_threshold: float = 0.0
    bullish_ratio_threshold: float = 1.15
    bearish_ratio_threshold: float = 0.87


@dataclass(frozen=True)
class VolatilityRegimePolicyConfig:
    """
    Purpose:
        Dataclass representing VolatilityRegimePolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        low_vol_threshold (float): Threshold used to classify or trigger low vol.
        normal_vol_threshold (float): Threshold used to classify or trigger normal vol.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    low_vol_threshold: float = 0.15
    normal_vol_threshold: float = 0.25


@dataclass(frozen=True)
class GammaFlipPolicyConfig:
    """
    Policy thresholds governing the gamma-regime classifier in gamma_flip.py.

    Attributes
    ----------
    neutral_band_pct : float
        Half-width of the neutral zone around the gamma-flip level, expressed
        as a percentage of the flip price.  When
        ``|spot - flip| / flip * 100 <= neutral_band_pct`` the regime is
        ``NEUTRAL_GAMMA`` rather than ``POSITIVE_GAMMA`` or ``NEGATIVE_GAMMA``.

        Calibration guidance
        --------------------
        * NIFTY weekly options: 0.5 % corresponds to roughly ±120 pts on a
          24000 index level — a reasonable dead zone where dealer net gamma
          flips sign frequently on consecutive ticks.
        * Tighten toward 0.25 % for higher-conviction regime labels; widen
          toward 1.0 % in low-liquidity sessions where the flip estimate is
          noisy.
    """
    neutral_band_pct: float = 0.50


@dataclass(frozen=True)
class IvHvSpreadPolicyConfig:
    """Policy controls for IV-HV relative richness classification."""

    rich_threshold_relative: float = 0.15
    cheap_threshold_relative: float = -0.15


@dataclass(frozen=True)
class DealerFlowPolicyConfig:
    """Policy controls for dealer-flow convexity blend weights."""

    gamma_weight: float = 0.50
    charm_weight: float = 0.25


def get_gamma_flip_policy_config() -> GammaFlipPolicyConfig:
    """Return the gamma-flip policy bundle used by analytics features."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.gamma_flip", GammaFlipPolicyConfig())


def get_iv_hv_spread_policy_config() -> IvHvSpreadPolicyConfig:
    """Return policy bundle used by IV-HV spread classification."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.iv_hv_spread", IvHvSpreadPolicyConfig())


def get_dealer_flow_policy_config() -> DealerFlowPolicyConfig:
    """Return policy bundle used by dealer-flow convexity aggregation."""
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.dealer_flow", DealerFlowPolicyConfig())


def get_flow_imbalance_policy_config() -> FlowImbalancePolicyConfig:
    """
    Purpose:
        Return the flow-imbalance policy bundle used by analytics features.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        FlowImbalancePolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.flow_imbalance", FlowImbalancePolicyConfig())


def get_smart_money_flow_policy_config() -> SmartMoneyFlowPolicyConfig:
    """
    Purpose:
        Return the smart-money-flow policy bundle used by analytics features.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        SmartMoneyFlowPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.smart_money_flow", SmartMoneyFlowPolicyConfig())


def get_volatility_regime_policy_config() -> VolatilityRegimePolicyConfig:
    """
    Purpose:
        Return the volatility-regime policy bundle used by analytics features.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        VolatilityRegimePolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("analytics.volatility_regime", VolatilityRegimePolicyConfig())
