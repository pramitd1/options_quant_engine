"""
Centralized feature-threshold configuration for analytics modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowImbalancePolicyConfig:
    neutral_default_imbalance: float = 1.0
    no_put_flow_fallback_imbalance: float = 2.0
    bullish_threshold: float = 1.20
    bearish_threshold: float = 0.83


@dataclass(frozen=True)
class SmartMoneyFlowPolicyConfig:
    unusual_volume_ratio_threshold: float = 1.0
    opening_activity_threshold: float = 0.0
    bullish_ratio_threshold: float = 1.15
    bearish_ratio_threshold: float = 0.87


@dataclass(frozen=True)
class VolatilityRegimePolicyConfig:
    low_vol_threshold: float = 0.01
    normal_vol_threshold: float = 0.03


def get_flow_imbalance_policy_config() -> FlowImbalancePolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("analytics.flow_imbalance", FlowImbalancePolicyConfig())


def get_smart_money_flow_policy_config() -> SmartMoneyFlowPolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("analytics.smart_money_flow", SmartMoneyFlowPolicyConfig())


def get_volatility_regime_policy_config() -> VolatilityRegimePolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("analytics.volatility_regime", VolatilityRegimePolicyConfig())
