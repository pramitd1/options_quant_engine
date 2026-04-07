"""
Module: signal_consistency_policy.py

Purpose:
    Define escalation policy for runtime consistency findings.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds,
    and governance controls for signal-engine consistency gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast


@dataclass(frozen=True)
class SignalConsistencyPolicyConfig:
    """Runtime policy for consistency-check escalation behavior.

    trade_escalation_regime_map keys use a compact condition grammar:
        "gamma=NEGATIVE_GAMMA;global_risk=RISK_OFF": "MEDIUM"

    Supported condition keys:
        - gamma
        - global_risk
        - vol
        - confirmation

    Severity ordering:
        NONE < LOW < MEDIUM < HIGH < CRITICAL
    """

    default_trade_escalation_min_severity: str = "HIGH"
    trade_escalation_regime_map: dict[str, str] = field(
        default_factory=lambda: {
            "gamma=NEGATIVE_GAMMA;global_risk=RISK_OFF": "MEDIUM",
            "gamma=NEGATIVE_GAMMA;vol=VOL_EXPANSION": "MEDIUM",
            "vol=NORMAL_VOL;confirmation=CONFIRMED": "HIGH",
            "vol=NORMAL_VOL;confirmation=STRONG_CONFIRMATION": "HIGH",
        }
    )


SIGNAL_CONSISTENCY_POLICY_CONFIG = SignalConsistencyPolicyConfig()


def get_signal_consistency_policy_config() -> SignalConsistencyPolicyConfig:
    """Return active signal-consistency policy configuration."""
    from config.policy_resolver import resolve_dataclass_config

    return cast(
        SignalConsistencyPolicyConfig,
        resolve_dataclass_config("signal_engine.consistency", SignalConsistencyPolicyConfig()),
    )
