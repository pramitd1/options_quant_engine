from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationRegimeConfig:
    elevated_event_risk_threshold: float = 45.0
    high_event_risk_threshold: float = 70.0


def get_validation_regime_config() -> ValidationRegimeConfig:
    return ValidationRegimeConfig()
