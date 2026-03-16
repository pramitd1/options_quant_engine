"""
Module: validation_regime_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by validation regime.

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
class ValidationRegimeConfig:
    """
    Purpose:
        Dataclass representing ValidationRegimeConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        elevated_event_risk_threshold (float): Threshold used to classify or trigger elevated event risk.
        high_event_risk_threshold (float): Threshold used to classify or trigger high event risk.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
    elevated_event_risk_threshold: float = 45.0
    high_event_risk_threshold: float = 70.0


def get_validation_regime_config() -> ValidationRegimeConfig:
    """
    Purpose:
        Return the validation-regime configuration used by tuning and governance workflows.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        ValidationRegimeConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    return ValidationRegimeConfig()
