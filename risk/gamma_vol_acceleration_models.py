"""
Module: gamma_vol_acceleration_models.py

Purpose:
    Define structured models returned by gamma vol acceleration workflows.

Role in the System:
    Part of the risk-overlay layer that measures destabilizing conditions and adjusts trade eligibility or sizing.

Key Outputs:
    Overlay states, feature diagnostics, and trade-adjustment decisions.

Downstream Usage:
    Consumed by the signal engine, trade construction, and research diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GammaVolAccelerationState:
    """
    Purpose:
        Dataclass representing GammaVolAccelerationState within the repository.
    
    Context:
        Used within the risk-overlay layer that scores destabilizing conditions and modifies trade decisions. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        gamma_vol_acceleration_score (int): Score value for gamma vol acceleration.
        squeeze_risk_state (str): State payload for squeeze risk.
        directional_convexity_state (str): State payload for directional convexity.
        upside_squeeze_risk (float): Upside squeeze-risk score derived by the current model.
        downside_airpocket_risk (float): Downside air-pocket-risk score derived by the current model.
        overnight_convexity_risk (float): Overnight convexity-risk score derived by the current model.
        overnight_hold_allowed (bool): Boolean flag controlling whether overnight hold allowed is active.
        overnight_hold_reason (str): Human-readable explanation for the overnight-hold decision.
        overnight_convexity_penalty (int): Penalty applied when convexity risk argues against an overnight hold.
        overnight_convexity_boost (int): Boost applied when convexity favors the current overnight thesis.
        gamma_vol_adjustment_score (int): Score value for gamma vol adjustment.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        gamma_vol_reasons (list[str]): Human-readable explanations for gamma vol.
        gamma_vol_features (dict[str, Any]): Structured mapping for gamma vol features.
        gamma_vol_diagnostics (dict[str, Any]): Structured mapping for gamma vol diagnostics.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    gamma_vol_acceleration_score: int
    squeeze_risk_state: str
    directional_convexity_state: str
    upside_squeeze_risk: float
    downside_airpocket_risk: float
    overnight_convexity_risk: float
    overnight_hold_allowed: bool
    overnight_hold_reason: str
    overnight_convexity_penalty: int
    overnight_convexity_boost: int
    gamma_vol_adjustment_score: int
    neutral_fallback: bool
    gamma_vol_reasons: list[str] = field(default_factory=list)
    gamma_vol_features: dict[str, Any] = field(default_factory=dict)
    gamma_vol_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `GammaVolAccelerationState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)
