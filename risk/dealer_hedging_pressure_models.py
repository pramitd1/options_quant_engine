"""
Module: dealer_hedging_pressure_models.py

Purpose:
    Define structured models returned by dealer hedging pressure workflows.

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
class DealerHedgingPressureState:
    """
    Purpose:
        Dataclass representing DealerHedgingPressureState within the repository.
    
    Context:
        Used within the risk-overlay layer that scores destabilizing conditions and modifies trade decisions. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        dealer_hedging_pressure_score (int): Score value for dealer hedging pressure.
        dealer_flow_state (str): State payload for dealer flow.
        upside_hedging_pressure (float): Upside dealer-hedging pressure score.
        downside_hedging_pressure (float): Downside dealer-hedging pressure score.
        pinning_pressure_score (float): Score value for pinning pressure.
        overnight_hedging_risk (float): Overnight hedging-risk score derived by the current model.
        overnight_hold_allowed (bool): Boolean flag controlling whether overnight hold allowed is active.
        overnight_hold_reason (str): Human-readable explanation for the overnight-hold decision.
        overnight_dealer_pressure_penalty (int): Penalty applied when dealer-hedging pressure raises overnight risk.
        overnight_dealer_pressure_boost (int): Boost applied when dealer-hedging pressure supports the overnight thesis.
        dealer_pressure_adjustment_score (int): Score value for dealer pressure adjustment.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        dealer_pressure_reasons (list[str]): Human-readable explanations for dealer pressure.
        dealer_pressure_features (dict[str, Any]): Structured mapping for dealer pressure features.
        dealer_pressure_diagnostics (dict[str, Any]): Structured mapping for dealer pressure diagnostics.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    dealer_hedging_pressure_score: int
    dealer_flow_state: str
    upside_hedging_pressure: float
    downside_hedging_pressure: float
    pinning_pressure_score: float
    overnight_hedging_risk: float
    overnight_hold_allowed: bool
    overnight_hold_reason: str
    overnight_dealer_pressure_penalty: int
    overnight_dealer_pressure_boost: int
    dealer_pressure_adjustment_score: int
    neutral_fallback: bool
    dealer_pressure_reasons: list[str] = field(default_factory=list)
    dealer_pressure_features: dict[str, Any] = field(default_factory=dict)
    dealer_pressure_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `DealerHedgingPressureState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)
