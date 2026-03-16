"""
Module: global_risk_models.py

Purpose:
    Define structured models returned by global risk workflows.

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
class HoldingContext:
    """
    Purpose:
        Dataclass representing HoldingContext within the repository.
    
    Context:
        Used within the risk-overlay layer that scores destabilizing conditions and modifies trade decisions. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        holding_profile (str): Value supplied for holding profile.
        overnight_relevant (bool): Whether the current context should apply overnight-risk logic.
        market_session (str): Market-session label used by the current state model.
        minutes_to_close (float | None): Minutes remaining until the relevant market close.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    holding_profile: str = "AUTO"
    overnight_relevant: bool = False
    market_session: str = "UNKNOWN"
    minutes_to_close: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `HoldingContext` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)


@dataclass(frozen=True)
class GlobalRiskState:
    """
    Purpose:
        Dataclass representing GlobalRiskState within the repository.
    
    Context:
        Used within the risk-overlay layer that scores destabilizing conditions and modifies trade decisions. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        global_risk_state (str): State payload for global risk.
        global_risk_score (int): Score value for global risk.
        overnight_gap_risk_score (int): Score value for overnight gap risk.
        volatility_expansion_risk_score (int): Score value for volatility expansion risk.
        overnight_hold_allowed (bool): Boolean flag controlling whether overnight hold allowed is active.
        overnight_hold_reason (str): Human-readable explanation for the overnight-hold decision.
        overnight_risk_penalty (int): Penalty applied when overnight risk is active.
        global_risk_adjustment_score (int): Score value for global risk adjustment.
        global_risk_veto (bool): Boolean flag controlling whether global risk veto is active.
        global_risk_position_size_multiplier (float): Multiplier applied to global risk position size.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        holding_context (dict[str, Any]): Structured mapping for holding context.
        global_risk_reasons (list[str]): Human-readable explanations for global risk.
        global_risk_features (dict[str, Any]): Structured mapping for global risk features.
        global_risk_diagnostics (dict[str, Any]): Structured mapping for global risk diagnostics.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    global_risk_state: str
    global_risk_score: int
    overnight_gap_risk_score: int
    volatility_expansion_risk_score: int
    overnight_hold_allowed: bool
    overnight_hold_reason: str
    overnight_risk_penalty: int
    global_risk_adjustment_score: int
    global_risk_veto: bool
    global_risk_position_size_multiplier: float
    neutral_fallback: bool
    holding_context: dict[str, Any] = field(default_factory=dict)
    global_risk_reasons: list[str] = field(default_factory=list)
    global_risk_features: dict[str, Any] = field(default_factory=dict)
    global_risk_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `GlobalRiskState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)
