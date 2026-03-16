"""
Module: option_efficiency_models.py

Purpose:
    Define structured models returned by option efficiency workflows.

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
class OptionEfficiencyState:
    """
    Purpose:
        Dataclass representing OptionEfficiencyState within the repository.
    
    Context:
        Used within the risk-overlay layer that scores destabilizing conditions and modifies trade decisions. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        expected_move_points (float | None): Expected move expressed in absolute points.
        expected_move_pct (float | None): Value supplied for expected move percentage.
        expected_move_quality (str): Qualitative label describing the quality of the expected-move estimate.
        target_distance_points (float | None): Distance, in points, between the current spot and the planned target.
        target_distance_pct (float | None): Value supplied for target distance percentage.
        expected_move_coverage_ratio (float | None): Ratio between expected move and target distance.
        target_reachability_score (int): Score value for target reachability.
        premium_efficiency_score (int): Score value for premium efficiency.
        strike_efficiency_score (int): Score value for strike efficiency.
        option_efficiency_score (int): Score value for option efficiency.
        option_efficiency_adjustment_score (int): Score value for option efficiency adjustment.
        overnight_hold_allowed (bool): Boolean flag controlling whether overnight hold allowed is active.
        overnight_hold_reason (str): Human-readable explanation for the overnight-hold decision.
        overnight_option_efficiency_penalty (int): Penalty applied when overnight option efficiency is active.
        strike_moneyness_bucket (str): Moneyness bucket assigned to the selected strike.
        strike_distance_from_spot (float | None): Distance between the selected strike and current spot.
        payoff_efficiency_hint (str): Qualitative hint describing the contract's payoff efficiency.
        neutral_fallback (bool): Boolean flag controlling whether neutral fallback is active.
        option_efficiency_reasons (list[str]): Human-readable explanations for option efficiency.
        option_efficiency_features (dict[str, Any]): Structured mapping for option efficiency features.
        option_efficiency_diagnostics (dict[str, Any]): Structured mapping for option efficiency diagnostics.
    
    Notes:
        The structured record keeps important state explicit, serializable, and easy to audit across live, replay, and research workflows.
    """
    expected_move_points: float | None
    expected_move_pct: float | None
    expected_move_quality: str
    target_distance_points: float | None
    target_distance_pct: float | None
    expected_move_coverage_ratio: float | None
    target_reachability_score: int
    premium_efficiency_score: int
    strike_efficiency_score: int
    option_efficiency_score: int
    option_efficiency_adjustment_score: int
    overnight_hold_allowed: bool
    overnight_hold_reason: str
    overnight_option_efficiency_penalty: int
    strike_moneyness_bucket: str
    strike_distance_from_spot: float | None
    payoff_efficiency_hint: str
    neutral_fallback: bool
    option_efficiency_reasons: list[str] = field(default_factory=list)
    option_efficiency_features: dict[str, Any] = field(default_factory=dict)
    option_efficiency_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Purpose:
            Serialize the object into a plain dictionary for persistence and review.
        
        Context:
            Method on `OptionEfficiencyState` that makes the object easy to persist in tuning artifacts, logs, or serialized payloads.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Dictionary representation of the object suitable for serialization or persistence.
        
        Notes:
            The serialized shape is kept stable so artifacts can be diffed, persisted, and inspected outside Python.
        """
        return asdict(self)
