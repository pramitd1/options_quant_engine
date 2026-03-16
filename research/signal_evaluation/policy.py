"""
Module: policy.py

Purpose:
    Implement policy utilities for signal evaluation, reporting, or research diagnostics.

Role in the System:
    Part of the research layer that records signal-evaluation datasets and diagnostic reports.

Key Outputs:
    Signal-evaluation datasets, reports, and comparison artifacts.

Downstream Usage:
    Consumed by tuning, governance reviews, and post-trade analysis.
"""

from __future__ import annotations


CAPTURE_POLICY_TRADE_ONLY = "TRADE_ONLY"
CAPTURE_POLICY_ACTIONABLE = "ACTIONABLE"
CAPTURE_POLICY_ALL = "ALL_SIGNALS"

VALID_CAPTURE_POLICIES = {
    CAPTURE_POLICY_TRADE_ONLY,
    CAPTURE_POLICY_ACTIONABLE,
    CAPTURE_POLICY_ALL,
}


def normalize_capture_policy(policy: str | None) -> str:
    """
    Purpose:
        Normalize capture policy into the repository-standard form.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        policy (str | None): Input associated with policy.
    
    Returns:
        str: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    normalized = str(policy or "").upper().strip()
    if normalized in VALID_CAPTURE_POLICIES:
        return normalized
    return CAPTURE_POLICY_ALL


def should_capture_signal(trade: dict | None, policy: str | None = None) -> bool:
    """
    Purpose:
        Process should capture signal for downstream use.
    
    Context:
        Public function within the research layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade (dict | None): Input associated with trade.
        policy (str | None): Input associated with policy.
    
    Returns:
        bool: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not isinstance(trade, dict) or not trade:
        return False

    normalized_policy = normalize_capture_policy(policy)
    trade_status = str(trade.get("trade_status") or "").upper().strip()

    if normalized_policy == CAPTURE_POLICY_TRADE_ONLY:
        return trade_status == "TRADE"

    if normalized_policy == CAPTURE_POLICY_ACTIONABLE:
        return trade_status in {"TRADE", "WATCHLIST"}

    return bool(trade_status)
