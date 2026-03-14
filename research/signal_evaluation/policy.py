"""
Capture policy helpers for signal evaluation persistence.
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
    normalized = str(policy or "").upper().strip()
    if normalized in VALID_CAPTURE_POLICIES:
        return normalized
    return CAPTURE_POLICY_ALL


def should_capture_signal(trade: dict | None, policy: str | None = None) -> bool:
    if not isinstance(trade, dict) or not trade:
        return False

    normalized_policy = normalize_capture_policy(policy)
    trade_status = str(trade.get("trade_status") or "").upper().strip()

    if normalized_policy == CAPTURE_POLICY_TRADE_ONLY:
        return trade_status == "TRADE"

    if normalized_policy == CAPTURE_POLICY_ACTIONABLE:
        return trade_status in {"TRADE", "WATCHLIST"}

    return bool(trade_status)
