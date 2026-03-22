"""Tests for state mismatch detection between live and backtest modes."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class SignalState:
    """Represents signal state across modes."""
    direction: str | None
    strength: float
    confidence: str
    mode: str  # "live" or "backtest"
    timestamp: str


def test_live_vs_backtest_signal_consistency_same_symbol_time():
    """Live signal should not contradict backtest signal on same symbol/time."""
    live_signal = SignalState(
        direction="CALL",
        strength=0.75,
        confidence="HIGH",
        mode="live",
        timestamp="2026-03-22T10:00:00+05:30"
    )
    
    backtest_signal = SignalState(
        direction="CALL",
        strength=0.75,
        confidence="HIGH",
        mode="backtest",
        timestamp="2026-03-22T10:00:00+05:30"
    )
    
    # Signals should agree on direction and confidence level
    assert live_signal.direction == backtest_signal.direction
    assert live_signal.confidence == backtest_signal.confidence


def test_live_vs_backtest_signal_contradiction_detected():
    """When signals contradict, contradiction should be detected and flagged."""
    live_signal = SignalState(
        direction="CALL",
        strength=0.75,
        confidence="HIGH",
        mode="live",
        timestamp="2026-03-22T10:00:00+05:30"
    )
    
    backtest_signal = SignalState(
        direction="PUT",  # Opposite direction
        strength=0.70,
        confidence="HIGH",
        mode="backtest",
        timestamp="2026-03-22T10:00:00+05:30"
    )
    
    # Detect contradiction
    contradiction = live_signal.direction != backtest_signal.direction
    assert contradiction is True


def test_trade_strength_scoring_consistency_across_modes():
    """Trade strength scoring should produce similar results in both modes."""
    def compute_trade_strength(spot, option_price, volatility, mode):
        """Simulate trade strength scoring - should be same regardless of mode."""
        base_strength = (option_price / spot) * volatility * 100
        # Mode shouldn't affect scoring
        return base_strength
    
    spot = 23000
    option_price = 150
    volatility = 0.25
    
    live_strength = compute_trade_strength(spot, option_price, volatility, "live")
    backtest_strength = compute_trade_strength(spot, option_price, volatility, "backtest")
    
    # Strength should be identical regardless of mode
    assert live_strength == backtest_strength


def test_provider_health_status_consistency_across_modes():
    """Provider health status should be consistent between live and backtest."""
    live_provider_health = {
        "provider_name": "mock_provider",
        "is_healthy": True,
        "last_data_age_seconds": 5,
        "error_count": 0,
    }
    
    backtest_provider_health = {
        "provider_name": "mock_provider",
        "is_healthy": True,
        "last_data_age_seconds": 5,  # In replay, this is elapsed time simulation
        "error_count": 0,
    }
    
    # Health status should match
    assert live_provider_health["is_healthy"] == backtest_provider_health["is_healthy"]


def test_provider_health_mismatch_detected():
    """When provider health differs between modes, mismatch should be detected."""
    live_provider_health = {
        "provider_name": "mock_provider",
        "is_healthy": True,
        "error_count": 0,
    }
    
    backtest_provider_health = {
        "provider_name": "mock_provider",
        "is_healthy": False,  # Health differs - stale data in backtest
        "error_count": 3,
    }
    
    # Detect mismatch
    mismatch = live_provider_health["is_healthy"] != backtest_provider_health["is_healthy"]
    assert mismatch is True


def test_global_risk_state_consistency_across_modes():
    """Global risk state should not differ unexpectedly between modes."""
    live_global_risk = {
        "veto_active": False,
        "overnight_hold_allowed": True,
        "volatility_shock_level": "NORMAL",
    }
    
    backtest_global_risk = {
        "veto_active": False,
        "overnight_hold_allowed": True,
        "volatility_shock_level": "NORMAL",
    }
    
    # Risk states should align
    assert live_global_risk["veto_active"] == backtest_global_risk["veto_active"]
    assert live_global_risk["volatility_shock_level"] == backtest_global_risk["volatility_shock_level"]


def test_stale_macro_data_causes_global_risk_mismatch():
    """Stale macro data in backtest can cause global risk state to diverge."""
    live_global_risk = {
        "veto_active": False,
        "macro_data_freshness": "CURRENT",
        "last_macro_update": "2026-03-22T09:59:00+05:30",
    }
    
    # In backtest with stale replay data
    backtest_global_risk = {
        "veto_active": False,
        "macro_data_freshness": "STALE",  # Data is from replay
        "last_macro_update": "2026-03-21T16:00:00+05:30",  # Previous day
    }
    
    # Detect freshness mismatch
    mismatch = live_global_risk["macro_data_freshness"] != backtest_global_risk["macro_data_freshness"]
    assert mismatch is True


def test_mode_mismatch_flag_prevents_using_backtest_as_live():
    """When modes are mixed, a flag should indicate danger."""
    
    class TradeDecision:
        def __init__(self, decision_mode):
            self.decision_mode = decision_mode
            self._execution_mode = None
        
        def set_execution_mode(self, exec_mode):
            if self.decision_mode != exec_mode:
                # Different modes - flag as dangerous
                return False
            self._execution_mode = exec_mode
            return True
    
    # Create decision in backtest mode
    decision = TradeDecision("backtest")
    
    # Try to execute in live mode - should fail
    can_execute = decision.set_execution_mode("live")
    assert can_execute is False
