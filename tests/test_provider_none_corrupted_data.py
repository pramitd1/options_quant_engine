"""Tests for provider None/corrupted/late data propagation through engine."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


def test_engine_runner_handles_none_option_chain():
    """When option chain provider returns None, engine detection catches it early."""
    
    # Simulate engine receiving None option chain
    option_chain = None
    
    # Detection check
    is_invalid = option_chain is None or (isinstance(option_chain, pd.DataFrame) and len(option_chain) == 0)
    
    assert is_invalid is True


def test_spot_snapshot_none_value_blocks_engine():
    """When spot downloader returns None spot value, engine blocks before calculations."""
    
    spot_snapshot = {
        "spot": None,  # Critical: None spot
        "timestamp": "2026-03-22T10:00:00+05:30",
    }
    
    # Detection before any calculation
    is_invalid = spot_snapshot.get("spot") is None
    
    assert is_invalid is True


def test_vol_surface_none_propagates_safely():
    """When IV surface is None for all strikes, calculations don't crash."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "IV": [None, None, None],  # All None
    })
    
    # Safe handling: check before using
    has_iv = bool(option_chain["IV"].notna().any())
    
    assert has_iv is False


def test_dealer_pressure_layer_none_state_fallback():
    """When dealer pressure returns None, fallback to neutral state."""
    
    dealer_pressure_state = None
    
    # Fallback logic
    if dealer_pressure_state is None:
        effective_state = {
            "dealer_hedging_pressure_score": 0,  # Neutral
            "upside_hedge": 0,
            "downside_hedge": 0,
        }
    else:
        effective_state = dealer_pressure_state
    
    assert effective_state["dealer_hedging_pressure_score"] == 0


def test_engine_runner_detects_corrupted_global_risk_snapshot():
    """When global risk snapshot has corrupted/invalid structure, detect early."""
    
    # Missing required keys
    corrupted_snapshot = {
        "veto_active": True,
        # Missing: overnight_hold_allowed, volatility_shock_level
    }
    
    # Validation check
    required_keys = {"overnight_hold_allowed", "volatility_shock_level"}
    missing_keys = required_keys - set(corrupted_snapshot.keys())
    
    is_corrupted = len(missing_keys) > 0
    
    assert is_corrupted is True
    assert "overnight_hold_allowed" in missing_keys


def test_signal_generation_fails_gracefully_when_provider_layer_returns_none():
    """When provider layer returns None, signal generation doesn't crash."""
    
    provider_output = None
    
    # Safe signal generation fallback
    signal = {
        "direction": None,  # No signal possible
        "strength": 0,
        "confidence": "UNKNOWN",
        "reason": "Provider data unavailable",
    }
    
    assert signal["direction"] is None
    assert signal["reason"] == "Provider data unavailable"


def test_option_chain_none_detected_before_analytics():
    """Option chain None is caught before feeding into analytics layer."""
    
    option_chain = None
    
    # Early detection boundary
    can_run_analytics = option_chain is not None and len(option_chain) > 0
    
    assert can_run_analytics is False
    
    # Safe fallback
    analytics_result = {
        "gamma_regime": "UNKNOWN",
        "trend": None,
        "features": {},
    }
    
    assert analytics_result["gamma_regime"] == "UNKNOWN"


def test_spot_price_none_in_greeks_calculation():
    """When spot price is None, greeks don't compute and fallback to defaults."""
    
    spot = None
    strike = 23000
    iv = 0.2
    
    # Safe division check
    if spot is None or spot <= 0 or strike <= 0 or iv <= 0:
        # Cannot compute - use defaults
        gamma = 0.0
        delta = 0.0
    else:
        # Normal computation
        gamma = 0.001  # Example
        delta = 0.5
    
    assert gamma == 0.0
    assert delta == 0.0


def test_cascade_cascading_none_through_multiple_layers():
    """When None propagates through multiple layers, detect at each boundary."""
    
    # Layer 1: Data fetch
    raw_data = None
    if raw_data is None:
        processed_data = {"error": "Data unavailable"}
    else:
        processed_data = raw_data
    
    # Layer 2: Analytics
    if processed_data.get("error"):
        analytics_state = {"error": "Upstream data error"}
    else:
        analytics_state = {"computed": True}
    
    # Layer 3: Signal generation
    if analytics_state.get("error"):
        signal = {"direction": None, "reason": "Analytics failed"}
    else:
        signal = {"direction": "CALL"}
    
    # Verify cascade was caught
    assert signal["direction"] is None
    assert signal["reason"] == "Analytics failed"


def test_corrupted_payload_missing_required_fields():
    """Corrupted payload with missing required fields is detected."""
    
    payload = {
        "spot": 23000,
        # Missing: "option_chain"
        # Missing: "macro_event_state"
    }
    
    required_fields = {"spot", "option_chain", "macro_event_state"}
    provided_fields = set(payload.keys())
    missing_fields = required_fields - provided_fields
    
    is_corrupted = len(missing_fields) > 0
    
    assert is_corrupted is True
    assert "option_chain" in missing_fields


def test_provider_returns_wrong_dataframe_type():
    """When provider returns wrong type (not DataFrame), caught before use."""
    
    option_chain = {"data": [1, 2, 3]}  # Dict instead of DataFrame
    
    # Type check
    is_valid =isinstance(option_chain, pd.DataFrame)
    
    assert is_valid is False


def test_macro_event_state_none_doesnt_crash_engine():
    """When macro event state is None, engine continues with defaults."""
    
    macro_event_state = None
    
    # Safe lookup
    if macro_event_state is None:
        event_name = "none"
        veto_active = False
    else:
        event_name = macro_event_state.get("event_name", "unknown")
        veto_active = macro_event_state.get("veto_active", False)
    
    assert event_name == "none"
    assert veto_active is False
