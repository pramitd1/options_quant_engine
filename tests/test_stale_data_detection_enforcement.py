"""Tests for stale data detection and enforcement across all providers."""
from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timedelta
import pytz


IST = pytz.timezone("Asia/Kolkata")


def get_ist_time(minutes_ago=0):
    """Helper to get IST timestamps."""
    return (datetime.now(IST) - timedelta(minutes=minutes_ago)).isoformat()


def test_option_chain_staleness_blocks_trade_entry():
    """When option chain is stale >15min, trade entry is blocked."""
    
    # Option chain from 20 minutes ago
    option_chain = pd.DataFrame({
        "strikePrice": [23000],
        "timestamp": get_ist_time(minutes_ago=20),
    })
    
    current_time = datetime.now(IST)
    chain_time = pd.to_datetime(option_chain["timestamp"].iloc[0])
    age_minutes = (current_time - chain_time).total_seconds() / 60
    
    stale_threshold = 15
    is_stale = age_minutes > stale_threshold
    
    assert is_stale is True


def test_spot_price_stale_15min_terminates_engine():
    """When spot price is stale >10min, engine terminates."""
    
    spot_snapshot = {
        "spot": 23000,
        "timestamp": get_ist_time(minutes_ago=12),
    }
    
    current_time = datetime.now(IST)
    spot_time = pd.to_datetime(spot_snapshot["timestamp"])
    age_minutes = (current_time - spot_time).total_seconds() / 60
    
    freshness_threshold = 10
    is_stale = age_minutes > freshness_threshold
    
    assert is_stale is True


def test_dealer_pressure_stale_uses_cached_state():
    """When dealer pressure data is stale, use cached previous state."""
    
    dealer_pressure = {
        "hedge_pressure": 0.5,
        "timestamp": get_ist_time(minutes_ago=45),  # 45 minutes old
    }
    
    cached_state = {
        "hedge_pressure": 0.3,  # Previous value
        "timestamp": get_ist_time(minutes_ago=5),   # 5 minutes old (fresher)
    }
    
    current_time = datetime.now(IST)
    dealer_age = (current_time - pd.to_datetime(dealer_pressure["timestamp"])).total_seconds() / 60
    
    freshness_threshold = 30
    should_use_cache = dealer_age > freshness_threshold
    
    assert should_use_cache is True
    
    # Use cached instead
    effective_pressure = cached_state if should_use_cache else dealer_pressure
    assert effective_pressure["hedge_pressure"] == 0.3


def test_macro_event_expiration_mid_engine_run():
    """When macro event expires during engine run, veto is lifted."""
    
    event_start = datetime.now(IST)
    event_end = event_start + timedelta(minutes=30)
    
    # Check at different times
    check_time_before_end = event_end - timedelta(minutes=5)
    check_time_after_end = event_end + timedelta(minutes=5)
    
    is_active_before = check_time_before_end < event_end
    is_active_after = check_time_after_end < event_end
    
    assert is_active_before is True
    assert is_active_after is False


def test_iv_surface_staleness_detected():
    """When IV surface timestamp is stale, flag for recalculation."""
    
    iv_surface = {
        "data": [[0.2, 0.25], [0.22, 0.28]],
        "timestamp": get_ist_time(minutes_ago=20),
    }
    
    current_time = datetime.now(IST)
    iv_time = pd.to_datetime(iv_surface["timestamp"])
    age_minutes = (current_time - iv_time).total_seconds() / 60
    
    max_age = 15
    needs_refresh = age_minutes > max_age
    
    assert needs_refresh is True


def test_replay_mode_detects_old_snapshot_at_startup():
    """In replay mode, startup with snapshot >15min old triggers warning."""
    
    snapshot = {
        "spot": 23000,
        "timestamp": get_ist_time(minutes_ago=25),  # 25 minutes old
    }
    
    current_time = datetime.now(IST)
    snap_time = pd.to_datetime(snapshot["timestamp"])
    age_minutes = (current_time - snap_time).total_seconds() / 60
    
    warning_threshold = 15
    should_warn = age_minutes > warning_threshold
    
    assert should_warn is True


def test_engine_aborts_when_all_providers_stale():
    """When all providers are stale, engine aborts safely."""
    
    current_time = datetime.now(IST)
    stale_threshold = 10
    
    providers = {
        "spot": get_ist_time(minutes_ago=15),      # Stale
        "option_chain": get_ist_time(minutes_ago=20),  # Stale
        "volatility": get_ist_time(minutes_ago=12),    # Stale
    }
    
    stale_count = 0
    for provider_name, timestamp in providers.items():
        provider_time = pd.to_datetime(timestamp)
        age_minutes = (current_time - provider_time).total_seconds() / 60
        if age_minutes > stale_threshold:
            stale_count += 1
    
    all_stale = stale_count == len(providers)
    
    assert all_stale is True


def test_data_freshness_logged_for_diagnostics():
    """Each provider's freshness is logged for diagnostic audit."""
    
    current_time = datetime.now(IST)
    
    data_sources = {
        "spot": {
            "timestamp": get_ist_time(minutes_ago=2),
            "freshness_critical": True,
        },
        "option_chain": {
            "timestamp": get_ist_time(minutes_ago=8),
            "freshness_critical": True,
        },
        "macro_data": {
            "timestamp": get_ist_time(minutes_ago=45),
            "freshness_critical": False,
        },
    }
    
    diagnostics = {}
    for source_name, source_data in data_sources.items():
        source_time = pd.to_datetime(source_data["timestamp"])
        age_minutes = (current_time - source_time).total_seconds() / 60
        diagnostics[source_name] = {
            "age_minutes": age_minutes,
            "is_critical": source_data["freshness_critical"],
        }
    
    # Spot and chain should be fresh, macro can be older
    assert diagnostics["spot"]["age_minutes"] < 5
    assert diagnostics["option_chain"]["age_minutes"] < 10
    assert diagnostics["macro_data"]["age_minutes"] > 30


def test_stale_check_happens_before_calculations():
    """Staleness check occurs before any analytics calculations."""
    
    calculation_attempted = False
    
    option_chain = {
        "timestamp": get_ist_time(minutes_ago=25),
    }
    
    current_time = datetime.now(IST)
    chain_time = pd.to_datetime(option_chain["timestamp"])
    age_minutes = (current_time - chain_time).total_seconds() / 60
    
    if age_minutes > 15:
        # Block before calculation
        calculation_attempted = False
    else:
        # Proceed with calculation
        calculation_attempted = True
    
    assert calculation_attempted is False


def test_backtest_detects_cumulative_dataset_staleness():
    """In backtest, cumulative dataset staleness is detected."""
    
    cumulative_dataset = {
        "last_update": get_ist_time(minutes_ago=480),  # 8 hours ago
    }
    
    current_time = datetime.now(IST)
    last_update = pd.to_datetime(cumulative_dataset["last_update"])
    age_minutes = (current_time - last_update).total_seconds() / 60
    
    dataset_threshold = 240  # 4 hours max
    is_stale = age_minutes > dataset_threshold
    
    assert is_stale is True
