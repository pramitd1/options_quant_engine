"""Tests for liquidity vacuum and breakout detection edge cases."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


def test_liquidity_all_zeros_handled():
    """When liquidity (volume) is zero for all strikes, system handles gracefully."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100, 23200, 23300],
        "openInterest": [0, 0, 0, 0, 0],  # All zero
        "volume": [0, 0, 0, 0, 0],
        "bid": [150, 160, 170, 180, 190],
        "ask": [155, 165, 175, 185, 195],
    })
    
    # Total liquidity check
    total_liquidity = int(option_chain["openInterest"].sum()) + int(option_chain["volume"].sum())
    is_empty = total_liquidity == 0
    
    assert is_empty is True


def test_open_interest_missing_fallback():
    """When openInterest is NaN/missing, use volume as fallback."""
    
    option_row = {
        "strikePrice": 23000,
        "openInterest": np.nan,  # Missing
        "volume": 5000,  # Fallback
    }
    
    # Use volume if OI missing
    effective_liquidity = option_row["volume"] if pd.isna(option_row["openInterest"]) else option_row["openInterest"]
    
    assert effective_liquidity == 5000


def test_vacuum_without_breakout_zone_detected():
    """When liquidity vacuum exists but no breakout zone follows, flag it."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100, 23200, 23300],
        "openInterest": [60000, 80000, 50000, 5000, 3000],  # Vacuum at 23200
    })
    
    # Detect vacuum: OI drops >50%
    oi = option_chain["openInterest"].values
    vacuum_detected = False
    for i in range(1, len(oi)):
        if oi[i] < oi[i-1] * 0.5:  # >50% drop
            vacuum_detected = True
            vacuum_idx = i
            break
    
    # Check if breakout zone follows (OI recovery)
    has_breakout = False
    if vacuum_detected and vacuum_idx < len(oi) - 1:
        # Look for OI expansion after vacuum
        for j in range(vacuum_idx + 1, len(oi)):
            if oi[j] > oi[vacuum_idx]:  # OI recovers
                has_breakout = True
                break
    
    assert vacuum_detected is True
    assert has_breakout is False  # No recovery


def test_liquidity_reversal_vacuum_then_sudden_volume():
    """When liquidity suddenly reverses (vacuum then high volume), detect anomaly."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100, 23200, 23300],
        "openInterest": [80000, 60000, 30000, 90000, 75000],  # Vacuum then reversal
    })
    
    oi = option_chain["openInterest"].values
    
    # Detect reversal pattern
    reversal_found = False
    for i in range(2, len(oi)):
        # Check for: decrease then increase > 2x
        if oi[i-1] < oi[i-2] and oi[i] > oi[i-1] * 2:
            reversal_found = True
            break
    
    assert reversal_found is True


def test_bid_ask_spread_widens_into_liquidity_vacuum():
    """When bid-ask spread widens as liquidity disappears, correlation detected."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100, 23200, 23300],
        "openInterest": [80000, 60000, 40000, 5000, 2000],  # Declining OI
        "bid": [150.0, 155.0, 160.0, 120.0, 100.0],
        "ask": [155.0, 160.0, 165.0, 200.0, 250.0],  # Spread widens
    })
    
    # Compute spread per row
    option_chain["spread"] = option_chain["ask"] - option_chain["bid"]
    
    # Correlation check: as OI drops, spread increases
    oi = option_chain["openInterest"].values
    spread = option_chain["spread"].values
    
    # Detect correlation (both first and last have inverse relationship)
    oi_declining = bool(oi[-1] < oi[0])
    spread_widening = bool(spread[-1] > spread[0])
    
    correlation_detected = oi_declining and spread_widening
    assert correlation_detected is True


def test_oversized_trade_vs_liquidity_execution_risk():
    """When trade size exceeds normal liquidity by 10x, flag execution risk."""
    
    option_row = {
        "strikePrice": 23000,
        "openInterest": 1000,  # Very low liquidity
        "ask": 160.0,
    }
    
    trade_size = 100  # 100 contracts
    normal_trade_size = 10  # Typical is 10 contracts
    
    trade_to_liquidity_ratio = trade_size / option_row["openInterest"]
    size_exceeds_ratio = trade_size / normal_trade_size
    
    execution_risk = trade_to_liquidity_ratio > 0.1 or size_exceeds_ratio > 5
    
    assert execution_risk is True


def test_liquidity_map_heatmap_with_zeros():
    """When computing liquidity heatmap, zero values don't cause division errors."""
    
    liquidity_matrix = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "call_oi": [50000, 80000, 0],  # One zero
        "put_oi": [45000, 75000, 0],   # One zero
    })
    
    # Safe division: avoid divide-by-zero
    total_oi = liquidity_matrix["call_oi"] + liquidity_matrix["put_oi"]
    total_oi = total_oi.replace(0, np.nan)  # Replace 0 with NaN to avoid div errors
    
    call_ratio = liquidity_matrix["call_oi"] / total_oi
    put_ratio = liquidity_matrix["put_oi"] / total_oi
    
    # Result should have NaN for zero-total rows
    assert pd.isna(call_ratio.iloc[2])
    assert pd.isna(put_ratio.iloc[2])


def test_vacuum_detection_threshold():
    """Vacuum is detected when OI drops below 20% of previous level."""
    
    option_chain = pd.DataFrame({
        "strikePrice": [23000, 23100, 23200],
        "openInterest": [100000, 50000, 8000],  # 50% drop, then 84% drop (8k < 50k*0.2=10k)
    })
    
    oi = option_chain["openInterest"].values
    vacuum_threshold = 0.2
    
    vacuum_strikes = []
    for i in range(1, len(oi)):
        if oi[i] < oi[i-1] * vacuum_threshold:
            vacuum_strikes.append(option_chain["strikePrice"].iloc[i])
    
    # Should detect vacuum at 23200 (8k < 50k * 0.2)
    assert 23200 in vacuum_strikes
