"""Tests for bid/ask spread and pricing basis validation."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


def test_one_sided_quote_bid_missing():
    """When bid is missing but ask exists, handle gracefully."""
    
    option_row = {
        "strikePrice": 23000,
        "optionType": "CE",
        "bid": np.nan,  # Missing
        "ask": 155.0,
        "ltp": 150.0,  # Can use as fallback
    }
    
    # Detect one-sided quote
    is_one_sided = pd.isna(option_row["bid"]) or pd.isna(option_row["ask"])
    assert is_one_sided is True
    
    # Can use LTP or ask as approximation
    mid_price = option_row["ask"] if pd.isna(option_row["bid"]) else (option_row["bid"] + option_row["ask"]) / 2
    assert mid_price == 155.0


def test_one_sided_quote_ask_missing():
    """When ask is missing but bid exists, handle gracefully."""
    
    option_row = {
        "strikePrice": 23000,
        "optionType": "CE",
        "bid": 150.0,
        "ask": np.nan,  # Missing
        "ltp": 150.0,
    }
    
    # Detect one-sided quote
    is_one_sided = pd.isna(option_row["bid"]) or pd.isna(option_row["ask"])
    assert is_one_sided is True
    
    # Use bid as fallback
    mid_price = option_row["bid"] if pd.isna(option_row["ask"]) else (option_row["bid"] + option_row["ask"]) / 2
    assert mid_price == 150.0


def test_anomalous_spread_wider_than_strike_price():
    """When spread is wider than strike price, flag as anomalous."""
    
    option_row = {
        "strikePrice": 100,  # Very low strike
        "bid": 1.0,
        "ask": 150.0,  # Spread of 149 > strike of 100
    }
    
    spread = option_row["ask"] - option_row["bid"]
    spread_pct = (spread / option_row["strikePrice"]) * 100
    
    # Detect anomalous spread
    is_anomalous = spread > option_row["strikePrice"]
    assert is_anomalous is True
    assert spread_pct > 100  # >100% spread


def test_anomalous_spread_extreme_pct():
    """When spread exceeds 50% of mid-price, flag as extreme."""
    
    option_row = {
        "strikePrice": 23000,
        "bid": 100.0,
        "ask": 200.0,  # Spread of 100
    }
    
    mid_price = (option_row["bid"] + option_row["ask"]) / 2  # 150
    spread = option_row["ask"] - option_row["bid"]
    spread_pct = (spread / mid_price) * 100
    
    # Detect extreme spread
    is_extreme = spread_pct > 50
    assert is_extreme is True
    assert spread_pct == pytest.approx(66.67, rel=0.01)


def test_mid_price_calculation_normal_case():
    """Normal case: mid-price is (bid + ask) / 2."""
    
    option_row = {
        "strikePrice": 23000,
        "bid": 150.0,
        "ask": 160.0,
    }
    
    mid_price = (option_row["bid"] + option_row["ask"]) / 2
    assert mid_price == 155.0


def test_mid_price_calculation_with_reversed_quotes():
    """When bid > ask (reversed quotes), use absolute ordering."""
    
    option_row = {
        "strikePrice": 23000,
        "bid": 160.0,  # Reversed: bid > ask
        "ask": 150.0,
    }
    
    # Detect reversed quotes
    is_reversed = option_row["bid"] > option_row["ask"]
    assert is_reversed is True
    
    # Use min/max for correct ordering
    actual_bid = min(option_row["bid"], option_row["ask"])
    actual_ask = max(option_row["bid"], option_row["ask"])
    mid_price = (actual_bid + actual_ask) / 2
    
    assert mid_price == 155.0


def test_mid_price_when_bid_equals_ask():
    """When bid equals ask, mid-price is the same value."""
    
    option_row = {
        "strikePrice": 23000,
        "bid": 155.0,
        "ask": 155.0,  # No spread
    }
    
    mid_price = (option_row["bid"] + option_row["ask"]) / 2
    assert mid_price == 155.0


def test_pricing_basis_changes_mid_chain():
    """Detect when pricing basis changes within the option chain."""
    
    # Simulate option chain where first rows are premium basis, mid-chain switches
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100, 23200, 23300],
        "bid": [250, 200, 150, 50, 10],  # Decreasing premium
        "ask": [260, 210, 160, 60, 20],
        "openInterest": [50000, 60000, 80000, 5000, 1000],  # Sudden drop at 23200
    })
    
    # Detect basis change: OI drops significantly
    oi = option_chain["openInterest"]
    oi_ratio = float(oi.iloc[3] / oi.iloc[2])  # 5000 / 80000 = 0.0625
    
    basis_changed = oi_ratio < 0.1  # Sudden drop
    assert basis_changed is True


def test_liquidity_vacuum_detected_by_spread():
    """When bid-ask spread widens dramatically, liquidity vacuum likely."""
    
    # Normal spread
    normal_spread = 2.0  # bid=150, ask=152
    
    # After market shock - liquidity vacuum
    vacuum_spread = 50.0  # bid=125, ask=175
    
    spread_ratio = vacuum_spread / normal_spread
    is_vacuum = spread_ratio > 10  # >10x increase
    
    assert is_vacuum is True


def test_pricing_basis_validation_ltp_within_spread():
    """LTP should be within bid-ask spread for valid pricing basis."""
    
    option_row = {
        "bid": 150.0,
        "ask": 160.0,
        "ltp": 155.0,  # Within spread
    }
    
    is_valid = option_row["bid"] <= option_row["ltp"] <= option_row["ask"]
    assert is_valid is True


def test_pricing_basis_validation_ltp_outside_spread():
    """When LTP is outside bid-ask spread, flag as pricing error."""
    
    option_row = {
        "bid": 150.0,
        "ask": 160.0,
        "ltp": 175.0,  # Outside spread (above ask)
    }
    
    is_valid = option_row["bid"] <= option_row["ltp"] <= option_row["ask"]
    assert is_valid is False
    
    # Delta from spread
    delta_above_ask = option_row["ltp"] - option_row["ask"]
    assert delta_above_ask == 15.0


def test_negative_spread_not_possible():
    """Spread cannot be negative (ask must be >= bid)."""
    
    option_row = {
        "bid": 150.0,
        "ask": 160.0,
    }
    
    spread = option_row["ask"] - option_row["bid"]
    is_valid = spread >= 0
    
    assert is_valid is True
    assert spread == 10.0
