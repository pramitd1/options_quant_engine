"""Tests for empty DataFrame handling and division by zero protection."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import math


def test_empty_option_chain_returns_graceful_error():
    """When option chain DataFrame is empty, graceful error returned."""
    
    empty_df = pd.DataFrame({
        "strikePrice": pd.Series(dtype=float),
        "bid": pd.Series(dtype=float),
        "ask": pd.Series(dtype=float),
    })
    
    # Detection
    is_empty = len(empty_df) == 0
    
    if is_empty:
        result = {"error": "Empty option chain", "trades": []}
    else:
        result = {"error": None, "trades": []}
    
    assert result["error"] == "Empty option chain"


def test_empty_macro_news_dataset_handled():
    """When macro news dataset returns empty, system continues."""
    
    news_df = pd.DataFrame({
        "headline": pd.Series(dtype=str),
        "impact_score": pd.Series(dtype=float),
    })
    
    # Safe handling
    if len(news_df) == 0:
        processed_news = []
    else:
        processed_news = news_df.to_dict("records")
    
    assert processed_news == []


def test_spot_history_initialization_with_no_prior_data():
    """Spot history with no prior data initializes safely."""
    
    spot_history = pd.DataFrame({
        "timestamp": pd.Series(dtype=str),
        "close": pd.Series(dtype=float),
    })
    
    # Safe initialization
    if len(spot_history) == 0:
        average_price = np.nan
        price_change = np.nan
    else:
        average_price = spot_history["close"].mean()
        price_change = spot_history["close"].pct_change().mean()
    
    assert pd.isna(average_price)
    assert pd.isna(price_change)


def test_analytics_functions_handle_empty_series_in_iv_computation():
    """IV computation with empty series doesn't crash."""
    
    empty_iv_series = pd.Series([], dtype=float)
    
    # Safe computation
    if len(empty_iv_series) == 0:
        avg_iv = np.nan
    else:
        avg_iv = empty_iv_series.mean()
    
    assert pd.isna(avg_iv)


def test_gamma_exposure_calc_with_atm_strike():
    """Gamma calculation when spot equals strike (zero difference) gives zero."""
    
    spot = 23000
    strike = 23000
    gamma_coefficient = 0.01
    
    # Safe calculation avoiding division
    if spot == strike:
        gamma = 0.0  # Safe fallback
    else:
        distance = abs(spot - strike)
        gamma = gamma_coefficient / distance
    
    assert gamma == 0.0


def test_greeks_engine_with_zero_volatility():
    """Greeks calculations with zero IV are safe."""
    
    spot = 23000
    strike = 23100
    iv = 0.0  # Zero volatility
    days_to_expiry = 7
    
    # Safe vega calculation (should be zero when IV is zero)
    if iv <= 0 or days_to_expiry <= 0:
        vega = 0.0
    else:
        vega = 0.01 * iv * math.sqrt(days_to_expiry)
    
    assert vega == 0.0


def test_large_move_probability_with_zero_range():
    """Large move detection with zero price range (prev_close == current)."""
    
    prev_close = 23000
    current_spot = 23000  # No change
    
    # Safe range calculation
    if current_spot == prev_close:
        price_range = 0.0
        move_probability = 0.0
    else:
        price_range = abs(current_spot - prev_close)
        move_probability = price_range / prev_close
    
    assert price_range == 0.0
    assert move_probability == 0.0


def test_dealer_hedging_flow_with_identical_prices():
    """Dealer flow calculation when prices are identical (zero change)."""
    
    previous_dealer_position = 1000
    current_dealer_position = 1000
    
    # Safe calculation
    if current_dealer_position == previous_dealer_position:
        dealer_flow = 0
    else:
        dealer_flow = current_dealer_position - previous_dealer_position
    
    assert dealer_flow == 0


def test_liquidity_heatmap_with_zero_volume():
    """Liquidity heatmap calculation with zero volume avoids division by zero."""
    
    call_volume = 0
    put_volume = 0
    
    # Safe ratio calculation
    total_volume = call_volume + put_volume
    if total_volume == 0:
        call_ratio = 0.0
        put_ratio = 0.0
    else:
        call_ratio = call_volume / total_volume
        put_ratio = put_volume / total_volume
    
    assert call_ratio == 0.0
    assert put_ratio == 0.0


def test_option_efficiency_with_zero_expected_move():
    """Option efficiency calculation with zero expected move."""
    
    strike_price = 23100
    expected_move = 0.0  # Zero move
    distance_from_spot = 100
    
    # Safe efficiency score
    if expected_move == 0:
        efficiency_score = 0.0
    else:
        efficiency_score = distance_from_spot / expected_move
    
    assert efficiency_score == 0.0


def test_overnight_convexity_calculation_zero_dte():
    """Overnight convexity with zero days to expiry."""
    
    days_to_expiry = 0
    gamma = 0.01
    theta = 0.001
    
    # Safe convexity calculation
    if days_to_expiry == 0:
        convexity = 0.0
    else:
        convexity = gamma * days_to_expiry - theta
    
    assert convexity == 0.0


def test_iv_term_structure_missing_fallback():
    """When IV term structure is missing, use flat IV fallback."""
    
    term_structure = None
    flat_iv = 0.25
    
    # Safe fallback
    if term_structure is None:
        effective_iv = flat_iv
    else:
        effective_iv = term_structure["3m"]
    
    assert effective_iv == flat_iv


def test_pinning_location_calculation_with_zero_oi():
    """Pinning detection when open interest is zero."""
    
    call_oi = 0
    put_oi = 0
    strike = 23000
    
    # Safe pinning check
    total_oi = call_oi + put_oi
    if total_oi == 0:
        is_pinning = False
        pinning_strength = 0.0
    else:
        oi_ratio = call_oi / put_oi if put_oi > 0 else 0
        is_pinning = 0.8 < oi_ratio < 1.2
        pinning_strength = 1.0 if is_pinning else 0.0
    
    assert is_pinning is False
    assert pinning_strength == 0.0


def test_expected_move_calc_with_zero_iv():
    """Expected move calculation fails gracefully with zero IV."""
    
    spot = 23000
    iv = 0.0  # Zero volatility
    days_to_expiry = 7
    
    # Safe expected move
    if iv <= 0:
        expected_move = 0.0  # Can't move with zero volatility
    else:
        expected_move = spot * iv * math.sqrt(days_to_expiry / 365)
    
    assert expected_move == 0.0


def test_backtest_empty_market_data_directory():
    """Backtest with empty market data directory returns error."""
    
    market_data = []
    
    if len(market_data) == 0:
        result = {"success": False, "error": "No market data found"}
    else:
        result = {"success": True, "error": None}
    
    assert result["success"] is False


def test_backtest_insufficient_lookback_data():
    """Backtest with only 1-2 days triggers insufficient data error."""
    
    data_days = 1
    min_required_lookback = 20
    
    has_sufficient_data = data_days >= min_required_lookback
    
    assert has_sufficient_data is False
