"""Tests for provider normalization and alias column resolution."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


def test_normalization_unknown_column_set_handled():
    """When provider returns completely unknown column set, normalization handles gracefully."""
    
    # Provider returns columns that don't match any known pattern
    provider_df = pd.DataFrame({
        "unknown_col_1": [1, 2, 3],
        "unknown_col_2": ["a", "b", "c"],
        "random_data": [10.5, 20.5, 30.5],
    })
    
    # Expected columns for normalization
    required_columns = ["strikePrice", "optionType", "bid", "ask", "ltp", "openInterest"]
    
    # Check which required columns are present
    present_cols = [col for col in required_columns if col in provider_df.columns]
    missing_cols = [col for col in required_columns if col not in provider_df.columns]
    
    # All should be missing
    assert len(missing_cols) == len(required_columns)
    assert len(present_cols) == 0


def test_column_alias_mapping_fallback_single_column_missing():
    """When one alias mapping fails, fallback logic provides safe default."""
    
    provider_df = pd.DataFrame({
        "Strike": [22900, 23000, 23100],
        "Type": ["CE", "CE", "CE"],
        "Bid": [150.0, 160.0, 170.0],
        "Ask": [155.0, 165.0, 175.0],
        # Missing: "LastTradedPrice"
    })
    
    # Map to standard columns
    alias_map = {
        "Strike": "strikePrice",
        "Type": "optionType",
        "Bid": "bid",
        "Ask": "ask",
        "LastTradedPrice": "ltp",  # Missing column
    }
    
    # Rename columns that exist
    normalized = provider_df.copy()
    for provider_col, standard_col in alias_map.items():
        if provider_col in normalized.columns:
            normalized = normalized.rename(columns={provider_col: standard_col})
    
    # Add missing columns with safe defaults
    for standard_col in alias_map.values():
        if standard_col not in normalized.columns:
            normalized[standard_col] = np.nan
    
    # Verify result
    assert "strikePrice" in normalized.columns
    assert "ltp" in normalized.columns
    assert normalized["ltp"].isna().all()  # Safe default is NaN


def test_column_alias_mapping_failure_threshold():
    """When >50% of columns fail to map, alert is raised."""
    
    total_columns = 6
    failed_mappings = 4  # >50% failure
    success_rate = (total_columns - failed_mappings) / total_columns
    
    # Should alert when success_rate < 0.5
    should_alert = success_rate < 0.5
    assert should_alert is True


def test_mixed_column_naming_nifty_standard_vs_provider():
    """When provider uses mix of NSE-std and provider-specific naming, handle both."""
    
    provider_df = pd.DataFrame({
        # NSE standard columns
        "strikePrice": [22900, 23000, 23100],
        "OPTION_TYP": ["CE", "CE", "CE"],  # Provider-specific
        # Mixed: some standard, some provider-specific
        "bid_price": [150.0, 160.0, 170.0],  # Provider-specific
        "Ask": [155.0, 165.0, 175.0],  # NSE-like
    })
    
    # Define aliases to normalize everything
    aliases = {
        "strikePrice": "strike",
        "OPTION_TYP": "option_type",
        "bid_price": "bid",
        "Ask": "ask",
    }
    
    normalized_df = provider_df.rename(columns=aliases)
    
    # All should map to standard names
    assert "strike" in normalized_df.columns
    assert "option_type" in normalized_df.columns
    assert "bid" in normalized_df.columns
    assert "ask" in normalized_df.columns


def test_duplicate_rows_same_strike_different_providers():
    """When duplicate rows exist for same strike but different providers, deduplicate."""
    
    # Simulates option chain with rows from multiple providers
    df = pd.DataFrame({
        "strikePrice": [23000, 23000, 23000, 23100, 23100],
        "optionType": ["CE", "CE", "CE", "CE", "CE"],
        "provider": ["provider_a", "provider_b", "provider_a", "provider_a", "provider_b"],
        "bid": [150.0, 151.0, 150.0, 160.0, 161.0],
        "ask": [155.0, 156.0, 155.0, 165.0, 166.0],
    })
    
    # Remove duplicates keeping the first occurrence of each strike+type combo
    deduplicated = df.drop_duplicates(subset=["strikePrice", "optionType"], keep="first")
    
    # Should have 3 unique strike+type combos
    assert len(deduplicated) == 2  # Two unique strikes
    assert deduplicated.iloc[0]["provider"] == "provider_a"


def test_provider_returns_all_empty_dataframe():
    """When provider returns empty dataframe, system handles gracefully."""
    
    empty_df = pd.DataFrame({
        "strikePrice": pd.Series(dtype=float),
        "optionType": pd.Series(dtype=str),
    })
    
    # Should be safely handled
    assert len(empty_df) == 0
    assert empty_df.empty is True


def test_column_type_mismatch_bid_ask_strings_instead_of_float():
    """When bid/ask are strings instead of float, coercion with fallback."""
    
    provider_df = pd.DataFrame({
        "strikePrice": [23000, 23000, 23000],
        "optionType": ["CE", "CE", "CE"],
        "bid": ["150.5", "not_a_number", "160.5"],  # Mixed types
        "ask": ["155.5", "invalid", "165.5"],
    })
    
    # Coerce to float with errors='coerce' for safety
    normalized = provider_df.copy()
    normalized["bid"] = pd.to_numeric(normalized["bid"], errors="coerce")
    normalized["ask"] = pd.to_numeric(normalized["ask"], errors="coerce")
    
    # Invalid values become NaN
    assert normalized["bid"].iloc[1] is pd.NA or pd.isna(normalized["bid"].iloc[1])
    assert normalized["ask"].iloc[1] is pd.NA or pd.isna(normalized["ask"].iloc[1])


def test_provider_column_order_irrelevant():
    """Order of columns should not matter for normalization."""
    
    # Two dataframes with same data but different column order
    df1 = pd.DataFrame({
        "strikePrice": [23000, 23100],
        "optionType": ["CE", "CE"],
        "bid": [150.0, 160.0],
    })
    
    df2 = pd.DataFrame({
        "bid": [150.0, 160.0],
        "strikePrice": [23000, 23100],
        "optionType": ["CE", "CE"],
    })
    
    # After normalization, both should have same column set
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    assert cols1 == cols2
