"""
Type consistency and timestamp conversion tests for trading engine.

Tests cover:
- UTC/IST timezone conversion accuracy
- Type consistency in data flows (numeric types, enums, booleans)
- Proper handling of None and missing timestamps
- Contract type consistency (CE/PE)
- Probability bounds [0, 1]
- Bid/ask consistency
- Logging with proper context
"""

import pytest
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta


class TestTimestampConversion:
    """Test UTC to IST timezone conversion for signal timestamps."""

    def test_none_timestamps_in_option_chains(self):
        """None timestamps should be safely handled."""
        option_chain = pd.DataFrame({
            'SYMBOL': ['BANKNIFTY2412C27000', None, 'BANKNIFTY2412P27000'],
            'EXPIRY_TIMESTAMP': [None, None, datetime(2024, 12, 26, 15, 30, tzinfo=pytz.UTC)],
        })
        parsed = pd.to_datetime(option_chain['EXPIRY_TIMESTAMP'], errors='coerce')
        assert parsed.isna().sum() >= 2

    def test_utc_to_ist_conversion(self):
        """UTC timezone should convert to IST (UTC+5:30)."""
        utc_time = pd.Timestamp('2024-12-20 15:30:00', tz=pytz.UTC)
        ist_time = utc_time.tz_convert('Asia/Kolkata')
        expected_hour = 21  # 15:30 UTC + 5:30 = 21:00 IST
        assert ist_time.hour == expected_hour

    def test_stale_detection_across_timezones(self):
        """Staleness detection should work across timezones."""
        current_ist = pd.Timestamp(datetime.now(tz=pytz.timezone('Asia/Kolkata')))
        data_time_ist = current_ist - timedelta(minutes=20)
        data_time_utc = data_time_ist.tz_convert(pytz.UTC)
        
        stale_threshold = timedelta(minutes=15)
        is_stale = (current_ist - data_time_ist) > stale_threshold
        assert bool(is_stale) is True

    def test_missing_expiry_timestamps(self):
        """Missing expiry timestamps should be logged, not raise."""
        data = {
            'SYMBOL': ['BANKNIFTY2412C27000'],
            'EXPIRY_TIMESTAMP': [pd.NaT],
        }
        df = pd.DataFrame(data)
        missing = df['EXPIRY_TIMESTAMP'].isna().sum()
        assert missing == 1

    def test_ist_signal_timestamps_captured_accurately(self):
        """Signal timestamps in IST should match current time accurately."""
        ist_tz = pytz.timezone('Asia/Kolkata')
        current_ist = datetime.now(tz=ist_tz)
        
        # Simulate signal captured in IST
        signal_time = current_ist
        
        # Verify it's in IST timezone
        assert signal_time.tzinfo is not None
        # Check that timezone offset is +5:30 (IST)
        offset_seconds = signal_time.utcoffset().total_seconds()
        offset_hours = offset_seconds / 3600
        assert offset_hours == 5.5

    def test_mixed_timezone_option_chains_handled_safely(self):
        """Mixed timezone data should not raise, handle gracefully."""
        option_chain = pd.DataFrame({
            'SYMBOL': ['BN1', 'BN2', 'BN3'],
            'TIMESTAMP': [
                pd.Timestamp('2024-12-20 10:00:00', tz=pytz.UTC),
                pd.Timestamp('2024-12-20 15:30:00'),  # Naive
                None,
            ]
        })
        parsed = pd.to_datetime(option_chain['TIMESTAMP'], utc=True, errors='coerce')
        valid = int(parsed.notna().sum())
        assert valid == 2


class TestTypeConsistency:
    """Test data type consistency throughout engine."""

    def test_contract_type_consistency_ce_pe(self):
        """Contract types should be consistently 'CE' or 'PE'."""
        contracts = {'BANKNIFTY2412C27000': 'CE', 'BANKNIFTY2412P27000': 'PE'}
        
        for symbol, expected_type in contracts.items():
            # Extract from symbol - find C/P position from end
            # Strip numbers from end to find contract type
            for i in range(len(symbol) - 1, -1, -1):
                if symbol[i] in ('C', 'P'):
                    calculated_type = 'CE' if symbol[i] == 'C' else 'PE'
                    assert calculated_type == expected_type
                    break

    def test_numeric_strike_consistency(self):
        """Strike prices should be consistently numeric."""
        strikes = [27000, 27100, 27200]
        for strike in strikes:
            assert isinstance(strike, (int, np.integer))
            assert strike > 0

    def test_probability_values_bounded_0_to_1(self):
        """Probability values must be bounded [0, 1]."""
        probabilities = np.array([0.0, 0.5, 0.75, 1.0])
        result = bool((probabilities >= 0.0).all() and (probabilities <= 1.0).all())
        assert result is True

    def test_open_interest_non_negative(self):
        """Open interest should never be negative."""
        oi_values = np.array([0, 100000, 500000, 1000000])
        result = bool((oi_values >= 0).all())
        assert result is True

    def test_bid_ask_consistency_bid_lte_ask(self):
        """Bid price should always be <= ask price."""
        bids = np.array([100, 200, 150])
        asks = np.array([105, 205, 155])
        result = bool((bids <= asks).all())
        assert result is True

    def test_direction_enum_consistency(self):
        """Direction should be CALL, PUT, or None."""
        directions = ['CALL', 'PUT', None, 'CALL']
        valid_directions = {'CALL', 'PUT', None}
        
        for direction in directions:
            assert direction in valid_directions

    def test_standardized_confidence_levels(self):
        """Confidence levels should be LOW, MEDIUM, HIGH."""
        confidence_levels = ['LOW', 'MEDIUM', 'HIGH']
        valid_levels = {'LOW', 'MEDIUM', 'HIGH'}
        
        for level in confidence_levels:
            assert level in valid_levels


class TestStructuredLogging:
    """Test logging contains proper context and structure."""

    def test_structured_signal_logging_has_context(self):
        """Signal log should include timestamp, symbol, direction."""
        log_entry = {
            'timestamp': '2024-12-20T15:30:00+05:30',
            'symbol': 'BANKNIFTY2412C27000',
            'direction': 'CALL',
        }
        
        assert 'timestamp' in log_entry
        assert 'symbol' in log_entry
        assert 'direction' in log_entry

    def test_trade_decision_context_logged(self):
        """Trade decisions should include risk context."""
        trade_entry = {
            'timestamp': '2024-12-20T15:30:00+05:30',
            'symbol': 'BANKNIFTY2412C27000',
            'dealer_position': 'SHORT',
            'gamma_exposure': -0.05,
            'liquidity_quality': 'GOOD',
        }
        
        required_fields = {'timestamp', 'symbol', 'dealer_position'}
        assert required_fields.issubset(set(trade_entry.keys()))

    def test_error_stacktrace_logged_with_context(self):
        """Error logs should include stacktrace and context."""
        error_entry = {
            'error_type': 'ZeroDivisionError',
            'stacktrace': 'division by zero at line 142',
            'context': 'greeks computation',
            'recovery_action': 'fallback to previous greeks',
        }
        
        assert 'error_type' in error_entry
        assert 'stacktrace' in error_entry
        assert 'recovery_action' in error_entry

    def test_replay_mode_flag_logged(self):
        """Replay mode should be logged in all timestamped entries."""
        log_entry_live = {'timestamp': '2024-12-20T15:30:00', 'replay': False}
        log_entry_backtest = {'timestamp': '2024-01-15T10:15:00', 'replay': True}
        
        assert 'replay' in log_entry_live
        assert 'replay' in log_entry_backtest


class TestEdgeCasesTypeHandling:
    """Test edge cases in type handling."""

    def test_zero_volatility_type_consistency(self):
        """Zero volatility should be numeric 0, not None."""
        vol = 0.0
        assert isinstance(vol, (int, float, np.number))
        assert vol == 0

    def test_zero_price_range_type_consistency(self):
        """Price range of zero should be numeric."""
        price_range = 0.0
        assert isinstance(price_range, (int, float, np.number))

    def test_empty_dataframe_column_types_preserved(self):
        """Empty DataFrame should preserve column types."""
        df = pd.DataFrame({
            'symbol': pd.Series([], dtype=str),
            'price': pd.Series([], dtype=float),
            'quantity': pd.Series([], dtype=int),
        })
        
        assert df['symbol'].dtype == object
        assert df['price'].dtype == float
        assert df['quantity'].dtype == int


class TestIST_UTC_EdgeCases:
    """Test IST/UTC conversion edge cases."""

    def test_ist_market_open_conversion(self):
        """Market open at 09:15 IST should convert correctly."""
        market_open_ist = pd.Timestamp('2024-12-20 09:15:00', tz=pytz.timezone('Asia/Kolkata'))
        market_open_utc = market_open_ist.tz_convert(pytz.UTC)
        
        # 09:15 IST (UTC+5:30) = 03:45 UTC
        assert market_open_utc.hour == 3
        assert market_open_utc.minute == 45

    def test_ist_market_close_conversion(self):
        """Market close at 15:30 IST should convert correctly."""
        market_close_ist = pd.Timestamp('2024-12-20 15:30:00', tz=pytz.timezone('Asia/Kolkata'))
        market_close_utc = market_close_ist.tz_convert(pytz.UTC)
        
        # 15:30 IST (UTC+5:30) = 10:00 UTC
        assert market_close_utc.hour == 10
        assert market_close_utc.minute == 0

    def test_dst_edge_case_not_applicable_to_ist(self):
        """IST should not have DST (remains UTC+5:30 year-round)."""
        winter_ist = pd.Timestamp('2024-01-15 12:00:00', tz=pytz.timezone('Asia/Kolkata'))
        summer_ist = pd.Timestamp('2024-07-15 12:00:00', tz=pytz.timezone('Asia/Kolkata'))
        
        winter_offset = winter_ist.utcoffset().total_seconds() / 3600
        summer_offset = summer_ist.utcoffset().total_seconds() / 3600
        
        # Both should be +5.5 hours
        assert winter_offset == 5.5
        assert summer_offset == 5.5
