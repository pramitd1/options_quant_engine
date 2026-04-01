"""
Test suite for pre-market engine components.

Tests cover:
- Pre-market session detection
- Overnight dealer setup
- Morning volatility initialization
- Pre-market signal readiness validation
- Signal adjustments based on pre-market state
- Integration with signal engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.pre_market_policy import get_pre_market_policy_config
from engine.trading_support.pre_market_state import (
    is_pre_market_session,
    build_overnight_dealer_context,
    build_morning_volatility_context,
    validate_pre_market_readiness,
    apply_pre_market_signal_adjustments,
)
from engine.pre_market_engine import (
    initialize_pre_market_context,
    apply_pre_market_adjustments_to_signal,
    build_pre_market_diagnostic_report,
)


IST_TIMEZONE = "Asia/Kolkata"


class TestPreMarketSessionDetection:
    """Test pre-market session detection logic."""
    
    def test_is_pre_market_session_during_pre_market_hours(self):
        """Verify that 08:30 IST is detected as pre-market."""
        pre_market_ts = pd.Timestamp("2026-04-01 08:30:00", tz=IST_TIMEZONE)
        assert is_pre_market_session(pre_market_ts) is True
    
    def test_is_pre_market_session_outside_pre_market_hours(self):
        """Verify that intraday hours (10:00 IST) are not pre-market."""
        intraday_ts = pd.Timestamp("2026-04-01 10:00:00", tz=IST_TIMEZONE)
        assert is_pre_market_session(intraday_ts) is False
    
    def test_is_pre_market_session_before_pre_market_start(self):
        """Verify that before 08:00 IST is not pre-market."""
        early_ts = pd.Timestamp("2026-04-01 07:00:00", tz=IST_TIMEZONE)
        assert is_pre_market_session(early_ts) is False
    
    def test_is_pre_market_session_at_exact_boundary(self):
        """Verify boundary handling at 09:15 IST."""
        boundary_ts = pd.Timestamp("2026-04-01 09:15:00", tz=IST_TIMEZONE)
        assert is_pre_market_session(boundary_ts) is False  # At/after 09:15 is intraday
    
    def test_is_pre_market_session_one_minute_before_boundary(self):
        """Verify one minute before boundary is pre-market."""
        before_boundary_ts = pd.Timestamp("2026-04-01 09:14:59", tz=IST_TIMEZONE)
        assert is_pre_market_session(before_boundary_ts) is True


class TestOvernightDealerSetup:
    """Test overnight dealer position context building."""
    
    def test_overnight_dealer_context_with_current_position(self):
        """Verify dealer context with current session position available."""
        ctx = build_overnight_dealer_context(
            current_dealer_position="Long Gamma",
            current_dealer_basis="OI_CHANGE",
            previous_session_position="Short Gamma",
            call_oi=1000.0,
            put_oi=800.0,
        )
        
        assert ctx["position"] == "Long Gamma"
        assert ctx["basis"] == "OI_CHANGE"
        assert ctx["confidence"] == 1.0
        assert ctx["session_origin"] == "current"
        assert ctx["is_bootstrapped"] is False
    
    def test_overnight_dealer_context_with_previous_position_only(self):
        """Verify dealer context falls back to previous session."""
        ctx = build_overnight_dealer_context(
            current_dealer_position=None,
            current_dealer_basis=None,
            previous_session_position="Long Gamma",
            call_oi=0.0,
            put_oi=0.0,
        )
        
        assert ctx["position"] == "Long Gamma"
        assert ctx["confidence"] < 1.0  # Reduced confidence for carryover
        assert ctx["session_origin"] == "previous_session"
        assert ctx["is_bootstrapped"] is True
    
    def test_overnight_dealer_context_neutral_fallback(self):
        """Verify neutral fallback when no position data available."""
        ctx = build_overnight_dealer_context(
            current_dealer_position=None,
            current_dealer_basis=None,
            previous_session_position=None,
            call_oi=0.0,
            put_oi=0.0,
        )
        
        assert ctx["position"] == "NEUTRAL"
        assert ctx["confidence"] == 0.0
        assert ctx["is_bootstrapped"] is True


class TestMorningVolatilityInitialization:
    """Test morning volatility context building."""
    
    def test_morning_volatility_context_with_current_regime(self):
        """Verify volatility context with current IV regime."""
        ctx = build_morning_volatility_context(
            current_iv_regime="VOL_EXPANSION",
            previous_iv_regime="VOL_NEUTRAL",
            implied_vol_median=0.25,
            realized_vol_5d=0.20,
            realized_vol_30d=0.22,
            iv_percentile=0.75,
        )
        
        assert ctx["regime"] == "VOL_EXPANSION"
        assert ctx["confidence"] > 0.8
        assert ctx["session_origin"] == "current"
        assert ctx["iv_level"] == "HIGH"
        assert ctx["is_bootstrapped"] is False
    
    def test_morning_volatility_context_with_percentiles(self):
        """Verify IV level classification by percentile."""
        # High IV (percentile >= 0.7)
        ctx_high = build_morning_volatility_context(
            implied_vol_median=0.30,
            iv_percentile=0.75,
        )
        assert ctx_high["iv_level"] == "HIGH"
        
        # Low IV (percentile <= 0.3)
        ctx_low = build_morning_volatility_context(
            implied_vol_median=0.10,
            iv_percentile=0.25,
        )
        assert ctx_low["iv_level"] == "LOW"
        
        # Normal IV
        ctx_normal = build_morning_volatility_context(
            implied_vol_median=0.20,
            iv_percentile=0.50,
        )
        assert ctx_normal["iv_level"] == "NORMAL"


class TestPreMarketReadinessValidation:
    """Test pre-market data quality and readiness checks."""
    
    def test_validate_pre_market_readiness_all_good(self):
        """Verify readiness passes all checks."""
        result = validate_pre_market_readiness(
            market_snapshot_quality=85.0,
            option_chain_iv_count=50,
            global_market_staleness_minutes=15.0,
            has_dealer_positioning=True,
            has_volatility_data=True,
        )
        
        assert result["ready"] is True
        all_checks_pass = all(
            check.get("pass", True)
            for check in result["checks"].values()
            if check.get("required", True)
        )
        assert all_checks_pass is True
    
    def test_validate_pre_market_readiness_low_quality(self):
        """Verify readiness fails with low quality score."""
        cfg = get_pre_market_policy_config()
        result = validate_pre_market_readiness(
            market_snapshot_quality=50.0,  # Below minimum
            option_chain_iv_count=30,
            global_market_staleness_minutes=15.0,
        )
        
        assert result["ready"] is False
        assert result["checks"]["quality_score"]["pass"] is False
    
    def test_validate_pre_market_readiness_stale_market(self):
        """Verify readiness fails with stale global market."""
        cfg = get_pre_market_policy_config()
        result = validate_pre_market_readiness(
            market_snapshot_quality=80.0,
            option_chain_iv_count=30,
            global_market_staleness_minutes=120.0,  # 2 hours, too stale
        )
        
        assert result["ready"] is False
        assert result["checks"]["global_market_staleness"]["pass"] is False


class TestPreMarketSignalAdjustments:
    """Test signal strength adjustments for pre-market."""
    
    def test_apply_pre_market_signal_adjustments_when_enabled(self):
        """Verify signal adjustments with pre-market enabled."""
        # Since enable_pre_market_signals is False by default,
        # test that adjustments work when the logic is applied
        from unittest.mock import patch
        
        with patch('engine.trading_support.pre_market_state.get_pre_market_policy_config') as mock_cfg:
            mock_config = mock_cfg.return_value
            mock_config.enable_pre_market_signals = True
            mock_config.pre_market_signal_quality_boost = 1.25
            mock_config.pre_market_min_trade_strength = 40
            
            result = apply_pre_market_signal_adjustments(
                base_trade_strength=60.0,
                data_quality_score=85.0,
                is_pre_market=True,
            )
            
            assert result["signal_eligible"] is True
            assert result["quality_multiplier"] == 1.25
            assert result["adjusted_trade_strength"] == 75.0
    
    def test_apply_pre_market_signal_adjustments_when_disabled(self):
        """Verify no signals when pre-market disabled by policy."""
        from unittest.mock import patch
        
        with patch('engine.trading_support.pre_market_state.get_pre_market_policy_config') as mock_cfg:
            mock_config = mock_cfg.return_value
            mock_config.enable_pre_market_signals = False
            
            result = apply_pre_market_signal_adjustments(
                base_trade_strength=60.0,
                data_quality_score=85.0,
                is_pre_market=True,
            )
            
            assert result["signal_eligible"] is False
            assert result["adjusted_trade_strength"] == 0.0
    
    def test_apply_pre_market_signal_adjustments_outside_pre_market(self):
        """Verify no adjustment outside pre-market hours."""
        result = apply_pre_market_signal_adjustments(
            base_trade_strength=60.0,
            data_quality_score=85.0,
            is_pre_market=False,
        )
        
        assert result["signal_eligible"] is True
        assert result["quality_multiplier"] == 1.0
        assert result["adjusted_trade_strength"] == 60.0


class TestPreMarketContextInitialization:
    """Test full pre-market context initialization."""
    
    def create_sample_option_chain(self):
        """Create a sample option chain dataframe for testing."""
        return pd.DataFrame({
            "strikePrice": [22000, 22100, 22200, 22000, 22100, 22200],
            "OPTION_TYP": ["CE", "CE", "CE", "PE", "PE", "PE"],
            "openInterest": [1000, 800, 600, 1200, 1000, 800],
            "changeinOI": [50, -20, 10, -30, 40, 25],
            "IV": [0.25, 0.26, 0.27, 0.24, 0.25, 0.26],
            "impliedVolatility": [0.25, 0.26, 0.27, 0.24, 0.25, 0.26],
        })
    
    def create_sample_global_market_snapshot(self):
        """Create a sample global market snapshot."""
        return {
            "data_available": True,
            "as_of": pd.Timestamp.now(tz=IST_TIMEZONE).isoformat(),
            "market_inputs": {
                "realized_vol_5d": 0.20,
                "realized_vol_30d": 0.22,
            },
        }
    
    def test_initialize_pre_market_context_full(self):
        """Verify full pre-market context initialization."""
        option_chain = self.create_sample_option_chain()
        global_snapshot = self.create_sample_global_market_snapshot()
        
        ts = pd.Timestamp("2026-04-01 08:30:00", tz=IST_TIMEZONE)
        
        ctx = initialize_pre_market_context(
            option_chain=option_chain,
            spot=22050.0,
            global_market_snapshot=global_snapshot,
            previous_session_state=None,
            as_of=ts,
        )
        
        assert ctx["session"] == "PRE_MARKET"
        assert ctx["is_pre_market"] is True
        assert ctx["spot"] == 22050.0
        assert "dealer_context" in ctx
        assert "volatility_context" in ctx
        assert "readiness" in ctx
        assert ctx["readiness"]["ready"] is not None
    
    def test_initialize_pre_market_context_with_previous_session_state(self):
        """Verify context carries forward previous session state."""
        option_chain = self.create_sample_option_chain()
        global_snapshot = self.create_sample_global_market_snapshot()
        
        previous_state = {
            "dealer_context": {"position": "Short Gamma"},
            "volatility_context": {"regime": "VOL_COMPRESSION"},
        }
        
        ts = pd.Timestamp("2026-04-01 08:30:00", tz=IST_TIMEZONE)
        
        ctx = initialize_pre_market_context(
            option_chain=None,  # No current option chain
            spot=22050.0,
            global_market_snapshot=None,
            previous_session_state=previous_state,
            as_of=ts,
        )
        
        # When no current data, should fallback to previous session
        assert ctx["dealer_context"]["is_bootstrapped"] is True
        assert ctx["volatility_context"]["is_bootstrapped"] is True


class TestPreMarketDiagnosticReport:
    """Test pre-market diagnostic report generation."""
    
    def test_build_pre_market_diagnostic_report(self):
        """Verify diagnostic report generation."""
        pre_market_context = {
            "timestamp": pd.Timestamp.now(tz=IST_TIMEZONE).isoformat(),
            "session": "PRE_MARKET",
            "spot": 22050.0,
            "dealer_context": {
                "position": "Long Gamma",
                "basis": "OI_CHANGE",
                "confidence": 0.9,
                "session_origin": "current",
                "call_oi": 1000.0,
                "put_oi": 800.0,
            },
            "volatility_context": {
                "regime": "VOL_EXPANSION",
                "confidence": 0.85,
                "implied_vol_median": 0.25,
                "realized_vol_5d": 0.20,
                "realized_vol_30d": 0.22,
                "iv_level": "HIGH",
                "iv_percentile": 0.75,
            },
            "readiness": {
                "ready": True,
                "enable_signals": True,
                "checks": {
                    "quality_score": {"pass": True, "value": 85.0},
                    "option_chain_iv_count": {"pass": True, "value": 50},
                },
            },
        }
        
        report = build_pre_market_diagnostic_report(
            pre_market_context=pre_market_context,
        )
        
        assert isinstance(report, str)
        assert "PRE-MARKET DIAGNOSTIC REPORT" in report
        assert "Dealer Setup" in report
        assert "Volatility Setup" in report
        assert "Readiness" in report
        assert "Long Gamma" in report
        assert "VOL_EXPANSION" in report


class TestPreMarketIntegration:
    """Integration tests for pre-market components."""
    
    def test_pre_market_signal_eligibility_flow(self):
        """Verify complete pre-market signal eligibility flow."""
        # Mock low data quality
        result = validate_pre_market_readiness(
            market_snapshot_quality=60.0,  # Below threshold
            option_chain_iv_count=5,  # Below minimum
        )
        assert result["ready"] is False
        
        # Now with good data
        result = validate_pre_market_readiness(
            market_snapshot_quality=90.0,
            option_chain_iv_count=40,
            global_market_staleness_minutes=20.0,
            has_dealer_positioning=True,
            has_volatility_data=True,
        )
        assert result["ready"] is True
        assert result["enable_signals"] == get_pre_market_policy_config().enable_pre_market_signals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
