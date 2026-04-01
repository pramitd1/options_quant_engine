"""
Module: pre_market_state.py

Purpose:
    Provide pre-market state helpers and context management for signal readiness and overnight setup.

Role in the System:
    Part of the signal engine that manages pre-market state transitions and readiness validation.

Key Outputs:
    Pre-market state payloads, readiness checks, and dealer/volatility initialization contexts.

Downstream Usage:
    Consumed by engine_runner (pre-market signals), signal_engine (readiness checks), and research workflows.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
import pandas as pd

from config.pre_market_policy import get_pre_market_policy_config
from utils.numerics import safe_float as _safe_float

_LOG = logging.getLogger(__name__)
IST_TIMEZONE = "Asia/Kolkata"


def _coerce_timestamp(value):
    """Parse flexible timestamp inputs into timezone-aware timestamps."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return None
    try:
        return parsed.tz_convert(IST_TIMEZONE)
    except Exception:
        return None


def is_pre_market_session(as_of: Any = None) -> bool:
    """
    Purpose:
        Determine if the given timestamp falls within pre-market hours.
    
    Context:
        Used by engine_runner and signal_engine to route pre-market logic.
    
    Inputs:
        as_of (Any): Timestamp to check; defaults to now.
    
    Returns:
        bool: True if in pre-market window (08:00-09:15 IST).
    
    Notes:
        Pre-market runs 08:00 IST to 09:15 IST (45 minutes).
    """
    cfg = get_pre_market_policy_config()
    ts = _coerce_timestamp(as_of)
    if ts is None:
        ts = pd.Timestamp.now(tz=IST_TIMEZONE)
    
    pre_market_start = ts.replace(
        hour=cfg.pre_market_start_hour,
        minute=cfg.pre_market_start_minute,
        second=0,
        microsecond=0,
    )
    pre_market_end = ts.replace(
        hour=cfg.pre_market_end_hour,
        minute=cfg.pre_market_end_minute,
        second=0,
        microsecond=0,
    )
    
    return pre_market_start <= ts < pre_market_end


def build_overnight_dealer_context(
    *,
    current_dealer_position: Optional[str] = None,
    current_dealer_basis: Optional[str] = None,
    previous_session_position: Optional[str] = None,
    call_oi: float = 0.0,
    put_oi: float = 0.0,
) -> dict:
    """
    Purpose:
        Build overnight dealer context that bridges previous session to current pre-market.
    
    Context:
        Called during pre-market initialization to inherit dealer positioning from overnight.
    
    Inputs:
        current_dealer_position (str|None): Current session dealer position estimate.
        current_dealer_basis (str|None): How the position was computed (OI_CHANGE vs OPEN_INTEREST).
        previous_session_position (str|None): Previous session's final dealer position.
        call_oi (float): Current session call open interest.
        put_oi (float): Current session put open interest.
    
    Returns:
        dict: Overnight dealer context with position, confidence, and basis.
    
    Notes:
        Used for pre-market dealer setup and to bootstrap gamma regime estimates.
    """
    cfg = get_pre_market_policy_config()
    
    # Prefer basis on OI_CHANGE (more recent), fallback to OPEN_INTEREST
    preferred_basis = "OI_CHANGE" if current_dealer_basis == "OI_CHANGE" else "OPEN_INTEREST"
    
    # Use current position if available, else carryover previous
    dealer_pos = current_dealer_position
    confidence = 1.0  # Full confidence if we have current data
    
    if dealer_pos is None and cfg.use_previous_session_dealer_position:
        dealer_pos = previous_session_position
        confidence = 0.7  # Reduced confidence for overnight carryover
    
    if dealer_pos is None:
        dealer_pos = "NEUTRAL"
        confidence = 0.0  # No position estimate available
    
    return {
        "position": str(dealer_pos or "NEUTRAL"),
        "basis": preferred_basis,
        "confidence": confidence,  # Return unadjusted confidence based on data source
        "session_origin": "current" if current_dealer_position else "previous_session",
        "call_oi": _safe_float(call_oi, 0.0),
        "put_oi": _safe_float(put_oi, 0.0),
        "is_bootstrapped": current_dealer_position is None,
    }


def build_morning_volatility_context(
    *,
    current_iv_regime: Optional[str] = None,
    previous_iv_regime: Optional[str] = None,
    implied_vol_median: float = 0.0,
    realized_vol_5d: float = 0.0,
    realized_vol_30d: float = 0.0,
    iv_percentile: float = 0.5,
) -> dict:
    """
    Purpose:
        Build morning volatility context for IV surface initialization.
    
    Context:
        Called during pre-market to anchor volatility state for the trading day.
    
    Inputs:
        current_iv_regime (str|None): Current session IV regime estimate.
        previous_iv_regime (str|None): Previous session's closing IV regime.
        implied_vol_median (float): Median IV from current option chain.
        realized_vol_5d (float): 5-day realized volatility.
        realized_vol_30d (float): 30-day realized volatility.
        iv_percentile (float): IV percentile vs historical (0-1).
    
    Returns:
        dict: Morning volatility context with regime, anchors, and initialization flags.
    
    Notes:
        Morning vol context seeds the full intraday vol regime machine.
    """
    cfg = get_pre_market_policy_config()
    
    # Use current IV regime if computed, else carryover previous
    vol_regime = current_iv_regime
    confidence = 0.9 if current_iv_regime else 0.0
    
    if vol_regime is None and cfg.use_previous_iv_regime:
        vol_regime = previous_iv_regime or "VOL_NEUTRAL"
        confidence = 0.6  # Lower confidence for overnight carryover
    
    if vol_regime is None:
        vol_regime = "VOL_NEUTRAL"
        confidence = 0.0
    
    # IV percentile to classification
    vol_level = "HIGH" if iv_percentile >= 0.7 else ("LOW" if iv_percentile <= 0.3 else "NORMAL")
    
    return {
        "regime": str(vol_regime),
        "confidence": confidence,
        "session_origin": "current" if current_iv_regime else "previous_session",
        "implied_vol_median": _safe_float(implied_vol_median, 0.0),
        "realized_vol_5d": _safe_float(realized_vol_5d, 0.0),
        "realized_vol_30d": _safe_float(realized_vol_30d, 0.0),
        "iv_percentile": _safe_float(iv_percentile, 0.5),
        "iv_level": vol_level,
        "is_bootstrapped": current_iv_regime is None,
    }


def validate_pre_market_readiness(
    *,
    market_snapshot_quality: float = 0.0,
    option_chain_iv_count: int = 0,
    global_market_staleness_minutes: float = 999999.0,
    has_dealer_positioning: bool = False,
    has_volatility_data: bool = False,
) -> dict:
    """
    Purpose:
        Validate pre-market data quality and signal readiness.
    
    Context:
        Called before enabling pre-market signal generation to check data sufficiency.
    
    Inputs:
        market_snapshot_quality (float): Overall market snapshot quality score (0-100).
        option_chain_iv_count (int): Number of IV observations in option chain.
        global_market_staleness_minutes (float): Age of global market snapshot.
        has_dealer_positioning (bool): Whether dealer position data is available.
        has_volatility_data (bool): Whether volatility data is available.
    
    Returns:
        dict: Readiness status with overall pass/fail, individual checks, and diagnostics.
    
    Notes:
        Readiness checks are stricter than intraday thresholds to ensure robust pre-open signals.
    """
    cfg = get_pre_market_policy_config()
    
    checks = {
        "quality_score": {
            "value": _safe_float(market_snapshot_quality, 0.0),
            "threshold": cfg.min_data_quality_score,
            "pass": _safe_float(market_snapshot_quality, 0.0) >= cfg.min_data_quality_score,
        },
        "option_chain_iv_count": {
            "value": int(option_chain_iv_count or 0),
            "threshold": cfg.min_option_chain_iv_count,
            "pass": int(option_chain_iv_count or 0) >= cfg.min_option_chain_iv_count,
        },
        "global_market_staleness": {
            "value": _safe_float(global_market_staleness_minutes, 999999.0),
            "threshold": cfg.max_global_market_staleness_minutes,
            "pass": _safe_float(global_market_staleness_minutes, 999999.0) <= cfg.max_global_market_staleness_minutes,
        },
        "dealer_positioning_available": {
            "value": bool(has_dealer_positioning),
            "required": False,
            "pass": True,  # Optional but informative
        },
        "volatility_data_available": {
            "value": bool(has_volatility_data),
            "required": False,
            "pass": True,
        },
    }
    
    # Overall readiness: all required checks must pass
    all_pass = all(
        check.get("pass", True)
        for check in checks.values()
        if check.get("required", True)
    )
    
    return {
        "ready": all_pass,
        "enable_signals": all_pass and cfg.enable_pre_market_signals,
        "checks": checks,
        "timestamp": pd.Timestamp.now(tz=IST_TIMEZONE).isoformat(),
    }


def apply_pre_market_signal_adjustments(
    *,
    base_trade_strength: float = 0.0,
    data_quality_score: float = 0.0,
    is_pre_market: bool = False,
) -> dict:
    """
    Purpose:
        Apply pre-market-specific adjustments to base signal strength and thresholds.
    
    Context:
        Called by signal_engine during pre-market to apply heightened quality gates.
    
    Inputs:
        base_trade_strength (float): Unadjusted trade strength (0-100).
        data_quality_score (float): Overall data quality (0-100).
        is_pre_market (bool): Whether we are in pre-market session.
    
    Returns:
        dict: Adjusted trade strength, quality multiplier, and readiness status.
    
    Notes:
        Pre-market signals are only enabled if quality thresholds are met and confidence is high.
    """
    cfg = get_pre_market_policy_config()
    
    if not is_pre_market:
        return {
            "adjusted_trade_strength": _safe_float(base_trade_strength, 0.0),
            "quality_multiplier": 1.0,
            "signal_eligible": True,
        }
    
    if not cfg.enable_pre_market_signals:
        return {
            "adjusted_trade_strength": 0.0,
            "quality_multiplier": 0.0,
            "signal_eligible": False,
            "reason": "pre_market_signals_disabled_by_policy",
        }
    
    # Apply quality boost (stricter quality requirement)
    quality_multiplier = cfg.pre_market_signal_quality_boost
    quality_adjusted_strength = _safe_float(base_trade_strength, 0.0) * quality_multiplier
    
    # Pre-market uses lower trade strength threshold
    eligible = quality_adjusted_strength >= cfg.pre_market_min_trade_strength
    
    return {
        "adjusted_trade_strength": round(quality_adjusted_strength, 2),
        "quality_multiplier": quality_multiplier,
        "min_trade_strength_threshold": cfg.pre_market_min_trade_strength,
        "signal_eligible": eligible,
        "data_quality_score": _safe_float(data_quality_score, 0.0),
    }
