"""
Module: pre_market_engine.py

Purpose:
    Orchestrate pre-market initialization including overnight dealer setup, morning volatility 
    initialization, and pre-open signal readiness validation.

Role in the System:
    Part of the signal engine that prepares market state and signal machinery for trading day open.

Key Outputs:
    Pre-market context payloads, readiness status, and bootstrapped dealer/volatility state.

Downstream Usage:
    Consumed by engine_runner (pre-market entry point), signal_engine (readiness checks), 
    and research workflows.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

import pandas as pd

from config.pre_market_policy import get_pre_market_policy_config
from engine.trading_support.pre_market_state import (
    is_pre_market_session,
    build_overnight_dealer_context,
    build_morning_volatility_context,
    validate_pre_market_readiness,
    apply_pre_market_signal_adjustments,
)
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


def initialize_pre_market_context(
    *,
    option_chain: Optional[pd.DataFrame] = None,
    spot: float = 0.0,
    global_market_snapshot: Optional[dict] = None,
    previous_session_state: Optional[dict] = None,
    as_of: Any = None,
) -> dict:
    """
    Purpose:
        Initialize full pre-market context for the trading day.
    
    Context:
        Called by engine_runner at market open (or during pre-market replay) to bootstrap
        dealer positioning, volatility regime, and signal readiness state.
    
    Inputs:
        option_chain (pd.DataFrame|None): Current option-chain snapshot.
        spot (float): Current underlying spot price.
        global_market_snapshot (dict|None): Global market data (macro, FX, commodities).
        previous_session_state (dict|None): State carry-over from previous trading day.
        as_of (Any): Timestamp context; defaults to now.
    
    Returns:
        dict: Comprehensive pre-market context with dealer, volatility, and readiness substates.
    
    Notes:
        This is the entry point for pre-market initialization. All downstream pre-market
        logic uses the returned context to make readiness and signal decisions.
    """
    cfg = get_pre_market_policy_config()
    ts = _coerce_timestamp(as_of) or pd.Timestamp.now(tz=IST_TIMEZONE)
    
    if cfg.log_pre_market_debug:
        _LOG.debug(
            "Initializing pre-market context at %s for spot=%.2f",
            ts.isoformat(),
            _safe_float(spot, 0.0),
        )
    
    # Extract dealer positioning from current and previous session
    dealer_context = _extract_dealer_context(
        option_chain=option_chain,
        previous_session_state=previous_session_state,
    )
    
    # Extract volatility state from current and global market
    volatility_context = _extract_volatility_context(
        option_chain=option_chain,
        global_market_snapshot=global_market_snapshot,
        previous_session_state=previous_session_state,
    )
    
    # Validate readiness for signal generation
    readiness = _validate_readiness(
        option_chain=option_chain,
        global_market_snapshot=global_market_snapshot,
        dealer_context=dealer_context,
        volatility_context=volatility_context,
    )
    
    return {
        "timestamp": ts.isoformat(),
        "session": "PRE_MARKET",
        "is_pre_market": is_pre_market_session(ts),
        "spot": _safe_float(spot, 0.0),
        "dealer_context": dealer_context,
        "volatility_context": volatility_context,
        "readiness": readiness,
        "market_snapshot_available": global_market_snapshot is not None,
        "option_chain_available": option_chain is not None and not option_chain.empty,
    }


def _extract_dealer_context(
    *,
    option_chain: Optional[pd.DataFrame] = None,
    previous_session_state: Optional[dict] = None,
) -> dict:
    """
    Purpose:
        Extract overnight dealer positioning from current and previous session data.
    
    Returns:
        dict: Dealer context with position, confidence, and basis.
    """
    cfg = get_pre_market_policy_config()
    
    # Current session dealer position (if option chain available)
    current_pos = None
    current_basis = None
    call_oi = 0.0
    put_oi = 0.0
    
    if option_chain is not None and not option_chain.empty:
        try:
            calls = option_chain[option_chain["OPTION_TYP"] == "CE"]
            puts = option_chain[option_chain["OPTION_TYP"] == "PE"]
            
            call_oi = _safe_float(
                calls.get("openInterest", pd.Series(dtype=float)).sum(),
                0.0,
            )
            put_oi = _safe_float(
                puts.get("openInterest", pd.Series(dtype=float)).sum(),
                0.0,
            )
            
            # Check for OI change data (more recent position estimate)
            if "changeinOI" in option_chain.columns:
                call_oi_change = _safe_float(
                    calls.get("changeinOI", pd.Series(dtype=float)).sum(),
                    0.0,
                )
                put_oi_change = _safe_float(
                    puts.get("changeinOI", pd.Series(dtype=float)).sum(),
                    0.0,
                )
                total_change = abs(call_oi_change) + abs(put_oi_change)
                
                if total_change > 0:
                    # OI_CHANGE basis is preferred (more recent flow)
                    net_change_bias = put_oi_change - call_oi_change
                    current_pos = "Short Gamma" if net_change_bias >= 0 else "Long Gamma"
                    current_basis = "OI_CHANGE"
            
            # Fallback to absolute OI basis
            if current_pos is None:
                current_pos = "Long Gamma" if call_oi > put_oi else "Short Gamma"
                current_basis = "OPEN_INTEREST"
        
        except Exception as exc:
            _LOG.warning(
                "Failed to extract dealer position from option chain: %s",
                exc,
            )
    
    # Extract previous session dealer position
    previous_pos = None
    if previous_session_state is not None:
        try:
            prev_dealer = previous_session_state.get("dealer_context", {})
            previous_pos = prev_dealer.get("position")
        except Exception as exc:
            _LOG.debug("Could not extract previous session dealer position: %s", exc)
    
    return build_overnight_dealer_context(
        current_dealer_position=current_pos,
        current_dealer_basis=current_basis,
        previous_session_position=previous_pos,
        call_oi=call_oi,
        put_oi=put_oi,
    )


def _extract_volatility_context(
    *,
    option_chain: Optional[pd.DataFrame] = None,
    global_market_snapshot: Optional[dict] = None,
    previous_session_state: Optional[dict] = None,
) -> dict:
    """
    Purpose:
        Extract morning volatility state from current and historical data.
    
    Returns:
        dict: Volatility context with regime, anchors, and initialization flags.
    """
    cfg = get_pre_market_policy_config()
    
    # Current session IV regime and median
    current_iv_regime = None
    implied_vol_median = 0.0
    
    if option_chain is not None and not option_chain.empty:
        try:
            # Compute median IV
            iv_col = "IV" if "IV" in option_chain.columns else "impliedVolatility"
            iv_series = pd.to_numeric(option_chain.get(iv_col), errors="coerce")
            iv_series = iv_series.replace([float('inf'), float('-inf')], float('nan')).dropna()
            
            if not iv_series.empty:
                implied_vol_median = float(iv_series.median())
                
                # Simple regime classification based on IV level
                # In production, this would use the full volatility_regime module
                iv_25 = iv_series.quantile(0.25)
                iv_75 = iv_series.quantile(0.75)
                
                if implied_vol_median > iv_75:
                    current_iv_regime = "VOL_EXPANSION"
                elif implied_vol_median < iv_25:
                    current_iv_regime = "VOL_COMPRESSION"
                else:
                    current_iv_regime = "VOL_NEUTRAL"
        
        except Exception as exc:
            _LOG.warning("Failed to extract IV regime from option chain: %s", exc)
    
    # Extract realized vol from global market snapshot
    realized_vol_5d = 0.0
    realized_vol_30d = 0.0
    
    if global_market_snapshot is not None:
        try:
            market_inputs = global_market_snapshot.get("market_inputs", {})
            realized_vol_5d = _safe_float(market_inputs.get("realized_vol_5d"), 0.0)
            realized_vol_30d = _safe_float(market_inputs.get("realized_vol_30d"), 0.0)
        except Exception as exc:
            _LOG.debug("Failed to extract realized vol from global market: %s", exc)
    
    # Extract previous session IV regime
    previous_iv_regime = None
    if previous_session_state is not None:
        try:
            prev_vol = previous_session_state.get("volatility_context", {})
            previous_iv_regime = prev_vol.get("regime")
        except Exception as exc:
            _LOG.debug("Could not extract previous session IV regime: %s", exc)
    
    # IV percentile: compare current IV to realized vol distribution
    iv_percentile = 0.5  # Default neutral
    if realized_vol_5d > 0:
        # Simple heuristic: IV near 30d realized vol is "normal"
        if implied_vol_median > realized_vol_30d * 1.2:
            iv_percentile = 0.7  # High IV
        elif implied_vol_median < realized_vol_30d * 0.8:
            iv_percentile = 0.3  # Low IV
    
    return build_morning_volatility_context(
        current_iv_regime=current_iv_regime,
        previous_iv_regime=previous_iv_regime,
        implied_vol_median=implied_vol_median,
        realized_vol_5d=realized_vol_5d,
        realized_vol_30d=realized_vol_30d,
        iv_percentile=iv_percentile,
    )


def _validate_readiness(
    *,
    option_chain: Optional[pd.DataFrame] = None,
    global_market_snapshot: Optional[dict] = None,
    dealer_context: dict,
    volatility_context: dict,
) -> dict:
    """
    Purpose:
        Validate pre-market data readiness for signal generation.
    
    Returns:
        dict: Readiness status with detailed checks.
    """
    cfg = get_pre_market_policy_config()
    
    # Count IV observations in option chain
    option_chain_iv_count = 0
    if option_chain is not None and not option_chain.empty:
        try:
            iv_col = "IV" if "IV" in option_chain.columns else "impliedVolatility"
            iv_series = pd.to_numeric(option_chain.get(iv_col), errors="coerce")
            option_chain_iv_count = int(iv_series.notna().sum())
        except Exception:
            pass
    
    # Check global market staleness
    global_market_staleness_minutes = 999999.0
    if global_market_snapshot is not None:
        try:
            as_of_ts = pd.to_datetime(
                global_market_snapshot.get("as_of"),
                utc=True,
            ).tz_convert(IST_TIMEZONE)
            now_ts = pd.Timestamp.now(tz=IST_TIMEZONE)
            global_market_staleness_minutes = (now_ts - as_of_ts).total_seconds() / 60.0
        except Exception:
            pass
    
    # Estimate overall message quality score (0-100)
    # Based on data availability and recency
    quality_score = 50.0  # Base score
    
    if option_chain is not None and not option_chain.empty:
        quality_score += 20.0
    if global_market_snapshot is not None and global_market_snapshot.get("data_available"):
        quality_score += 15.0
    if dealer_context.get("confidence", 0.0) > 0.5:
        quality_score += 10.0
    if volatility_context.get("confidence", 0.0) > 0.5:
        quality_score += 10.0
    
    # Cap at 100
    quality_score = min(quality_score, 100.0)
    
    return validate_pre_market_readiness(
        market_snapshot_quality=quality_score,
        option_chain_iv_count=option_chain_iv_count,
        global_market_staleness_minutes=global_market_staleness_minutes,
        has_dealer_positioning=dealer_context.get("confidence", 0.0) > 0.0,
        has_volatility_data=volatility_context.get("confidence", 0.0) > 0.0,
    )


def apply_pre_market_adjustments_to_signal(
    *,
    base_trade_strength: float = 0.0,
    pre_market_context: Optional[dict] = None,
) -> dict:
    """
    Purpose:
        Apply pre-market-specific adjustments to signal strength and eligibility.
    
    Context:
        Called by signal_engine when generating pre-market signals.
    
    Inputs:
        base_trade_strength (float): Unadjusted signal strength (0-100).
        pre_market_context (dict|None): Pre-market context from initialize_pre_market_context.
    
    Returns:
        dict: Adjusted signal with pre-market multipliers and eligibility checks.
    
    Notes:
        Pre-market signals require higher quality thresholds and may use lower trade-strength
        minimums to enable warming-up during the pre-open window.
    """
    if pre_market_context is None:
        pre_market_context = {}
    
    readiness = pre_market_context.get("readiness", {})
    quality_score = readiness.get("checks", {}).get("quality_score", {}).get("value", 0.0)
    
    return apply_pre_market_signal_adjustments(
        base_trade_strength=base_trade_strength,
        data_quality_score=_safe_float(quality_score, 0.0),
        is_pre_market=pre_market_context.get("is_pre_market", False),
    )


def build_pre_market_diagnostic_report(
    *,
    pre_market_context: dict,
) -> str:
    """
    Purpose:
        Generate a human-readable diagnostic report of pre-market state.
    
    Context:
        Used for logging, debugging, and research review.
    
    Inputs:
        pre_market_context (dict): Full pre-market context from initialize_pre_market_context.
    
    Returns:
        str: Multi-line diagnostic report.
    
    Notes:
        Useful for pre-market signal validation and troubleshooting.
    """
    cfg = get_pre_market_policy_config()
    
    dealer = pre_market_context.get("dealer_context", {})
    vol = pre_market_context.get("volatility_context", {})
    readiness = pre_market_context.get("readiness", {})
    
    lines = [
        "=== PRE-MARKET DIAGNOSTIC REPORT ===",
        f"Timestamp: {pre_market_context.get('timestamp')}",
        f"Spot: {pre_market_context.get('spot'):.2f}",
        "",
        "--- Dealer Setup ---",
        f"Position: {dealer.get('position')} (basis={dealer.get('basis')})",
        f"Confidence: {dealer.get('confidence', 0.0):.2f}",
        f"Session Origin: {dealer.get('session_origin')}",
        f"Call OI: {dealer.get('call_oi', 0.0):.2f}, Put OI: {dealer.get('put_oi', 0.0):.2f}",
        "",
        "--- Volatility Setup ---",
        f"Regime: {vol.get('regime')} (confidence={vol.get('confidence', 0.0):.2f})",
        f"IV Median: {vol.get('implied_vol_median', 0.0):.4f}",
        f"Realized Vol (5d/30d): {vol.get('realized_vol_5d', 0.0):.4f} / {vol.get('realized_vol_30d', 0.0):.4f}",
        f"IV Level: {vol.get('iv_level')} (percentile={vol.get('iv_percentile', 0.5):.2f})",
        "",
        "--- Readiness ---",
        f"Overall Ready: {readiness.get('ready')}",
        f"Enable Signals: {readiness.get('enable_signals')}",
    ]
    
    checks = readiness.get("checks", {})
    for check_name, check_data in checks.items():
        req = "required" if check_data.get("required", True) else "optional"
        passfail = "✓" if check_data.get("pass", True) else "✗"
        value = check_data.get("value", "N/A")
        lines.append(f"  [{passfail}] {check_name} ({req}): {value}")
    
    return "\n".join(lines)
