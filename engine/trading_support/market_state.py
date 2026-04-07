"""
Module: market_state.py

Purpose:
    Provide market state helpers used during market-state, probability, or signal assembly.

Role in the System:
    Part of the signal engine that turns analytics, probability estimates, and overlays into final trade decisions.

Key Outputs:
    Trade decisions, intermediate state bundles, and signal diagnostics.

Downstream Usage:
    Consumed by the live runtime loop, backtests, shadow mode, and signal-evaluation logging.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from analytics import dealer_gamma_path as dealer_gamma_path_mod
from analytics import dealer_hedging_flow as dealer_hedging_flow_mod
from analytics import dealer_hedging_simulator as dealer_hedging_simulator_mod
from analytics import dealer_inventory as dealer_inventory_mod
from analytics import gamma_exposure as gamma_exposure_mod
from analytics import gamma_flip as gamma_flip_mod
from analytics import gamma_walls as gamma_walls_mod
from analytics import intraday_gamma_shift as intraday_gamma_shift_mod
from analytics import liquidity_heatmap as liquidity_heatmap_mod
from analytics import liquidity_vacuum as liquidity_vacuum_mod
from analytics import liquidity_void as liquidity_void_mod
from analytics import market_gamma_map as market_gamma_map_mod
from analytics import options_flow_imbalance as options_flow_imbalance_mod
from analytics import smart_money_flow as smart_money_flow_mod
from analytics import volatility_regime as volatility_regime_mod
from analytics import volatility_surface as volatility_surface_mod
from analytics.dealer_liquidity_map import build_dealer_liquidity_map
from analytics.greeks_engine import summarize_greek_exposures

from analytics import max_pain as max_pain_mod
from analytics import oi_velocity as oi_velocity_mod
from analytics import volume_pcr as volume_pcr_mod
from .common import _call_first, _clean_zone_list, _safe_float, _to_python_number
from .signal_state import classify_spot_vs_flip_for_symbol, normalize_flow_signal


_LOG = logging.getLogger(__name__)


def _top_timing_steps(step_timings: dict[str, float], limit: int = 5) -> list[dict[str, float | str]]:
    ranked = sorted(step_timings.items(), key=lambda item: item[1], reverse=True)
    return [
        {"step": name, "elapsed_ms": elapsed_ms}
        for name, elapsed_ms in ranked[:limit]
    ]


def _summarize_market_gamma(market_gex):
    """
    Purpose:
        Summarize market gamma into a compact diagnostic payload.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        market_gex (Any): Input associated with market gex.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if isinstance(market_gex, dict):
        summary = {}
        for key in ("total_gamma", "call_gamma", "put_gamma", "net_gamma"):
            value = market_gex.get(key)
            if value is not None:
                summary[key] = round(_safe_float(value), 2)
        return summary or market_gex
    if isinstance(market_gex, pd.Series):
        numeric = pd.to_numeric(market_gex, errors="coerce").dropna()
        if numeric.empty:
            return {"strike_count": int(len(market_gex))}

        peak_abs_idx = numeric.abs().idxmax()
        summary = {
            "strike_count": int(len(numeric)),
            "net_gamma": round(float(numeric.sum()), 2),
            "peak_abs_strike": _to_python_number(peak_abs_idx),
            "peak_abs_gamma": round(_safe_float(numeric.loc[peak_abs_idx]), 2),
        }

        positive = numeric[numeric > 0]
        negative = numeric[numeric < 0]
        if not positive.empty:
            pos_idx = positive.idxmax()
            summary["peak_positive_strike"] = _to_python_number(pos_idx)
            summary["peak_positive_gamma"] = round(_safe_float(positive.loc[pos_idx]), 2)
        if not negative.empty:
            neg_idx = negative.idxmin()
            summary["peak_negative_strike"] = _to_python_number(neg_idx)
            summary["peak_negative_gamma"] = round(_safe_float(negative.loc[neg_idx]), 2)
        return summary
    if isinstance(market_gex, pd.DataFrame):
        numeric_cols = [
            col for col in market_gex.columns
            if pd.api.types.is_numeric_dtype(market_gex[col])
        ]
        if "GAMMA_EXPOSURE" in market_gex.columns:
            series = market_gex["GAMMA_EXPOSURE"]
            index = market_gex.get("strikePrice")
            if index is not None:
                series = pd.Series(series.to_numpy(), index=index)
            return _summarize_market_gamma(series)
        if numeric_cols:
            return {
                "row_count": int(len(market_gex)),
                "numeric_columns": numeric_cols[:5],
            }
    return market_gex


def _collect_market_state(
    df,
    spot,
    symbol=None,
    prev_df=None,
    days_to_expiry=None,
    india_vix_level=None,
    fallback_iv=None,
):
    """
    Purpose:
        Collect the cross-analytic market state for the current option-chain
        snapshot.

    Context:
        This helper centralizes the analytics fan-out step. It keeps the signal
        engine from knowing the details of each analytics module while still
        returning a rich, explainable state payload.

    Inputs:
        df (Any): Normalized option-chain dataframe supplied to the routine.
        spot (Any): Current underlying spot price.
        symbol (Any): Trading symbol or index identifier.
        prev_df (Any): Previous normalized dataframe used for comparison.
        days_to_expiry (Any): Calendar days remaining to the selected expiry.

    Returns:
        dict: Consolidated market-state payload used by probability models,
        trade-strength scoring, overlays, and final reporting.

    Notes:
        Each `_call_first` invocation preserves compatibility with legacy module
        names so analytics can evolve without breaking the engine contract.
    """
    fanout_started_at = time.perf_counter()
    step_timings: dict[str, float] = {}

    def _timed_step(name, callback, *args, **kwargs):
        started_at = time.perf_counter()
        value = callback(*args, **kwargs)
        step_timings[name] = round((time.perf_counter() - started_at) * 1000.0, 3)
        return value

    # The first block captures the market's structural gamma regime: net gamma,
    # the flip level, and dealer positioning.
    gamma = _timed_step(
        "gamma_exposure",
        _call_first,
        gamma_exposure_mod,
        ["calculate_gamma_exposure", "calculate_gex"],
        df,
        spot,
        default=0,
    )
    flip = _timed_step(
        "gamma_flip",
        _call_first,
        gamma_flip_mod,
        ["gamma_flip_level", "find_gamma_flip"],
        df,
        spot=spot,
        default=None,
    )

    dealer_metrics = _timed_step(
        "dealer_inventory_metrics",
        _call_first,
        dealer_inventory_mod,
        ["dealer_inventory_metrics"],
        df,
        default={},
    ) or {}
    dealer_pos = dealer_metrics.get("position")
    if not dealer_pos:
        dealer_pos = _timed_step(
            "dealer_inventory_position",
            _call_first,
            dealer_inventory_mod,
            ["dealer_inventory_position", "dealer_inventory"],
            df,
            default="Unknown",
        )

    vol_regime = _timed_step(
        "volatility_regime",
        _call_first,
        volatility_regime_mod,
        ["detect_volatility_regime", "volatility_regime"],
        df,
        india_vix_level=india_vix_level,
        fallback_iv=fallback_iv,
        default="UNKNOWN",
    )

    gamma_path_result = _timed_step(
        "dealer_gamma_path",
        _call_first,
        dealer_gamma_path_mod,
        ["simulate_gamma_path"],
        df,
        spot,
        default=([], []),
    )
    if isinstance(gamma_path_result, tuple) and len(gamma_path_result) == 2:
        prices, gamma_curve = gamma_path_result
    else:
        prices, gamma_curve = [], []

    gamma_event = _timed_step(
        "gamma_event",
        _call_first,
        dealer_gamma_path_mod,
        ["detect_gamma_squeeze"],
        prices,
        gamma_curve,
        default="NORMAL",
    )

    # Flow combines transaction imbalance and "smart money" activity so the
    # engine can separate broad flow from more selective institutional flow.
    flow_signal_value = _timed_step(
        "options_flow_signal",
        _call_first,
        options_flow_imbalance_mod,
        ["flow_signal", "calculate_flow_signal"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )
    smart_money_signal_value = _timed_step(
        "smart_money_flow",
        _call_first,
        smart_money_flow_mod,
        ["smart_money_signal", "classify_flow"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )
    final_flow_signal = normalize_flow_signal(flow_signal_value, smart_money_signal_value)

    # Liquidity maps are reused later by the strike selector and by trade
    # strength scoring to understand where dealer pinning or air pockets exist.
    liquidity_levels = _timed_step(
        "liquidity_heatmap",
        _call_first,
        liquidity_heatmap_mod,
        ["strongest_liquidity_levels", "build_liquidity_heatmap"],
        df,
        default=[],
    )
    if isinstance(liquidity_levels, pd.Series):
        liquidity_levels = list(liquidity_levels.index[:5])
    elif isinstance(liquidity_levels, pd.Index):
        liquidity_levels = list(liquidity_levels[:5])
    elif liquidity_levels is None:
        liquidity_levels = []
    liquidity_levels = [_to_python_number(x) for x in liquidity_levels]

    voids = _timed_step(
        "liquidity_voids",
        _call_first,
        liquidity_void_mod,
        ["detect_liquidity_voids", "detect_liquidity_void"],
        df,
        default=[],
    )
    void_signal = _timed_step(
        "liquidity_void_signal",
        _call_first,
        liquidity_void_mod,
        ["liquidity_void_signal"],
        spot,
        voids,
        default=None,
    )

    vacuum_zones = _timed_step(
        "liquidity_vacuum",
        _call_first,
        liquidity_vacuum_mod,
        ["detect_liquidity_vacuum"],
        df,
        default=[],
    )
    vacuum_zones = _clean_zone_list(vacuum_zones)
    vacuum_state = _timed_step(
        "liquidity_vacuum_direction",
        _call_first,
        liquidity_vacuum_mod,
        ["vacuum_direction"],
        spot,
        vacuum_zones,
        default="NORMAL",
    )

    walls = _timed_step(
        "gamma_walls",
        _call_first,
        gamma_walls_mod,
        ["classify_walls"],
        df,
        default={},
    ) or {}
    support_wall = _to_python_number(walls.get("support_wall") if isinstance(walls, dict) else None)
    resistance_wall = _to_python_number(walls.get("resistance_wall") if isinstance(walls, dict) else None)

    market_gex = _timed_step(
        "market_gamma_map",
        _call_first,
        market_gamma_map_mod,
        ["calculate_market_gamma"],
        df,
        default=None,
    )
    market_gamma_summary = _summarize_market_gamma(market_gex)
    gamma_regime = _timed_step(
        "market_gamma_regime",
        _call_first,
        market_gamma_map_mod,
        ["market_gamma_regime"],
        market_gex,
        default=None,
    )
    gamma_clusters = _timed_step(
        "gamma_clusters",
        _call_first,
        market_gamma_map_mod,
        ["largest_gamma_strikes"],
        market_gex,
        spot=spot,
        default=[],
    )
    gamma_clusters = [_to_python_number(x) for x in gamma_clusters] if gamma_clusters else []

    # Greek exposures provide second-order context beyond plain gamma so the
    # direction engine can reason about vanna/charm support and decay dynamics.
    greek_exposures = _timed_step("greek_exposures", summarize_greek_exposures, df)
    if gamma_regime is None:
        if flip is None:
            gamma_regime = "UNKNOWN"
        elif spot > flip:
            gamma_regime = "LONG_GAMMA_ZONE"
        else:
            gamma_regime = "SHORT_GAMMA_ZONE"

    hedging_flow = _timed_step(
        "dealer_hedging_flow",
        _call_first,
        dealer_hedging_flow_mod,
        ["dealer_hedging_flow"],
        df,
        default=None,
    )
    hedging_sim = _timed_step(
        "dealer_hedging_simulation",
        _call_first,
        dealer_hedging_simulator_mod,
        ["simulate_dealer_hedging"],
        df,
        default={},
    )
    hedging_bias = _timed_step(
        "dealer_hedging_bias",
        _call_first,
        dealer_hedging_simulator_mod,
        ["hedging_bias"],
        hedging_sim,
        default=None,
    )

    intraday_gamma_state = None
    if prev_df is not None:
        intraday_gamma_state = _timed_step(
            "intraday_gamma_shift",
            _call_first,
            intraday_gamma_shift_mod,
            ["gamma_shift_signal", "detect_gamma_shift"],
            prev_df,
            df,
            spot,
            default=None,
        )

    atm_iv = _timed_step(
        "atm_vol",
        _call_first,
        volatility_surface_mod,
        ["atm_vol"],
        df,
        spot,
        default=None,
    )
    surface_regime = None
    if atm_iv is not None:
        surface_regime = _timed_step(
            "vol_surface_regime",
            _call_first,
            volatility_surface_mod,
            ["vol_regime"],
            atm_iv,
            default=None,
        )

    # ATM straddle price and market-implied expected move in index points.
    straddle_data = _timed_step(
        "atm_straddle",
        _call_first,
        volatility_surface_mod,
        ["compute_atm_straddle_price"],
        df,
        spot,
        default={},
    ) or {}

    # Max pain — expiry-gravity strike computed from full option-chain holder payout.
    max_pain_data = _timed_step(
        "max_pain",
        _call_first,
        max_pain_mod,
        ["compute_max_pain"],
        df,
        spot,
        default={},
    ) or {}

    # Volume PCR — real-time put/call sentiment ratio (full chain + near-ATM).
    volume_pcr_data = _timed_step(
        "volume_pcr",
        _call_first,
        volume_pcr_mod,
        ["compute_volume_pcr"],
        df,
        spot,
        default={},
    ) or {}

    # Risk-reversal skew and momentum (front expiry), informative for direction.
    rr_data = _timed_step(
        "risk_reversal",
        _call_first,
        volatility_surface_mod,
        ["compute_risk_reversal"],
        df,
        spot,
        default={},
    ) or {}
    rr_momentum_data = {}
    if prev_df is not None:
        prev_rr_data = _timed_step(
            "risk_reversal_previous",
            _call_first,
            volatility_surface_mod,
            ["compute_risk_reversal"],
            prev_df,
            spot,
            default={},
        ) or {}
        rr_momentum_data = _timed_step(
            "risk_reversal_velocity",
            _call_first,
            volatility_surface_mod,
            ["risk_reversal_velocity"],
            rr_data.get("rr_value"),
            prev_rr_data.get("rr_value"),
            default={},
        ) or {}

    # OI velocity — fresh positioning speed from rolling OI changes.
    oi_velocity_data = {}
    oi_velocity_regime = "BALANCED"
    if prev_df is not None:
        oi_velocity_data = _timed_step(
            "oi_velocity",
            _call_first,
            oi_velocity_mod,
            ["compute_oi_velocity"],
            [prev_df, df],
            spot=spot,
            default={},
        ) or {}
        oi_velocity_regime = _timed_step(
            "oi_velocity_regime",
            _call_first,
            oi_velocity_mod,
            ["oi_velocity_regime"],
            oi_velocity_data.get("velocity_score", 0.0),
            default="BALANCED",
        )

    # Gamma flip drift — direction and magnitude the flip level moved vs prior snapshot.
    # Requires two snapshots; gracefully absent when prev_df is None.
    gamma_flip_drift: dict = {}
    if prev_df is not None:
        prev_flip = _timed_step(
            "gamma_flip_previous",
            _call_first,
            gamma_flip_mod,
            ["gamma_flip_level", "find_gamma_flip"],
            prev_df,
            # Use spot-independent selection for previous snapshot to avoid
            # injecting current-spot drift into historical flip estimation.
            spot=None,
            default=None,
        )
        if prev_flip is not None and flip is not None:
            drift = round(float(flip) - float(prev_flip), 2)
            if abs(drift) < 1e-6:
                drift_direction = "STABLE"
            elif drift > 0:
                drift_direction = "RISING"
            else:
                drift_direction = "FALLING"
            gamma_flip_drift = {
                "drift": drift,
                "drift_direction": drift_direction,
                "prev_flip": float(prev_flip),
            }

    spot_vs_flip = classify_spot_vs_flip_for_symbol(symbol, spot, flip)
    dealer_liquidity_map = build_dealer_liquidity_map(
        spot=spot,
        gamma_flip=flip,
        liquidity_levels=liquidity_levels,
        support_wall=support_wall,
        resistance_wall=resistance_wall,
        gamma_clusters=gamma_clusters,
        vacuum_zones=vacuum_zones,
    )

    total_elapsed_ms = round((time.perf_counter() - fanout_started_at) * 1000.0, 3)
    market_state_timings = {
        "total_ms": total_elapsed_ms,
        "slowest_steps": _top_timing_steps(step_timings),
        "step_ms": step_timings,
    }
    _LOG.debug(
        "market_state fan-out completed for %s in %.3f ms; slowest=%s",
        symbol or "UNKNOWN",
        total_elapsed_ms,
        market_state_timings["slowest_steps"],
    )

    # The returned state is intentionally denormalized: downstream modules favor
    # explicit fields over repeated recomputation during scoring and reporting.
    return {
        "gamma": gamma,
        "flip": flip,
        "dealer_metrics": dealer_metrics,
        "dealer_pos": dealer_pos,
        "vol_regime": vol_regime,
        "gamma_event": gamma_event,
        "flow_signal_value": flow_signal_value,
        "smart_money_signal_value": smart_money_signal_value,
        "final_flow_signal": final_flow_signal,
        "liquidity_levels": liquidity_levels,
        "voids": voids,
        "void_signal": void_signal,
        "vacuum_zones": vacuum_zones,
        "vacuum_state": vacuum_state,
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "market_gamma_summary": market_gamma_summary,
        "gamma_regime": gamma_regime,
        "gamma_clusters": gamma_clusters,
        "greek_exposures": greek_exposures,
        "hedging_flow": hedging_flow,
        "hedging_bias": hedging_bias,
        "intraday_gamma_state": intraday_gamma_state,
        "atm_iv": atm_iv,
        "surface_regime": surface_regime,
        "spot_vs_flip": spot_vs_flip,
        "dealer_liquidity_map": dealer_liquidity_map,
        # New enrichment signals
        "max_pain": max_pain_data.get("max_pain"),
        "max_pain_dist": max_pain_data.get("max_pain_dist"),
        "max_pain_zone": max_pain_data.get("max_pain_zone", "UNAVAILABLE"),
        "atm_straddle_price": straddle_data.get("atm_straddle_price"),
        "expected_move_up": straddle_data.get("expected_move_up"),
        "expected_move_down": straddle_data.get("expected_move_down"),
        "expected_move_pct": straddle_data.get("expected_move_pct"),
        "volume_pcr": volume_pcr_data.get("volume_pcr"),
        "volume_pcr_atm": volume_pcr_data.get("volume_pcr_atm"),
        "volume_pcr_regime": volume_pcr_data.get("volume_pcr_regime", "UNAVAILABLE"),
        "rr_value": rr_data.get("rr_value"),
        "rr_momentum": rr_momentum_data.get("rr_momentum", "UNAVAILABLE"),
        "rr_velocity": rr_momentum_data.get("rr_velocity"),
        "oi_velocity_score": oi_velocity_data.get("velocity_score"),
        "oi_velocity_regime": oi_velocity_regime,
        "gamma_flip_drift": gamma_flip_drift,
        "days_to_expiry": _safe_float(days_to_expiry, None),
        "market_state_timings": market_state_timings,
    }
