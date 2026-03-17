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

from .common import _call_first, _clean_zone_list, _safe_float, _to_python_number
from .signal_state import classify_spot_vs_flip_for_symbol, normalize_flow_signal


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
    return market_gex


def _collect_market_state(df, spot, symbol=None, prev_df=None):
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

    Returns:
        dict: Consolidated market-state payload used by probability models,
        trade-strength scoring, overlays, and final reporting.

    Notes:
        Each `_call_first` invocation preserves compatibility with legacy module
        names so analytics can evolve without breaking the engine contract.
    """
    # The first block captures the market's structural gamma regime: net gamma,
    # the flip level, and dealer positioning.
    gamma = _call_first(
        gamma_exposure_mod,
        ["calculate_gamma_exposure", "calculate_gex"],
        df,
        spot,
        default=0,
    )
    flip = _call_first(
        gamma_flip_mod,
        ["gamma_flip_level", "find_gamma_flip"],
        df,
        spot=spot,
        default=None,
    )

    dealer_metrics = _call_first(
        dealer_inventory_mod,
        ["dealer_inventory_metrics"],
        df,
        default={},
    ) or {}
    dealer_pos = dealer_metrics.get("position") or _call_first(
        dealer_inventory_mod,
        ["dealer_inventory_position", "dealer_inventory"],
        df,
        default="Unknown",
    )

    vol_regime = _call_first(
        volatility_regime_mod,
        ["detect_volatility_regime", "volatility_regime"],
        df,
        default="UNKNOWN",
    )

    gamma_path_result = _call_first(
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

    gamma_event = _call_first(
        dealer_gamma_path_mod,
        ["detect_gamma_squeeze"],
        prices,
        gamma_curve,
        default="NORMAL",
    )

    # Flow combines transaction imbalance and "smart money" activity so the
    # engine can separate broad flow from more selective institutional flow.
    flow_signal_value = _call_first(
        options_flow_imbalance_mod,
        ["flow_signal", "calculate_flow_signal"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )
    smart_money_signal_value = _call_first(
        smart_money_flow_mod,
        ["smart_money_signal", "classify_flow"],
        df,
        spot=spot,
        default="NEUTRAL_FLOW",
    )
    final_flow_signal = normalize_flow_signal(flow_signal_value, smart_money_signal_value)

    # Liquidity maps are reused later by the strike selector and by trade
    # strength scoring to understand where dealer pinning or air pockets exist.
    liquidity_levels = _call_first(
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

    voids = _call_first(
        liquidity_void_mod,
        ["detect_liquidity_voids", "detect_liquidity_void"],
        df,
        default=[],
    )
    void_signal = _call_first(
        liquidity_void_mod,
        ["liquidity_void_signal"],
        spot,
        voids,
        default=None,
    )

    vacuum_zones = _call_first(
        liquidity_vacuum_mod,
        ["detect_liquidity_vacuum"],
        df,
        default=[],
    )
    vacuum_zones = _clean_zone_list(vacuum_zones)
    vacuum_state = _call_first(
        liquidity_vacuum_mod,
        ["vacuum_direction"],
        spot,
        vacuum_zones,
        default="NORMAL",
    )

    walls = _call_first(gamma_walls_mod, ["classify_walls"], df, default={}) or {}
    support_wall = _to_python_number(walls.get("support_wall") if isinstance(walls, dict) else None)
    resistance_wall = _to_python_number(walls.get("resistance_wall") if isinstance(walls, dict) else None)

    market_gex = _call_first(
        market_gamma_map_mod,
        ["calculate_market_gamma"],
        df,
        default=None,
    )
    market_gamma_summary = _summarize_market_gamma(market_gex)
    gamma_regime = _call_first(
        market_gamma_map_mod,
        ["market_gamma_regime"],
        market_gex,
        default=None,
    )
    gamma_clusters = _call_first(
        market_gamma_map_mod,
        ["largest_gamma_strikes"],
        market_gex,
        spot=spot,
        default=[],
    )
    gamma_clusters = [_to_python_number(x) for x in gamma_clusters] if gamma_clusters else []

    # Greek exposures provide second-order context beyond plain gamma so the
    # direction engine can reason about vanna/charm support and decay dynamics.
    greek_exposures = summarize_greek_exposures(df)
    if gamma_regime is None:
        if flip is None:
            gamma_regime = "UNKNOWN"
        elif spot > flip:
            gamma_regime = "LONG_GAMMA_ZONE"
        else:
            gamma_regime = "SHORT_GAMMA_ZONE"

    hedging_flow = _call_first(
        dealer_hedging_flow_mod,
        ["dealer_hedging_flow"],
        df,
        default=None,
    )
    hedging_sim = _call_first(
        dealer_hedging_simulator_mod,
        ["simulate_dealer_hedging"],
        df,
        default={},
    )
    hedging_bias = _call_first(
        dealer_hedging_simulator_mod,
        ["hedging_bias"],
        hedging_sim,
        default=None,
    )

    intraday_gamma_state = None
    if prev_df is not None:
        intraday_gamma_state = _call_first(
            intraday_gamma_shift_mod,
            ["gamma_shift_signal", "detect_gamma_shift"],
            prev_df,
            df,
            spot,
            default=None,
        )

    atm_iv = _call_first(
        volatility_surface_mod,
        ["atm_vol"],
        df,
        spot,
        default=None,
    )
    surface_regime = None
    if atm_iv is not None:
        surface_regime = _call_first(
            volatility_surface_mod,
            ["vol_regime"],
            atm_iv,
            default=None,
        )

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
    }
