"""
Module: parameter_sweep.py

Purpose:
    Implement parameter sweep logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""
from itertools import product

from config.settings import (
    SWEEP_SIGNAL_PERSISTENCE_GRID,
    SWEEP_MAX_HOLD_BARS_GRID,
    SWEEP_TP_GRID,
    SWEEP_SL_GRID
)


def build_parameter_grid():
    """
    Build parameter combinations for sweep.
    """
    return list(product(
        SWEEP_SIGNAL_PERSISTENCE_GRID,
        SWEEP_MAX_HOLD_BARS_GRID,
        SWEEP_TP_GRID,
        SWEEP_SL_GRID
    ))


def summarize_sweep_results(results):
    """
    Purpose:
        Summarize sweep results into a compact diagnostic payload.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        results (Any): Input associated with results.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not results:
        return []

    return sorted(
        results,
        key=lambda x: (
            x.get("total_pnl", 0),
            x.get("sharpe_ratio", 0),
            x.get("profit_factor", 0)
        ),
        reverse=True
    )
