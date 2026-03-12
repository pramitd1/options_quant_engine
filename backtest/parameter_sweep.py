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
    Note:
    tp/sl are currently recorded in results for ranking/logging.
    They are not yet wired into live exit override logic.
    """
    return list(product(
        SWEEP_SIGNAL_PERSISTENCE_GRID,
        SWEEP_MAX_HOLD_BARS_GRID,
        SWEEP_TP_GRID,
        SWEEP_SL_GRID
    ))


def summarize_sweep_results(results):
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