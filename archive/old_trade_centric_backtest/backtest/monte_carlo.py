"""
Module: monte_carlo.py

Purpose:
    Implement monte carlo logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""
import random


def monte_carlo_reshuffle(trade_log, simulations=1000):
    """
    Shuffle realized trade PnLs to estimate robustness of path dependency.
    """
    if not trade_log:
        return {
            "simulations": simulations,
            "mean_total_pnl": 0.0,
            "best_total_pnl": 0.0,
            "worst_total_pnl": 0.0
        }

    pnls = [float(t.get("pnl", 0.0)) for t in trade_log]
    totals = []

    for _ in range(simulations):
        shuffled = pnls[:]
        random.shuffle(shuffled)
        totals.append(sum(shuffled))

    return {
        "simulations": simulations,
        "mean_total_pnl": round(sum(totals) / len(totals), 2),
        "best_total_pnl": round(max(totals), 2),
        "worst_total_pnl": round(min(totals), 2)
    }