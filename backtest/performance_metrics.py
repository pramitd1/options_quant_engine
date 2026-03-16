"""
Module: performance_metrics.py

Purpose:
    Implement performance metrics logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""
import math


def _equity_curve(pnls):
    """
    Purpose:
        Process equity curve for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        pnls (Any): Input associated with pnls.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    equity = []
    running = 0.0
    for p in pnls:
        running += p
        equity.append(running)
    return equity


def _max_drawdown(equity):
    """
    Purpose:
        Process max drawdown for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        equity (Any): Input associated with equity.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    peak = float("-inf")
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return max_dd


def compute_performance_metrics(trade_log, starting_capital=500000):
    """
    Purpose:
        Compute performance metrics from the supplied inputs.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade_log (Any): Input associated with trade log.
        starting_capital (Any): Input associated with starting capital.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not trade_log:
        return {
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "median_pnl": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "total_wins": 0,
            "total_losses": 0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "return_on_capital": 0.0,
            "sharpe_ratio": 0.0,
            "expectancy": 0.0
        }

    pnls = [float(t.get("pnl", 0.0)) for t in trade_log]
    total_trades = len(pnls)

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    total_pnl = sum(pnls)
    average_pnl = total_pnl / total_trades
    median_pnl = sorted(pnls)[total_trades // 2]
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = total_wins / total_trades if total_trades else 0.0
    loss_rate = total_losses / total_trades if total_trades else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    equity = _equity_curve(pnls)
    max_drawdown = _max_drawdown(equity)

    if total_trades > 1:
        variance = sum((x - average_pnl) ** 2 for x in pnls) / (total_trades - 1)
        std_pnl = math.sqrt(variance)
        sharpe_ratio = average_pnl / std_pnl if std_pnl > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + loss_rate * avg_loss
    roc = total_pnl / starting_capital if starting_capital > 0 else 0.0

    return {
        "total_pnl": round(total_pnl, 2),
        "average_pnl": round(average_pnl, 2),
        "median_pnl": round(median_pnl, 2),
        "win_rate": round(win_rate, 4),
        "loss_rate": round(loss_rate, 4),
        "total_wins": total_wins,
        "total_losses": total_losses,
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "max_drawdown": round(max_drawdown, 2),
        "return_on_capital": round(roc, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "expectancy": round(expectancy, 2)
    }


def performance_metrics(trade_log, starting_capital=500000):
    """
    Purpose:
        Process performance metrics for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade_log (Any): Input associated with trade log.
        starting_capital (Any): Input associated with starting capital.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return compute_performance_metrics(trade_log, starting_capital)