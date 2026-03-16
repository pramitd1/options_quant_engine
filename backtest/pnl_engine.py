"""
Module: pnl_engine.py

Purpose:
    Implement pnl engine logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""
from config.settings import (
    BACKTEST_ENTRY_SLIPPAGE_BPS,
    BACKTEST_EXIT_SLIPPAGE_BPS,
    BACKTEST_SPREAD_BPS,
    BACKTEST_COMMISSION_PER_ORDER,
)


def _bps_adjust(price: float, bps: float, side: str) -> float:
    """
    Purpose:
        Process bps adjust for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        price (float): Input associated with price.
        bps (float): Input associated with bps.
        side (str): Input associated with side.
    
    Returns:
        float: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    mult = bps / 10000.0
    if side == "BUY":
        return price * (1 + mult)
    if side == "SELL":
        return price * (1 - mult)
    return price


def _find_option_row(option_chain, strike, option_type):
    """
    Purpose:
        Process find option row for downstream use.
    
    Context:
        Internal helper within the backtest layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        strike (Any): Input associated with strike.
        option_type (Any): Input associated with option type.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    rows = option_chain[
        (option_chain["strikePrice"] == strike) &
        (option_chain["OPTION_TYP"] == option_type)
    ]
    return None if rows.empty else rows.iloc[0]


def calculate_trade_pnl(trade: dict, exit_snapshot):
    """
    Purpose:
        Calculate trade pnl from the supplied inputs.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade (dict): Input associated with trade.
        exit_snapshot (Any): Input associated with exit snapshot.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if not trade:
        return {
            "pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "exit_price": None,
            "exit_reason": "NO_TRADE",
            "charges": 0.0
        }

    strike = trade.get("strike")
    option_type = trade.get("option_type")
    selected_expiry = trade.get("selected_expiry")
    raw_entry_price = float(trade.get("entry_price", 0))
    target = float(trade.get("target", 0))
    stop_loss = float(trade.get("stop_loss", 0))
    lot_size = int(trade.get("lot_size", 1))
    number_of_lots = int(trade.get("optimized_lots", trade.get("number_of_lots", 1)))

    if (
        selected_expiry
        and exit_snapshot is not None
        and "EXPIRY_DT" in exit_snapshot.columns
    ):
        expiry_values = exit_snapshot["EXPIRY_DT"].astype(str).str.strip()
        filtered_snapshot = exit_snapshot.loc[expiry_values.eq(str(selected_expiry).strip())].copy()
        if not filtered_snapshot.empty:
            exit_snapshot = filtered_snapshot

    row = _find_option_row(exit_snapshot, strike, option_type)
    if row is None:
        return {
            "pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "exit_price": None,
            "exit_reason": "OPTION_NOT_FOUND",
            "charges": 0.0
        }

    raw_current_price = float(row.get("lastPrice", 0))

    entry_price = _bps_adjust(
        raw_entry_price,
        BACKTEST_ENTRY_SLIPPAGE_BPS + BACKTEST_SPREAD_BPS / 2,
        "BUY"
    )
    tradable_exit_price = _bps_adjust(
        raw_current_price,
        BACKTEST_EXIT_SLIPPAGE_BPS + BACKTEST_SPREAD_BPS / 2,
        "SELL"
    )

    if tradable_exit_price >= target:
        exit_price = target
        exit_reason = "TARGET_HIT"
    elif tradable_exit_price <= stop_loss:
        exit_price = stop_loss
        exit_reason = "STOP_LOSS_HIT"
    else:
        exit_price = tradable_exit_price
        exit_reason = "TIME_EXIT"

    gross_pnl = (exit_price - entry_price) * lot_size * number_of_lots
    charges = 2 * BACKTEST_COMMISSION_PER_ORDER
    net_pnl = gross_pnl - charges

    return {
        "pnl": round(net_pnl, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "charges": round(charges, 2)
    }


def pnl_engine(trade: dict, exit_snapshot):
    """
    Purpose:
        Process pnl engine for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        trade (dict): Input associated with trade.
        exit_snapshot (Any): Input associated with exit snapshot.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return calculate_trade_pnl(trade, exit_snapshot)
