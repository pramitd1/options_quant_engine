"""
Module: exit_model.py

Purpose:
    Implement exit model logic used during trade construction.

Role in the System:
    Part of the strategy layer that converts directional intent into executable option trades.

Key Outputs:
    Strike rankings, trade-construction inputs, and exit or sizing recommendations.

Downstream Usage:
    Consumed by the signal engine and by research tooling that inspects trade construction choices.
"""
from config.settings import TARGET_PROFIT_PERCENT, STOP_LOSS_PERCENT


def calculate_exit(
    entry_price,
    target_profit_percent=TARGET_PROFIT_PERCENT,
    stop_loss_percent=STOP_LOSS_PERCENT,
):

    """
    Purpose:
        Compute the initial target and stop-loss prices for a selected option
        contract.

    Context:
        Called after strike selection so the engine can package a complete trade
        plan with entry, target, and risk limits.

    Inputs:
        entry_price (Any): Option premium used as the trade entry.
        target_profit_percent (Any): Percentage profit target applied to the
        entry premium.
        stop_loss_percent (Any): Percentage stop-loss applied to the entry
        premium.

    Returns:
        tuple[float, float]: Target price and stop-loss price for the trade.

    Notes:
        The function applies percentage-based exits only; it does not model path
        dependence or dynamic trailing behavior.
    """
    target = entry_price * (1 + target_profit_percent / 100)

    stop = entry_price * (1 - stop_loss_percent / 100)

    return target, stop
