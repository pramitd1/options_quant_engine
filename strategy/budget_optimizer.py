"""
Budget Optimizer

Optimizes number of lots under a capital constraint.
"""

import math


def optimize_lots(
    entry_price,
    lot_size,
    max_capital,
    requested_lots=1
):
    """
    Optimize lots based on budget.

    Parameters
    ----------
    entry_price : float
        Premium of one option.
    lot_size : int
        Contract lot size.
    max_capital : float
        Maximum budget allowed.
    requested_lots : int
        User-requested lots.

    Returns
    -------
    dict
    """

    capital_per_lot = entry_price * lot_size

    if capital_per_lot <= 0:
        return {
            "lot_size": lot_size,
            "requested_lots": requested_lots,
            "optimized_lots": 0,
            "capital_per_lot": 0,
            "capital_required": 0,
            "max_affordable_lots": 0,
            "budget_ok": False
        }

    max_affordable_lots = math.floor(max_capital / capital_per_lot)
    optimized_lots = min(requested_lots, max_affordable_lots)
    capital_required = optimized_lots * capital_per_lot
    budget_ok = optimized_lots >= 1

    return {
        "lot_size": int(lot_size),
        "requested_lots": int(requested_lots),
        "optimized_lots": int(optimized_lots),
        "capital_per_lot": round(capital_per_lot, 2),
        "capital_required": round(capital_required, 2),
        "max_affordable_lots": int(max_affordable_lots),
        "budget_ok": bool(budget_ok)
    }