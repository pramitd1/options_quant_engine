"""
Dealer Inventory Model

This module estimates whether dealers are likely
LONG GAMMA or SHORT GAMMA based on open interest.

Institutional interpretation:

Dealers LONG gamma:
    • market tends to mean revert
    • volatility suppressed

Dealers SHORT gamma:
    • market tends to trend
    • volatility expands
"""

import pandas as pd


def dealer_inventory_position(option_chain: pd.DataFrame):
    """
    Estimate dealer positioning using call vs put open interest.

    Parameters
    ----------
    option_chain : DataFrame
        Option chain containing columns:
        strikePrice
        OPTION_TYP
        openInterest

    Returns
    -------
    str
        "Long Gamma" or "Short Gamma"
    """

    if option_chain.empty:
        return "Unknown"

    calls = option_chain[
        option_chain["OPTION_TYP"] == "CE"
    ]["openInterest"].sum()

    puts = option_chain[
        option_chain["OPTION_TYP"] == "PE"
    ]["openInterest"].sum()

    if calls > puts:
        return "Long Gamma"
    else:
        return "Short Gamma"