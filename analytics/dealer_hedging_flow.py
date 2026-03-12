import numpy as np


def dealer_hedging_flow(option_chain):

    """
    Estimate hedging pressure from dealer positions.
    """

    option_chain["delta_flow"] = (
        option_chain["DELTA"]
        * option_chain["OPEN_INT"]
    )

    flow = option_chain["delta_flow"].sum()

    if flow > 0:

        return "BUY_FUTURES"

    else:

        return "SELL_FUTURES"