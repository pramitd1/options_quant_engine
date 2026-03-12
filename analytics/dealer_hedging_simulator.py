import numpy as np


def simulate_dealer_hedging(option_chain, price_move=50):
    """
    Estimate hedging flows if price moves up or down.

    Dealers hedge delta exposure by buying/selling futures.
    """

    option_chain["DELTA_EXPOSURE"] = (
        option_chain["DELTA"]
        * option_chain["OPEN_INT"]
    )

    total_delta = option_chain["DELTA_EXPOSURE"].sum()

    hedge_up = total_delta * price_move

    hedge_down = total_delta * -price_move

    return {

        "hedge_if_up": hedge_up,
        "hedge_if_down": hedge_down
    }


def hedging_bias(simulation):
    """
    Determine if hedging will amplify move.
    """

    if simulation["hedge_if_up"] > abs(simulation["hedge_if_down"]):

        return "UPSIDE_ACCELERATION"

    else:

        return "DOWNSIDE_ACCELERATION"