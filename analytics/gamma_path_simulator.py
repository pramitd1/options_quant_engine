import numpy as np
import pandas as pd


def simulate_gamma_path(option_chain, spot):

    """
    Simulate gamma exposure if price moves.
    """

    price_grid = np.arange(
        spot - 300,
        spot + 300,
        50
    )

    gamma_profile = {}

    for price in price_grid:

        option_chain["GEX"] = (
            option_chain["GAMMA"]
            * option_chain["OPEN_INT"]
            * price
        )

        total_gamma = option_chain["GEX"].sum()

        gamma_profile[price] = total_gamma

    return gamma_profile


def gamma_acceleration_zone(gamma_profile):
    """
    Detect zone where gamma flips sharply.
    """

    prices = list(gamma_profile.keys())

    values = list(gamma_profile.values())

    gradient = np.gradient(values)

    max_change_idx = np.argmax(
        abs(gradient)
    )

    return prices[max_change_idx]