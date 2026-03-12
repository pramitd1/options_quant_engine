"""
Dealer Gamma Path Simulator

Simulates how dealer hedging flows change
as price moves across strikes.

This helps detect large potential moves.
"""

import numpy as np


def simulate_gamma_path(option_chain, spot, step=25, range_points=500):

    strikes = option_chain["strikePrice"].unique()

    prices = np.arange(
        spot - range_points,
        spot + range_points,
        step
    )

    gamma_curve = []

    for price in prices:

        gamma_total = 0

        for _, row in option_chain.iterrows():

            strike = row["strikePrice"]
            oi = row["openInterest"]

            distance = abs(strike - price)

            gamma = 1 / (1 + distance)

            if row["OPTION_TYP"] == "PE":
                gamma *= -1

            gamma_total += gamma * oi

        gamma_curve.append(gamma_total)

    return prices, gamma_curve


def detect_gamma_squeeze(prices, gamma_curve):

    if len(gamma_curve) < 2:
        return None

    slope = np.gradient(gamma_curve)

    max_slope = max(abs(slope))

    if max_slope > 10000:
        return "GAMMA_SQUEEZE"

    return "NORMAL"