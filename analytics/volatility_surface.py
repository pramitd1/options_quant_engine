import pandas as pd
import numpy as np


def build_vol_surface(option_chain):
    """
    Build implied volatility surface
    across strikes and expiries.
    """

    surface = option_chain.pivot_table(
        values="IV",
        index="STRIKE_PR",
        columns="EXPIRY_DT",
        aggfunc="mean"
    )

    return surface


def atm_vol(option_chain, spot):
    """
    Compute ATM implied volatility.
    """

    option_chain["DIST"] = abs(
        option_chain["STRIKE_PR"] - spot
    )

    atm_row = option_chain.sort_values(
        "DIST"
    ).iloc[0]

    return atm_row["IV"]


def vol_regime(atm_iv):
    """
    Determine volatility regime.
    """

    if atm_iv > 25:
        return "HIGH_VOL"

    if atm_iv < 15:
        return "LOW_VOL"

    return "NORMAL_VOL"