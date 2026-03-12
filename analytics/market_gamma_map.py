import pandas as pd


def calculate_market_gamma(option_chain):
    """
    Calculate market-wide gamma exposure
    """

    option_chain["GAMMA_EXPOSURE"] = (
        option_chain["GAMMA"]
        * option_chain["OPEN_INT"]
        * option_chain["STRIKE_PR"]
    )

    gex = option_chain.groupby(
        "STRIKE_PR"
    )["GAMMA_EXPOSURE"].sum()

    return gex


def market_gamma_regime(gex):
    """
    Determine overall gamma regime
    """

    total_gex = gex.sum()

    if total_gex > 0:
        return "POSITIVE_GAMMA"

    else:
        return "NEGATIVE_GAMMA"


def largest_gamma_strikes(gex, top_n=5):
    """
    Find strikes with largest gamma concentration
    """

    walls = gex.abs().sort_values(
        ascending=False
    ).head(top_n)

    return list(walls.index)