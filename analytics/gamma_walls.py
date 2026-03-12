import pandas as pd


def detect_gamma_walls(option_chain, top_n=3):
    """
    Detect strikes with highest open interest.

    These act as gamma walls (strong support/resistance).
    """

    oi_by_strike = option_chain.groupby(
        "STRIKE_PR"
    )["OPEN_INT"].sum()

    walls = oi_by_strike.sort_values(
        ascending=False
    ).head(top_n)

    return list(walls.index)


def classify_walls(option_chain):
    """
    Classify support and resistance walls.
    """

    call_oi = option_chain[
        option_chain["OPTION_TYP"] == "CE"
    ].groupby("STRIKE_PR")["OPEN_INT"].sum()

    put_oi = option_chain[
        option_chain["OPTION_TYP"] == "PE"
    ].groupby("STRIKE_PR")["OPEN_INT"].sum()

    resistance = call_oi.idxmax()

    support = put_oi.idxmax()

    return {

        "support_wall": support,
        "resistance_wall": resistance
    }