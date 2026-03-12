import numpy as np


def detect_liquidity_vacuum(option_chain):
    """
    Detect gaps in open interest across strikes.

    These zones create fast price movement.
    """

    oi = option_chain.groupby(
        "STRIKE_PR"
    )["OPEN_INT"].sum()

    strikes = oi.index.values

    vacuum_zones = []

    for i in range(len(strikes) - 1):

        current_oi = oi.iloc[i]
        next_oi = oi.iloc[i + 1]

        if next_oi < current_oi * 0.25:

            vacuum_zones.append(
                (strikes[i], strikes[i + 1])
            )

    return vacuum_zones


def vacuum_direction(spot, vacuum_zones):
    """
    Determine if spot is entering vacuum zone.
    """

    for low, high in vacuum_zones:

        if low < spot < high:

            return "BREAKOUT_ZONE"

    return "NORMAL"