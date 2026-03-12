"""
Options Flow Imbalance Detector

Detects abnormal call/put activity.

Used by professional traders to detect
smart money positioning.
"""

import pandas as pd


def calculate_flow_imbalance(option_chain):

    if option_chain.empty:
        return 0

    calls = option_chain[
        option_chain["OPTION_TYP"] == "CE"
    ]

    puts = option_chain[
        option_chain["OPTION_TYP"] == "PE"
    ]

    call_volume = calls["totalTradedVolume"].sum()
    put_volume = puts["totalTradedVolume"].sum()

    if put_volume == 0:
        return 0

    imbalance = call_volume / put_volume

    return imbalance


def flow_signal(option_chain):

    imbalance = calculate_flow_imbalance(option_chain)

    if imbalance > 1.3:
        return "BULLISH_FLOW"

    if imbalance < 0.7:
        return "BEARISH_FLOW"

    return "NEUTRAL_FLOW"