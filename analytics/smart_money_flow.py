import pandas as pd


def detect_unusual_volume(option_chain):
    """
    Detect unusual options activity based on volume spikes.
    """

    option_chain["VOL_OI_RATIO"] = (
        option_chain["VOLUME"] /
        (option_chain["OPEN_INT"] + 1)
    )

    spikes = option_chain[
        option_chain["VOL_OI_RATIO"] > 1.5
    ]

    return spikes


def classify_flow(spikes):
    """
    Classify institutional flow direction.
    """

    if len(spikes) == 0:
        return "NO_FLOW"

    call_flow = spikes[
        spikes["OPTION_TYP"] == "CE"
    ]

    put_flow = spikes[
        spikes["OPTION_TYP"] == "PE"
    ]

    if len(call_flow) > len(put_flow):
        return "BULLISH_FLOW"

    if len(put_flow) > len(call_flow):
        return "BEARISH_FLOW"

    return "MIXED_FLOW"


def smart_money_signal(option_chain):
    """
    Generate signal from options flow.
    """

    spikes = detect_unusual_volume(option_chain)

    flow = classify_flow(spikes)

    return flow