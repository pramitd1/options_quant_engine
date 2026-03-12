import pandas as pd
import numpy as np


def build_synthetic_chain(spot, strikes=20, step=50):
    """
    Generates synthetic option chain for backtesting
    when historical chain not available.
    """

    strike_list = [
        spot + (i - strikes//2) * step
        for i in range(strikes)
    ]

    rows = []

    for strike in strike_list:

        for typ in ["CE", "PE"]:

            rows.append({

                "STRIKE_PR": strike,
                "OPTION_TYP": typ,
                "OPEN_INT": np.random.randint(1000,10000),
                "IV": np.random.uniform(10,30),
                "DELTA": np.random.uniform(-1,1),
                "GAMMA": np.random.uniform(0.001,0.01),
                "UNDERLYING_VALUE": spot

            })

    return pd.DataFrame(rows)