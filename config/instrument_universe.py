"""
Instrument discovery utilities
"""

import pandas as pd


def load_zerodha_instruments():

    url = "https://api.kite.trade/instruments"

    df = pd.read_csv(url)

    return df


def get_derivative_symbols(df):

    derivatives = df[
        df["segment"].isin(["NFO-OPT", "NFO-FUT"])
    ]

    return derivatives["name"].unique()