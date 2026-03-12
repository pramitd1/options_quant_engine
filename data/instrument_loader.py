import pandas as pd


def load_instruments():

    """
    Download Zerodha instrument master
    """

    url = "https://api.kite.trade/instruments"

    df = pd.read_csv(url)

    return df