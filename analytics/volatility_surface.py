import pandas as pd
import numpy as np


def build_vol_surface(option_chain):
    """
    Build implied volatility surface
    across strikes and expiries.
    """

    clean_chain = option_chain.copy()
    clean_chain["IV"] = pd.to_numeric(clean_chain["IV"], errors="coerce")
    clean_chain = clean_chain[clean_chain["IV"] > 0]

    surface = clean_chain.pivot_table(
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
    clean_chain = option_chain.copy()
    clean_chain["STRIKE_PR"] = pd.to_numeric(clean_chain["STRIKE_PR"], errors="coerce")
    clean_chain["IV"] = pd.to_numeric(clean_chain["IV"], errors="coerce")
    clean_chain = clean_chain.dropna(subset=["STRIKE_PR", "IV"])
    clean_chain = clean_chain[clean_chain["IV"] > 0]

    if clean_chain.empty:
        return None

    clean_chain["DIST"] = abs(clean_chain["STRIKE_PR"] - spot)
    atm_row = clean_chain.sort_values("DIST").iloc[0]
    return float(atm_row["IV"])


def vol_regime(atm_iv):
    """
    Determine volatility regime.
    """

    if atm_iv is None:
        return "UNKNOWN"

    if atm_iv > 25:
        return "HIGH_VOL"

    if atm_iv < 15:
        return "LOW_VOL"

    return "NORMAL_VOL"
