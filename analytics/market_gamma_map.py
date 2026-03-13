import pandas as pd


def calculate_market_gamma(option_chain):
    """
    Calculate strike-wise signed gamma exposure proxy.
    """
    if option_chain is None or len(option_chain) == 0:
        return pd.Series(dtype=float)

    df = option_chain.copy()
    strike_col = "STRIKE_PR" if "STRIKE_PR" in df.columns else "strikePrice"
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else "openInterest"

    gamma = pd.to_numeric(df.get("GAMMA"), errors="coerce").fillna(0.0)
    oi = pd.to_numeric(df.get(oi_col), errors="coerce").fillna(0.0)
    strikes = pd.to_numeric(df.get(strike_col), errors="coerce").fillna(0.0)
    option_type = df.get("OPTION_TYP", pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    signed = option_type.map({"CE": 1.0, "PE": -1.0}).fillna(0.0)

    df["GAMMA_EXPOSURE"] = gamma * oi * strikes * signed

    return df.groupby(strike_col)["GAMMA_EXPOSURE"].sum()


def market_gamma_regime(gex):
    """
    Determine overall gamma regime
    """

    if gex is None or len(gex) == 0:
        return "UNKNOWN"

    total_gex = gex.sum()
    gross_gex = gex.abs().sum()

    if gross_gex == 0 or abs(total_gex) <= gross_gex * 0.05:
        return "NEUTRAL_GAMMA"

    if total_gex > 0:
        return "POSITIVE_GAMMA"

    return "NEGATIVE_GAMMA"


def largest_gamma_strikes(gex, top_n=5):
    """
    Find strikes with largest gamma concentration
    """

    walls = gex.abs().sort_values(
        ascending=False
    ).head(top_n)

    return list(walls.index)
