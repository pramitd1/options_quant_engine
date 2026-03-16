"""
Module: dealer_hedging_simulator.py

Purpose:
    Compute dealer hedging simulator analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import pandas as pd


def _to_numeric(series_or_value, default=0.0):
    """
    Purpose:
        Convert numeric into the representation expected downstream.
    
    Context:
        Internal helper within the analytics layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        series_or_value (Any): Input associated with series or value.
        default (Any): Input associated with default.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if isinstance(series_or_value, pd.Series):
        return pd.to_numeric(series_or_value, errors="coerce").fillna(default)
    return pd.to_numeric(pd.Series([series_or_value]), errors="coerce").fillna(default).iloc[0]


def _dealer_gamma_sign_proxy(df: pd.DataFrame) -> float:
    """
    Infer whether open-interest positioning behaves more like long or short gamma.
    """
    change_col = "CHG_IN_OI" if "CHG_IN_OI" in df.columns else "changeinOI" if "changeinOI" in df.columns else None
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else "openInterest"

    if change_col:
        call_change = _to_numeric(df.loc[df["OPTION_TYP"] == "CE", change_col]).sum()
        put_change = _to_numeric(df.loc[df["OPTION_TYP"] == "PE", change_col]).sum()
        total_change = abs(call_change) + abs(put_change)
        if total_change > 0:
            return 1.0 if (put_change - call_change) >= 0 else -1.0

    call_oi = _to_numeric(df.loc[df["OPTION_TYP"] == "CE", oi_col]).sum()
    put_oi = _to_numeric(df.loc[df["OPTION_TYP"] == "PE", oi_col]).sum()
    return 1.0 if put_oi >= call_oi else -1.0


def simulate_dealer_hedging(option_chain, price_move=50):
    """
    Estimate dealer hedge targets after an up/down move.
    """
    if option_chain is None or len(option_chain) == 0:
        return {
            "hedge_if_up": 0.0,
            "hedge_if_down": 0.0,
            "current_hedge": 0.0,
            "dealer_gamma_sign": 0.0,
            "gross_gamma_sensitivity": 0.0,
        }

    df = option_chain.copy()
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else "openInterest"

    delta = _to_numeric(df.get("DELTA"))
    gamma = _to_numeric(df.get("GAMMA"))
    oi = _to_numeric(df.get(oi_col))

    net_delta = float((delta * oi).sum())
    gross_gamma_sensitivity = float((gamma.abs() * oi).sum())
    dealer_gamma_sign = _dealer_gamma_sign_proxy(df)

    up_delta = net_delta + (dealer_gamma_sign * gross_gamma_sensitivity * float(price_move))
    down_delta = net_delta - (dealer_gamma_sign * gross_gamma_sensitivity * float(price_move))

    return {
        "hedge_if_up": -up_delta,
        "hedge_if_down": -down_delta,
        "current_hedge": -net_delta,
        "dealer_gamma_sign": dealer_gamma_sign,
        "gross_gamma_sensitivity": gross_gamma_sensitivity,
    }


def hedging_bias(simulation):
    """
    Classify whether dealer hedging is more likely to pin or amplify moves.
    """
    hedge_up = float(simulation.get("hedge_if_up", 0.0) or 0.0)
    hedge_down = float(simulation.get("hedge_if_down", 0.0) or 0.0)
    dealer_gamma_sign = float(simulation.get("dealer_gamma_sign", 0.0) or 0.0)

    up_mag = abs(hedge_up)
    down_mag = abs(hedge_down)
    tolerance = max(up_mag, down_mag, 1.0) * 0.02

    if abs(up_mag - down_mag) <= tolerance:
        return "PINNING"

    up_dominant = up_mag > down_mag

    if dealer_gamma_sign >= 0:
        return "UPSIDE_PINNING" if up_dominant else "DOWNSIDE_PINNING"

    return "UPSIDE_ACCELERATION" if up_dominant else "DOWNSIDE_ACCELERATION"
