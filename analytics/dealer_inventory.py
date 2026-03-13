"""
Dealer Inventory Model

Prefers OI-change-based positioning when available, with static OI as fallback.
"""

import pandas as pd


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def dealer_inventory_metrics(option_chain: pd.DataFrame):
    if option_chain is None or option_chain.empty:
        return {
            "position": "Unknown",
            "basis": "NO_DATA",
            "call_oi": 0.0,
            "put_oi": 0.0,
            "call_oi_change": 0.0,
            "put_oi_change": 0.0,
            "net_oi_change_bias": 0.0,
        }

    df = option_chain.copy()

    calls = df[df["OPTION_TYP"] == "CE"].copy()
    puts = df[df["OPTION_TYP"] == "PE"].copy()

    call_oi = _safe_float(calls.get("openInterest", pd.Series(dtype=float)).sum(), 0.0)
    put_oi = _safe_float(puts.get("openInterest", pd.Series(dtype=float)).sum(), 0.0)

    if "changeinOI" in df.columns:
        call_oi_change = _safe_float(calls.get("changeinOI", pd.Series(dtype=float)).sum(), 0.0)
        put_oi_change = _safe_float(puts.get("changeinOI", pd.Series(dtype=float)).sum(), 0.0)
    else:
        call_oi_change = 0.0
        put_oi_change = 0.0

    total_abs_change = abs(call_oi_change) + abs(put_oi_change)

    if total_abs_change > 0:
        # More put OI build than call OI build is treated as a stronger downside
        # inventory skew; more call OI build is treated as a stronger upside skew.
        net_oi_change_bias = put_oi_change - call_oi_change
        position = "Long Gamma" if net_oi_change_bias >= 0 else "Short Gamma"
        basis = "OI_CHANGE"
    else:
        net_oi_change_bias = put_oi - call_oi
        position = "Long Gamma" if call_oi > put_oi else "Short Gamma"
        basis = "OPEN_INTEREST"

    return {
        "position": position,
        "basis": basis,
        "call_oi": round(call_oi, 2),
        "put_oi": round(put_oi, 2),
        "call_oi_change": round(call_oi_change, 2),
        "put_oi_change": round(put_oi_change, 2),
        "net_oi_change_bias": round(net_oi_change_bias, 2),
    }


def dealer_inventory_position(option_chain: pd.DataFrame):
    return dealer_inventory_metrics(option_chain)["position"]
