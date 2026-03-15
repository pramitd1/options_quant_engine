from __future__ import annotations

import pandas as pd

from analytics.greeks_engine import enrich_chain_with_greeks


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_div(a, b, default=0.0):
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


def normalize_option_chain(option_chain, spot=None, valuation_time=None):
    df = option_chain.copy()

    rename_map = {
        "strikePrice": "STRIKE_PR",
        "openInterest": "OPEN_INT",
        "impliedVolatility": "IV",
        "totalTradedVolume": "VOLUME",
        "lastPrice": "LAST_PRICE",
        "changeinOI": "CHG_IN_OI",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "STRIKE_PR" in df.columns and "strikePrice" not in df.columns:
        df["strikePrice"] = df["STRIKE_PR"]

    if "OPEN_INT" in df.columns and "openInterest" not in df.columns:
        df["openInterest"] = df["OPEN_INT"]

    if "IV" in df.columns and "impliedVolatility" not in df.columns:
        df["impliedVolatility"] = df["IV"]

    if "VOLUME" in df.columns and "totalTradedVolume" not in df.columns:
        df["totalTradedVolume"] = df["VOLUME"]

    if "LAST_PRICE" in df.columns and "lastPrice" not in df.columns:
        df["lastPrice"] = df["LAST_PRICE"]

    if "EXPIRY_DT" not in df.columns:
        df["EXPIRY_DT"] = None

    if spot is None:
        spot = df["strikePrice"].median() if "strikePrice" in df.columns else None

    greek_cols = ["DELTA", "GAMMA", "THETA", "VEGA", "RHO", "TTE"]
    has_usable_greeks = all(col in df.columns for col in greek_cols)
    if has_usable_greeks:
        gamma_valid = pd.to_numeric(df["GAMMA"], errors="coerce").notna().any()
        delta_valid = pd.to_numeric(df["DELTA"], errors="coerce").notna().any()
        tte_valid = pd.to_numeric(df["TTE"], errors="coerce").notna().any()
        has_usable_greeks = gamma_valid and delta_valid and tte_valid

    if not has_usable_greeks:
        df = enrich_chain_with_greeks(df, spot=spot, valuation_time=valuation_time)

    return df


def _call_first(module, candidate_names, *args, default=None, **kwargs):
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue
    return default


def _to_python_number(x):
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass

    try:
        if isinstance(x, float) and x.is_integer():
            return int(x)
    except Exception:
        pass

    return x


def _clean_zone_list(zones):
    if not zones:
        return []

    cleaned = []
    for zone in zones:
        try:
            low, high = zone
            cleaned.append((_to_python_number(low), _to_python_number(high)))
        except Exception:
            continue

    return cleaned


def _normalize_validation_dict(validation):
    return validation if isinstance(validation, dict) else {}
