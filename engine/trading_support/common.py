"""
Module: common.py

Purpose:
    Provide common helpers used during market-state, probability, or signal assembly.

Role in the System:
    Part of the signal engine that turns analytics, probability estimates, and overlays into final trade decisions.

Key Outputs:
    Trade decisions, intermediate state bundles, and signal diagnostics.

Downstream Usage:
    Consumed by the live runtime loop, backtests, shadow mode, and signal-evaluation logging.
"""
from __future__ import annotations

import logging

import pandas as pd

from analytics.greeks_engine import enrich_chain_with_greeks

# ── Shared numeric helpers ──────────────────────────────────────────────
# Canonical implementations live in utils/numerics.py.  The private names
# are kept here as thin re-exports so every existing import site works
# without changes.
from utils.numerics import clip as _clip  # noqa: F401
from utils.numerics import safe_float as _safe_float  # noqa: F401
from utils.numerics import safe_div as _safe_div  # noqa: F401


_LOG = logging.getLogger(__name__)


def normalize_option_chain(option_chain, spot=None, valuation_time=None):
    """
    Purpose:
        Normalize an option-chain snapshot into the canonical schema expected by
        analytics, strategy, and risk modules.

    Context:
        Data providers expose different column names and may omit Greeks. This
        helper is the schema boundary that makes the rest of the trading system
        provider-agnostic.

    Inputs:
        option_chain (Any): Option-chain snapshot in dataframe form.
        spot (Any): Current underlying spot price.
        valuation_time (Any): Timestamp used when valuing contracts and Greeks.

    Returns:
        pd.DataFrame: Normalized option-chain dataframe with canonical columns
        and usable Greeks.

    Notes:
        When Greeks are missing or incomplete, the function enriches the chain
        so downstream analytics can assume a consistent feature surface.
    """
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

    # Downstream analytics rely on a minimally complete Greeks surface. If the
    # provider did not supply one, enrich the snapshot before continuing.
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
    """
    Purpose:
        Process call first for downstream use.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        module (Any): Input associated with module.
        candidate_names (Any): Input associated with candidate names.
        args (Any): Input associated with args.
        default (Any): Input associated with default.
        kwargs (Any): Input associated with kwargs.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                _LOG.debug(
                    "Candidate analytics call %s.%s rejected by signature/type error: %s",
                    getattr(module, "__name__", module.__class__.__name__),
                    name,
                    exc,
                    exc_info=True,
                )
                continue
            except Exception as exc:
                _LOG.debug(
                    "Candidate analytics call %s.%s failed; trying fallback: %s",
                    getattr(module, "__name__", module.__class__.__name__),
                    name,
                    exc,
                    exc_info=True,
                )
                continue
    return default


from utils.numerics import to_python_number as _to_python_number  # noqa: F401,E501


def _clean_zone_list(zones):
    """
    Purpose:
        Process clean zone list for downstream use.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        zones (Any): Input associated with zones.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
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
    """
    Purpose:
        Normalize validation dict into the repository-standard form.
    
    Context:
        Internal helper within the signal-engine layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        validation (Any): Input associated with validation.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    return validation if isinstance(validation, dict) else {}
