"""
Module: volatility_regime.py

Purpose:
    Compute volatility regime analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

import pandas as pd
import numpy as np

from config.analytics_feature_policy import get_volatility_regime_policy_config
from utils.regime_normalization import normalize_iv_decimal


def compute_atm_iv_level(option_chain):
    """
    Estimate the current implied-volatility level from a single option-chain
    snapshot.

    Implementation note
    -------------------
    This function works on a **single cross-sectional snapshot**, not on a
    time-series of underlying returns.  It therefore returns the **median
    implied volatility** across all strikes in the chain (normalized to
    decimal form), which is a robust, stale-IV-resistant proxy for the current
    vol level.

    This is intentionally *not* historical realized volatility (which would
    require ``std(log_returns) * sqrt(252)`` over a return series).  The name
    ``compute_realized_volatility`` that appeared in earlier versions of this
    module was misleading and has been corrected.

    Returns
    -------
    float | None
        Median IV in decimal units (e.g. 0.14 for 14 % annualized vol).
        Returns None when the chain is empty or all IV values are invalid.
    """
    if option_chain is None or option_chain.empty:
        return None

    # This module is invoked on single option-chain snapshots, so using
    # cross-strike premium pct-changes is not a valid time-series volatility
    # estimate. We instead infer the current volatility level from normalized
    # implied vol observations.
    iv_col = "IV" if "IV" in option_chain.columns else "impliedVolatility"
    iv_series = pd.to_numeric(option_chain.get(iv_col), errors="coerce")
    iv_series = iv_series.replace([np.inf, -np.inf], np.nan).dropna()
    if iv_series.empty:
        return None

    normalized = iv_series.apply(lambda value: normalize_iv_decimal(value, default=np.nan))
    normalized = pd.to_numeric(normalized, errors="coerce")
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    normalized = normalized[normalized > 0]
    if normalized.empty:
        return None

    # Median is robust to stale/deep-OTM tails and gives a stable per-snapshot
    # volatility level in decimal units.
    return float(normalized.median())


def detect_volatility_regime(option_chain, *, india_vix_level=None, fallback_iv=None):
    """
    Purpose:
        Detect the volatility regime from the available market signals.

    Context:
        Function inside the `volatility regime` module. The module sits in the analytics layer that turns option-chain and market-structure data into tradable features.

    Inputs:
        option_chain (Any): Option-chain snapshot in dataframe form.

    Returns:
        str | Any: Classification label returned by the current logic.

    Notes:
        Part of the module API used by downstream runtime, research, backtest, or governance workflows.
    """
    cfg = get_volatility_regime_policy_config()

    vol = compute_atm_iv_level(option_chain)
    if vol is None:
        vol = normalize_iv_decimal(india_vix_level, default=None)
    if vol is None:
        vol = normalize_iv_decimal(fallback_iv, default=None)
    if vol is None:
        return "UNKNOWN_VOL"

    if vol < cfg.low_vol_threshold:
        return "LOW_VOL"

    if vol < cfg.normal_vol_threshold:
        return "NORMAL_VOL"

    return "VOL_EXPANSION"
