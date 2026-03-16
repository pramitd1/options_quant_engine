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


def compute_realized_volatility(option_chain):

    """
    Purpose:
        Compute realized volatility from the supplied inputs.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
    
    Returns:
        Any: Computed value returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    if option_chain.empty:
        return 0

    prices = pd.to_numeric(option_chain["lastPrice"], errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.where(prices > 0).dropna()

    if len(prices) < 2:
        return 0

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) == 0:
        return 0

    vol = np.std(returns)

    return vol


def detect_volatility_regime(option_chain):
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

    vol = compute_realized_volatility(option_chain)

    if vol < cfg.low_vol_threshold:
        return "LOW_VOL"

    if vol < cfg.normal_vol_threshold:
        return "NORMAL_VOL"

    return "VOL_EXPANSION"
