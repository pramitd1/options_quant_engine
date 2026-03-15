"""
Volatility Regime Model

Identifies whether the market is in:

LOW VOL
NORMAL VOL
EXPANSION

Institutional desks use volatility regime
to determine strategy type.

Low Vol:
    option selling

Expansion:
    option buying
"""

import pandas as pd
import numpy as np

from config.analytics_feature_policy import get_volatility_regime_policy_config


def compute_realized_volatility(option_chain):

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
    cfg = get_volatility_regime_policy_config()

    vol = compute_realized_volatility(option_chain)

    if vol < cfg.low_vol_threshold:
        return "LOW_VOL"

    if vol < cfg.normal_vol_threshold:
        return "NORMAL_VOL"

    return "VOL_EXPANSION"
