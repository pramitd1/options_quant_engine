"""
Directional options flow based on front-expiry, near-ATM traded premium.
"""

import pandas as pd

from config.analytics_feature_policy import get_flow_imbalance_policy_config
from analytics.flow_utils import front_expiry_atm_slice


def calculate_flow_imbalance(option_chain, spot=None):
    cfg = get_flow_imbalance_policy_config()
    if option_chain is None or option_chain.empty:
        return cfg.neutral_default_imbalance

    df = front_expiry_atm_slice(option_chain, spot=spot, strike_window_steps=4)
    if df is None or df.empty:
        return cfg.neutral_default_imbalance

    calls = df[df["OPTION_TYP"] == "CE"].copy()
    puts = df[df["OPTION_TYP"] == "PE"].copy()

    call_notional = (
        pd.to_numeric(calls.get("totalTradedVolume"), errors="coerce").fillna(0.0) *
        pd.to_numeric(calls.get("lastPrice"), errors="coerce").fillna(0.0)
    ).sum()

    put_notional = (
        pd.to_numeric(puts.get("totalTradedVolume"), errors="coerce").fillna(0.0) *
        pd.to_numeric(puts.get("lastPrice"), errors="coerce").fillna(0.0)
    ).sum()

    if call_notional <= 0 and put_notional <= 0:
        return cfg.neutral_default_imbalance

    if put_notional <= 0:
        return cfg.no_put_flow_fallback_imbalance

    return float(call_notional / put_notional)


def flow_signal(option_chain, spot=None):
    cfg = get_flow_imbalance_policy_config()
    imbalance = calculate_flow_imbalance(option_chain, spot=spot)

    if imbalance >= cfg.bullish_threshold:
        return "BULLISH_FLOW"

    if imbalance <= cfg.bearish_threshold:
        return "BEARISH_FLOW"

    return "NEUTRAL_FLOW"
