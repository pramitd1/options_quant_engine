"""
Module: dealer_hedging_flow.py

Purpose:
    Compute dealer hedging flow analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import numpy as np

from config.analytics_feature_policy import get_dealer_flow_policy_config


def dealer_hedging_flow(
    option_chain,
    *,
    gamma_weight: float | None = None,
    charm_weight: float | None = None,
):

    """
    Estimate hedging pressure from dealer positions.
    """

    if option_chain is None or len(option_chain) == 0:
        return "SELL_FUTURES"

    policy = get_dealer_flow_policy_config()
    gamma_weight = float(policy.gamma_weight if gamma_weight is None else gamma_weight)
    charm_weight = float(policy.charm_weight if charm_weight is None else charm_weight)

    df = option_chain.copy()
    delta = np.nan_to_num(df.get("DELTA", 0.0), nan=0.0).astype(float)
    gamma = np.nan_to_num(df.get("GAMMA", 0.0), nan=0.0).astype(float)
    charm = np.nan_to_num(df.get("CHARM", 0.0), nan=0.0).astype(float)
    open_int = np.nan_to_num(df.get("OPEN_INT", df.get("openInterest", 0.0)), nan=0.0).astype(float)

    flow = float(
        (delta * open_int).sum()
        + float(gamma_weight) * (gamma * open_int).sum()
        + float(charm_weight) * (charm * open_int).sum()
    )

    if flow > 0:
        return "BUY_FUTURES"

    else:
        return "SELL_FUTURES"
