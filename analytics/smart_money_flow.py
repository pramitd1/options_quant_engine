"""
Module: smart_money_flow.py

Purpose:
    Compute smart money flow analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
import math

import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from config.analytics_feature_policy import get_smart_money_flow_policy_config


def detect_unusual_volume(option_chain, spot=None):
    """
    Detect unusual near-ATM front-expiry activity using both volume/OI and OI change.

    The opening-activity contribution is normalized by existing open interest and
    bounded to avoid raw OI-change counts overpowering all other evidence.
    """
    cfg = get_smart_money_flow_policy_config()

    df = front_expiry_atm_slice(option_chain, spot=spot, strike_window_steps=4)
    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()

    working["VOLUME"] = pd.to_numeric(
        working.get("VOLUME", working.get("totalTradedVolume")),
        errors="coerce",
    ).fillna(0.0)
    working["OPEN_INT"] = pd.to_numeric(
        working.get("OPEN_INT", working.get("openInterest")),
        errors="coerce",
    ).fillna(0.0)
    working["CHG_IN_OI"] = pd.to_numeric(
        working.get("CHG_IN_OI", working.get("changeinOI")),
        errors="coerce",
    ).fillna(0.0)
    working["LAST_PRICE"] = pd.to_numeric(
        working.get("LAST_PRICE", working.get("lastPrice")),
        errors="coerce",
    ).fillna(0.0)

    working["VOL_OI_RATIO"] = working["VOLUME"] / (working["OPEN_INT"] + 1.0)
    working["OPENING_ACTIVITY"] = working["CHG_IN_OI"].clip(lower=0.0)
    working["OPENING_ACTIVITY_RATIO"] = working["OPENING_ACTIVITY"] / (working["OPEN_INT"] + 1.0)

    ratio_cap = max(float(getattr(cfg, "opening_activity_ratio_cap", 3.0) or 3.0), 1e-6)
    weight_cap = max(float(getattr(cfg, "opening_activity_weight_cap", 4.0) or 4.0), 1.0)
    scaled_ratio = working["OPENING_ACTIVITY_RATIO"].clip(lower=0.0, upper=ratio_cap)
    working["OPENING_ACTIVITY_WEIGHT"] = 1.0 + (
        scaled_ratio.apply(math.log1p) / math.log1p(ratio_cap)
    ) * (weight_cap - 1.0)

    # Delta-weight the flow notional so large-notional OTM activity doesn't
    # dominate when it carries little directional exposure.
    if "DELTA" in working.columns:
        delta_abs = pd.to_numeric(working["DELTA"], errors="coerce").abs().fillna(0.5)
    else:
        delta_abs = pd.Series(0.5, index=working.index)
    working["FLOW_NOTIONAL"] = working["VOLUME"] * working["LAST_PRICE"] * delta_abs

    spikes = working[
        (working["VOL_OI_RATIO"] >= cfg.unusual_volume_ratio_threshold) |
        (working["OPENING_ACTIVITY_RATIO"] > cfg.opening_activity_threshold)
    ].copy()

    return spikes


def classify_flow(spikes):
    """
    Classify directional flow using notional activity, not just contract counts.

    Opening activity is incorporated through a bounded weight so unusual OI
    changes support the classification without numerically dominating it.
    """
    cfg = get_smart_money_flow_policy_config()

    if spikes is None or len(spikes) == 0:
        return "NO_FLOW"

    call_flow = spikes[spikes["OPTION_TYP"] == "CE"].copy()
    put_flow = spikes[spikes["OPTION_TYP"] == "PE"].copy()

    def _weights(frame: pd.DataFrame) -> pd.Series:
        if "OPENING_ACTIVITY_WEIGHT" in frame.columns:
            return pd.to_numeric(
                frame["OPENING_ACTIVITY_WEIGHT"],
                errors="coerce",
            ).fillna(1.0).clip(lower=1.0)

        raw_opening = pd.to_numeric(
            frame.get("OPENING_ACTIVITY"),
            errors="coerce",
        ).fillna(0.0)
        open_int = pd.to_numeric(
            frame.get("OPEN_INT", frame.get("openInterest")),
            errors="coerce",
        ).fillna(0.0)
        ratio = raw_opening.clip(lower=0.0) / (open_int + 1.0)
        ratio_cap = max(float(getattr(cfg, "opening_activity_ratio_cap", 3.0) or 3.0), 1e-6)
        weight_cap = max(float(getattr(cfg, "opening_activity_weight_cap", 4.0) or 4.0), 1.0)
        scaled = ratio.clip(lower=0.0, upper=ratio_cap)
        return 1.0 + (scaled.apply(math.log1p) / math.log1p(ratio_cap)) * (weight_cap - 1.0)

    call_score = (call_flow["FLOW_NOTIONAL"] * _weights(call_flow)).sum()
    put_score = (put_flow["FLOW_NOTIONAL"] * _weights(put_flow)).sum()

    if call_score <= 0 and put_score <= 0:
        return "NO_FLOW"

    if put_score <= 0:
        return "BULLISH_FLOW"

    ratio = float(call_score / put_score)

    if ratio >= cfg.bullish_ratio_threshold:
        return "BULLISH_FLOW"

    if ratio <= cfg.bearish_ratio_threshold:
        return "BEARISH_FLOW"

    return "MIXED_FLOW"



def smart_money_signal(option_chain, spot=None):
    """
    Purpose:
        Process smart money signal for downstream use.
    
    Context:
        Public function within the analytics layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        spot (Any): Input associated with spot.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        Keeping this step explicit makes it easier to audit how the final feature, score, or trade decision was assembled.
    """
    spikes = detect_unusual_volume(option_chain, spot=spot)
    return classify_flow(spikes)
