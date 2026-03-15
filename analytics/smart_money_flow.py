import pandas as pd

from analytics.flow_utils import front_expiry_atm_slice
from config.analytics_feature_policy import get_smart_money_flow_policy_config


def detect_unusual_volume(option_chain, spot=None):
    """
    Detect unusual near-ATM front-expiry activity using both volume/OI and OI change.
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
    working["FLOW_NOTIONAL"] = working["VOLUME"] * working["LAST_PRICE"]

    spikes = working[
        (working["VOL_OI_RATIO"] >= cfg.unusual_volume_ratio_threshold) |
        (working["OPENING_ACTIVITY"] > cfg.opening_activity_threshold)
    ].copy()

    return spikes


def classify_flow(spikes):
    """
    Classify directional flow using notional activity, not just contract counts.
    """
    cfg = get_smart_money_flow_policy_config()

    if spikes is None or len(spikes) == 0:
        return "NO_FLOW"

    call_flow = spikes[spikes["OPTION_TYP"] == "CE"].copy()
    put_flow = spikes[spikes["OPTION_TYP"] == "PE"].copy()

    call_score = (call_flow["FLOW_NOTIONAL"] * (1.0 + call_flow["OPENING_ACTIVITY"])).sum()
    put_score = (put_flow["FLOW_NOTIONAL"] * (1.0 + put_flow["OPENING_ACTIVITY"])).sum()

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
    spikes = detect_unusual_volume(option_chain, spot=spot)
    return classify_flow(spikes)
