"""
Strict tradable-data evaluation utilities.

This module separates data usability for analytics from data usability for
execution suggestions while remaining broker-agnostic.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.policy_resolver import get_parameter_value


def _resolve_column_name(option_chain: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in option_chain.columns:
            return name
    return None


def _safe_numeric_series(option_chain: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = _resolve_column_name(option_chain, candidates)
    if column is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(option_chain[column], errors="coerce")


def _safe_datetime_series(option_chain: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = _resolve_column_name(option_chain, candidates)
    if column is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(option_chain[column], errors="coerce", utc=True)


def evaluate_tradable_data_layer(option_chain: pd.DataFrame | None) -> dict[str, Any]:
    row_count = int(len(option_chain)) if isinstance(option_chain, pd.DataFrame) else 0
    if option_chain is None or not isinstance(option_chain, pd.DataFrame) or option_chain.empty:
        return {
            "analytics_usable": False,
            "execution_suggestion_usable": False,
            "quote_freshness": {"status": "UNKNOWN", "stale_ratio": None, "max_age_seconds": None},
            "crossed_locked": {
                "crossed_or_locked_rows": 0,
                "crossed_or_locked_ratio": 0.0,
                "status": "UNKNOWN",
            },
            "outlier_rejection": {
                "outlier_rows": 0,
                "outlier_ratio": 0.0,
                "status": "UNKNOWN",
            },
            "per_strike_confidence": {
                "status": "UNKNOWN",
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "strike_count": 0,
            },
            "reasons": ["option_chain_empty"],
            "row_count": row_count,
        }

    min_rows_for_analytics = int(float(get_parameter_value("tradable_data_layer.analytics.min_rows", 20)))
    min_rows_for_execution = int(float(get_parameter_value("tradable_data_layer.execution.min_rows", 40)))
    max_stale_ratio = float(get_parameter_value("tradable_data_layer.execution.max_stale_ratio", 0.20))
    max_crossed_locked_ratio = float(get_parameter_value("tradable_data_layer.execution.max_crossed_locked_ratio", 0.10))
    max_outlier_ratio = float(get_parameter_value("tradable_data_layer.execution.max_outlier_ratio", 0.20))
    outlier_mad_threshold = float(get_parameter_value("tradable_data_layer.outlier.mad_threshold", 8.0))

    bid = _safe_numeric_series(option_chain, ["bidPrice", "BID_PRICE", "bestBidPrice", "bid"])
    ask = _safe_numeric_series(option_chain, ["askPrice", "ASK_PRICE", "bestAskPrice", "ask"])
    ltp = _safe_numeric_series(option_chain, ["lastPrice", "LAST_PRICE"])
    strike = _safe_numeric_series(option_chain, ["strikePrice", "STRIKE_PR"])

    quote_time = _safe_datetime_series(option_chain, ["quote_timestamp", "timestamp", "updated_at", "last_trade_time"])
    stale_ratio = None
    freshness_status = "UNKNOWN"
    max_age_seconds = None
    if not quote_time.empty:
        age_seconds = (pd.Timestamp.utcnow(tz="UTC") - quote_time).dt.total_seconds()
        max_age_seconds = float(age_seconds.max()) if not age_seconds.dropna().empty else None
        stale_cutoff_seconds = float(get_parameter_value("tradable_data_layer.quote.stale_seconds", 180.0))
        stale_mask = age_seconds > stale_cutoff_seconds
        stale_ratio = float(stale_mask.mean()) if not stale_mask.empty else 0.0
        freshness_status = "STALE" if stale_ratio > max_stale_ratio else "FRESH"

    crossed_or_locked_mask = pd.Series(False, index=option_chain.index)
    if not bid.empty and not ask.empty:
        crossed_or_locked_mask = bid.gt(0) & ask.gt(0) & bid.ge(ask)
    crossed_or_locked_rows = int(crossed_or_locked_mask.sum())
    crossed_or_locked_ratio = crossed_or_locked_rows / max(row_count, 1)
    crossed_locked_status = "WEAK" if crossed_or_locked_ratio > max_crossed_locked_ratio else "GOOD"

    outlier_mask = pd.Series(False, index=option_chain.index)
    if not ltp.empty:
        ltp_valid = ltp[ltp.gt(0)].dropna()
        if not ltp_valid.empty:
            median = float(ltp_valid.median())
            mad = float((ltp_valid - median).abs().median())
            if mad > 0:
                robust_z = (ltp - median).abs() / (1.4826 * mad)
                outlier_mask = robust_z > outlier_mad_threshold
    outlier_rows = int(outlier_mask.sum())
    outlier_ratio = outlier_rows / max(row_count, 1)
    outlier_status = "WEAK" if outlier_ratio > max_outlier_ratio else "GOOD"

    # Strike-level confidence starts at 1.0 and decays for poor quote integrity and outliers.
    strike_confidence = pd.Series(1.0, index=option_chain.index, dtype="float64")
    if not bid.empty and not ask.empty:
        spread = (ask - bid).clip(lower=0)
        mid = ((ask + bid) / 2.0).replace(0, float("nan"))
        spread_ratio = (spread / mid).astype("float64").fillna(0.0)
        strike_confidence -= spread_ratio.clip(lower=0.0, upper=0.8)
    strike_confidence -= crossed_or_locked_mask.astype("float64") * 0.5
    strike_confidence -= outlier_mask.astype("float64") * 0.3
    strike_confidence = strike_confidence.clip(lower=0.0, upper=1.0)

    strike_count = int(strike.dropna().nunique()) if not strike.empty else 0
    per_strike_status = "GOOD" if float(strike_confidence.mean()) >= 0.55 else "WEAK"

    reasons: list[str] = []
    analytics_usable = row_count >= min_rows_for_analytics
    if not analytics_usable:
        reasons.append("analytics_min_rows_failed")

    execution_suggestion_usable = row_count >= min_rows_for_execution
    if not execution_suggestion_usable:
        reasons.append("execution_min_rows_failed")

    if freshness_status == "STALE":
        execution_suggestion_usable = False
        reasons.append("quote_freshness_stale")

    if crossed_or_locked_ratio > max_crossed_locked_ratio:
        execution_suggestion_usable = False
        reasons.append("crossed_locked_ratio_high")

    if outlier_ratio > max_outlier_ratio:
        execution_suggestion_usable = False
        reasons.append("outlier_ratio_high")

    if float(strike_confidence.mean()) < 0.45:
        execution_suggestion_usable = False
        reasons.append("per_strike_confidence_low")

    return {
        "analytics_usable": analytics_usable,
        "execution_suggestion_usable": execution_suggestion_usable,
        "quote_freshness": {
            "status": freshness_status,
            "stale_ratio": round(stale_ratio, 4) if stale_ratio is not None else None,
            "max_age_seconds": round(max_age_seconds, 2) if max_age_seconds is not None else None,
        },
        "crossed_locked": {
            "crossed_or_locked_rows": crossed_or_locked_rows,
            "crossed_or_locked_ratio": round(crossed_or_locked_ratio, 4),
            "status": crossed_locked_status,
        },
        "outlier_rejection": {
            "outlier_rows": outlier_rows,
            "outlier_ratio": round(outlier_ratio, 4),
            "status": outlier_status,
        },
        "per_strike_confidence": {
            "status": per_strike_status,
            "mean": round(float(strike_confidence.mean()), 4),
            "min": round(float(strike_confidence.min()), 4),
            "max": round(float(strike_confidence.max()), 4),
            "strike_count": strike_count,
        },
        "reasons": reasons,
        "row_count": row_count,
    }
