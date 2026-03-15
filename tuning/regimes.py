"""
Deterministic regime labeling for validation and robustness analysis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.validation_regime_policy import get_validation_regime_config


REGIME_COLUMNS = [
    "vol_regime_bucket",
    "gamma_regime_bucket",
    "macro_regime_bucket",
    "global_risk_bucket",
    "overnight_bucket",
    "squeeze_risk_bucket",
    "event_risk_bucket",
]


def _normalize(value: Any) -> str:
    return str(value or "").upper().strip()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().str.strip()


def _vol_bucket(row: pd.Series) -> str:
    state = _normalize(row.get("volatility_regime"))
    global_state = _normalize(row.get("global_risk_state"))
    if global_state == "VOL_SHOCK":
        return "HIGH_VOL"
    if state in {"VOL_EXPANSION", "HIGH_VOL", "VOL_SHOCK"}:
        return "HIGH_VOL"
    if state in {"VOL_SUPPRESSION", "LOW_VOL"}:
        return "LOW_VOL"
    if state in {"NORMAL_VOL", "BALANCED_VOL"}:
        return "NORMAL_VOL"
    return "UNKNOWN_VOL"


def _gamma_bucket(row: pd.Series) -> str:
    state = _normalize(row.get("gamma_regime"))
    if state in {"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}:
        return "SHORT_GAMMA"
    if state in {"POSITIVE_GAMMA", "LONG_GAMMA_ZONE"}:
        return "LONG_GAMMA"
    if state in {"NEUTRAL_GAMMA"}:
        return "NEUTRAL_GAMMA"
    return "UNKNOWN_GAMMA"


def _macro_bucket(row: pd.Series) -> str:
    state = _normalize(row.get("macro_regime"))
    if state in {"EVENT_LOCKDOWN"}:
        return "EVENT_LOCKDOWN"
    if state in {"RISK_OFF"}:
        return "RISK_OFF"
    if state in {"RISK_ON"}:
        return "RISK_ON"
    if state in {"MACRO_NEUTRAL"}:
        return "MACRO_NEUTRAL"
    return "MACRO_UNKNOWN"


def _global_risk_bucket(row: pd.Series) -> str:
    state = _normalize(row.get("global_risk_state"))
    if state in {"VOL_SHOCK", "EVENT_LOCKDOWN", "RISK_OFF", "RISK_ON", "GLOBAL_NEUTRAL"}:
        return state
    return "GLOBAL_RISK_UNKNOWN"


def _overnight_bucket(row: pd.Series) -> str:
    value = row.get("overnight_hold_allowed")
    raw = str(value).strip().lower() if value is not None and str(value) != "<NA>" else ""
    if raw in {"true", "1"}:
        return "OVERNIGHT_ALLOWED"
    if raw in {"false", "0"}:
        return "OVERNIGHT_BLOCKED"
    return "OVERNIGHT_UNKNOWN"


def _squeeze_bucket(row: pd.Series) -> str:
    state = _normalize(row.get("squeeze_risk_state"))
    if state in {
        "LOW_ACCELERATION_RISK",
        "MODERATE_ACCELERATION_RISK",
        "HIGH_ACCELERATION_RISK",
        "EXTREME_ACCELERATION_RISK",
    }:
        return state

    dealer_state = _normalize(row.get("dealer_flow_state"))
    if dealer_state in {"PINNING_DOMINANT"}:
        return "PINNING_REGIME"
    if dealer_state in {"UPSIDE_HEDGING_ACCELERATION", "DOWNSIDE_HEDGING_ACCELERATION", "TWO_SIDED_INSTABILITY"}:
        return "CONVEXITY_ACTIVE"
    return "SQUEEZE_UNKNOWN"


def _event_risk_bucket(row: pd.Series) -> str:
    event_score = _safe_float(row.get("macro_event_risk_score"), None)
    if event_score is None:
        return "EVENT_RISK_UNKNOWN"
    if event_score >= 70.0:
        return "HIGH_EVENT_RISK"
    if event_score >= 45.0:
        return "ELEVATED_EVENT_RISK"
    return "LOW_EVENT_RISK"


def label_validation_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy() if frame is not None else pd.DataFrame()
    if labeled.empty:
        for column in REGIME_COLUMNS:
            labeled[column] = pd.Series(dtype="object")
        return labeled

    cfg = get_validation_regime_config()
    vol_state = _normalize_series(labeled.get("volatility_regime", pd.Series("", index=labeled.index)))
    global_state = _normalize_series(labeled.get("global_risk_state", pd.Series("", index=labeled.index)))
    gamma_state = _normalize_series(labeled.get("gamma_regime", pd.Series("", index=labeled.index)))
    macro_state = _normalize_series(labeled.get("macro_regime", pd.Series("", index=labeled.index)))
    dealer_state = _normalize_series(labeled.get("dealer_flow_state", pd.Series("", index=labeled.index)))
    squeeze_state = _normalize_series(labeled.get("squeeze_risk_state", pd.Series("", index=labeled.index)))
    overnight_raw = labeled.get("overnight_hold_allowed", pd.Series("", index=labeled.index)).astype("string").fillna("").str.lower()
    event_score = pd.to_numeric(labeled.get("macro_event_risk_score", pd.Series(index=labeled.index)), errors="coerce")

    labeled["vol_regime_bucket"] = "UNKNOWN_VOL"
    labeled.loc[global_state.eq("VOL_SHOCK"), "vol_regime_bucket"] = "HIGH_VOL"
    labeled.loc[vol_state.isin({"VOL_EXPANSION", "HIGH_VOL", "VOL_SHOCK"}), "vol_regime_bucket"] = "HIGH_VOL"
    labeled.loc[vol_state.isin({"VOL_SUPPRESSION", "LOW_VOL"}), "vol_regime_bucket"] = "LOW_VOL"
    labeled.loc[vol_state.isin({"NORMAL_VOL", "BALANCED_VOL"}), "vol_regime_bucket"] = "NORMAL_VOL"

    labeled["gamma_regime_bucket"] = "UNKNOWN_GAMMA"
    labeled.loc[gamma_state.isin({"NEGATIVE_GAMMA", "SHORT_GAMMA_ZONE"}), "gamma_regime_bucket"] = "SHORT_GAMMA"
    labeled.loc[gamma_state.isin({"POSITIVE_GAMMA", "LONG_GAMMA_ZONE"}), "gamma_regime_bucket"] = "LONG_GAMMA"
    labeled.loc[gamma_state.eq("NEUTRAL_GAMMA"), "gamma_regime_bucket"] = "NEUTRAL_GAMMA"

    labeled["macro_regime_bucket"] = "MACRO_UNKNOWN"
    labeled.loc[macro_state.eq("EVENT_LOCKDOWN"), "macro_regime_bucket"] = "EVENT_LOCKDOWN"
    labeled.loc[macro_state.eq("RISK_OFF"), "macro_regime_bucket"] = "RISK_OFF"
    labeled.loc[macro_state.eq("RISK_ON"), "macro_regime_bucket"] = "RISK_ON"
    labeled.loc[macro_state.eq("MACRO_NEUTRAL"), "macro_regime_bucket"] = "MACRO_NEUTRAL"

    labeled["global_risk_bucket"] = "GLOBAL_RISK_UNKNOWN"
    labeled.loc[
        global_state.isin({"VOL_SHOCK", "EVENT_LOCKDOWN", "RISK_OFF", "RISK_ON", "GLOBAL_NEUTRAL"}),
        "global_risk_bucket",
    ] = global_state

    labeled["overnight_bucket"] = "OVERNIGHT_UNKNOWN"
    labeled.loc[overnight_raw.isin({"true", "1"}), "overnight_bucket"] = "OVERNIGHT_ALLOWED"
    labeled.loc[overnight_raw.isin({"false", "0"}), "overnight_bucket"] = "OVERNIGHT_BLOCKED"

    labeled["squeeze_risk_bucket"] = "SQUEEZE_UNKNOWN"
    labeled.loc[dealer_state.eq("PINNING_DOMINANT"), "squeeze_risk_bucket"] = "PINNING_REGIME"
    labeled.loc[
        dealer_state.isin({"UPSIDE_HEDGING_ACCELERATION", "DOWNSIDE_HEDGING_ACCELERATION", "TWO_SIDED_INSTABILITY"}),
        "squeeze_risk_bucket",
    ] = "CONVEXITY_ACTIVE"
    labeled.loc[
        squeeze_state.isin(
            {
                "LOW_ACCELERATION_RISK",
                "MODERATE_ACCELERATION_RISK",
                "HIGH_ACCELERATION_RISK",
                "EXTREME_ACCELERATION_RISK",
            }
        ),
        "squeeze_risk_bucket",
    ] = squeeze_state

    labeled["event_risk_bucket"] = "EVENT_RISK_UNKNOWN"
    labeled.loc[event_score < cfg.elevated_event_risk_threshold, "event_risk_bucket"] = "LOW_EVENT_RISK"
    labeled.loc[event_score >= cfg.elevated_event_risk_threshold, "event_risk_bucket"] = "ELEVATED_EVENT_RISK"
    labeled.loc[event_score >= cfg.high_event_risk_threshold, "event_risk_bucket"] = "HIGH_EVENT_RISK"
    return labeled
