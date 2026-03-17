"""
Module: state.py

Purpose:
    Manage Streamlit URL query-parameter persistence and session-state seeding.

Role in the System:
    Part of the application layer. Keeps state management separate from
    rendering logic so the workstation can survive timed page reloads without
    losing operator-selected controls.
"""
from __future__ import annotations

import streamlit as st

from config.settings import (
    DATA_SOURCE_OPTIONS,
    DEFAULT_DATA_SOURCE,
    DEFAULT_SYMBOL,
    LOT_SIZE,
    MAX_CAPITAL_PER_TRADE,
    NUMBER_OF_LOTS,
)


def query_param_value(name: str, default: str | None = None) -> str | None:
    """Read a single query parameter value from the current Streamlit URL state."""
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def query_param_bool(name: str, default: bool) -> bool:
    """Decode a boolean control value from query parameters."""
    value = (query_param_value(name) or "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def query_param_int(name: str, default: int) -> int:
    """Decode an integer control value from query parameters."""
    value = query_param_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def query_param_float(name: str, default: float) -> float:
    """Decode a float control value from query parameters."""
    value = query_param_value(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def seed_control_state():
    """Seed Streamlit widget state from URL parameters on first load."""
    defaults = {
        "control_mode": query_param_value("mode", "LIVE") or "LIVE",
        "control_symbol": (query_param_value("symbol", DEFAULT_SYMBOL) or DEFAULT_SYMBOL).strip().upper(),
        "control_source": (query_param_value("source", DEFAULT_DATA_SOURCE) or DEFAULT_DATA_SOURCE).strip().upper(),
        "control_save_live_snapshots": query_param_bool("save_live_snapshots", False),
        "control_auto_refresh": query_param_bool("auto_refresh", False),
        "control_refresh_seconds": max(10, min(query_param_int("refresh_seconds", 30), 300)),
        "control_apply_budget_constraint": query_param_bool("apply_budget_constraint", False),
        "control_lot_size": max(1, query_param_int("lot_size", int(LOT_SIZE))),
        "control_requested_lots": max(1, query_param_int("requested_lots", int(NUMBER_OF_LOTS))),
        "control_max_capital": max(0.0, query_param_float("max_capital", float(MAX_CAPITAL_PER_TRADE))),
        "control_replay_dir": query_param_value("replay_dir", "debug_samples") or "debug_samples",
        "control_replay_spot": query_param_value("replay_spot", "") or "",
        "control_replay_chain": query_param_value("replay_chain", "") or "",
    }

    if defaults["control_source"] not in DATA_SOURCE_OPTIONS and defaults["control_mode"] == "LIVE":
        defaults["control_source"] = DEFAULT_DATA_SOURCE

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_control_state(mode: str, *, symbol: str | None = None, source: str | None = None, replay_dir: str | None = None):
    """Persist non-sensitive sidebar controls into the URL query string."""
    st.query_params["mode"] = st.session_state.get("control_mode", mode)
    st.query_params["symbol"] = symbol or st.session_state.get("control_symbol", DEFAULT_SYMBOL)
    st.query_params["source"] = source or st.session_state.get("control_source", DEFAULT_DATA_SOURCE)
    st.query_params["save_live_snapshots"] = "1" if st.session_state.get("control_save_live_snapshots") else "0"
    st.query_params["auto_refresh"] = "1" if st.session_state.get("control_auto_refresh") else "0"
    st.query_params["refresh_seconds"] = str(st.session_state.get("control_refresh_seconds", 30))
    if not st.session_state.get("control_auto_refresh"):
        st.query_params.pop("auto_run", None)
    st.query_params["apply_budget_constraint"] = "1" if st.session_state.get("control_apply_budget_constraint") else "0"
    st.query_params["lot_size"] = str(st.session_state.get("control_lot_size", int(LOT_SIZE)))
    st.query_params["requested_lots"] = str(st.session_state.get("control_requested_lots", int(NUMBER_OF_LOTS)))
    st.query_params["max_capital"] = str(st.session_state.get("control_max_capital", float(MAX_CAPITAL_PER_TRADE)))

    if mode == "REPLAY":
        st.query_params["replay_dir"] = replay_dir or st.session_state.get("control_replay_dir", "debug_samples")
        st.query_params["replay_spot"] = st.session_state.get("control_replay_spot", "")
        st.query_params["replay_chain"] = st.session_state.get("control_replay_chain", "")
    else:
        for key in ("replay_dir", "replay_spot", "replay_chain"):
            st.query_params.pop(key, None)
