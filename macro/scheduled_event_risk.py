"""
Scheduled macro event risk layer.

Stage 1 scope:
- configurable local event schedule
- provider-agnostic loading interface
- conservative pre/post-event risk scoring
- interpretable outputs for the trading engine
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from macro.scope_utils import normalize_scope, symbol_scope_matches
from config.event_window_policy import get_event_window_policy_config
from config.settings import (
    BASE_DIR,
    DEFAULT_MACRO_EVENT_SCHEDULE,
    MACRO_EVENT_FILTER_ENABLED,
    MACRO_EVENT_SCHEDULE_FILE,
)


def _severity_to_base_risk():
    cfg = get_event_window_policy_config()
    return {
        "CRITICAL": cfg.severity_risk_critical,
        "MAJOR": cfg.severity_risk_major,
        "MEDIUM": cfg.severity_risk_medium,
        "MINOR": cfg.severity_risk_minor,
    }

IST_TIMEZONE = "Asia/Kolkata"


def _coerce_timestamp(value):
    if value is None or value == "":
        return None

    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None

    if ts.tzinfo is None:
        try:
            return ts.tz_localize(IST_TIMEZONE)
        except Exception:
            return None

    try:
        return ts.tz_convert(IST_TIMEZONE)
    except Exception:
        return None


def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _resolve_schedule_path(path_value: str | None):
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        path = Path(BASE_DIR) / path

    return path


def load_scheduled_macro_events(schedule_path: str | None = None, default_events=None) -> list[dict]:
    cfg = get_event_window_policy_config()
    default_events = default_events if default_events is not None else DEFAULT_MACRO_EVENT_SCHEDULE
    resolved_path = _resolve_schedule_path(schedule_path or MACRO_EVENT_SCHEDULE_FILE)

    raw_events = None
    if resolved_path and resolved_path.exists():
        try:
            raw_events = json.loads(resolved_path.read_text(encoding="utf-8"))
        except Exception:
            raw_events = None

    if raw_events is None:
        raw_events = default_events

    normalized = []
    for raw in raw_events or []:
        if not isinstance(raw, dict):
            continue

        event_time = _coerce_timestamp(raw.get("timestamp") or raw.get("event_time") or raw.get("start_time"))
        if event_time is None:
            continue

        severity = str(raw.get("severity", "MAJOR")).strip().upper()
        if severity not in _severity_to_base_risk():
            severity = "MAJOR"

        normalized.append({
            "name": str(raw.get("name", "UNNAMED_EVENT")).strip() or "UNNAMED_EVENT",
            "timestamp": event_time,
            "severity": severity,
            "scope": normalize_scope(raw.get("scope")),
            "warning_minutes": _coerce_int(raw.get("warning_minutes"), cfg.pre_event_warning_minutes),
            "lockdown_minutes": _coerce_int(raw.get("lockdown_minutes"), cfg.pre_event_lockdown_minutes),
            "event_duration_minutes": _coerce_int(raw.get("event_duration_minutes"), cfg.event_duration_minutes),
            "cooldown_minutes": _coerce_int(raw.get("cooldown_minutes"), cfg.post_event_cooldown_minutes),
            "source": str(raw.get("source", "LOCAL_CONFIG")).strip() or "LOCAL_CONFIG",
        })

    normalized.sort(key=lambda event: event["timestamp"])
    return normalized


def _normalize_event_list(events) -> list[dict]:
    cfg = get_event_window_policy_config()
    if not events:
        return []

    normalized = []
    for raw in events:
        if not isinstance(raw, dict):
            continue

        event_time = raw.get("timestamp")
        if not isinstance(event_time, pd.Timestamp):
            event_time = _coerce_timestamp(raw.get("timestamp") or raw.get("event_time") or raw.get("start_time"))

        if event_time is None:
            continue

        severity = str(raw.get("severity", "MAJOR")).strip().upper()
        if severity not in _severity_to_base_risk():
            severity = "MAJOR"

        normalized.append({
            "name": str(raw.get("name", "UNNAMED_EVENT")).strip() or "UNNAMED_EVENT",
            "timestamp": event_time,
            "severity": severity,
            "scope": normalize_scope(raw.get("scope")),
            "warning_minutes": _coerce_int(raw.get("warning_minutes"), cfg.pre_event_warning_minutes),
            "lockdown_minutes": _coerce_int(raw.get("lockdown_minutes"), cfg.pre_event_lockdown_minutes),
            "event_duration_minutes": _coerce_int(raw.get("event_duration_minutes"), cfg.event_duration_minutes),
            "cooldown_minutes": _coerce_int(raw.get("cooldown_minutes"), cfg.post_event_cooldown_minutes),
            "source": str(raw.get("source", "LOCAL_CONFIG")).strip() or "LOCAL_CONFIG",
        })

    normalized.sort(key=lambda event: event["timestamp"])
    return normalized


def _neutral_event_state(status="NO_EVENT_DATA"):
    event_data_available = status not in {"NO_EVENT_DATA", "EVENT_FILTER_DISABLED"}
    return {
        "macro_event_risk_score": 0,
        "event_window_status": status,
        "event_lockdown_flag": False,
        "minutes_to_next_event": None,
        "next_event_name": None,
        "active_event_name": None,
        "event_data_available": event_data_available,
        "event_source": None,
    }


def _risk_from_pre_watch(base_risk: int, minutes_until: float, warning_minutes: int, lockdown_minutes: int) -> int:
    cfg = get_event_window_policy_config()
    span = max(warning_minutes - lockdown_minutes, 1)
    proximity = 1.0 - max(minutes_until - lockdown_minutes, 0.0) / span
    return int(round(base_risk * (cfg.pre_watch_base_multiplier + (cfg.pre_watch_proximity_multiplier * proximity))))


def _risk_from_pre_lockdown(base_risk: int, minutes_until: float, lockdown_minutes: int) -> int:
    cfg = get_event_window_policy_config()
    span = max(lockdown_minutes, 1)
    proximity = 1.0 - max(minutes_until, 0.0) / span
    return int(round(base_risk * (cfg.pre_lockdown_base_multiplier + (cfg.pre_lockdown_proximity_multiplier * proximity))))


def _risk_from_post_cooldown(base_risk: int, minutes_since: float, cooldown_minutes: int) -> int:
    cfg = get_event_window_policy_config()
    span = max(cooldown_minutes, 1)
    decay = 1.0 - max(minutes_since, 0.0) / span
    decay = max(0.0, min(1.0, decay))
    return int(round(base_risk * (cfg.post_cooldown_base_multiplier + (cfg.post_cooldown_decay_multiplier * decay))))


def evaluate_scheduled_event_risk(
    symbol: str,
    as_of=None,
    events: list[dict] | None = None,
    enabled: bool = MACRO_EVENT_FILTER_ENABLED,
):
    cfg = get_event_window_policy_config()
    if not enabled:
        return _neutral_event_state(status="EVENT_FILTER_DISABLED")

    events = events if events is not None else load_scheduled_macro_events()
    events = _normalize_event_list(events)
    if not events:
        return _neutral_event_state(status="NO_EVENT_DATA")

    as_of_ts = _coerce_timestamp(as_of)
    if as_of_ts is None:
        as_of_ts = pd.Timestamp.now(tz=IST_TIMEZONE)

    relevant_events = [event for event in events if symbol_scope_matches(symbol, event["scope"])]
    if not relevant_events:
        return _neutral_event_state(status="NO_EVENT_DATA")

    next_event = None
    active_state = None

    for event in relevant_events:
        event_time = event["timestamp"]
        minutes_until = (event_time - as_of_ts).total_seconds() / 60.0
        minutes_since = (as_of_ts - event_time).total_seconds() / 60.0
        base_risk = _severity_to_base_risk().get(event["severity"], 80)

        if minutes_until >= 0 and (next_event is None or event_time < next_event["timestamp"]):
            next_event = event

        candidate = None

        if 0 <= minutes_until <= event["lockdown_minutes"]:
            candidate = {
                "status": "PRE_EVENT_LOCKDOWN",
                "risk": _risk_from_pre_lockdown(base_risk, minutes_until, event["lockdown_minutes"]),
                "lockdown": True,
                "event": event,
            }
        elif event["lockdown_minutes"] < minutes_until <= event["warning_minutes"]:
            candidate = {
                "status": "PRE_EVENT_WATCH",
                "risk": _risk_from_pre_watch(base_risk, minutes_until, event["warning_minutes"], event["lockdown_minutes"]),
                "lockdown": False,
                "event": event,
            }
        elif 0 <= minutes_since <= event["event_duration_minutes"]:
            candidate = {
                "status": "LIVE_EVENT",
                "risk": min(100, base_risk + cfg.live_event_risk_bonus),
                "lockdown": True,
                "event": event,
            }
        elif event["event_duration_minutes"] < minutes_since <= (event["event_duration_minutes"] + event["cooldown_minutes"]):
            candidate = {
                "status": "POST_EVENT_COOLDOWN",
                "risk": _risk_from_post_cooldown(base_risk, minutes_since - event["event_duration_minutes"], event["cooldown_minutes"]),
                "lockdown": False,
                "event": event,
            }

        if candidate is None:
            continue

        if active_state is None:
            active_state = candidate
            continue

        current_priority = (active_state["lockdown"], active_state["risk"])
        new_priority = (candidate["lockdown"], candidate["risk"])
        if new_priority > current_priority:
            active_state = candidate

    minutes_to_next_event = None
    next_event_name = None
    if next_event is not None:
        minutes_to_next_event = int(round((next_event["timestamp"] - as_of_ts).total_seconds() / 60.0))
        minutes_to_next_event = max(minutes_to_next_event, 0)
        next_event_name = next_event["name"]

    if active_state is None:
        return {
            "macro_event_risk_score": 0,
            "event_window_status": "CLEAR",
            "event_lockdown_flag": False,
            "minutes_to_next_event": minutes_to_next_event,
            "next_event_name": next_event_name,
            "active_event_name": None,
            "event_data_available": True,
            "event_source": next_event["source"] if next_event is not None else "LOCAL_CONFIG",
        }

    return {
        "macro_event_risk_score": int(active_state["risk"]),
        "event_window_status": active_state["status"],
        "event_lockdown_flag": bool(active_state["lockdown"]),
        "minutes_to_next_event": minutes_to_next_event,
        "next_event_name": next_event_name,
        "active_event_name": active_state["event"]["name"],
        "event_data_available": True,
        "event_source": active_state["event"]["source"],
    }
