"""
Centralized configuration for scheduled macro-event window handling.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import (
    MACRO_EVENT_EVENT_DURATION_MINUTES,
    MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES,
    MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES,
    MACRO_EVENT_PRE_EVENT_WARNING_MINUTES,
    MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD,
    MACRO_EVENT_WATCH_RISK_THRESHOLD,
)


@dataclass(frozen=True)
class EventWindowPolicyConfig:
    pre_event_warning_minutes: int = MACRO_EVENT_PRE_EVENT_WARNING_MINUTES
    pre_event_lockdown_minutes: int = MACRO_EVENT_PRE_EVENT_LOCKDOWN_MINUTES
    event_duration_minutes: int = MACRO_EVENT_EVENT_DURATION_MINUTES
    post_event_cooldown_minutes: int = MACRO_EVENT_POST_EVENT_COOLDOWN_MINUTES
    severity_risk_minor: int = 30
    severity_risk_medium: int = 55
    severity_risk_major: int = 80
    severity_risk_critical: int = 95
    watch_risk_threshold: int = MACRO_EVENT_WATCH_RISK_THRESHOLD
    strong_watch_risk_threshold: int = MACRO_EVENT_STRONG_WATCH_RISK_THRESHOLD
    pre_event_watch_penalty_high: int = -6
    pre_event_watch_penalty_normal: int = -3
    post_event_cooldown_penalty_high: int = -4
    post_event_cooldown_penalty_normal: int = -2
    lockdown_penalty: int = -10
    pre_watch_base_multiplier: float = 0.35
    pre_watch_proximity_multiplier: float = 0.25
    pre_lockdown_base_multiplier: float = 0.75
    pre_lockdown_proximity_multiplier: float = 0.25
    post_cooldown_base_multiplier: float = 0.35
    post_cooldown_decay_multiplier: float = 0.35
    live_event_risk_bonus: int = 10


EVENT_WINDOW_POLICY_CONFIG = EventWindowPolicyConfig()


def get_event_window_policy_config() -> EventWindowPolicyConfig:
    from tuning.runtime import resolve_dataclass_config

    return resolve_dataclass_config("event_windows.core", EventWindowPolicyConfig())
