"""
Module: event_window_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by event window.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
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
    """
    Purpose:
        Dataclass representing EventWindowPolicyConfig within the repository.
    
    Context:
        Used within the configuration layer that centralizes policy defaults and thresholds. The class keeps configuration or structured state explicit for downstream consumers.
    
    Attributes:
        pre_event_warning_minutes (int): Number of minutes used for pre event warning.
        pre_event_lockdown_minutes (int): Number of minutes used for pre event lockdown.
        event_duration_minutes (int): Number of minutes used for event duration.
        post_event_cooldown_minutes (int): Number of minutes used for post event cooldown.
        severity_risk_minor (int): Value supplied for severity risk minor.
        severity_risk_medium (int): Value supplied for severity risk medium.
        severity_risk_major (int): Value supplied for severity risk major.
        severity_risk_critical (int): Value supplied for severity risk critical.
        watch_risk_threshold (int): Threshold used to classify or trigger watch risk.
        strong_watch_risk_threshold (int): Threshold used to classify or trigger strong watch risk.
        pre_event_watch_penalty_high (int): Value supplied for pre event watch penalty high.
        pre_event_watch_penalty_normal (int): Value supplied for pre event watch penalty normal.
        post_event_cooldown_penalty_high (int): Value supplied for post event cooldown penalty high.
        post_event_cooldown_penalty_normal (int): Value supplied for post event cooldown penalty normal.
        lockdown_penalty (int): Penalty applied when lockdown is active.
        pre_watch_base_multiplier (float): Multiplier applied to pre watch base.
        pre_watch_proximity_multiplier (float): Multiplier applied to pre watch proximity.
        pre_lockdown_base_multiplier (float): Multiplier applied to pre lockdown base.
        pre_lockdown_proximity_multiplier (float): Multiplier applied to pre lockdown proximity.
        post_cooldown_base_multiplier (float): Multiplier applied to post cooldown base.
        post_cooldown_decay_multiplier (float): Multiplier applied to post cooldown decay.
        live_event_risk_bonus (int): Bonus applied when live event risk is active.
    
    Notes:
        Explicit field-level documentation makes policy tuning safer because threshold and weighting semantics stay visible at the point of definition.
    """
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
    """
    Purpose:
        Return the event-window policy bundle used by macro-event overlays.
    
    Context:
        Public function in the configuration layer. It exposes a stable policy bundle for runtime, research, or governance code.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        EventWindowPolicyConfig: Configuration object used by downstream runtime, research, or governance code.
    
    Notes:
        Centralizing policy access behind getters keeps live, replay, research, and tuning workflows aligned on the same defaults.
    """
    from config.policy_resolver import resolve_dataclass_config

    return resolve_dataclass_config("event_windows.core", EventWindowPolicyConfig())
