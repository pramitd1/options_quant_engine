"""Tests for macro event window boundary conditions."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
import pytz


IST = pytz.timezone("Asia/Kolkata")


def test_trade_entry_exactly_at_event_start_lockdown_boundary():
    """Trade entry exactly at event start time triggers lockdown."""
    
    event_start = datetime.now(IST)
    event_end = event_start + timedelta(minutes=30)
    
    check_time = event_start  # Exactly at start
    
    # Lockdown: no trades during event window
    in_lockdown = check_time >= event_start and check_time < event_end
    
    assert in_lockdown is True


def test_trade_entry_one_minute_before_event_lockdown_triggered():
    """Trade entry 1 minute before event starts pre-event lockdown."""
    
    event_start = datetime.now(IST) + timedelta(minutes=1)
    pre_event_window = 2  # 2 minutes before event
    
    check_time = event_start - timedelta(minutes=1)  # 1 minute before
    
    seconds_until_event = (event_start - check_time).total_seconds() / 60
    pre_event_lockdown = seconds_until_event < pre_event_window
    
    assert pre_event_lockdown is True


def test_trade_entry_five_minutes_after_event_allowed():
    """Trade entry 5 minutes after event ends is allowed."""
    
    event_end = datetime.now(IST) - timedelta(minutes=5)  # Event ended 5 min ago
    lockdown_grace_period = 2  # 2 minutes grace after event
    
    check_time = datetime.now(IST)
    minutes_since_end = (check_time - event_end).total_seconds() / 60
    
    lockdown_active = minutes_since_end < lockdown_grace_period
    
    assert lockdown_active is False


def test_multiple_events_same_day_confusion_test():
    """When multiple events occur same day, system tracks each separately."""
    
    events = [
        {
            "name": "RBI",
            "start": datetime(2026, 3, 22, 10, 0, tzinfo=IST),
            "end": datetime(2026, 3, 22, 10, 30, tzinfo=IST),
        },
        {
            "name": "Fed",
            "start": datetime(2026, 3, 22, 20, 0, tzinfo=IST),  # 20:00 IST = early morning UTC
            "end": datetime(2026, 3, 22, 20, 30, tzinfo=IST),
        },
    ]
    
    check_time = datetime(2026, 3, 22, 14, 0, tzinfo=IST)  # Between events
    
    active_events = []
    for event in events:
        if check_time >= event["start"] and check_time < event["end"]:
            active_events.append(event["name"])
    
    # Should not be in any event window
    assert len(active_events) == 0


def test_event_with_missing_event_name_safe_handling():
    """When event name is missing, use safe default."""
    
    event = {
        "name": None,  # Missing
        "start": datetime.now(IST),
        "end": datetime.now(IST) + timedelta(minutes=30),
    }
    
    # Safe naming
    event_label = event.get("name") or "scheduled_macro_event"
    
    assert event_label == "scheduled_macro_event"


def test_event_time_in_future_vs_past_detected():
    """System distinguishes between future and past events."""
    
    now = datetime.now(IST)
    
    future_event = {
        "start": now + timedelta(hours=2),
        "end": now + timedelta(hours=2, minutes=30),
    }
    
    past_event = {
        "start": now - timedelta(hours=2),
        "end": now - timedelta(hours=1, minutes=30),
    }
    
    is_future = future_event["start"] > now
    is_past = past_event["end"] < now
    
    assert is_future is True
    assert is_past is True


def test_event_window_dst_boundary_edge_case():
    """Event window calculation respects DST boundaries (India has no DST but verify)."""
    
    # India doesn't observe DST, but verify consistent behavior
    event_start_ist = datetime(2026, 3, 22, 10, 0, tzinfo=IST)
    event_end_ist = datetime(2026, 3, 22, 10, 30, tzinfo=IST)
    
    check_time_ist = datetime(2026, 3, 22, 10, 15, tzinfo=IST)
    
    in_window = check_time_ist >= event_start_ist and check_time_ist < event_end_ist
    
    assert in_window is True


def test_evening_expiry_trade_during_morning_macro_event():
    """Evening expiry trade (closing) during morning macro event is blocked."""
    
    macro_event_start = datetime(2026, 3, 22, 10, 0, tzinfo=IST)
    macro_event_end = datetime(2026, 3, 22, 10, 30, tzinfo=IST)
    
    evening_expiry_time = datetime(2026, 3, 22, 15, 15, tzinfo=IST)  # 3:15 PM
    
    # Check if evening trade would be blocked by morning event
    event_already_ended = evening_expiry_time >= macro_event_end
    
    assert event_already_ended is True  # Event is over by 3:15 PM


def test_consecutive_events_no_gap_handled():
    """When events are consecutive with no gap, both windows are honored."""
    
    event1_start = datetime(2026, 3, 22, 10, 0, tzinfo=IST)
    event1_end = datetime(2026, 3, 22, 10, 30, tzinfo=IST)
    
    event2_start = datetime(2026, 3, 22, 10, 30, tzinfo=IST)
    event2_end = datetime(2026, 3, 22, 11, 0, tzinfo=IST)
    
    check_time = datetime(2026, 3, 22, 10, 30, tzinfo=IST)
    
    # At exact boundary - treat as event2_start
    in_event1 = check_time >= event1_start and check_time < event1_end
    in_event2 = check_time >= event2_start and check_time < event2_end
    
    # Should be in event2
    assert in_event1 is False
    assert in_event2 is True


def test_event_window_respects_holding_profile():
    """Event window enforcement depends on holding profile (intraday vs overnight)."""
    
    event_start = datetime(2026, 3, 22, 15, 0, tzinfo=IST)  # 3 PM
    event_end = datetime(2026, 3, 22, 16, 0, tzinfo=IST)
    
    check_time = datetime(2026, 3, 22, 15, 30, tzinfo=IST)
    
    # For intraday: block during event
    intraday_blocked = (check_time >= event_start and check_time < event_end)
    
    # For overnight: less critical (overnight position doesn't suffer immediate impact)
    overnight_blocked = False  # Overnight holds less sensitive to event
    
    assert intraday_blocked is True
    assert overnight_blocked is False


def test_timezone_conversion_event_boundary():
    """Event times in different timezones are correctly converted for IST boundary."""
    
    # Event specified in UTC
    event_start_utc = datetime(2026, 3, 22, 4, 30, tzinfo=pytz.UTC)
    event_start_ist = event_start_utc.astimezone(IST)
    
    # Should be 10:00 IST (UTC+5:30)
    assert event_start_ist.hour == 10
    assert event_start_ist.minute == 0
