"""Tests for replay snapshot consistency and timestamp validation."""
from __future__ import annotations

import pandas as pd
from data.replay_loader import validate_snapshot_consistency


def test_validate_snapshot_consistency_accepts_aligned_timestamps():
    """When spot and chain timestamps are close, consistency check passes."""
    spot_snapshot = {
        "spot": 23000,
        "timestamp": "2026-03-22T10:00:00+05:30",
    }
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "OPTION_TYP": ["CE", "CE", "CE"],
        "TIMESTAMP": ["2026-03-22T10:00:05+05:30"] * 3,
    })

    result = validate_snapshot_consistency(spot_snapshot, option_chain)

    assert result["is_consistent"] is True
    assert len(result["warnings"]) == 0


def test_validate_snapshot_consistency_warns_on_timestamp_mismatch():
    """When spot and chain timestamps diverge >30s, consistency warning is raised."""
    spot_snapshot = {
        "spot": 23000,
        "timestamp": "2026-03-22T10:00:00+05:30",
    }
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "OPTION_TYP": ["CE", "CE", "CE"],
        "TIMESTAMP": ["2026-03-22T10:01:00+05:30"] * 3,  # 60 seconds later
    })

    result = validate_snapshot_consistency(spot_snapshot, option_chain)

    assert result["is_consistent"] is False
    assert any("mismatch_delta_sec" in str(w) for w in result["warnings"])


def test_validate_snapshot_consistency_handles_missing_timestamps():
    """When timestamps are missing, consistency is still True (no data to conflict)."""
    spot_snapshot = {
        "spot": 23000,
        "timestamp": None,
    }
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "OPTION_TYP": ["CE", "CE", "CE"],
    })

    result = validate_snapshot_consistency(spot_snapshot, option_chain)

    assert result["is_consistent"] is True


def test_validate_snapshot_consistency_handles_none_snapshots():
    """When either snapshot is None, check passes without error."""
    result = validate_snapshot_consistency(None, None)
    assert result["is_consistent"] is True
    assert len(result["warnings"]) == 0

    result = validate_snapshot_consistency({"spot": 23000, "timestamp": "2026-03-22T10:00:00+05:30"}, None)
    assert result["is_consistent"] is True


def test_validate_snapshot_consistency_warns_on_unparseable_spot_timestamp():
    """When spot timestamp cannot be parsed, a warning is recorded."""
    spot_snapshot = {
        "spot": 23000,
        "timestamp": "invalid-timestamp",
    }
    option_chain = pd.DataFrame({
        "strikePrice": [22900, 23000, 23100],
        "OPTION_TYP": ["CE", "CE", "CE"],
        "TIMESTAMP": ["2026-03-22T10:00:00+05:30", "2026-03-22T10:00:00+05:30", "2026-03-22T10:00:00+05:30"],
    })

    result = validate_snapshot_consistency(spot_snapshot, option_chain)

    assert any("unparseable" in str(w) for w in result["warnings"])
