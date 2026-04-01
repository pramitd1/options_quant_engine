from __future__ import annotations

import json

import pytest

from data.replay_loader import (
    list_replay_chain_snapshots,
    load_option_chain_snapshot,
    validate_snapshot_consistency,
    resolve_replay_snapshot_paths,
)


def test_resolve_replay_snapshot_paths_source_aware_and_skips_invalid(tmp_path):
    # Spot snapshots
    (tmp_path / "NIFTY_spot_snapshot_2026-03-20T12-00-00+05-30.json").write_text(
        json.dumps({"spot": 23000, "timestamp": "2026-03-20T12:00:00+05:30"}),
        encoding="utf-8",
    )

    # One valid ICICI chain, one newer but empty ICICI chain.
    (tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-10-00+05-30.csv").write_text(
        "strikePrice,OPTION_TYP,lastPrice\n23000,CE,100\n",
        encoding="utf-8",
    )
    (tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-20-00+05-30.csv").write_text(
        "\n",
        encoding="utf-8",
    )

    selection = resolve_replay_snapshot_paths(
        "NIFTY",
        replay_dir=str(tmp_path),
        source_label="ICICI",
    )

    assert selection["spot_path"] is not None
    assert selection["chain_path"].endswith("NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-10-00+05-30.csv")
    assert selection["selection_reason"] == "latest_valid_for_source"
    assert any(item.get("reason") == "empty_file" for item in selection["skipped_chain_files"])


def test_list_replay_chain_snapshots_filters_by_source(tmp_path):
    (tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-10-00+05-30.csv").write_text(
        "strikePrice,OPTION_TYP,lastPrice\n23000,CE,100\n",
        encoding="utf-8",
    )
    (tmp_path / "NIFTY_NSE_option_chain_snapshot_2026-03-20T12-12-00+05-30.csv").write_text(
        "strikePrice,OPTION_TYP,lastPrice\n23000,CE,101\n",
        encoding="utf-8",
    )

    chain_paths, skipped = list_replay_chain_snapshots(
        "NIFTY",
        replay_dir=str(tmp_path),
        source_label="ICICI",
    )

    assert len(chain_paths) == 1
    assert chain_paths[0].endswith("NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-10-00+05-30.csv")
    assert skipped == []


def test_load_option_chain_snapshot_raises_clear_error_for_empty_csv(tmp_path):
    empty_file = tmp_path / "NIFTY_ICICI_option_chain_snapshot_2026-03-20T12-10-00+05-30.csv"
    empty_file.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty or malformed"):
        load_option_chain_snapshot(str(empty_file))


def test_validate_snapshot_consistency_ignores_expiry_columns_for_timestamp_checks():
    spot_snapshot = {"timestamp": "2026-03-20T09:20:00+05:30"}
    chain = __import__("pandas").DataFrame(
        {
            "EXPIRY_DT": ["2026-03-27"],
            "strikePrice": [23000],
            "OPTION_TYP": ["CE"],
        }
    )

    result = validate_snapshot_consistency(spot_snapshot, chain)
    assert result["is_consistent"] is True
    assert result["warnings"] == []
