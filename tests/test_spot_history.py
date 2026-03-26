"""Tests for the local spot history persistence layer."""
from __future__ import annotations

import pandas as pd
import pytest

from data.spot_history import append_spot_observation, load_spot_history


class TestAppendSpotObservation:
    def test_creates_daily_csv_with_header(self, tmp_path):
        path = append_spot_observation(
            "NIFTY", 23100.5, "2026-03-16T12:30:00+05:30", base_dir=tmp_path
        )
        assert path.exists()
        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "timestamp,spot"
        assert "23100.5" in lines[1]

    def test_appends_multiple_observations(self, tmp_path):
        append_spot_observation("NIFTY", 23100.0, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23110.0, "2026-03-16T12:35:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23120.0, "2026-03-16T12:40:00+05:30", base_dir=tmp_path)

        df = load_spot_history("NIFTY", base_dir=tmp_path)
        assert len(df) == 3
        assert list(df.columns) == ["timestamp", "spot"]
        assert df["spot"].tolist() == [23100.0, 23110.0, 23120.0]

    def test_separate_files_per_date(self, tmp_path):
        append_spot_observation("NIFTY", 23100.0, "2026-03-15T14:00:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23200.0, "2026-03-16T10:00:00+05:30", base_dir=tmp_path)

        files = list((tmp_path / "NIFTY").glob("*.csv"))
        assert len(files) == 2

    def test_symbol_case_insensitive(self, tmp_path):
        append_spot_observation("nifty", 23100.0, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)
        df = load_spot_history("Nifty", base_dir=tmp_path)
        assert len(df) == 1


class TestLoadSpotHistory:
    def test_empty_when_no_data(self, tmp_path):
        df = load_spot_history("NIFTY", base_dir=tmp_path)
        assert df.empty
        assert list(df.columns) == ["timestamp", "spot"]

    def test_filters_by_time_range(self, tmp_path):
        for i in range(6):
            ts = f"2026-03-16T{12+i}:00:00+05:30"
            append_spot_observation("NIFTY", 23000.0 + i * 10, ts, base_dir=tmp_path)

        start = pd.Timestamp("2026-03-16T13:00:00+05:30")
        end = pd.Timestamp("2026-03-16T15:00:00+05:30")
        df = load_spot_history("NIFTY", start_ts=start, end_ts=end, base_dir=tmp_path)
        assert len(df) == 3

    def test_deduplicates_timestamps(self, tmp_path):
        append_spot_observation("NIFTY", 23100.0, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23100.5, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)

        df = load_spot_history("NIFTY", base_dir=tmp_path)
        assert len(df) == 1

    def test_dedup_guard_keeps_latest_spot_for_same_timestamp(self, tmp_path):
        append_spot_observation("NIFTY", 23100.0, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23150.0, "2026-03-16T12:30:00+05:30", base_dir=tmp_path)

        df = load_spot_history("NIFTY", base_dir=tmp_path)
        assert len(df) == 1
        assert df["spot"].iloc[0] == 23150.0

    def test_spans_multiple_dates(self, tmp_path):
        append_spot_observation("NIFTY", 23100.0, "2026-03-15T15:00:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23200.0, "2026-03-16T09:15:00+05:30", base_dir=tmp_path)
        append_spot_observation("NIFTY", 23250.0, "2026-03-16T10:00:00+05:30", base_dir=tmp_path)

        df = load_spot_history("NIFTY", base_dir=tmp_path)
        assert len(df) == 3
        assert df["spot"].iloc[0] == 23100.0
        assert df["spot"].iloc[-1] == 23250.0

    def test_dedupes_duplicate_timestamps_across_files_in_window(self, tmp_path):
        symbol_dir = tmp_path / "NIFTY"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        day_one = symbol_dir / "NIFTY_2026-03-15.csv"
        day_two = symbol_dir / "NIFTY_2026-03-16.csv"

        day_one.write_text(
            "\n".join(
                [
                    "timestamp,spot",
                    "2026-03-15T15:20:00+05:30,23090.0",
                    "2026-03-16T09:15:00+05:30,23100.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        day_two.write_text(
            "\n".join(
                [
                    "timestamp,spot",
                    "2026-03-16T09:15:00+05:30,23125.0",
                    "2026-03-16T09:20:00+05:30,23140.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        start = pd.Timestamp("2026-03-16T09:15:00+05:30")
        end = pd.Timestamp("2026-03-16T09:20:00+05:30")
        df = load_spot_history("NIFTY", start_ts=start, end_ts=end, base_dir=tmp_path)

        assert len(df) == 2
        assert df["timestamp"].iloc[0].isoformat() == "2026-03-16T09:15:00+05:30"
        assert df["spot"].iloc[0] == 23125.0
        assert df["timestamp"].iloc[1].isoformat() == "2026-03-16T09:20:00+05:30"
