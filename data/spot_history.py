"""
Module: spot_history.py

Purpose:
    Persist and retrieve intraday spot observations as a continuous time-series.

Role in the System:
    Part of the data layer that accumulates a local spot history during live sessions.
    This provides a reliable fallback for outcome enrichment when external providers
    (yfinance) are unavailable or delayed.

Key Outputs:
    Append-only CSV files with (timestamp, spot, symbol) rows, one file per symbol per date.

Downstream Usage:
    Consumed by the signal-evaluation enrichment pipeline as a primary local source
    of realized spot paths, reducing dependence on yfinance backfills.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.market_data_policy import IST_TIMEZONE

logger = logging.getLogger(__name__)

SPOT_HISTORY_DIR = Path("data_store") / "spot_history"


def _safe_dir_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return -1


@lru_cache(maxsize=128)
def _cached_symbol_csv_files(symbol_dir_str: str, dir_mtime_ns: int) -> tuple[Path, ...]:
    # dir_mtime_ns participates in cache key to invalidate after appends.
    _ = dir_mtime_ns
    symbol_dir = Path(symbol_dir_str)
    return tuple(sorted(symbol_dir.glob("*.csv")))


@lru_cache(maxsize=4096)
def _cached_file_date_from_stem(stem: str):
    if "_" not in stem:
        return None
    candidate = stem.rsplit("_", 1)[-1]
    try:
        return pd.Timestamp(candidate).date()
    except Exception:
        return None


def _history_path(symbol: str, date: str, *, base_dir: Path = SPOT_HISTORY_DIR) -> Path:
    return base_dir / symbol.upper() / f"{symbol.upper()}_{date}.csv"


def append_spot_observation(
    symbol: str,
    spot: float,
    timestamp,
    *,
    base_dir: Path = SPOT_HISTORY_DIR,
) -> Path:
    """Append a single (timestamp, spot) row to the daily spot history file."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST_TIMEZONE)
    else:
        ts = ts.tz_convert(IST_TIMEZONE)

    date_str = ts.strftime("%Y-%m-%d")
    path = _history_path(symbol, date_str, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts_iso = ts.isoformat()
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,spot\n")
        f.write(f"{ts_iso},{round(float(spot), 4)}\n")

    return path


def load_spot_history(
    symbol: str,
    *,
    start_ts=None,
    end_ts=None,
    base_dir: Path = SPOT_HISTORY_DIR,
) -> pd.DataFrame:
    """Load local spot history for a symbol between start_ts and end_ts.

    Scans daily CSV files in the symbol's directory to build a contiguous
    (timestamp, spot) DataFrame.  Returns an empty frame if no local data exists.
    """
    symbol_dir = base_dir / symbol.upper()
    if not symbol_dir.exists():
        return pd.DataFrame(columns=["timestamp", "spot"])

    def _to_ist(ts_value):
        if ts_value is None:
            return None
        ts = pd.Timestamp(ts_value)
        if ts.tzinfo is None:
            return ts.tz_localize(IST_TIMEZONE)
        return ts.tz_convert(IST_TIMEZONE)

    start = _to_ist(start_ts)
    end = _to_ist(end_ts)

    candidate_files = list(_cached_symbol_csv_files(str(symbol_dir), _safe_dir_mtime_ns(symbol_dir)))
    if start is not None or end is not None:
        lower_date = (start.normalize() - pd.Timedelta(days=1)).date() if start is not None else None
        upper_date = (end.normalize() + pd.Timedelta(days=1)).date() if end is not None else None

        filtered: list[Path] = []
        for csv_file in candidate_files:
            file_date = _cached_file_date_from_stem(csv_file.stem)

            # If filename date cannot be parsed, keep file to avoid false negatives.
            if file_date is None:
                filtered.append(csv_file)
                continue
            if lower_date is not None and file_date < lower_date:
                continue
            if upper_date is not None and file_date > upper_date:
                continue
            filtered.append(csv_file)
        candidate_files = filtered

    frames: list[pd.DataFrame] = []
    for csv_file in candidate_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            logger.warning("Skipping corrupt spot history file: %s", csv_file)
            continue

        if df.empty or "timestamp" not in df.columns or "spot" not in df.columns:
            continue

        # Support mixed timestamp precision (e.g. seconds and fractional seconds)
        # across appenders while remaining deterministic for replay/evaluation.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        ).dt.tz_convert(IST_TIMEZONE)
        df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
        df = df.dropna(subset=["timestamp", "spot"])

        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "spot"])

    combined = pd.concat(frames, ignore_index=True)
    # Keep the latest appended row per timestamp so corrections/late writes win.
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return combined[["timestamp", "spot"]]
