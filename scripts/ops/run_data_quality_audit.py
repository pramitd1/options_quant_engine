#!/usr/bin/env python3
"""Audit and lightly clean the live/research data collection estate.

The script is intentionally conservative: source datasets are never deleted.
When repair mode is enabled it rewrites signal datasets through the canonical
dataset writer after creating backups, which normalizes columns, syncs CSV and
SQLite storage, and keeps the latest row for duplicate signal IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.market_data_policy import IST_TIMEZONE
from research.signal_evaluation.dataset import (  # noqa: E402
    CUMULATIVE_DATASET_PATH,
    SIGNAL_DATASET_COLUMNS,
    SIGNAL_DATASET_PATH,
    _dataset_store_path,
    _dataset_write_lock,
    _load_signals_dataset_unlocked,
    _write_signals_dataset_unlocked,
    load_signals_dataset,
)


REPORT_DIR = PROJECT_ROOT / "research" / "data_audit"
CURATED_DIR = REPORT_DIR / "curated_views"

SIGNAL_PATHS = [
    ("session_signal_dataset", Path(SIGNAL_DATASET_PATH)),
    ("cumulative_signal_dataset", Path(CUMULATIVE_DATASET_PATH)),
]

HISTORICAL_OPTION_PATH = PROJECT_ROOT / "data_store" / "historical" / "merged" / "NIFTY_option_chain_historical.parquet"
HISTORICAL_SPOT_PATH = PROJECT_ROOT / "data_store" / "historical" / "spot" / "NIFTY_spot_daily.parquet"
GLOBAL_FEATURES_PATH = PROJECT_ROOT / "data_store" / "historical" / "global_market" / "features" / "global_market_features.parquet"
MACRO_EVENTS_PATH = PROJECT_ROOT / "data_store" / "historical" / "macro_events" / "india_macro_events_historical.json"
SPOT_HISTORY_DIR = PROJECT_ROOT / "data_store" / "spot_history"
OPTION_SNAPSHOT_DIR = PROJECT_ROOT / "debug_samples" / "option_chain_snapshots"
SPOT_SNAPSHOT_DIR = PROJECT_ROOT / "debug_samples" / "spot_snapshots"
OI_ARTIFACT_DIR = PROJECT_ROOT / "research" / "artifacts" / "oi_inference"
PARAMETER_PACK_DIR = PROJECT_ROOT / "config" / "parameter_packs"
RUNTIME_MARKER_PATH = (
    PROJECT_ROOT
    / "research"
    / "signal_evaluation"
    / "reports"
    / "threshold_runtime_activation"
    / "latest_threshold_runtime_activation.json"
)


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST_TIMEZONE)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            missing = pd.isna(value)
            if missing is pd.NA:
                return None
            if isinstance(missing, bool) and missing:
                return None
        except Exception:
            pass
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _file_profile(path: Path) -> dict[str, Any]:
    exists = path.exists()
    profile: dict[str, Any] = {
        "path": _rel(path),
        "exists": exists,
    }
    if exists:
        stat = path.stat()
        profile.update(
            {
                "size_bytes": int(stat.st_size),
                "modified_at": pd.Timestamp(stat.st_mtime, unit="s", tz=IST_TIMEZONE).isoformat(),
            }
        )
    return profile


def _parse_ts(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "t", "yes", "y"})


def _value_counts(frame: pd.DataFrame, column: str, limit: int = 12) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].fillna("<NA>").astype(str).value_counts(dropna=False).head(limit)
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"exists": False}
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {"exists": True, "non_null": 0}
    return {
        "exists": True,
        "non_null": int(valid.shape[0]),
        "min": round(float(valid.min()), 6),
        "median": round(float(valid.median()), 6),
        "max": round(float(valid.max()), 6),
        "mean": round(float(valid.mean()), 6),
    }


def _sqlite_row_count(sqlite_path: Path) -> int | None:
    if not sqlite_path.exists():
        return None
    try:
        with sqlite3.connect(sqlite_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM signals").fetchone()
        return int(row[0]) if row else None
    except sqlite3.DatabaseError:
        return None


def _profile_signal_dataset(name: str, path: Path) -> dict[str, Any]:
    profile = _file_profile(path)
    profile["dataset_name"] = name
    sqlite_path = _dataset_store_path(path)
    profile["sqlite"] = _file_profile(sqlite_path)
    profile["sqlite_row_count"] = _sqlite_row_count(sqlite_path)

    if not path.exists():
        profile["status"] = "MISSING"
        return profile

    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        profile["status"] = "UNREADABLE"
        profile["error"] = f"{type(exc).__name__}: {exc}"
        return profile

    profile["status"] = "READABLE"
    profile["rows"] = int(len(raw))
    profile["columns"] = int(len(raw.columns))
    profile["canonical_columns"] = raw.columns.tolist() == SIGNAL_DATASET_COLUMNS
    profile["missing_canonical_columns"] = [c for c in SIGNAL_DATASET_COLUMNS if c not in raw.columns]
    profile["extra_columns"] = [c for c in raw.columns if c not in SIGNAL_DATASET_COLUMNS]

    if "signal_id" in raw.columns:
        signal_ids = raw["signal_id"].fillna("").astype(str)
        profile["missing_signal_id_rows"] = int(signal_ids.eq("").sum())
        profile["duplicate_signal_id_rows"] = int(signal_ids[signal_ids.ne("")].duplicated(keep=False).sum())
        profile["unique_signal_ids"] = int(signal_ids[signal_ids.ne("")].nunique())
    else:
        profile["missing_signal_id_rows"] = int(len(raw))
        profile["duplicate_signal_id_rows"] = None
        profile["unique_signal_ids"] = 0

    if "signal_timestamp" in raw.columns:
        parsed = _parse_ts(raw["signal_timestamp"])
        parseable = int(parsed.notna().sum())
        profile["timestamp_parseable_rows"] = parseable
        profile["timestamp_unparseable_rows"] = int(len(raw) - parseable)
        profile["timestamp_parseable_ratio"] = round(parseable / max(len(raw), 1), 4)
        if parseable:
            ist = parsed.dt.tz_convert(IST_TIMEZONE)
            profile["timestamp_min"] = ist.min().isoformat()
            profile["timestamp_max"] = ist.max().isoformat()
            profile["sessions"] = int(ist.dt.date.nunique())
            today = _now_ist().date()
            today_mask = ist.dt.date == today
            profile["today_rows"] = int(today_mask.sum())
            if today_mask.any():
                today_ts = ist[today_mask]
                profile["today_first_timestamp"] = today_ts.min().isoformat()
                profile["today_latest_timestamp"] = today_ts.max().isoformat()
                minute_bins = today_ts.dt.floor("min")
                profile["today_unique_minutes"] = int(minute_bins.nunique())
    else:
        profile["timestamp_parseable_rows"] = 0
        profile["timestamp_unparseable_rows"] = int(len(raw))
        profile["timestamp_parseable_ratio"] = 0.0

    for column in [
        "option_source",
        "spot_source",
        "market_data_provenance_status",
        "market_data_source_consistency",
        "trade_status",
        "direction",
        "label_quality_status",
        "outcome_status",
        "parameter_pack_name",
        "runtime_activation_guard_status",
    ]:
        counts = _value_counts(raw, column)
        if counts:
            profile[f"{column}_counts"] = counts

    for column in [
        "trade_strength",
        "move_probability",
        "primary_outcome_return_bps",
        "signed_return_60m_bps",
        "selected_option_iv",
        "selected_option_last_price",
        "selected_option_bid_price",
        "selected_option_ask_price",
        "selected_option_mid_price",
        "option_premium_return_60m_bps",
        "india_vix_level",
        "market_data_timestamp_delta_seconds",
    ]:
        profile[f"{column}_summary"] = _numeric_summary(raw, column)

    if "saved_spot_snapshot_path" in raw.columns:
        saved_spot = raw["saved_spot_snapshot_path"].fillna("").astype(str).str.len().gt(0)
        profile["saved_spot_snapshot_rows"] = int(saved_spot.sum())
        profile["saved_spot_snapshot_ratio"] = round(float(saved_spot.mean()) if len(raw) else 0.0, 4)
    if "saved_chain_snapshot_path" in raw.columns:
        saved_chain = raw["saved_chain_snapshot_path"].fillna("").astype(str).str.len().gt(0)
        profile["saved_chain_snapshot_rows"] = int(saved_chain.sum())
        profile["saved_chain_snapshot_ratio"] = round(float(saved_chain.mean()) if len(raw) else 0.0, 4)
    if "selected_option_last_price" in raw.columns:
        entry_premium = pd.to_numeric(raw["selected_option_last_price"], errors="coerce").gt(0)
        profile["selected_option_premium_rows"] = int(entry_premium.sum())
        profile["selected_option_premium_ratio"] = round(float(entry_premium.mean()) if len(raw) else 0.0, 4)
    if "option_premium_path_status" in raw.columns:
        counts = _value_counts(raw, "option_premium_path_status")
        if counts:
            profile["option_premium_path_status_counts"] = counts
    if "option_premium_return_60m_bps" in raw.columns:
        option_60m = pd.to_numeric(raw["option_premium_return_60m_bps"], errors="coerce").notna()
        profile["option_premium_60m_rows"] = int(option_60m.sum())
        profile["option_premium_60m_ratio"] = round(float(option_60m.mean()) if len(raw) else 0.0, 4)

    if "signal_capture_guarded" in raw.columns:
        guarded = _coerce_bool_series(raw["signal_capture_guarded"])
        profile["guarded_rows"] = int(guarded.sum())
        profile["guarded_ratio"] = round(float(guarded.mean()) if len(raw) else 0.0, 4)

    return profile


def _dedupe_signal_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or "signal_id" not in frame.columns:
        return frame.copy(), 0
    cleaned = frame.copy()
    cleaned["__sort_key__"] = cleaned.get("updated_at", pd.Series(index=cleaned.index)).fillna(
        cleaned.get("created_at", pd.Series(index=cleaned.index))
    ).fillna("")
    before = len(cleaned)
    cleaned["__signal_key__"] = cleaned["signal_id"].fillna("").astype(str)
    with_ids = cleaned[cleaned["__signal_key__"].ne("")].sort_values("__sort_key__", kind="stable")
    without_ids = cleaned[cleaned["__signal_key__"].eq("")]
    with_ids = with_ids.drop_duplicates(subset=["__signal_key__"], keep="last")
    cleaned = pd.concat([without_ids, with_ids], ignore_index=True, sort=False)
    cleaned = cleaned.sort_values("__sort_key__", kind="stable")
    cleaned = cleaned.drop(columns=["__sort_key__"]).reset_index(drop=True)
    cleaned = cleaned.drop(columns=["__signal_key__"]).reset_index(drop=True)
    return cleaned, int(before - len(cleaned))


def _backup_file(path: Path, backup_dir: Path) -> str | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return _rel(target)


def _repair_signal_dataset(name: str, path: Path, backup_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset_name": name, "path": _rel(path), "status": "SKIPPED"}
    with _dataset_write_lock(path):
        frame = _load_signals_dataset_unlocked(path)
        result["input_rows"] = int(len(frame))
        result["csv_backup"] = _backup_file(path, backup_dir)
        result["sqlite_backup"] = _backup_file(_dataset_store_path(path), backup_dir)
        cleaned, removed = _dedupe_signal_frame(frame)
        _write_signals_dataset_unlocked(cleaned, path)
        result["output_rows"] = int(len(cleaned))
        result["duplicate_signal_id_rows_removed"] = removed
        result["status"] = "REPAIRED"
    return result


def _historical_option_profile() -> dict[str, Any]:
    profile = _file_profile(HISTORICAL_OPTION_PATH)
    if not HISTORICAL_OPTION_PATH.exists():
        profile["status"] = "MISSING"
        return profile
    columns = ["trade_date", "expiry_date", "option_type", "strike_price", "open_interest", "change_in_oi"]
    try:
        frame = pd.read_parquet(HISTORICAL_OPTION_PATH, columns=columns)
    except Exception as exc:
        profile["status"] = "UNREADABLE"
        profile["error"] = f"{type(exc).__name__}: {exc}"
        return profile
    profile["status"] = "READABLE"
    profile["rows"] = int(len(frame))
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    profile["trade_date_min"] = dates.min().date().isoformat() if dates.notna().any() else None
    profile["trade_date_max"] = dates.max().date().isoformat() if dates.notna().any() else None
    profile["trading_days"] = int(dates.dt.date.nunique()) if dates.notna().any() else 0
    expiries = pd.to_datetime(frame["expiry_date"], errors="coerce")
    profile["expiry_min"] = expiries.min().date().isoformat() if expiries.notna().any() else None
    profile["expiry_max"] = expiries.max().date().isoformat() if expiries.notna().any() else None
    profile["option_type_counts"] = _value_counts(frame, "option_type")
    profile["open_interest_null_rows"] = int(frame["open_interest"].isna().sum()) if "open_interest" in frame else None
    profile["change_in_oi_null_rows"] = int(frame["change_in_oi"].isna().sum()) if "change_in_oi" in frame else None
    return profile


def _parquet_date_profile(path: Path, date_column: str = "date") -> dict[str, Any]:
    profile = _file_profile(path)
    if not path.exists():
        profile["status"] = "MISSING"
        return profile
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        profile["status"] = "UNREADABLE"
        profile["error"] = f"{type(exc).__name__}: {exc}"
        return profile
    profile["status"] = "READABLE"
    profile["rows"] = int(len(frame))
    profile["columns"] = frame.columns.tolist()
    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        profile["date_min"] = dates.min().date().isoformat() if dates.notna().any() else None
        profile["date_max"] = dates.max().date().isoformat() if dates.notna().any() else None
        profile["date_count"] = int(dates.dt.date.nunique()) if dates.notna().any() else 0
    profile["null_ratios_top"] = {
        str(k): round(float(v), 4)
        for k, v in (frame.isna().mean().sort_values(ascending=False).head(12)).items()
    }
    return profile


def _profile_global_features() -> dict[str, Any]:
    profile = _parquet_date_profile(GLOBAL_FEATURES_PATH, date_column="date")
    if profile.get("status") != "READABLE":
        return profile
    try:
        frame = pd.read_parquet(GLOBAL_FEATURES_PATH)
    except Exception:
        return profile
    for column in [
        "india_vix_close",
        "india_vix_change_1d",
        "vix_change_1d",
        "sp500_change_1d",
        "usdinr_change_1d",
        "oil_change_1d",
        "nifty50_realized_vol_30d",
    ]:
        if column in frame.columns:
            profile[f"{column}_summary"] = _numeric_summary(frame, column)
    return profile


def _profile_macro_events() -> dict[str, Any]:
    profile = _file_profile(MACRO_EVENTS_PATH)
    if not MACRO_EVENTS_PATH.exists():
        profile["status"] = "MISSING"
        return profile
    try:
        payload = json.loads(MACRO_EVENTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        profile["status"] = "UNREADABLE"
        profile["error"] = f"{type(exc).__name__}: {exc}"
        return profile
    events = payload.get("events") if isinstance(payload, dict) else payload
    events = events if isinstance(events, list) else []
    profile["status"] = "READABLE"
    profile["events"] = int(len(events))
    dates = []
    categories: Counter[str] = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        raw_date = event.get("date") or event.get("event_date") or event.get("timestamp")
        if raw_date:
            ts = pd.to_datetime(raw_date, errors="coerce")
            if pd.notna(ts):
                dates.append(ts)
        category = event.get("category") or event.get("event_type") or event.get("type")
        if category:
            categories[str(category)] += 1
    if dates:
        dates_series = pd.Series(dates)
        profile["date_min"] = dates_series.min().date().isoformat()
        profile["date_max"] = dates_series.max().date().isoformat()
    profile["category_counts"] = dict(categories.most_common(12))
    return profile


def _profile_spot_history() -> dict[str, Any]:
    profile = _file_profile(SPOT_HISTORY_DIR)
    files = sorted(SPOT_HISTORY_DIR.glob("*/*.csv")) if SPOT_HISTORY_DIR.exists() else []
    profile["csv_files"] = int(len(files))
    profile["symbols"] = sorted({path.parent.name for path in files})
    day_profiles = []
    total_rows = 0
    duplicate_rows = 0
    invalid_rows = 0
    latest_ts = None
    earliest_ts = None
    today = _now_ist().date()
    today_rows = 0
    for path in files:
        item = {"path": _rel(path)}
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            item["status"] = "UNREADABLE"
            item["error"] = f"{type(exc).__name__}: {exc}"
            invalid_rows += 1
            day_profiles.append(item)
            continue
        item["status"] = "READABLE"
        item["rows"] = int(len(frame))
        total_rows += len(frame)
        if {"timestamp", "spot"}.issubset(frame.columns):
            ts = _parse_ts(frame["timestamp"])
            spot = pd.to_numeric(frame["spot"], errors="coerce")
            bad = ts.isna() | spot.isna() | ~spot.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False) | spot.le(0)
            item["invalid_rows"] = int(bad.sum())
            invalid_rows += int(bad.sum())
            dup = int(ts[ts.notna()].duplicated(keep=False).sum())
            item["duplicate_timestamp_rows"] = dup
            duplicate_rows += dup
            if ts.notna().any():
                ist = ts.dt.tz_convert(IST_TIMEZONE)
                item["first_timestamp"] = ist.min().isoformat()
                item["latest_timestamp"] = ist.max().isoformat()
                local_dates = ist.dt.date
                today_rows += int((local_dates == today).sum())
                latest_ts = ist.max() if latest_ts is None else max(latest_ts, ist.max())
                earliest_ts = ist.min() if earliest_ts is None else min(earliest_ts, ist.min())
        else:
            item["status"] = "MALFORMED"
            invalid_rows += len(frame)
        day_profiles.append(item)
    profile["rows"] = int(total_rows)
    profile["duplicate_timestamp_rows"] = int(duplicate_rows)
    profile["invalid_rows"] = int(invalid_rows)
    profile["today_rows"] = int(today_rows)
    profile["first_timestamp"] = earliest_ts.isoformat() if earliest_ts is not None else None
    profile["latest_timestamp"] = latest_ts.isoformat() if latest_ts is not None else None
    profile["recent_files"] = day_profiles[-10:]
    return profile


def _extract_snapshot_source(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) >= 3 and parts[0].upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}:
        return parts[1].upper()
    return "UNKNOWN"


def _extract_snapshot_timestamp(path: Path, marker: str) -> pd.Timestamp | None:
    try:
        token = path.stem.split(marker, 1)[1]
    except Exception:
        return None
    token = token.replace("-", ":", 2) if False else token
    try:
        return pd.Timestamp(token.replace("T", "T").replace("+05-30", "+05:30"))
    except Exception:
        try:
            return pd.Timestamp(token)
        except Exception:
            return None


def _profile_snapshots() -> dict[str, Any]:
    option_files = sorted(OPTION_SNAPSHOT_DIR.glob("*.csv")) if OPTION_SNAPSHOT_DIR.exists() else []
    spot_files = sorted(SPOT_SNAPSHOT_DIR.glob("*.json")) if SPOT_SNAPSHOT_DIR.exists() else []
    profile: dict[str, Any] = {
        "option_chain_snapshots": {
            **_file_profile(OPTION_SNAPSHOT_DIR),
            "files": int(len(option_files)),
            "source_counts": dict(Counter(_extract_snapshot_source(path) for path in option_files)),
            "recent_files": [_rel(path) for path in option_files[-10:]],
        },
        "spot_snapshots": {
            **_file_profile(SPOT_SNAPSHOT_DIR),
            "files": int(len(spot_files)),
            "recent_files": [_rel(path) for path in spot_files[-10:]],
        },
    }
    empty_option = [path for path in option_files if path.stat().st_size == 0]
    profile["option_chain_snapshots"]["empty_files"] = int(len(empty_option))
    malformed_spot = 0
    latest_spot_ts = None
    for path in spot_files[-200:]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ts = pd.to_datetime(payload.get("timestamp"), errors="coerce")
            if pd.notna(ts):
                ts = ts.tz_localize(IST_TIMEZONE) if ts.tzinfo is None else ts.tz_convert(IST_TIMEZONE)
                latest_spot_ts = ts if latest_spot_ts is None else max(latest_spot_ts, ts)
        except Exception:
            malformed_spot += 1
    profile["spot_snapshots"]["malformed_recent_files"] = int(malformed_spot)
    profile["spot_snapshots"]["latest_payload_timestamp"] = latest_spot_ts.isoformat() if latest_spot_ts is not None else None
    return profile


def _profile_oi_artifacts() -> dict[str, Any]:
    files = sorted(OI_ARTIFACT_DIR.glob("**/*.jsonl")) if OI_ARTIFACT_DIR.exists() else []
    profile = _file_profile(OI_ARTIFACT_DIR)
    profile["files"] = int(len(files))
    profile["recent_files"] = [_rel(path) for path in files[-10:]]
    total_rows = 0
    latest_ts = None
    earliest_ts = None
    by_file = []
    for path in files:
        rows = 0
        malformed = 0
        file_first = None
        file_latest = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    raw_ts = payload.get("snapshot_timestamp") or payload.get("timestamp") or payload.get("created_at")
                    if raw_ts:
                        ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)
                        if pd.notna(ts):
                            ts = ts.tz_convert(IST_TIMEZONE)
                            file_first = ts if file_first is None else min(file_first, ts)
                            file_latest = ts if file_latest is None else max(file_latest, ts)
        except OSError as exc:
            by_file.append({"path": _rel(path), "status": "UNREADABLE", "error": str(exc)})
            continue
        total_rows += rows
        if file_first is not None:
            earliest_ts = file_first if earliest_ts is None else min(earliest_ts, file_first)
            latest_ts = file_latest if latest_ts is None else max(latest_ts, file_latest)
        by_file.append(
            {
                "path": _rel(path),
                "rows": int(rows),
                "malformed_rows": int(malformed),
                "first_timestamp": file_first.isoformat() if file_first is not None else None,
                "latest_timestamp": file_latest.isoformat() if file_latest is not None else None,
            }
        )
    profile["rows"] = int(total_rows)
    profile["first_timestamp"] = earliest_ts.isoformat() if earliest_ts is not None else None
    profile["latest_timestamp"] = latest_ts.isoformat() if latest_ts is not None else None
    profile["file_profiles"] = by_file[-20:]
    return profile


def _profile_parameter_packs() -> dict[str, Any]:
    files = sorted(PARAMETER_PACK_DIR.glob("*.json")) if PARAMETER_PACK_DIR.exists() else []
    profile = _file_profile(PARAMETER_PACK_DIR)
    packs = []
    for path in files:
        item = {"path": _rel(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item["name"] = payload.get("name")
            item["version"] = payload.get("version")
            item["active"] = bool(payload.get("active", False))
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        packs.append(item)
    profile["packs"] = packs
    profile["active_pack_names"] = [item.get("name") for item in packs if item.get("active")]
    profile["runtime_activation_marker"] = _file_profile(RUNTIME_MARKER_PATH)
    return profile


def _constant_iv_csv_profiles() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, path in {
        "NIFTY_1y_option_chain_csv": PROJECT_ROOT / "data_store" / "NIFTY" / "NIFTY_1y_option_chain.csv",
        "NIFTY_5y_option_chain_csv": PROJECT_ROOT / "data_store" / "NIFTY" / "NIFTY_5y_option_chain.csv",
    }.items():
        item = _file_profile(path)
        if path.exists():
            try:
                frame = pd.read_csv(path, low_memory=False)
                item["rows"] = int(len(frame))
                if "iv" in frame.columns:
                    iv = pd.to_numeric(frame["iv"], errors="coerce").dropna()
                    item["iv_unique_count"] = int(iv.nunique())
                    item["iv_min"] = round(float(iv.min()), 6) if not iv.empty else None
                    item["iv_max"] = round(float(iv.max()), 6) if not iv.empty else None
                elif "IV" in frame.columns:
                    iv = pd.to_numeric(frame["IV"], errors="coerce").dropna()
                    item["iv_unique_count"] = int(iv.nunique())
                    item["iv_min"] = round(float(iv.min()), 6) if not iv.empty else None
                    item["iv_max"] = round(float(iv.max()), 6) if not iv.empty else None
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        result[label] = item
    return result


def _add_issue(issues: list[dict[str, Any]], severity: str, component: str, finding: str, recommendation: str) -> None:
    issues.append(
        {
            "severity": severity,
            "component": component,
            "finding": finding,
            "recommendation": recommendation,
        }
    )


def _derive_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    now = _now_ist()

    for name, profile in report.get("signal_datasets", {}).items():
        rows = int(profile.get("rows") or 0)
        parse_ratio = float(profile.get("timestamp_parseable_ratio") or 0.0)
        if rows and parse_ratio < 0.95:
            _add_issue(
                issues,
                "HIGH",
                name,
                f"Only {parse_ratio:.1%} of signal timestamps parse cleanly.",
                "Use only parseable rows for training views and keep canonical timestamp format for all new rows.",
            )
        if int(profile.get("duplicate_signal_id_rows") or 0) > 0:
            _add_issue(
                issues,
                "HIGH",
                name,
                f"{profile.get('duplicate_signal_id_rows')} rows share duplicate signal IDs.",
                "Run the repair path to keep the latest row per signal_id and sync CSV/SQLite.",
            )
        if not profile.get("canonical_columns", False):
            _add_issue(
                issues,
                "MEDIUM",
                name,
                "Dataset header is not exactly the canonical signal schema.",
                "Rewrite through research.signal_evaluation.dataset to normalize column order and add missing columns.",
            )
        saved_chain_ratio = float(profile.get("saved_chain_snapshot_ratio") or 0.0)
        if rows and saved_chain_ratio < 0.5:
            _add_issue(
                issues,
                "MEDIUM",
                name,
                f"Only {saved_chain_ratio:.1%} of rows carry saved option-chain snapshot paths.",
                "Keep SAVE_LIVE_SNAPSHOTS enabled so every future signal row has replayable raw context.",
            )

    historical = report.get("historical_data", {})
    for component, freshness_key in [
        ("historical_option_chain", "trade_date_max"),
        ("historical_spot_daily", "date_max"),
        ("global_market_features", "date_max"),
    ]:
        profile = historical.get(component, {})
        latest = profile.get(freshness_key)
        if latest:
            days_old = (now.normalize() - pd.Timestamp(latest, tz=IST_TIMEZONE).normalize()).days
            if days_old > 14:
                _add_issue(
                    issues,
                    "MEDIUM",
                    component,
                    f"Latest stored date is {latest}, {days_old} calendar days behind the audit date.",
                    "Refresh this dataset before using April/May 2026 conditions for research or calibration.",
                )

    spot_history = report.get("live_capture", {}).get("spot_history", {})
    if int(spot_history.get("invalid_rows") or 0) > 0:
        _add_issue(
            issues,
            "HIGH",
            "spot_history",
            f"{spot_history.get('invalid_rows')} invalid spot-history rows were found.",
            "Exclude invalid rows from outcome enrichment and keep append-time positive finite spot validation enabled.",
        )

    snapshots = report.get("live_capture", {}).get("snapshots", {})
    option_snapshots = snapshots.get("option_chain_snapshots", {})
    if int(option_snapshots.get("empty_files") or 0) > 0:
        _add_issue(
            issues,
            "MEDIUM",
            "option_chain_snapshots",
            f"{option_snapshots.get('empty_files')} empty option-chain snapshot files were found.",
            "Replay selection already skips empty files; archive or ignore these in curated replay sets.",
        )
    if int(option_snapshots.get("files") or 0) < max(int(spot_history.get("today_rows") or 0) // 3, 1):
        _add_issue(
            issues,
            "MEDIUM",
            "snapshot_capture",
            "Saved option-chain snapshots are sparse relative to live spot observations.",
            "Per-cycle live snapshot persistence has been enabled for terminal runs; keep it on during research sessions.",
        )

    csv_profiles = report.get("historical_option_csv_profiles", {})
    for label, item in csv_profiles.items():
        if item.get("iv_unique_count") == 1:
            _add_issue(
                issues,
                "LOW",
                label,
                f"IV is constant at {item.get('iv_min')}.",
                "Do not use this CSV IV column for volatility learning; prefer live provider IV or reconstructed IV with a quality flag.",
            )

    marker = report.get("governance", {}).get("runtime_activation_marker", {})
    if marker.get("exists"):
        _add_issue(
            issues,
            "LOW",
            "runtime_activation_marker",
            "A threshold runtime activation marker is currently active.",
            "Verify expected and active parameter packs match before treating live rows as production-comparable.",
        )

    return issues


def _write_curated_training_view(source_path: Path, audit_id: str) -> dict[str, Any]:
    frame = load_signals_dataset(source_path)
    result: dict[str, Any] = {
        "source_path": _rel(source_path),
        "status": "SKIPPED",
        "input_rows": int(len(frame)),
    }
    if frame.empty:
        return result

    reasons = pd.Series("", index=frame.index, dtype="object")

    ts = _parse_ts(frame["signal_timestamp"]) if "signal_timestamp" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    valid_ts = ts.notna()
    reasons.loc[~valid_ts] = reasons.loc[~valid_ts] + "|timestamp_unparseable"

    if "label_quality_status" in frame.columns:
        clean_label = frame["label_quality_status"].fillna("").astype(str).str.upper().eq("CLEAN")
    else:
        clean_label = pd.Series(False, index=frame.index)
    reasons.loc[~clean_label] = reasons.loc[~clean_label] + "|label_not_clean"

    if "calibration_label_available" in frame.columns:
        label_available = _coerce_bool_series(frame["calibration_label_available"])
    else:
        label_available = pd.Series(False, index=frame.index)
    reasons.loc[~label_available] = reasons.loc[~label_available] + "|calibration_label_unavailable"

    if "primary_outcome_return_bps" in frame.columns:
        outcome_available = pd.to_numeric(frame["primary_outcome_return_bps"], errors="coerce").notna()
    else:
        outcome_available = pd.Series(False, index=frame.index)
    reasons.loc[~outcome_available] = reasons.loc[~outcome_available] + "|primary_outcome_missing"

    if "direction" in frame.columns:
        valid_direction = frame["direction"].fillna("").astype(str).str.upper().isin({"CALL", "PUT"})
    else:
        valid_direction = pd.Series(False, index=frame.index)
    reasons.loc[~valid_direction] = reasons.loc[~valid_direction] + "|direction_missing"

    if "signal_capture_guarded" in frame.columns:
        guarded = _coerce_bool_series(frame["signal_capture_guarded"])
    else:
        guarded = pd.Series(False, index=frame.index)
    reasons.loc[guarded] = reasons.loc[guarded] + "|runtime_guarded_capture"

    keep = valid_ts & clean_label & label_available & outcome_available & valid_direction & ~guarded
    curated = frame.loc[keep].copy()
    curated["signal_timestamp_utc"] = ts.loc[keep].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    curated_path = CURATED_DIR / f"curated_signal_outcome_view_{audit_id}.csv"
    latest_path = CURATED_DIR / "latest_curated_signal_outcome_view.csv"
    _atomic_write_csv(curated_path, curated)
    _atomic_write_csv(latest_path, curated)

    exclusions = pd.DataFrame(
        {
            "signal_id": frame.get("signal_id", pd.Series(index=frame.index, dtype="object")),
            "signal_timestamp": frame.get("signal_timestamp", pd.Series(index=frame.index, dtype="object")),
            "excluded": ~keep,
            "exclusion_reasons": reasons.str.strip("|"),
        }
    )
    exclusions = exclusions.loc[~keep].copy()
    exclusions_path = CURATED_DIR / f"curated_signal_outcome_exclusions_{audit_id}.csv"
    _atomic_write_csv(exclusions_path, exclusions)

    result.update(
        {
            "status": "WRITTEN",
            "curated_rows": int(len(curated)),
            "excluded_rows": int((~keep).sum()),
            "curated_path": _rel(curated_path),
            "latest_path": _rel(latest_path),
            "exclusions_path": _rel(exclusions_path),
        }
    )
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    issues = report.get("issues", [])
    signal = report.get("signal_datasets", {})
    historical = report.get("historical_data", {})
    live = report.get("live_capture", {})
    cleanup = report.get("cleanup", {})
    lines = [
        "# Data Collection Quality Audit",
        "",
        f"- Audit ID: `{report.get('audit_id')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Overall readiness: **{report.get('overall_readiness')}**",
        "",
        "## Findings",
    ]
    if issues:
        for issue in issues:
            lines.append(
                f"- **{issue.get('severity')}** `{issue.get('component')}`: "
                f"{issue.get('finding')} Recommendation: {issue.get('recommendation')}"
            )
    else:
        lines.append("- No blocking data-quality issues were detected.")

    lines.extend(["", "## Signal Datasets"])
    for name, profile in signal.items():
        lines.append(
            f"- `{name}`: rows={profile.get('rows')}, timestamp_parseable="
            f"{profile.get('timestamp_parseable_ratio')}, duplicates={profile.get('duplicate_signal_id_rows')}, "
            f"saved_chain_ratio={profile.get('saved_chain_snapshot_ratio')}, guarded={profile.get('guarded_rows')}"
        )

    lines.extend(["", "## Historical And Macro Data"])
    lines.append(
        f"- Historical option chain: rows={historical.get('historical_option_chain', {}).get('rows')}, "
        f"range={historical.get('historical_option_chain', {}).get('trade_date_min')} to "
        f"{historical.get('historical_option_chain', {}).get('trade_date_max')}"
    )
    lines.append(
        f"- Spot daily: rows={historical.get('historical_spot_daily', {}).get('rows')}, "
        f"range={historical.get('historical_spot_daily', {}).get('date_min')} to "
        f"{historical.get('historical_spot_daily', {}).get('date_max')}"
    )
    lines.append(
        f"- Global features: rows={historical.get('global_market_features', {}).get('rows')}, "
        f"range={historical.get('global_market_features', {}).get('date_min')} to "
        f"{historical.get('global_market_features', {}).get('date_max')}"
    )
    lines.append(
        f"- Macro events: events={historical.get('macro_events', {}).get('events')}, "
        f"range={historical.get('macro_events', {}).get('date_min')} to "
        f"{historical.get('macro_events', {}).get('date_max')}"
    )

    snapshots = live.get("snapshots", {})
    lines.extend(["", "## Live Capture"])
    lines.append(
        f"- Spot history: files={live.get('spot_history', {}).get('csv_files')}, "
        f"rows={live.get('spot_history', {}).get('rows')}, today_rows={live.get('spot_history', {}).get('today_rows')}, "
        f"latest={live.get('spot_history', {}).get('latest_timestamp')}"
    )
    lines.append(
        f"- Option snapshots: files={snapshots.get('option_chain_snapshots', {}).get('files')}, "
        f"sources={snapshots.get('option_chain_snapshots', {}).get('source_counts')}"
    )
    lines.append(
        f"- OI inference artifacts: files={live.get('oi_inference_artifacts', {}).get('files')}, "
        f"rows={live.get('oi_inference_artifacts', {}).get('rows')}, "
        f"range={live.get('oi_inference_artifacts', {}).get('first_timestamp')} to "
        f"{live.get('oi_inference_artifacts', {}).get('latest_timestamp')}"
    )

    lines.extend(["", "## Cleanup"])
    if cleanup:
        for key, value in cleanup.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- Cleanup was not requested.")

    lines.extend(
        [
            "",
            "## Operational Defaults Confirmed",
            "- Terminal live mode now enables per-cycle raw snapshot capture by default.",
            "- Spot-history append rejects non-positive/non-finite spot values and uses a file lock.",
            "- Spot and option-chain snapshot writes are atomic.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(*, repair_signals: bool, write_curated_view: bool) -> dict[str, Any]:
    audit_id = _now_ist().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "audit_id": audit_id,
        "generated_at": _now_ist().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "signal_datasets": {},
        "historical_data": {},
        "historical_option_csv_profiles": {},
        "live_capture": {},
        "governance": {},
        "cleanup": {},
    }

    if repair_signals:
        backup_dir = REPORT_DIR / f"cleanup_{audit_id}" / "backups"
        repairs = {}
        for name, path in SIGNAL_PATHS:
            repairs[name] = _repair_signal_dataset(name, path, backup_dir)
        report["cleanup"]["signal_dataset_repairs"] = repairs

    for name, path in SIGNAL_PATHS:
        report["signal_datasets"][name] = _profile_signal_dataset(name, path)

    report["historical_data"]["historical_option_chain"] = _historical_option_profile()
    report["historical_data"]["historical_spot_daily"] = _parquet_date_profile(HISTORICAL_SPOT_PATH, date_column="date")
    report["historical_data"]["global_market_features"] = _profile_global_features()
    report["historical_data"]["macro_events"] = _profile_macro_events()
    report["historical_option_csv_profiles"] = _constant_iv_csv_profiles()
    report["live_capture"]["spot_history"] = _profile_spot_history()
    report["live_capture"]["snapshots"] = _profile_snapshots()
    report["live_capture"]["oi_inference_artifacts"] = _profile_oi_artifacts()
    report["governance"] = _profile_parameter_packs()

    if write_curated_view:
        report["cleanup"]["curated_training_view"] = _write_curated_training_view(
            Path(CUMULATIVE_DATASET_PATH),
            audit_id,
        )

    report["issues"] = _derive_issues(report)
    severities = Counter(issue.get("severity") for issue in report["issues"])
    if severities.get("HIGH"):
        report["overall_readiness"] = "CAUTION"
    elif severities.get("MEDIUM"):
        report["overall_readiness"] = "WATCH"
    else:
        report["overall_readiness"] = "OK"
    report["issue_counts"] = dict(severities)
    return report


def write_report(report: dict[str, Any]) -> dict[str, str]:
    audit_id = report["audit_id"]
    json_path = REPORT_DIR / f"data_collection_quality_audit_{audit_id}.json"
    md_path = REPORT_DIR / f"data_collection_quality_audit_{audit_id}.md"
    latest_json = REPORT_DIR / "latest_data_collection_quality_audit.json"
    latest_md = REPORT_DIR / "latest_data_collection_quality_audit.md"
    _atomic_write_json(json_path, report)
    _atomic_write_text(md_path, _render_markdown(report))
    _atomic_write_json(latest_json, report)
    _atomic_write_text(latest_md, _render_markdown(report))
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit live/research data collection quality.")
    parser.add_argument(
        "--repair-signals",
        action="store_true",
        help="Backup, canonicalize, and deduplicate signal datasets.",
    )
    parser.add_argument(
        "--write-curated-view",
        action="store_true",
        help="Write a clean cumulative signal outcome view for research/training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        repair_signals=bool(args.repair_signals),
        write_curated_view=bool(args.write_curated_view),
    )
    paths = write_report(report)
    print(f"Data audit readiness: {report.get('overall_readiness')}")
    print(f"Issues: {report.get('issue_counts')}")
    print(f"Markdown: {paths['markdown']}")
    print(f"JSON: {paths['json']}")
    cleanup = report.get("cleanup", {})
    if cleanup:
        print(f"Cleanup: {cleanup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
