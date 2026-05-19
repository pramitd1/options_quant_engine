"""Offline Heston diagnostics backfill from saved option-chain snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    BASE_DIR,
    HESTON_CALIBRATION_ERROR_REJECT,
    HESTON_CALIBRATION_MAX_ROWS,
    HESTON_CALIBRATION_MIN_ROWS,
    HESTON_CALIBRATION_TIMEOUT_SECONDS,
)
from data.replay_loader import load_option_chain_snapshot
from models.heston.heston_features import HESTON_FEATURE_COLUMNS, build_heston_research_features
from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset, write_signals_dataset
from research.signal_evaluation.pcr_backfill import DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR


OFFLINE_HESTON_BACKFILL_VERSION = "offline_heston_backfill_v1"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(number):
        return None
    return number


def _resolve_existing_path(path: Any, *, snapshot_dir: str | Path | None = None) -> Path | None:
    raw = _safe_text(path).strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path(BASE_DIR) / candidate)
        if snapshot_dir is not None:
            candidates.append(Path(snapshot_dir) / candidate.name)
    for item in candidates:
        if item.exists():
            return item
    return None


def _row_spot(row: pd.Series) -> float | None:
    for field in ("spot_at_signal", "spot", "underlying_spot", "index_spot"):
        value = _safe_float(row.get(field))
        if value is not None and value > 0:
            return value
    return None


def _row_option_type(row: pd.Series) -> str | None:
    option_type = _safe_text(row.get("option_type")).upper().strip()
    if option_type in {"CE", "PE", "CALL", "PUT", "C", "P"}:
        return option_type
    direction = _safe_text(row.get("direction")).upper().strip()
    if direction in {"CALL", "CE", "UP", "BULLISH"}:
        return "CE"
    if direction in {"PUT", "PE", "DOWN", "BEARISH"}:
        return "PE"
    return None


def _already_backfilled(row: pd.Series) -> bool:
    status = _safe_text(row.get("heston_calibration_status")).upper().strip()
    if status in {"", "DISABLED", "PENDING_SELECTION"}:
        return False
    diagnostics = _safe_text(row.get("heston_diagnostics_json"))
    if OFFLINE_HESTON_BACKFILL_VERSION in diagnostics:
        return True
    return status in {"CALIBRATED", "REJECTED"}


def _with_offline_diagnostics(features: dict[str, Any], *, row: pd.Series, chain_path: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any]
    try:
        diagnostics = json.loads(_safe_text(features.get("heston_diagnostics_json"))) or {}
    except Exception:
        diagnostics = {}
    diagnostics.update(
        {
            "backfill_version": OFFLINE_HESTON_BACKFILL_VERSION,
            "backfill_source": "saved_option_chain_snapshot",
            "saved_chain_snapshot_path": str(chain_path),
            "signal_id": row.get("signal_id"),
            "signal_timestamp": row.get("signal_timestamp"),
            "live_trade_decision_unchanged": True,
        }
    )
    updated = dict(features)
    updated["heston_diagnostics_json"] = json.dumps(diagnostics, sort_keys=True, default=str)
    return updated


def enrich_heston_research_features_from_snapshots(
    frame: pd.DataFrame,
    *,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    min_rows: int = HESTON_CALIBRATION_MIN_ROWS,
    max_rows: int = HESTON_CALIBRATION_MAX_ROWS,
    reject_error: float = HESTON_CALIBRATION_ERROR_REJECT,
    timeout_seconds: float = HESTON_CALIBRATION_TIMEOUT_SECONDS,
    force: bool = False,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Populate Heston diagnostics by replaying saved option-chain snapshots."""

    working = pd.DataFrame(frame).copy()
    for column in HESTON_FEATURE_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA
        else:
            working[column] = working[column].astype("object")

    summary: dict[str, Any] = {
        "backfill_version": OFFLINE_HESTON_BACKFILL_VERSION,
        "rows_seen": int(len(working)),
        "rows_considered": 0,
        "rows_updated": 0,
        "rows_calibrated": 0,
        "rows_rejected": 0,
        "rows_failed": 0,
        "rows_skipped_existing": 0,
        "rows_missing_spot": 0,
        "rows_missing_chain_path": 0,
        "rows_missing_chain_file": 0,
        "snapshot_read_failures": 0,
        "limit": int(limit) if limit is not None else None,
    }
    if working.empty:
        return working, summary

    chain_cache: dict[str, pd.DataFrame] = {}
    processed = 0
    for idx, row in working.iterrows():
        if limit is not None and processed >= int(limit):
            break
        summary["rows_considered"] += 1

        if not force and _already_backfilled(row):
            summary["rows_skipped_existing"] += 1
            continue

        chain_path = _resolve_existing_path(row.get("saved_chain_snapshot_path"), snapshot_dir=snapshot_dir)
        if chain_path is None:
            if _safe_text(row.get("saved_chain_snapshot_path")):
                summary["rows_missing_chain_file"] += 1
            else:
                summary["rows_missing_chain_path"] += 1
            continue

        spot = _row_spot(row)
        if spot is None:
            summary["rows_missing_spot"] += 1
            continue

        key = str(chain_path)
        if key not in chain_cache:
            try:
                chain_cache[key] = load_option_chain_snapshot(key)
            except Exception:
                chain_cache[key] = pd.DataFrame()
                summary["snapshot_read_failures"] += 1
        option_chain = chain_cache[key]
        if option_chain.empty:
            summary["rows_failed"] += 1
            continue

        features = build_heston_research_features(
            option_chain,
            spot=spot,
            selected_strike=row.get("strike"),
            selected_option_type=_row_option_type(row),
            selected_expiry=row.get("selected_expiry"),
            selected_iv=row.get("selected_option_iv"),
            selected_iv_is_proxy=row.get("selected_option_iv_is_proxy"),
            selected_iv_proxy_source=row.get("selected_option_iv_proxy_source"),
            bs_delta=row.get("selected_option_delta"),
            bs_gamma=row.get("selected_option_gamma"),
            valuation_time=row.get("signal_timestamp"),
            enabled=True,
            min_rows=min_rows,
            max_rows=max_rows,
            reject_error=reject_error,
            timeout_seconds=timeout_seconds,
        )
        features = _with_offline_diagnostics(features, row=row, chain_path=chain_path)
        for column, value in features.items():
            working.at[idx, column] = value

        processed += 1
        summary["rows_updated"] += 1
        status = _safe_text(features.get("heston_calibration_status")).upper().strip()
        if status == "CALIBRATED":
            summary["rows_calibrated"] += 1
        elif status == "REJECTED":
            summary["rows_rejected"] += 1
        else:
            summary["rows_failed"] += 1

    return working, summary


def backfill_heston_research_dataset(
    *,
    dataset_path: str | Path = CUMULATIVE_DATASET_PATH,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    min_rows: int = HESTON_CALIBRATION_MIN_ROWS,
    max_rows: int = HESTON_CALIBRATION_MAX_ROWS,
    reject_error: float = HESTON_CALIBRATION_ERROR_REJECT,
    timeout_seconds: float = HESTON_CALIBRATION_TIMEOUT_SECONDS,
    force: bool = False,
    limit: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Backfill Heston research fields on a stored signal dataset."""

    frame = load_signals_dataset(dataset_path)
    updated, summary = enrich_heston_research_features_from_snapshots(
        frame,
        snapshot_dir=snapshot_dir,
        min_rows=min_rows,
        max_rows=max_rows,
        reject_error=reject_error,
        timeout_seconds=timeout_seconds,
        force=force,
        limit=limit,
    )
    if not dry_run:
        write_signals_dataset(updated, dataset_path)
    return {
        **summary,
        "dataset_path": str(dataset_path),
        "snapshot_dir": str(snapshot_dir),
        "min_rows": int(min_rows),
        "max_rows": int(max_rows),
        "reject_error": float(reject_error),
        "timeout_seconds": float(timeout_seconds),
        "force": bool(force),
        "dry_run": bool(dry_run),
    }
