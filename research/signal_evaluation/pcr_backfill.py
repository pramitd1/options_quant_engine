"""Backfill canonical PCR fields from saved option-chain snapshots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import BASE_DIR
from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset, write_signals_dataset
from utils.pcr import select_canonical_pcr


DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR = Path(BASE_DIR) / "debug_samples" / "option_chain_snapshots"

_SNAPSHOT_TS_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d\d-\d\d)T(?P<hour>\d\d)-(?P<minute>\d\d)-(?P<second>\d\d(?:\.\d+)?)\+05-30"
)


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _option_type_series(frame: pd.DataFrame) -> pd.Series | None:
    column = _first_column(frame, ("OPTION_TYP", "option_type", "optionType", "instrument_type"))
    if column is None:
        return None
    return frame[column].astype(str).str.upper().str.strip().replace({"CALL": "CE", "PUT": "PE"})


def _safe_pcr(numerator: float, denominator: float) -> float | None:
    if denominator > 0:
        return round(float(numerator) / float(denominator), 4)
    if numerator > 0:
        return 9.99
    return None


def _front_expiry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expiry_col = _first_column(frame, ("EXPIRY_DT", "expiry", "expiryDate", "expiry_date"))
    if expiry_col is None:
        return frame
    parsed = pd.to_datetime(frame[expiry_col], errors="coerce", format="mixed", dayfirst=True)
    if parsed.notna().sum() == 0:
        return frame
    first_expiry = parsed.dropna().min()
    return frame.loc[parsed == first_expiry].copy()


def _near_atm_slice(frame: pd.DataFrame, spot: Any) -> pd.DataFrame:
    spot_value = pd.to_numeric(pd.Series([spot]), errors="coerce").iloc[0]
    strike_col = _first_column(frame, ("strikePrice", "STRIKE_PR", "strike", "strike_price"))
    if pd.isna(spot_value) or strike_col is None:
        return frame.iloc[0:0].copy()

    working = frame.copy()
    working["_strike"] = pd.to_numeric(working[strike_col], errors="coerce")
    strikes = working["_strike"].dropna().sort_values().unique()
    if len(strikes) == 0:
        return frame.iloc[0:0].copy()
    atm_strike = min(strikes, key=lambda value: abs(float(value) - float(spot_value)))
    diffs = pd.Series(strikes).diff().dropna()
    strike_step = float(diffs[diffs > 0].median()) if (diffs > 0).any() else 50.0
    return working.loc[(working["_strike"] - atm_strike).abs() <= strike_step * 4].copy()


def compute_pcr_from_option_chain(option_chain: pd.DataFrame, *, spot: Any = None) -> dict[str, Any]:
    """Compute canonical PCR fields from a saved normalized option chain."""
    if option_chain is None or option_chain.empty:
        return select_canonical_pcr()

    frame = _front_expiry_frame(option_chain)
    option_type = _option_type_series(frame)
    if option_type is None:
        return select_canonical_pcr()

    oi_col = _first_column(frame, ("openInterest", "open_interest", "OPEN_INT", "oi"))
    vol_col = _first_column(frame, ("totalTradedVolume", "VOLUME", "volume"))

    open_interest_pcr = None
    if oi_col is not None:
        oi = pd.to_numeric(frame[oi_col], errors="coerce").fillna(0.0)
        open_interest_pcr = _safe_pcr(
            float(oi.loc[option_type == "PE"].sum()),
            float(oi.loc[option_type == "CE"].sum()),
        )

    volume_pcr = None
    volume_pcr_atm = None
    if vol_col is not None:
        volume = pd.to_numeric(frame[vol_col], errors="coerce").fillna(0.0)
        volume_pcr = _safe_pcr(
            float(volume.loc[option_type == "PE"].sum()),
            float(volume.loc[option_type == "CE"].sum()),
        )

        atm_frame = _near_atm_slice(frame, spot)
        atm_type = _option_type_series(atm_frame)
        if atm_type is not None and not atm_frame.empty:
            atm_volume = pd.to_numeric(atm_frame[vol_col], errors="coerce").fillna(0.0)
            volume_pcr_atm = _safe_pcr(
                float(atm_volume.loc[atm_type == "PE"].sum()),
                float(atm_volume.loc[atm_type == "CE"].sum()),
            )

    return select_canonical_pcr(
        open_interest_pcr=open_interest_pcr,
        volume_pcr_atm=volume_pcr_atm,
        volume_pcr=volume_pcr,
    )


def parse_snapshot_timestamp(path: str | Path) -> pd.Timestamp | None:
    """Parse the IST timestamp encoded in a saved snapshot filename."""
    match = _SNAPSHOT_TS_PATTERN.search(Path(path).name)
    if not match:
        return None
    text = (
        f"{match.group('date')}T{match.group('hour')}:{match.group('minute')}:"
        f"{match.group('second')}+05:30"
    )
    try:
        return pd.Timestamp(text)
    except Exception:
        return None


def _snapshot_index(snapshot_dir: Path) -> pd.DataFrame:
    rows = []
    if not snapshot_dir.exists():
        return pd.DataFrame(columns=["snapshot_timestamp", "snapshot_path"])
    for path in snapshot_dir.glob("*.csv"):
        timestamp = parse_snapshot_timestamp(path)
        if timestamp is not None:
            rows.append({"snapshot_timestamp": timestamp, "snapshot_path": str(path)})
    if not rows:
        return pd.DataFrame(columns=["snapshot_timestamp", "snapshot_path"])
    return pd.DataFrame(rows).sort_values("snapshot_timestamp", kind="mergesort").reset_index(drop=True)


def _resolve_existing_path(path: Any) -> Path | None:
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        rooted = Path(BASE_DIR) / candidate
        if rooted.exists():
            return rooted
    return None


def _nearest_past_snapshot(
    *,
    signal_timestamp: pd.Timestamp,
    snapshots: pd.DataFrame,
    max_age_seconds: int | None,
) -> tuple[str | None, float | None]:
    if snapshots.empty or signal_timestamp is None or pd.isna(signal_timestamp) or max_age_seconds is None:
        return None, None
    eligible = snapshots.loc[snapshots["snapshot_timestamp"] <= signal_timestamp]
    if eligible.empty:
        return None, None
    row = eligible.iloc[-1]
    age_seconds = (signal_timestamp - row["snapshot_timestamp"]).total_seconds()
    if age_seconds < 0 or age_seconds > max_age_seconds:
        return None, None
    return str(row["snapshot_path"]), round(float(age_seconds), 3)


def backfill_pcr_fields(
    frame: pd.DataFrame,
    *,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    max_age_seconds: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a copy of `frame` with missing canonical PCR fields populated."""
    working = frame.copy()
    for column in (
        "open_interest_pcr",
        "volume_pcr",
        "volume_pcr_atm",
        "pcr_value",
        "pcr_basis",
        "pcr_bucket",
        "pcr_data_source",
        "pcr_snapshot_age_seconds",
    ):
        if column not in working.columns:
            working[column] = pd.NA

    snapshots = _snapshot_index(Path(snapshot_dir))
    summary = {
        "rows": int(len(working)),
        "missing_before": int(pd.to_numeric(working["pcr_value"], errors="coerce").isna().sum()),
        "saved_path_backfilled": 0,
        "nearest_snapshot_backfilled": 0,
        "snapshot_read_failures": 0,
        "still_missing": 0,
    }

    timestamps = (
        pd.to_datetime(working["signal_timestamp"], errors="coerce", format="mixed")
        if "signal_timestamp" in working.columns
        else pd.Series(pd.NaT, index=working.index)
    )

    for index, row in working.iterrows():
        if pd.to_numeric(pd.Series([row.get("pcr_value")]), errors="coerce").notna().iloc[0]:
            continue

        path = _resolve_existing_path(row.get("saved_chain_snapshot_path"))
        source = "BACKFILL_SAVED_CHAIN"
        age_seconds = None
        if path is None:
            path, age_seconds = _nearest_past_snapshot(
                signal_timestamp=timestamps.loc[index],
                snapshots=snapshots,
                max_age_seconds=max_age_seconds,
            )
            source = "BACKFILL_NEAREST_CHAIN"
        if not path:
            continue

        try:
            chain = pd.read_csv(path)
        except Exception:
            summary["snapshot_read_failures"] += 1
            continue

        pcr = compute_pcr_from_option_chain(chain, spot=row.get("spot_at_signal"))
        if pcr.get("pcr_value") is None:
            continue

        for field, value in pcr.items():
            working.at[index, field] = value
        working.at[index, "pcr_data_source"] = source
        working.at[index, "pcr_snapshot_age_seconds"] = age_seconds
        if source == "BACKFILL_SAVED_CHAIN":
            summary["saved_path_backfilled"] += 1
        else:
            summary["nearest_snapshot_backfilled"] += 1

    summary["still_missing"] = int(pd.to_numeric(working["pcr_value"], errors="coerce").isna().sum())
    return working, summary


def backfill_pcr_dataset(
    *,
    dataset_path: str | Path = CUMULATIVE_DATASET_PATH,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    max_age_seconds: int | None = None,
    dry_run: bool = True,
) -> dict[str, int | str | bool]:
    """Backfill a signal dataset on disk, optionally as a dry run."""
    dataset_path = Path(dataset_path)
    frame = load_signals_dataset(dataset_path)
    updated, summary = backfill_pcr_fields(frame, snapshot_dir=snapshot_dir, max_age_seconds=max_age_seconds)
    if not dry_run:
        write_signals_dataset(updated, dataset_path)
    return {
        **summary,
        "dataset_path": str(dataset_path),
        "snapshot_dir": str(snapshot_dir),
        "max_age_seconds": max_age_seconds if max_age_seconds is not None else -1,
        "dry_run": dry_run,
    }
