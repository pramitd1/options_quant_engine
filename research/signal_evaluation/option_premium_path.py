"""Backfill selected-option premium paths from saved option-chain snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import BASE_DIR, LOT_SIZE
from config.signal_evaluation_policy import SIGNAL_EVALUATION_HORIZON_MINUTES
from research.signal_evaluation.dataset import CUMULATIVE_DATASET_PATH, load_signals_dataset, write_signals_dataset
from research.signal_evaluation.pcr_backfill import DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR, parse_snapshot_timestamp


OPTION_PREMIUM_HORIZONS = tuple(int(value) for value in SIGNAL_EVALUATION_HORIZON_MINUTES if int(value) <= 120)
OPTION_PREMIUM_PRICE_BASIS = "LTP"
OPTION_PREMIUM_MAX_LAG_SECONDS = 180


def _option_premium_columns() -> list[str]:
    columns = [
        "option_premium_path_status",
        "option_premium_path_snapshot_count",
        "option_premium_path_max_lag_seconds",
        "option_premium_path_reasons",
        "option_premium_path_last_updated_at",
    ]
    for horizon in OPTION_PREMIUM_HORIZONS:
        columns.extend(
            [
                f"option_premium_{horizon}m",
                f"option_premium_return_{horizon}m_pct",
                f"option_premium_return_{horizon}m_bps",
                f"option_premium_pnl_per_lot_{horizon}m",
            ]
        )
    return columns


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def _normalize_option_type(value: Any) -> str | None:
    token = _safe_text(value).upper().strip()
    if token in {"CE", "CALL", "C"}:
        return "CE"
    if token in {"PE", "PUT", "P"}:
        return "PE"
    return None


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


def _snapshot_identity(path: Path) -> tuple[str | None, str | None]:
    marker = "_option_chain_snapshot_"
    name = path.name
    if marker not in name:
        return None, None
    prefix = name.split(marker, 1)[0]
    if "_" not in prefix:
        return prefix.upper().strip() or None, None
    symbol, source = prefix.rsplit("_", 1)
    return symbol.upper().strip() or None, source.upper().strip() or None


def build_option_chain_snapshot_index(snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR) -> pd.DataFrame:
    """Index saved option-chain snapshots with parsed timestamps and provider labels."""
    root = Path(snapshot_dir)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(columns=["snapshot_timestamp", "snapshot_path", "symbol", "source"])
    for path in sorted(root.glob("*.csv")):
        timestamp = parse_snapshot_timestamp(path)
        if timestamp is None:
            continue
        symbol, source = _snapshot_identity(path)
        rows.append(
            {
                "snapshot_timestamp": _coerce_ts(timestamp),
                "snapshot_path": str(path),
                "symbol": symbol,
                "source": source,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["snapshot_timestamp", "snapshot_path", "symbol", "source"])
    return pd.DataFrame(rows).sort_values("snapshot_timestamp", kind="mergesort").reset_index(drop=True)


def _filter_snapshots_for_row(snapshots: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if snapshots.empty:
        return snapshots
    symbol = _safe_text(row.get("symbol")).upper().strip()
    source = ""
    for field in ("option_source", "requested_option_source", "source"):
        source = _safe_text(row.get(field)).upper().strip()
        if source:
            break

    filtered = snapshots
    if symbol and "symbol" in filtered.columns:
        symbol_filtered = filtered.loc[filtered["symbol"].fillna("").astype(str).str.upper() == symbol]
        if not symbol_filtered.empty:
            filtered = symbol_filtered
    if source and "source" in filtered.columns:
        source_filtered = filtered.loc[filtered["source"].fillna("").astype(str).str.upper() == source]
        if not source_filtered.empty:
            filtered = source_filtered
    return filtered


def _snapshot_at_or_after(
    snapshots: pd.DataFrame,
    target_ts: pd.Timestamp,
    *,
    max_lag_seconds: int,
) -> tuple[str | None, float | None]:
    if snapshots.empty or target_ts is None or pd.isna(target_ts):
        return None, None
    eligible = snapshots.loc[snapshots["snapshot_timestamp"] >= target_ts]
    if eligible.empty:
        return None, None
    row = eligible.iloc[0]
    lag_seconds = (row["snapshot_timestamp"] - target_ts).total_seconds()
    if lag_seconds < 0 or lag_seconds > max_lag_seconds:
        return None, None
    return str(row["snapshot_path"]), round(float(lag_seconds), 3)


def _expiry_matches(series: pd.Series, selected_expiry: Any) -> pd.Series:
    if selected_expiry in (None, ""):
        return pd.Series(True, index=series.index)
    raw_match = series.astype(str).str.strip() == str(selected_expiry).strip()
    parsed_series = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    selected_ts = pd.to_datetime(pd.Series([selected_expiry]), errors="coerce", format="mixed", dayfirst=True).iloc[0]
    if pd.isna(selected_ts):
        return raw_match
    date_match = parsed_series.dt.date == selected_ts.date()
    return raw_match | date_match.fillna(False)


def match_selected_contract(
    option_chain: pd.DataFrame,
    *,
    strike: Any,
    option_type: Any,
    selected_expiry: Any = None,
) -> pd.Series | None:
    """Return the selected contract row from a normalized or provider-native chain."""
    if option_chain is None or option_chain.empty:
        return None
    strike_value = _safe_float(strike)
    normalized_type = _normalize_option_type(option_type)
    if strike_value is None or normalized_type is None:
        return None

    strike_col = _first_column(option_chain, ("strikePrice", "STRIKE_PR", "strike", "strike_price"))
    type_col = _first_column(option_chain, ("OPTION_TYP", "option_type", "optionType", "instrument_type"))
    if strike_col is None or type_col is None:
        return None

    working = option_chain.copy()
    working["_strike"] = pd.to_numeric(working[strike_col], errors="coerce")
    working["_option_type"] = working[type_col].map(_normalize_option_type)
    mask = working["_strike"].sub(strike_value).abs().le(1e-6) & working["_option_type"].eq(normalized_type)

    expiry_col = _first_column(working, ("EXPIRY_DT", "expiry", "expiryDate", "expiry_date"))
    if expiry_col is not None and selected_expiry not in (None, ""):
        expiry_mask = _expiry_matches(working[expiry_col], selected_expiry)
        if (mask & expiry_mask).any():
            mask = mask & expiry_mask

    candidates = working.loc[mask].copy()
    if candidates.empty:
        return None

    sort_cols: list[str] = []
    ascending: list[bool] = []
    volume_col = _first_column(candidates, ("totalTradedVolume", "VOLUME", "volume"))
    oi_col = _first_column(candidates, ("openInterest", "OPEN_INT", "open_interest", "oi"))
    for column in (volume_col, oi_col):
        if column is not None:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
            sort_cols.append(column)
            ascending.append(False)
    if sort_cols:
        candidates = candidates.sort_values(sort_cols, ascending=ascending, kind="stable")
    return candidates.iloc[0]


def extract_option_premium(row: pd.Series, *, price_basis: str = OPTION_PREMIUM_PRICE_BASIS) -> float | None:
    """Extract a tradable premium mark from one option-chain row."""
    basis = str(price_basis or OPTION_PREMIUM_PRICE_BASIS).upper().strip()

    def first_value(candidates: tuple[str, ...]) -> Any:
        for column in candidates:
            if column in row.index:
                return row.get(column)
        return None

    ltp = _safe_float(first_value(("lastPrice", "LAST_PRICE", "ltp", "close")))
    bid = _safe_float(first_value(("bidPrice", "BID_PRICE", "bid")))
    ask = _safe_float(first_value(("askPrice", "ASK_PRICE", "ask")))
    bid = bid if bid is not None and bid > 0 else None
    ask = ask if ask is not None and ask > 0 else None

    if basis == "MID" and bid is not None and ask is not None:
        return round((bid + ask) / 2.0, 4)
    if basis == "BID" and bid is not None:
        return round(bid, 4)
    if basis == "ASK" and ask is not None:
        return round(ask, 4)
    return round(ltp, 4) if ltp is not None and ltp > 0 else None


def _entry_premium(row: pd.Series, chain_cache: dict[str, pd.DataFrame]) -> float | None:
    for field in ("option_entry_premium", "selected_option_last_price", "entry_price"):
        value = _safe_float(row.get(field))
        if value is not None and value > 0:
            return value
    path = _resolve_existing_path(row.get("saved_chain_snapshot_path"))
    if path is None:
        return None
    key = str(path)
    if key not in chain_cache:
        try:
            chain_cache[key] = pd.read_csv(path)
        except Exception:
            chain_cache[key] = pd.DataFrame()
    contract = match_selected_contract(
        chain_cache[key],
        strike=row.get("strike"),
        option_type=row.get("option_type"),
        selected_expiry=row.get("selected_expiry"),
    )
    if contract is None:
        return None
    return extract_option_premium(contract)


def enrich_option_premium_paths(
    frame: pd.DataFrame,
    *,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    max_lag_seconds: int = OPTION_PREMIUM_MAX_LAG_SECONDS,
    price_basis: str = OPTION_PREMIUM_PRICE_BASIS,
    as_of: Any = None,
    lot_size: int = LOT_SIZE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Populate selected-contract premium paths from future saved chain snapshots."""
    working = pd.DataFrame(frame).copy()
    for column in _option_premium_columns():
        if column not in working.columns:
            working[column] = pd.NA

    summary = {
        "rows_seen": int(len(working)),
        "rows_updated": 0,
        "premium_points_filled": 0,
        "rows_complete": 0,
        "rows_partial": 0,
        "rows_pending": 0,
        "rows_no_entry_premium": 0,
        "snapshot_read_failures": 0,
    }
    if working.empty:
        return working, summary

    snapshots = build_option_chain_snapshot_index(snapshot_dir)
    as_of_ts = _coerce_ts(as_of) if as_of is not None else None
    if as_of_ts is None and not snapshots.empty:
        as_of_ts = snapshots["snapshot_timestamp"].max()

    chain_cache: dict[str, pd.DataFrame] = {}

    for idx, row in working.iterrows():
        current_status = _safe_text(row.get("option_premium_path_status")).upper().strip()
        if current_status == "COMPLETE":
            summary["rows_complete"] += 1
            continue

        signal_ts = _coerce_ts(row.get("signal_timestamp"))
        reasons: list[str] = []
        if signal_ts is None:
            working.at[idx, "option_premium_path_status"] = "NO_SIGNAL_TIMESTAMP"
            working.at[idx, "option_premium_path_reasons"] = "missing_signal_timestamp"
            summary["rows_pending"] += 1
            continue

        entry = _entry_premium(row, chain_cache)
        if entry is None or entry <= 0:
            working.at[idx, "option_premium_path_status"] = "NO_ENTRY_PREMIUM"
            working.at[idx, "option_premium_path_reasons"] = "missing_entry_premium"
            summary["rows_no_entry_premium"] += 1
            continue

        row_snapshots = _filter_snapshots_for_row(snapshots, row)
        if row_snapshots.empty:
            working.at[idx, "option_premium_path_status"] = "NO_SNAPSHOTS"
            working.at[idx, "option_premium_path_reasons"] = "no_saved_chain_snapshots"
            summary["rows_pending"] += 1
            continue

        matured = 0
        filled = 0
        max_lag = None
        for horizon in OPTION_PREMIUM_HORIZONS:
            target_ts = signal_ts + pd.Timedelta(minutes=horizon)
            if as_of_ts is not None and target_ts > as_of_ts:
                reasons.append(f"{horizon}m_pending")
                continue
            matured += 1
            path, lag_seconds = _snapshot_at_or_after(
                row_snapshots,
                target_ts,
                max_lag_seconds=max_lag_seconds,
            )
            if path is None:
                reasons.append(f"{horizon}m_snapshot_missing")
                continue
            if path not in chain_cache:
                try:
                    chain_cache[path] = pd.read_csv(path)
                except Exception:
                    chain_cache[path] = pd.DataFrame()
                    summary["snapshot_read_failures"] += 1
            contract = match_selected_contract(
                chain_cache[path],
                strike=row.get("strike"),
                option_type=row.get("option_type"),
                selected_expiry=row.get("selected_expiry"),
            )
            if contract is None:
                reasons.append(f"{horizon}m_contract_missing")
                continue
            premium = extract_option_premium(contract, price_basis=price_basis)
            if premium is None:
                reasons.append(f"{horizon}m_premium_missing")
                continue

            return_pct = ((premium - entry) / entry) * 100.0
            return_bps = return_pct * 100.0
            working.at[idx, f"option_premium_{horizon}m"] = round(float(premium), 4)
            working.at[idx, f"option_premium_return_{horizon}m_pct"] = round(float(return_pct), 4)
            working.at[idx, f"option_premium_return_{horizon}m_bps"] = round(float(return_bps), 2)
            working.at[idx, f"option_premium_pnl_per_lot_{horizon}m"] = round(float(premium - entry) * float(lot_size), 2)
            filled += 1
            summary["premium_points_filled"] += 1
            if lag_seconds is not None:
                max_lag = lag_seconds if max_lag is None else max(max_lag, lag_seconds)

        if filled:
            summary["rows_updated"] += 1
        working.at[idx, "option_premium_path_snapshot_count"] = filled
        working.at[idx, "option_premium_path_max_lag_seconds"] = max_lag
        working.at[idx, "option_premium_path_last_updated_at"] = (as_of_ts or pd.Timestamp.now(tz="Asia/Kolkata")).isoformat()
        working.at[idx, "option_premium_path_reasons"] = "|".join(reasons)

        if filled == len(OPTION_PREMIUM_HORIZONS):
            status = "COMPLETE"
            summary["rows_complete"] += 1
        elif matured == 0:
            status = "PENDING"
            summary["rows_pending"] += 1
        elif filled > 0:
            status = "PARTIAL"
            summary["rows_partial"] += 1
        else:
            status = "NO_PREMIUM_MATCH"
            summary["rows_pending"] += 1
        working.at[idx, "option_premium_path_status"] = status

    return working, summary


def backfill_option_premium_dataset(
    *,
    dataset_path: str | Path = CUMULATIVE_DATASET_PATH,
    snapshot_dir: str | Path = DEFAULT_OPTION_CHAIN_SNAPSHOT_DIR,
    max_lag_seconds: int = OPTION_PREMIUM_MAX_LAG_SECONDS,
    price_basis: str = OPTION_PREMIUM_PRICE_BASIS,
    as_of: Any = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Backfill option-premium path fields on a signal dataset."""
    frame = load_signals_dataset(dataset_path)
    updated, summary = enrich_option_premium_paths(
        frame,
        snapshot_dir=snapshot_dir,
        max_lag_seconds=max_lag_seconds,
        price_basis=price_basis,
        as_of=as_of,
    )
    if not dry_run:
        write_signals_dataset(updated, dataset_path)
    return {
        **summary,
        "dataset_path": str(dataset_path),
        "snapshot_dir": str(snapshot_dir),
        "max_lag_seconds": int(max_lag_seconds),
        "price_basis": str(price_basis).upper().strip(),
        "dry_run": bool(dry_run),
    }
