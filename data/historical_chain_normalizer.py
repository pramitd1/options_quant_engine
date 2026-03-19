"""
Historical Chain Normalizer
============================
Unified loader that reads the year-partitioned NIFTY parquet files (2012-2026),
normalizes them into a consistent schema, recovers NaT expiry dates from raw
CSVs, fills underlying_price from spot_daily, and produces the backtest-ready
column layout.

This module does NOT modify raw data on disk. It reads raw data and returns
normalized DataFrames in memory, or optionally writes to the merged output
directory.

Column Contract (output):
    trade_date, symbol, instrument, expiry_date, strike_price, option_type,
    open, high, low, close, last_price, settle_price, underlying_price,
    contracts, open_interest, change_in_oi
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

_DATA = Path(DATA_DIR)
NSE_FO_DIR = _DATA / "historical" / "nse_fo"
RAW_CSV_DIR = NSE_FO_DIR / "raw"
SPOT_FILE = _DATA / "historical" / "spot" / "NIFTY_spot_daily.parquet"

CANONICAL_COLUMNS = [
    "trade_date", "symbol", "instrument", "expiry_date", "strike_price",
    "option_type", "open", "high", "low", "close", "last_price",
    "settle_price", "underlying_price", "contracts", "open_interest",
    "change_in_oi",
]


# Cache normalized yearly chains to avoid repeating expensive schema recovery
# and NaT-expiry reconstruction on every backtest day.
_normalized_chain_cache: dict[tuple, pd.DataFrame] = {}


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def load_normalized_chain(
    symbol: str = "NIFTY",
    years: list[int] | int | None = None,
    options_only: bool = True,
    recover_nat_expiry: bool = True,
    fill_underlying: bool = True,
) -> pd.DataFrame:
    """Load, normalize, and clean historical option chain data.

    Parameters
    ----------
    symbol : str
        Target symbol (default NIFTY).
    years : list[int] | int | None
        Specific year(s) to load, or None for all available.
    options_only : bool
        If True, filter to CE/PE rows only (drop futures).
    recover_nat_expiry : bool
        If True, recover NaT expiry dates from raw CSVs.
    fill_underlying : bool
        If True, fill missing underlying_price from spot_daily.

    Returns
    -------
    pd.DataFrame with CANONICAL_COLUMNS, sorted by trade_date.
    """
    if isinstance(years, int):
        years = [years]
    if years is not None:
        years = sorted(int(y) for y in years)

    cache_key = (
        symbol.strip().upper(),
        tuple(years) if years is not None else None,
        bool(options_only),
        bool(recover_nat_expiry),
        bool(fill_underlying),
    )
    cached = _normalized_chain_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    parquet_files = sorted(NSE_FO_DIR.glob(f"fo_bhav_{symbol}_*.parquet"))
    if not parquet_files:
        log.warning("No parquet files found for %s in %s", symbol, NSE_FO_DIR)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    frames = []
    for pf in parquet_files:
        year = int(pf.stem.split("_")[-1])
        if years is not None and year not in years:
            continue
        df = pd.read_parquet(pf)
        df = _normalize_schema(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.date
    combined["expiry_date"] = pd.to_datetime(combined["expiry_date"], errors="coerce")

    if options_only:
        combined = combined[combined["option_type"].isin(["CE", "PE"])].copy()

    if recover_nat_expiry:
        nat_count = combined["expiry_date"].isna().sum()
        if nat_count > 0:
            log.info("Recovering %d NaT expiry dates from raw CSVs ...", nat_count)
            combined = _recover_nat_expiry(combined, symbol)

    if fill_underlying:
        combined = _fill_underlying_price(combined)

    # Final dtype coercion
    for col in ("contracts", "open_interest", "change_in_oi"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)

    combined = combined.sort_values(["trade_date", "expiry_date", "strike_price", "option_type"])
    combined = combined.reset_index(drop=True)

    log.info("Loaded %d rows across %d trading days", len(combined),
             combined["trade_date"].nunique())

    _normalized_chain_cache[cache_key] = combined
    return combined.copy()


def load_normalized_day(
    trade_date: date | str,
    symbol: str = "NIFTY",
    options_only: bool = True,
) -> pd.DataFrame:
    """Load and normalize a single day's option chain.

    Efficient: only loads the relevant year's parquet.
    """
    if isinstance(trade_date, str):
        trade_date = pd.to_datetime(trade_date).date()

    year = trade_date.year
    result = load_normalized_chain(
        symbol=symbol,
        years=[year],
        options_only=options_only,
        recover_nat_expiry=True,
        fill_underlying=True,
    )

    return result[result["trade_date"] == trade_date].copy()


# ---------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all DataFrames have CANONICAL_COLUMNS with consistent types."""
    out = df.copy()

    # Add missing columns with NaN
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # Standardize instrument codes
    if "instrument" in out.columns:
        out["instrument"] = (
            out["instrument"].astype(str).str.strip().str.upper()
            .replace({"IDO": "OPTIDX", "IDF": "FUTIDX"})
        )

    # Standardize option_type
    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()
        out.loc[out["option_type"] == "XX", "option_type"] = "FUT"

    # Numeric columns
    for col in ("strike_price", "open", "high", "low", "close", "last_price",
                "settle_price", "underlying_price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fill last_price from close if missing
    if "last_price" in out.columns:
        out["last_price"] = out["last_price"].fillna(out["close"])
    else:
        out["last_price"] = out["close"]

    return out[CANONICAL_COLUMNS]


# ---------------------------------------------------------------
# NaT expiry recovery
# ---------------------------------------------------------------

def _recover_nat_expiry(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Recover NaT expiry_date values by matching rows to raw CSV data.

    Strategy:
      For each date with NaT rows, load the raw CSV and build a lookup
      table of (symbol, strike_price, option_type, close) → expiry_date.
      Match NaT rows against this lookup.
    """
    nat_mask = df["expiry_date"].isna()
    if not nat_mask.any():
        return df

    nat_dates = df.loc[nat_mask, "trade_date"].unique()
    recovered = 0
    total_nat = int(nat_mask.sum())

    for td in nat_dates:
        raw_path = RAW_CSV_DIR / f"fo_bhav_{td}.csv"
        if not raw_path.exists():
            continue

        try:
            raw = pd.read_csv(raw_path)
            raw.columns = [c.strip() for c in raw.columns]
        except Exception:
            continue

        if "INSTRUMENT" not in raw.columns:
            continue

        sym_upper = symbol.strip().upper()
        raw = raw[raw["SYMBOL"].astype(str).str.strip().str.upper() == sym_upper]
        raw = raw[raw["OPTION_TYP"].astype(str).str.strip().str.upper().isin(["CE", "PE"])]

        if raw.empty:
            continue

        # Build lookup: (strike, option_type, close) → expiry_date
        raw["_strike"] = pd.to_numeric(raw["STRIKE_PR"], errors="coerce")
        raw["_opt"] = raw["OPTION_TYP"].astype(str).str.strip().str.upper()
        raw["_close"] = pd.to_numeric(raw["CLOSE"], errors="coerce")
        raw["_expiry"] = pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce")

        # Build multi-key lookup
        lookup = {}
        for _, row in raw.iterrows():
            key = (row["_strike"], row["_opt"], round(float(row["_close"]), 2) if pd.notna(row["_close"]) else None)
            if pd.notna(row["_expiry"]):
                lookup[key] = row["_expiry"]

        # Also build a simpler (strike, option_type) → [expiries] mapping
        # for cases where close doesn't match exactly
        strike_opt_expiries: dict[tuple, list] = {}
        for _, row in raw.iterrows():
            if pd.notna(row["_expiry"]):
                k = (row["_strike"], row["_opt"])
                strike_opt_expiries.setdefault(k, []).append(row["_expiry"])

        # Apply to NaT rows for this date
        day_nat_mask = nat_mask & (df["trade_date"] == td)
        day_nat_idx = df.index[day_nat_mask]

        for idx in day_nat_idx:
            row = df.loc[idx]
            close_key = (row["strike_price"], row["option_type"],
                         round(float(row["close"]), 2) if pd.notna(row["close"]) else None)

            if close_key in lookup:
                df.at[idx, "expiry_date"] = lookup[close_key]
                recovered += 1
            else:
                # Fallback: match longest-dated expiry for this (strike, option_type)
                so_key = (row["strike_price"], row["option_type"])
                candidates = strike_opt_expiries.get(so_key, [])
                if candidates:
                    # NaT rows tend to be far-dated; pick the most distant expiry
                    # that hasn't been assigned to a non-NaT row yet
                    assigned_expiries = set(
                        df.loc[(df["trade_date"] == td) &
                               (df["strike_price"] == row["strike_price"]) &
                               (df["option_type"] == row["option_type"]) &
                               df["expiry_date"].notna(), "expiry_date"]
                    )
                    unassigned = [c for c in candidates if c not in assigned_expiries]
                    if unassigned:
                        df.at[idx, "expiry_date"] = max(unassigned)
                        recovered += 1

    remaining_nat = df["expiry_date"].isna().sum()
    log.info("  Recovered %d/%d NaT expiries (%d still missing)",
             recovered, total_nat, remaining_nat)

    # Drop rows where expiry could not be recovered
    if remaining_nat > 0:
        df = df[df["expiry_date"].notna()].copy()
        log.info("  Dropped %d unrecoverable NaT rows", remaining_nat)

    return df


# ---------------------------------------------------------------
# Underlying price fill
# ---------------------------------------------------------------

def _fill_underlying_price(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing underlying_price by joining spot_daily close."""
    if not SPOT_FILE.exists():
        log.warning("Spot file not found: %s", SPOT_FILE)
        return df

    missing_mask = df["underlying_price"].isna() | (df["underlying_price"] == 0)
    if not missing_mask.any():
        return df

    spot = pd.read_parquet(SPOT_FILE)
    spot["date"] = pd.to_datetime(spot["date"]).dt.date
    spot_lookup = dict(zip(spot["date"], spot["close"]))

    def _fill(row):
        if pd.isna(row["underlying_price"]) or row["underlying_price"] == 0:
            return spot_lookup.get(row["trade_date"], row["underlying_price"])
        return row["underlying_price"]

    # Vectorized approach
    dates = df["trade_date"]
    fill_values = dates.map(spot_lookup)
    df.loc[missing_mask, "underlying_price"] = fill_values[missing_mask]

    filled = int(missing_mask.sum() - df["underlying_price"].isna().sum())
    log.info("  Filled %d underlying_price values from spot_daily", filled)

    return df
