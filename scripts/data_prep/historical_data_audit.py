#!/usr/bin/env python3
"""
Historical Data Audit Framework
=================================
Institutional-grade audit of the NIFTY historical dataset covering
spot data, options chain data (CE+PE), and all derived fields.

Sections:
  1. Data Inventory & Structure Analysis
  2. Data Quality Audit
  3. Options Chain Integrity
  4. Point-in-Time Consistency
  5. Feature Reconstruction Plan
  6. Data Normalization & Standardization
  7. Replay-Ready Data Pipeline (design)
  8. Data Quality Scoring System
  9. Report Generation

Usage:
    python scripts/data_prep/historical_data_audit.py
    python scripts/data_prep/historical_data_audit.py --year 2024
    python scripts/data_prep/historical_data_audit.py --section 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_audit")

# ===========================================================================
#  Paths
# ===========================================================================
DATA_DIR = _PROJECT_ROOT / "data_store"
NSE_FO_DIR = DATA_DIR / "historical" / "nse_fo"
RAW_CSV_DIR = NSE_FO_DIR / "raw"
SPOT_FILE = DATA_DIR / "historical" / "spot" / "NIFTY_spot_daily.parquet"
REPORT_DIR = _PROJECT_ROOT / "research" / "data_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical schema for the parquet files (post-normalization target)
CANONICAL_COLUMNS = [
    "trade_date", "symbol", "instrument", "expiry_date", "strike_price",
    "option_type", "open", "high", "low", "close", "last_price",
    "settle_price", "underlying_price", "contracts", "open_interest",
    "change_in_oi",
]


# ===========================================================================
#  Data classes for structured results
# ===========================================================================
@dataclass
class YearProfile:
    year: int
    rows: int = 0
    trading_days: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    date_range: tuple[str, str] = ("", "")
    option_types: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    expiry_count: int = 0
    strike_range: tuple[float, float] = (0.0, 0.0)
    has_last_price: bool = False
    has_underlying_price: bool = False
    ce_rows: int = 0
    pe_rows: int = 0
    fut_rows: int = 0


@dataclass
class DayQuality:
    date: str
    total_rows: int = 0
    ce_rows: int = 0
    pe_rows: int = 0
    unique_strikes: int = 0
    unique_expiries: int = 0
    paired_strikes: int = 0
    zero_oi_pct: float = 0.0
    zero_volume_pct: float = 0.0
    zero_close_pct: float = 0.0
    negative_values: int = 0
    duplicate_rows: int = 0
    quality_score: float = 0.0  # 0-100


@dataclass
class AuditResults:
    """Container for all audit results across sections."""
    timestamp: str = ""
    # Section 1
    year_profiles: list[YearProfile] = field(default_factory=list)
    spot_profile: dict = field(default_factory=dict)
    raw_csv_count: int = 0
    raw_date_range: tuple[str, str] = ("", "")
    schema_changes: list[str] = field(default_factory=list)
    # Section 2
    missing_days: list[str] = field(default_factory=list)
    day_qualities: list[DayQuality] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    # Section 3
    chain_integrity_issues: list[dict] = field(default_factory=list)
    # Section 4
    pit_violations: list[dict] = field(default_factory=list)
    # Section 5
    feature_plan: list[dict] = field(default_factory=list)
    # Section 8
    yearly_scores: dict[int, dict] = field(default_factory=dict)
    overall_score: float = 0.0


# ===========================================================================
#  SECTION 1 — DATA INVENTORY & STRUCTURE ANALYSIS
# ===========================================================================
def section_1_inventory(results: AuditResults, target_year: int | None = None):
    """Scan entire dataset and report structure, schemas, coverage."""
    log.info("=" * 60)
    log.info("SECTION 1 — DATA INVENTORY & STRUCTURE ANALYSIS")
    log.info("=" * 60)

    # 1a. Raw CSV inventory
    raw_files = sorted(RAW_CSV_DIR.glob("fo_bhav_*.csv"))
    results.raw_csv_count = len(raw_files)
    if raw_files:
        first_date = raw_files[0].stem.replace("fo_bhav_", "")
        last_date = raw_files[-1].stem.replace("fo_bhav_", "")
        results.raw_date_range = (first_date, last_date)
    log.info("  Raw CSVs: %d files (%s → %s)", results.raw_csv_count,
             results.raw_date_range[0], results.raw_date_range[1])

    # 1b. Parquet year profiles
    parquet_files = sorted(NSE_FO_DIR.glob("fo_bhav_NIFTY_*.parquet"))
    for pf in parquet_files:
        year = int(pf.stem.split("_")[-1])
        if target_year and year != target_year:
            continue

        df = pd.read_parquet(pf)
        prof = YearProfile(year=year)
        prof.rows = len(df)
        prof.columns = list(df.columns)
        prof.dtypes = {c: str(df[c].dtype) for c in df.columns}

        # trade_date handling
        if "trade_date" in df.columns:
            dates = pd.to_datetime(df["trade_date"])
            prof.date_range = (str(dates.min().date()), str(dates.max().date()))
            prof.trading_days = dates.dt.date.nunique()

        if "option_type" in df.columns:
            prof.option_types = sorted(df["option_type"].dropna().unique().tolist())
            prof.ce_rows = int((df["option_type"] == "CE").sum())
            prof.pe_rows = int((df["option_type"] == "PE").sum())
            prof.fut_rows = int(df["option_type"].isin(["FUT", "XX"]).sum())

        if "instrument" in df.columns:
            prof.instruments = sorted(df["instrument"].dropna().unique().tolist())

        if "expiry_date" in df.columns:
            prof.expiry_count = df["expiry_date"].nunique()

        if "strike_price" in df.columns:
            opts = df[df["option_type"].isin(["CE", "PE"])] if "option_type" in df.columns else df
            if len(opts) > 0:
                prof.strike_range = (float(opts["strike_price"].min()),
                                     float(opts["strike_price"].max()))

        prof.has_last_price = "last_price" in df.columns
        prof.has_underlying_price = "underlying_price" in df.columns

        results.year_profiles.append(prof)
        log.info("  %d: %d rows, %d days, cols=%d, CE=%d PE=%d FUT=%d",
                 year, prof.rows, prof.trading_days, len(prof.columns),
                 prof.ce_rows, prof.pe_rows, prof.fut_rows)

    # 1c. Schema change detection
    results.schema_changes = _detect_schema_changes(results.year_profiles)
    for sc in results.schema_changes:
        log.warning("  SCHEMA CHANGE: %s", sc)

    # 1d. Spot data profile
    if SPOT_FILE.exists():
        spot = pd.read_parquet(SPOT_FILE)
        results.spot_profile = {
            "rows": len(spot),
            "columns": list(spot.columns),
            "date_range": (str(spot["date"].min()), str(spot["date"].max())) if "date" in spot.columns else ("?", "?"),
            "nan_counts": spot.isna().sum().to_dict(),
            "zero_volume_days": int((spot["volume"] == 0).sum()) if "volume" in spot.columns else 0,
        }
        log.info("  Spot: %d rows, %s → %s, zero_vol_days=%d",
                 results.spot_profile["rows"],
                 results.spot_profile["date_range"][0],
                 results.spot_profile["date_range"][1],
                 results.spot_profile.get("zero_volume_days", 0))

    # 1e. Raw CSV schema check (old vs new format)
    _check_raw_csv_schema_evolution(results)


def _detect_schema_changes(profiles: list[YearProfile]) -> list[str]:
    """Compare column sets across years to find schema transitions."""
    changes = []
    prev = None
    for prof in profiles:
        if prev:
            old_cols = set(prev.columns)
            new_cols = set(prof.columns)
            added = new_cols - old_cols
            removed = old_cols - new_cols
            if added:
                changes.append(f"{prev.year}→{prof.year}: Added columns {sorted(added)}")
            if removed:
                changes.append(f"{prev.year}→{prof.year}: Removed columns {sorted(removed)}")

            # Dtype changes for common columns
            common = old_cols & new_cols
            for col in sorted(common):
                if prev.dtypes.get(col) != prof.dtypes.get(col):
                    changes.append(
                        f"{prev.year}→{prof.year}: Column '{col}' type changed "
                        f"from {prev.dtypes[col]} to {prof.dtypes[col]}"
                    )

            # Instrument name changes
            if set(prev.instruments) != set(prof.instruments):
                changes.append(
                    f"{prev.year}→{prof.year}: Instrument values changed "
                    f"from {prev.instruments} to {prof.instruments}"
                )
        prev = prof
    return changes


def _check_raw_csv_schema_evolution(results: AuditResults):
    """Sample raw CSVs from different eras to detect format changes."""
    sample_dates = ["2012-01-02", "2016-01-04", "2020-01-01", "2024-01-02", "2025-01-01", "2026-01-02"]
    prev_header = None
    for dt in sample_dates:
        f = RAW_CSV_DIR / f"fo_bhav_{dt}.csv"
        if not f.exists():
            continue
        try:
            with open(f) as fh:
                header = fh.readline().strip()
            if prev_header and header != prev_header:
                results.schema_changes.append(
                    f"Raw CSV format change detected at {dt}: "
                    f"{len(header.split(','))} columns vs previous {len(prev_header.split(','))} columns"
                )
            prev_header = header
        except Exception:
            pass


# ===========================================================================
#  SECTION 2 — DATA QUALITY AUDIT
# ===========================================================================
def section_2_quality(results: AuditResults, target_year: int | None = None):
    """Missing data, anomalies, duplicates, outliers."""
    log.info("=" * 60)
    log.info("SECTION 2 — DATA QUALITY AUDIT")
    log.info("=" * 60)

    parquet_files = sorted(NSE_FO_DIR.glob("fo_bhav_NIFTY_*.parquet"))

    all_trading_dates: set[date] = set()
    all_anomalies: list[dict] = []

    for pf in parquet_files:
        year = int(pf.stem.split("_")[-1])
        if target_year and year != target_year:
            continue

        log.info("  Auditing %d ...", year)
        df = pd.read_parquet(pf)
        df["_date"] = pd.to_datetime(df["trade_date"]).dt.date

        # Options only (CE/PE)
        opts = df[df["option_type"].isin(["CE", "PE"])].copy()

        year_dates = sorted(df["_date"].unique())
        all_trading_dates.update(year_dates)

        for d in year_dates:
            day_df = opts[opts["_date"] == d]
            if len(day_df) == 0:
                continue

            dq = DayQuality(date=str(d))
            dq.total_rows = len(day_df)
            dq.ce_rows = int((day_df["option_type"] == "CE").sum())
            dq.pe_rows = int((day_df["option_type"] == "PE").sum())
            dq.unique_strikes = day_df["strike_price"].nunique()
            dq.unique_expiries = day_df["expiry_date"].nunique()

            # CE/PE pairing
            ce_strikes = set(day_df[day_df["option_type"] == "CE"]["strike_price"])
            pe_strikes = set(day_df[day_df["option_type"] == "PE"]["strike_price"])
            dq.paired_strikes = len(ce_strikes & pe_strikes)

            # Zero checks
            if "open_interest" in day_df.columns:
                dq.zero_oi_pct = round(float((day_df["open_interest"] == 0).mean()) * 100, 1)
            if "contracts" in day_df.columns:
                dq.zero_volume_pct = round(float((day_df["contracts"] == 0).mean()) * 100, 1)
            dq.zero_close_pct = round(float((day_df["close"] == 0).mean()) * 100, 1)

            # Negative values
            price_cols = [c for c in ["open", "high", "low", "close", "settle_price"] if c in day_df.columns]
            for c in price_cols:
                neg = (day_df[c] < 0).sum()
                if neg > 0:
                    dq.negative_values += int(neg)
                    all_anomalies.append({
                        "type": "NEGATIVE_PRICE",
                        "date": str(d),
                        "column": c,
                        "count": int(neg),
                        "severity": "CRITICAL",
                    })

            # Duplicate rows
            dup_subset = ["strike_price", "option_type", "expiry_date"]
            dup_count = day_df.duplicated(subset=dup_subset, keep=False).sum()
            dq.duplicate_rows = int(dup_count)
            if dup_count > 0:
                all_anomalies.append({
                    "type": "DUPLICATE_ROWS",
                    "date": str(d),
                    "count": int(dup_count),
                    "severity": "WARNING",
                })

            # Quality score (0-100)
            dq.quality_score = _compute_day_quality_score(dq)

            results.day_qualities.append(dq)

        # Year-level anomaly checks
        _check_year_anomalies(opts, year, all_anomalies)

        # NaT expiry date check (year-level)
        nat_expiry_count = int(opts["expiry_date"].isna().sum())
        if nat_expiry_count > 0:
            nat_pct = nat_expiry_count / max(1, len(opts)) * 100
            all_anomalies.append({
                "type": "NAT_EXPIRY_DATE",
                "year": year,
                "count": nat_expiry_count,
                "pct": f"{nat_pct:.1f}%",
                "severity": "CRITICAL",
                "note": "Rows with missing expiry_date — likely from preprocessing parse failures. "
                        "These rows have valid strikes and OHLC but no expiry, making them unusable "
                        "for IV computation and chain construction without recovery.",
            })

    results.anomalies = all_anomalies

    # Missing days analysis (compare with spot data)
    _find_missing_days(results, all_trading_dates)

    log.info("  Total anomalies found: %d", len(results.anomalies))
    log.info("  Missing trading days: %d", len(results.missing_days))


def _compute_day_quality_score(dq: DayQuality) -> float:
    """Score a single day's data quality on 0-100 scale."""
    score = 100.0

    # Row count penalty
    if dq.total_rows < 20:
        score -= 30
    elif dq.total_rows < 50:
        score -= 15
    elif dq.total_rows < 100:
        score -= 5

    # CE/PE balance
    if dq.ce_rows == 0 or dq.pe_rows == 0:
        score -= 25
    elif dq.paired_strikes < max(1, dq.unique_strikes * 0.5):
        score -= 10

    # Zero OI penalty (many zeros = illiquid data)
    if dq.zero_oi_pct > 80:
        score -= 15
    elif dq.zero_oi_pct > 50:
        score -= 8

    # Zero volume penalty
    if dq.zero_volume_pct > 90:
        score -= 10
    elif dq.zero_volume_pct > 70:
        score -= 5

    # Zero close penalty (likely stale data)
    if dq.zero_close_pct > 50:
        score -= 20
    elif dq.zero_close_pct > 20:
        score -= 10

    # Negative values
    if dq.negative_values > 0:
        score -= 20

    # Duplicates
    if dq.duplicate_rows > 0:
        dup_pct = dq.duplicate_rows / max(1, dq.total_rows) * 100
        if dup_pct > 10:
            score -= 15
        elif dup_pct > 2:
            score -= 5

    return max(0.0, round(score, 1))


def _check_year_anomalies(df: pd.DataFrame, year: int, anomalies: list[dict]):
    """Check for year-level outliers and spikes."""
    # Extreme OI changes
    if "change_in_oi" in df.columns and len(df) > 0:
        oi_std = df["change_in_oi"].std()
        if oi_std > 0:
            extreme_oi = df[df["change_in_oi"].abs() > 5 * oi_std]
            if len(extreme_oi) > 0:
                anomalies.append({
                    "type": "EXTREME_OI_CHANGE",
                    "year": year,
                    "count": len(extreme_oi),
                    "threshold": f"5σ = {5 * oi_std:.0f}",
                    "severity": "INFO",
                })

    # Zero-price rows (close = 0 for contracts with OI > 0)
    if "open_interest" in df.columns:
        stale = df[(df["close"] == 0) & (df["open_interest"] > 0)]
        if len(stale) > 0:
            anomalies.append({
                "type": "ZERO_CLOSE_WITH_OI",
                "year": year,
                "count": len(stale),
                "severity": "WARNING",
                "note": "Options with zero close but positive OI — likely untaded/illiquid",
            })

    # Settle price != close (potential EOD adjustment)
    if "settle_price" in df.columns:
        mismatch = df[(df["close"] > 0) & (df["settle_price"] > 0) &
                       (df["close"] != df["settle_price"])]
        if len(mismatch) > 0:
            pct = len(mismatch) / max(1, len(df)) * 100
            if pct > 5:
                anomalies.append({
                    "type": "CLOSE_SETTLE_MISMATCH",
                    "year": year,
                    "count": len(mismatch),
                    "pct": f"{pct:.1f}%",
                    "severity": "INFO",
                    "note": "Close != settle_price — normal for options with mark-to-market",
                })


def _find_missing_days(results: AuditResults, fo_dates: set[date]):
    """Compare FO trading days with spot data to find gaps."""
    if not SPOT_FILE.exists():
        return

    spot = pd.read_parquet(SPOT_FILE)
    spot_dates = set(pd.to_datetime(spot["date"]).dt.date)

    # Days in spot but not in FO
    fo_min = min(fo_dates) if fo_dates else date(2012, 1, 1)
    fo_max = max(fo_dates) if fo_dates else date(2026, 3, 17)
    spot_in_range = {d for d in spot_dates if fo_min <= d <= fo_max}

    missing_in_fo = sorted(spot_in_range - fo_dates)
    # Filter out weekends
    missing_in_fo = [d for d in missing_in_fo if d.weekday() < 5]

    results.missing_days = [str(d) for d in missing_in_fo]


# ===========================================================================
#  SECTION 3 — OPTIONS CHAIN INTEGRITY
# ===========================================================================
def section_3_chain_integrity(results: AuditResults, target_year: int | None = None):
    """CE/PE pairing, strike grid, expiry structure, OI continuity."""
    log.info("=" * 60)
    log.info("SECTION 3 — OPTIONS CHAIN INTEGRITY")
    log.info("=" * 60)

    issues = []
    parquet_files = sorted(NSE_FO_DIR.glob("fo_bhav_NIFTY_*.parquet"))

    for pf in parquet_files:
        year = int(pf.stem.split("_")[-1])
        if target_year and year != target_year:
            continue

        log.info("  Checking chain integrity for %d ...", year)
        df = pd.read_parquet(pf)
        df["_date"] = pd.to_datetime(df["trade_date"]).dt.date
        opts = df[df["option_type"].isin(["CE", "PE"])].copy()

        # Sample dates (every 5th trading day to keep runtime manageable)
        unique_dates = sorted(opts["_date"].unique())
        sample_dates = unique_dates[::5]  # Check every 5th day

        for d in sample_dates:
            day_df = opts[opts["_date"] == d]
            if len(day_df) == 0:
                continue

            # 3a. CE/PE pairing by strike and expiry
            for exp in day_df["expiry_date"].unique():
                exp_df = day_df[day_df["expiry_date"] == exp]
                ce_strikes = set(exp_df[exp_df["option_type"] == "CE"]["strike_price"])
                pe_strikes = set(exp_df[exp_df["option_type"] == "PE"]["strike_price"])

                unpaired_ce = ce_strikes - pe_strikes
                unpaired_pe = pe_strikes - ce_strikes

                if unpaired_ce or unpaired_pe:
                    if len(unpaired_ce) > 5 or len(unpaired_pe) > 5:
                        issues.append({
                            "type": "UNPAIRED_STRIKES",
                            "date": str(d),
                            "expiry": str(exp.date()) if hasattr(exp, "date") else str(exp),
                            "unpaired_ce": len(unpaired_ce),
                            "unpaired_pe": len(unpaired_pe),
                            "severity": "WARNING",
                        })

            # 3b. Strike grid uniformity (check for ATM coverage)
            if "underlying_price" in day_df.columns:
                spot_vals = day_df["underlying_price"].dropna()
                if len(spot_vals) > 0:
                    atm = spot_vals.iloc[0]
                else:
                    atm = None
            else:
                # Estimate ATM from futures
                fut_rows = df[(df["_date"] == d) & (df["option_type"].isin(["FUT", "XX"]))]
                atm = fut_rows["close"].iloc[0] if len(fut_rows) > 0 else None

            if atm and atm > 0:
                nearest_exp = day_df["expiry_date"].min()
                near_df = day_df[day_df["expiry_date"] == nearest_exp]
                strikes = sorted(near_df["strike_price"].unique())
                if len(strikes) >= 2:
                    # Check for gaps near ATM
                    atm_idx = min(range(len(strikes)),
                                  key=lambda i: abs(strikes[i] - atm))
                    lo = max(0, atm_idx - 3)
                    hi = min(len(strikes), atm_idx + 4)
                    near_atm = strikes[lo:hi]
                    if len(near_atm) >= 2:
                        steps = [near_atm[i+1] - near_atm[i] for i in range(len(near_atm)-1)]
                        # Check for irregular spacing
                        median_step = sorted(steps)[len(steps)//2]
                        irregular = [s for s in steps if abs(s - median_step) > median_step * 0.5]
                        if irregular and len(irregular) > len(steps) * 0.3:
                            issues.append({
                                "type": "IRREGULAR_STRIKE_GRID",
                                "date": str(d),
                                "atm": float(atm),
                                "steps": [float(s) for s in steps],
                                "severity": "INFO",
                            })

            # 3c. Missing strikes near ATM
            if atm and atm > 0:
                nearest_exp = day_df["expiry_date"].min()
                near_df = day_df[day_df["expiry_date"] == nearest_exp]
                strikes = sorted(near_df["strike_price"].unique())
                if len(strikes) >= 2:
                    # Typical strike step
                    diffs = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
                    step = sorted(diffs)[len(diffs)//2]
                    if step > 0:
                        # Expected strikes ±5 steps from ATM
                        expected = set()
                        base = round(atm / step) * step
                        for i in range(-5, 6):
                            expected.add(base + i * step)
                        missing_near_atm = expected - set(strikes)
                        if missing_near_atm:
                            near_missing = [s for s in missing_near_atm
                                            if abs(s - atm) <= 5 * step]
                            if len(near_missing) > 2:
                                issues.append({
                                    "type": "MISSING_ATM_STRIKES",
                                    "date": str(d),
                                    "atm": float(atm),
                                    "missing": [float(s) for s in sorted(near_missing)],
                                    "severity": "WARNING",
                                })

    results.chain_integrity_issues = issues
    log.info("  Chain integrity issues: %d", len(issues))
    issue_counts = Counter(i["type"] for i in issues)
    for t, c in issue_counts.most_common():
        log.info("    %s: %d", t, c)


# ===========================================================================
#  SECTION 4 — POINT-IN-TIME CONSISTENCY
# ===========================================================================
def section_4_point_in_time(results: AuditResults, target_year: int | None = None):
    """Check for lookahead bias and point-in-time correctness."""
    log.info("=" * 60)
    log.info("SECTION 4 — POINT-IN-TIME CONSISTENCY")
    log.info("=" * 60)

    violations = []

    # 4a. Data is EOD bhav copy → inherently point-in-time for daily granularity
    log.info("  Data nature: NSE Bhav Copy = End-of-Day settlement data")
    log.info("  This is inherently point-in-time for daily resolution.")
    log.info("  Checking for common PIT violations ...")

    parquet_files = sorted(NSE_FO_DIR.glob("fo_bhav_NIFTY_*.parquet"))

    for pf in parquet_files:
        year = int(pf.stem.split("_")[-1])
        if target_year and year != target_year:
            continue

        df = pd.read_parquet(pf)
        opts = df[df["option_type"].isin(["CE", "PE"])].copy()
        opts["_date"] = pd.to_datetime(opts["trade_date"]).dt.date

        # 4b. Check that expiry_date >= trade_date (no expired contracts in data)
        if "expiry_date" in opts.columns:
            expired = opts[opts["expiry_date"] < pd.to_datetime(opts["trade_date"])]
            if len(expired) > 0:
                violations.append({
                    "type": "EXPIRED_CONTRACT_IN_DATA",
                    "year": year,
                    "count": len(expired),
                    "severity": "CRITICAL",
                    "note": "Contracts where expiry_date < trade_date — should not exist in bhav copy",
                })

        # 4c. Check for future dates in data (data dated ahead of file date)
        unique_dates = sorted(opts["_date"].unique())
        for i in range(1, len(unique_dates)):
            if (unique_dates[i] - unique_dates[i-1]).days > 7:
                # Check if there's missing data or calendar jump
                violations.append({
                    "type": "LARGE_DATE_GAP",
                    "year": year,
                    "from": str(unique_dates[i-1]),
                    "to": str(unique_dates[i]),
                    "gap_days": (unique_dates[i] - unique_dates[i-1]).days,
                    "severity": "INFO",
                    "note": "Gap > 7 days — likely market holiday period",
                })

        # 4d. Check settle_price vs close relationship
        # In NSE bhav copy, settle_price can differ from close for options
        # but should not be a future value
        if "settle_price" in opts.columns and "close" in opts.columns:
            # settle > 2x close is suspicious
            valid = opts[(opts["close"] > 0) & (opts["settle_price"] > 0)]
            if len(valid) > 0:
                ratio = valid["settle_price"] / valid["close"]
                extreme = valid[ratio > 3.0]
                if len(extreme) > 0:
                    violations.append({
                        "type": "EXTREME_SETTLE_CLOSE_RATIO",
                        "year": year,
                        "count": len(extreme),
                        "max_ratio": float(ratio.max()),
                        "severity": "WARNING",
                        "note": "settle_price > 3x close — check if EOD mark-to-market or data error",
                    })

        # 4e. Underlying price consistency across same-day rows
        if "underlying_price" in opts.columns:
            up = opts.groupby("_date")["underlying_price"].nunique()
            multi_spot = up[up > 3]  # Allow 2-3 for rounding
            if len(multi_spot) > 0:
                violations.append({
                    "type": "MULTIPLE_UNDERLYING_PRICES",
                    "year": year,
                    "affected_days": len(multi_spot),
                    "max_unique": int(multi_spot.max()),
                    "severity": "INFO",
                    "note": "Multiple underlying prices on same day — may reflect intraday changes in source",
                })

    results.pit_violations = violations
    log.info("  PIT checks: %d issues found", len(violations))

    # 4f. Key PIT assessment
    log.info("  ASSESSMENT:")
    log.info("    - Data is NSE Bhav Copy (EOD) → inherently point-in-time correct")
    log.info("    - No intraday data → no intraday lookahead risk")
    log.info("    - IV is NOT pre-computed in raw data → will be reconstructed from available fields")
    log.info("    - Greeks are NOT in raw data → will be computed from IV + other inputs")
    log.info("    - Features derived from this data are safe if computed with same-day-only inputs")


# ===========================================================================
#  SECTION 5 — FEATURE RECONSTRUCTION PLAN
# ===========================================================================
def section_5_feature_plan(results: AuditResults):
    """Define what needs to be reconstructed and how."""
    log.info("=" * 60)
    log.info("SECTION 5 — FEATURE RECONSTRUCTION PLAN")
    log.info("=" * 60)

    plan = [
        {
            "feature": "Implied Volatility (IV)",
            "status": "MISSING_IN_RAW",
            "reconstruction": "Newton-Raphson on Black-Scholes using: close, strike_price, underlying_price/spot, expiry_days, risk-free rate",
            "required_inputs": ["close (option price)", "strike_price", "spot (underlying)", "days_to_expiry", "risk_free_rate (assumed 6-7%)"],
            "engine_location": "data/historical_data_adapter.py::_implied_vol_newton()",
            "notes": "Already implemented. Cap at 500%. Use settle_price as fallback if close=0.",
            "priority": "CRITICAL",
        },
        {
            "feature": "Greeks (Delta, Gamma, Theta, Vega, Rho)",
            "status": "MISSING_IN_RAW",
            "reconstruction": "Black-Scholes closed-form from IV, spot, strike, T, r",
            "required_inputs": ["IV (from reconstruction)", "spot", "strike_price", "days_to_expiry", "risk_free_rate"],
            "engine_location": "analytics/greeks_engine.py",
            "notes": "Compute after IV reconstruction. Standard BS formulas.",
            "priority": "HIGH",
        },
        {
            "feature": "Spot Price (underlying_price)",
            "status": "PARTIALLY_AVAILABLE",
            "reconstruction": "Available in 2025-2026 parquets. For 2012-2024: join with spot_daily parquet or use futures close as proxy.",
            "required_inputs": ["spot_daily parquet", "futures close from same bhav copy"],
            "engine_location": "Merge at normalization stage",
            "notes": "Futures close is a good proxy (basis typically <0.5%). Spot file covers 2007-2026.",
            "priority": "CRITICAL",
        },
        {
            "feature": "Last Price (last_price)",
            "status": "PARTIALLY_AVAILABLE",
            "reconstruction": "Available in 2024-2026 parquets. For 2012-2023: use close as proxy (bhav copy close is EOD last traded or theoretical).",
            "required_inputs": ["close column"],
            "engine_location": "Fill missing with close",
            "notes": "In NSE bhav copy, close is the official closing price. last_price adds LTP info.",
            "priority": "LOW",
        },
        {
            "feature": "Days to Expiry",
            "status": "DERIVABLE",
            "reconstruction": "(expiry_date - trade_date).days, floor at 0",
            "required_inputs": ["expiry_date", "trade_date"],
            "engine_location": "Computed in historical_data_adapter.py",
            "notes": "Calendar days, not trading days. Already implemented.",
            "priority": "HIGH",
        },
        {
            "feature": "Moneyness",
            "status": "DERIVABLE",
            "reconstruction": "strike_price / spot for CE, spot / strike_price for PE",
            "required_inputs": ["strike_price", "spot"],
            "engine_location": "Computed at signal generation time",
            "notes": "Standard definition. Log-moneyness also useful.",
            "priority": "MEDIUM",
        },
        {
            "feature": "Volume (totalTradedVolume)",
            "status": "AVAILABLE_AS_CONTRACTS",
            "reconstruction": "contracts column = number of contracts traded. Multiply by lot size for notional.",
            "required_inputs": ["contracts"],
            "engine_location": "Rename contracts → totalTradedVolume in adapter",
            "notes": "Lot size changed over time (50→75→25 for NIFTY). Track lot size schedule.",
            "priority": "MEDIUM",
        },
    ]

    results.feature_plan = plan
    for p in plan:
        log.info("  [%s] %s — %s", p["priority"], p["feature"], p["status"])


# ===========================================================================
#  SECTION 6 — DATA NORMALIZATION & STANDARDIZATION
# ===========================================================================
def section_6_normalization_report(results: AuditResults):
    """Report on what normalization is needed. Does NOT modify raw data."""
    log.info("=" * 60)
    log.info("SECTION 6 — NORMALIZATION REQUIREMENTS")
    log.info("=" * 60)

    log.info("  Normalization steps required:")
    log.info("  1. Unified column schema across all years (14→16 cols)")
    log.info("  2. trade_date → datetime type (currently date objects)")
    log.info("  3. Fill underlying_price from spot_daily for 2012-2024")
    log.info("  4. Fill last_price from close for 2012-2023")
    log.info("  5. Standardize instrument codes (OPTIDX/IDO → OPTION)")
    log.info("  6. Compute expiry_days = (expiry_date - trade_date).days")
    log.info("  7. Compute IV via Newton-Raphson")
    log.info("  8. Standardize option_type (XX/FUT → FUT)")
    log.info("  9. contracts dtype: float64 → int64 for 2012 data")


# ===========================================================================
#  SECTION 7 — REPLAY READINESS DESIGN
# ===========================================================================
def section_7_replay_design(results: AuditResults):
    """Design the replay system (report only - code in separate module)."""
    log.info("=" * 60)
    log.info("SECTION 7 — REPLAY-READY DATA PIPELINE DESIGN")
    log.info("=" * 60)

    log.info("  Replay API design:")
    log.info("    Input: date (YYYY-MM-DD)")
    log.info("    Output: {spot_snapshot, option_chain, derived_features}")
    log.info("  ")
    log.info("  Data flow:")
    log.info("    1. Load parquet for target year")
    log.info("    2. Filter to target date")
    log.info("    3. Join spot close from spot_daily")
    log.info("    4. Compute IV, Greeks, derived features")
    log.info("    5. Return deterministic snapshot")
    log.info("  ")
    log.info("  Point-in-time guarantee:")
    log.info("    - Only uses data available on or before target date")
    log.info("    - No forward-looking features")
    log.info("    - Deterministic: same date always produces same output")


# ===========================================================================
#  SECTION 8 — DATA QUALITY SCORING
# ===========================================================================
def section_8_scoring(results: AuditResults):
    """Compute per-year and overall data quality scores."""
    log.info("=" * 60)
    log.info("SECTION 8 — DATA QUALITY SCORING")
    log.info("=" * 60)

    # Group day_qualities by year
    by_year: dict[int, list[DayQuality]] = defaultdict(list)
    for dq in results.day_qualities:
        year = int(dq.date[:4])
        by_year[year].append(dq)

    for year in sorted(by_year.keys()):
        days = by_year[year]
        scores = [dq.quality_score for dq in days]
        avg_score = np.mean(scores) if scores else 0

        good_days = sum(1 for s in scores if s >= 80)
        partial_days = sum(1 for s in scores if 50 <= s < 80)
        poor_days = sum(1 for s in scores if s < 50)
        total = len(days)

        results.yearly_scores[year] = {
            "avg_score": round(float(avg_score), 1),
            "total_days": total,
            "good_days": good_days,
            "good_pct": round(good_days / max(1, total) * 100, 1),
            "partial_days": partial_days,
            "partial_pct": round(partial_days / max(1, total) * 100, 1),
            "poor_days": poor_days,
            "poor_pct": round(poor_days / max(1, total) * 100, 1),
            "min_score": round(float(min(scores)) if scores else 0, 1),
            "max_score": round(float(max(scores)) if scores else 0, 1),
        }

        log.info("  %d: avg=%.1f  good=%d (%.0f%%)  partial=%d (%.0f%%)  poor=%d (%.0f%%)",
                 year, avg_score, good_days, good_days/max(1,total)*100,
                 partial_days, partial_days/max(1,total)*100,
                 poor_days, poor_days/max(1,total)*100)

    # Overall score
    all_scores = [dq.quality_score for dq in results.day_qualities]
    results.overall_score = round(float(np.mean(all_scores)), 1) if all_scores else 0
    log.info("  OVERALL SCORE: %.1f / 100", results.overall_score)


# ===========================================================================
#  SECTION 9 — REPORT GENERATION
# ===========================================================================
def section_9_report(results: AuditResults):
    """Generate comprehensive markdown report."""
    log.info("=" * 60)
    log.info("SECTION 9 — REPORT GENERATION")
    log.info("=" * 60)

    report_lines = []

    def w(line=""):
        report_lines.append(line)

    w("# NIFTY Historical Data Audit Report")
    w(f"**Generated:** {results.timestamp}")
    w(f"**Overall Data Quality Score: {results.overall_score} / 100**")
    w()

    # ---- Section 1: Inventory ----
    w("## 1. Data Inventory & Structure")
    w()
    w("### 1.1 Raw Data Files")
    w(f"- **Raw CSV count:** {results.raw_csv_count}")
    w(f"- **Date range:** {results.raw_date_range[0]} → {results.raw_date_range[1]}")
    w()

    w("### 1.2 Processed Parquet Files (NIFTY)")
    w()
    w("| Year | Rows | Trading Days | Columns | CE Rows | PE Rows | FUT Rows | Has LTP | Has Underlying |")
    w("|------|------|-------------|---------|---------|---------|----------|---------|----------------|")
    total_rows = 0
    total_days = 0
    for p in results.year_profiles:
        total_rows += p.rows
        total_days += p.trading_days
        w(f"| {p.year} | {p.rows:,} | {p.trading_days} | {len(p.columns)} | "
          f"{p.ce_rows:,} | {p.pe_rows:,} | {p.fut_rows:,} | "
          f"{'Yes' if p.has_last_price else 'No'} | "
          f"{'Yes' if p.has_underlying_price else 'No'} |")
    w(f"| **TOTAL** | **{total_rows:,}** | **{total_days}** | — | — | — | — | — | — |")
    w()

    w("### 1.3 Spot Data")
    if results.spot_profile:
        sp = results.spot_profile
        w(f"- **Rows:** {sp['rows']:,}")
        w(f"- **Date range:** {sp['date_range'][0]} → {sp['date_range'][1]}")
        w(f"- **Columns:** {sp['columns']}")
        w(f"- **Zero volume days:** {sp['zero_volume_days']}")
    w()

    w("### 1.4 Schema Changes Detected")
    w()
    if results.schema_changes:
        for sc in results.schema_changes:
            w(f"- {sc}")
    else:
        w("- No schema changes detected")
    w()

    # ---- Section 2: Quality ----
    w("## 2. Data Quality Audit")
    w()

    w("### 2.1 Missing Trading Days")
    w(f"- **Missing days (spot exists but no FO data):** {len(results.missing_days)}")
    if results.missing_days and len(results.missing_days) <= 20:
        for d in results.missing_days:
            w(f"  - {d}")
    elif results.missing_days:
        w(f"  - First 10: {', '.join(results.missing_days[:10])}")
        w(f"  - Last 10: {', '.join(results.missing_days[-10:])}")
    w()

    w("### 2.2 Data Anomalies")
    w()
    anomaly_counts = Counter(a["type"] for a in results.anomalies)
    severity_counts = Counter(a["severity"] for a in results.anomalies)
    w(f"- **Total anomalies:** {len(results.anomalies)}")
    w(f"- **By severity:** CRITICAL={severity_counts.get('CRITICAL',0)}, "
      f"WARNING={severity_counts.get('WARNING',0)}, INFO={severity_counts.get('INFO',0)}")
    w()
    w("| Anomaly Type | Count | Severity |")
    w("|---|---|---|")
    for atype, count in anomaly_counts.most_common():
        sev = next((a["severity"] for a in results.anomalies if a["type"] == atype), "?")
        w(f"| {atype} | {count} | {sev} |")
    w()

    # Critical anomalies detail
    critical = [a for a in results.anomalies if a["severity"] == "CRITICAL"]
    if critical:
        w("### 2.3 Critical Issues")
        w()
        for a in critical:
            w(f"- **{a['type']}**: {json.dumps({k:v for k,v in a.items() if k not in ('type','severity')})}")
        w()

    # ---- Section 3: Chain Integrity ----
    w("## 3. Options Chain Integrity")
    w()
    w(f"- **Total integrity issues:** {len(results.chain_integrity_issues)}")
    ci_counts = Counter(i["type"] for i in results.chain_integrity_issues)
    w()
    w("| Issue Type | Count |")
    w("|---|---|")
    for t, c in ci_counts.most_common():
        w(f"| {t} | {c} |")
    w()

    # ---- Section 4: PIT ----
    w("## 4. Point-in-Time Consistency")
    w()
    if not results.pit_violations:
        w("**No critical point-in-time violations detected.**")
    else:
        w(f"- **Total PIT issues:** {len(results.pit_violations)}")
        pit_counts = Counter(v["type"] for v in results.pit_violations)
        w()
        w("| Issue Type | Count | Severity |")
        w("|---|---|---|")
        for t, c in pit_counts.most_common():
            sev = next((v["severity"] for v in results.pit_violations if v["type"] == t), "?")
            w(f"| {t} | {c} | {sev} |")
    w()
    w("### PIT Assessment")
    w()
    w("- **Data source:** NSE Bhav Copy (official End-of-Day settlement data)")
    w("- **Granularity:** Daily (no intraday data in raw files)")
    w("- **Lookahead risk:** LOW — data represents known EOD values")
    w("- **IV status:** Not in raw data — must be reconstructed from available fields")
    w("- **Greeks status:** Not in raw data — must be computed from IV")
    w("- **Recommendation:** All derived features (IV, Greeks, signals) must be computed using only same-day or prior-day inputs")
    w()

    # ---- Section 5: Feature Plan ----
    w("## 5. Feature Reconstruction Plan")
    w()
    w("| Feature | Status | Priority | Reconstruction Method |")
    w("|---|---|---|---|")
    for fp in results.feature_plan:
        w(f"| {fp['feature']} | {fp['status']} | {fp['priority']} | {fp['reconstruction'][:80]}... |")
    w()

    # ---- Section 6: Normalization ----
    w("## 6. Data Normalization Requirements")
    w()
    w("### Canonical Schema (target)")
    w()
    w("```")
    for col in CANONICAL_COLUMNS:
        w(f"  {col}")
    w("```")
    w()
    w("### Required Transformations")
    w()
    w("1. **Column unification:** 14→16 columns for 2012-2023 (add `last_price`, `underlying_price`)")
    w("2. **Type normalization:** `contracts` float64→int64 for 2012; `trade_date` to datetime")
    w("3. **Spot price fill:** Join `underlying_price` from spot_daily for pre-2025 data")
    w("4. **Instrument codes:** Unify `OPTIDX`/`IDO` → standard label")
    w("5. **Expiry format:** Ensure all `expiry_date` as datetime with consistent timezone handling")
    w("6. **option_type:** Standardize `XX`→`FUT`")
    w()

    # ---- Section 7: Replay Design ----
    w("## 7. Replay-Ready Data Pipeline")
    w()
    w("### Architecture")
    w()
    w("```")
    w("replay_historical_snapshot(date: str, symbol: str = 'NIFTY')")
    w("  → {")
    w("      'spot': {'date': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ...},")
    w("      'option_chain': pd.DataFrame with canonical columns + IV + Greeks,")
    w("      'quality_score': float,")
    w("      'metadata': {'data_source': 'NSE_BHAV', 'granularity': 'EOD', ...}")
    w("    }")
    w("```")
    w()
    w("### Point-in-Time Guarantees")
    w()
    w("1. Only uses data for the requested date (no future data)")
    w("2. IV computed from same-day close prices")
    w("3. Greeks computed from same-day IV")
    w("4. Spot from same-day spot_daily close")
    w("5. Deterministic output (same input → same output)")
    w()

    # ---- Section 8: Scoring ----
    w("## 8. Data Quality Scores")
    w()
    w("| Year | Avg Score | Good Days (≥80) | Partial (50-79) | Poor (<50) | Min | Max |")
    w("|------|----------|-----------------|-----------------|------------|-----|-----|")
    for year in sorted(results.yearly_scores.keys()):
        ys = results.yearly_scores[year]
        w(f"| {year} | {ys['avg_score']} | {ys['good_days']} ({ys['good_pct']}%) | "
          f"{ys['partial_days']} ({ys['partial_pct']}%) | {ys['poor_days']} ({ys['poor_pct']}%) | "
          f"{ys['min_score']} | {ys['max_score']} |")
    w()

    # ---- Data usability summary ----
    w("## 9. Data Usability Summary")
    w()
    total_d = len(results.day_qualities)
    good = sum(1 for d in results.day_qualities if d.quality_score >= 80)
    partial = sum(1 for d in results.day_qualities if 50 <= d.quality_score < 80)
    poor = sum(1 for d in results.day_qualities if d.quality_score < 50)
    w(f"- **Total trading days audited:** {total_d}")
    w(f"- **Usable (score ≥ 80):** {good} ({good/max(1,total_d)*100:.1f}%)")
    w(f"- **Requires cleaning (50-79):** {partial} ({partial/max(1,total_d)*100:.1f}%)")
    w(f"- **Unusable (< 50):** {poor} ({poor/max(1,total_d)*100:.1f}%)")
    w()

    # ---- Recommendations ----
    w("## 10. Recommendations")
    w()
    w("### Critical Actions")
    w()
    w("1. **Build normalization pipeline** — Unify all 15 year-files into canonical 16-column schema")
    w("2. **Fill underlying_price** — Join spot_daily close for 2012-2024 data")
    w("3. **Compute IV** — Use Newton-Raphson BS solver (already in `historical_data_adapter.py`)")
    w("4. **Build replay loader module** — `replay_historical_snapshot(date)` function")
    w("5. **Handle zero-close rows** — Use settle_price as fallback; flag as low-confidence")
    w()
    w("### Quality Improvements")
    w()
    w("6. Deduplicate rows per (date, strike, option_type, expiry)")
    w("7. Flag and quarantine days with quality score < 50")
    w("8. Validate strike grid continuity for ATM±10 strikes")
    w("9. Track NIFTY lot size changes for accurate volume normalization")
    w()
    w("### Data Gaps")
    w()
    for sc in results.schema_changes[:5]:
        w(f"- {sc}")
    w()

    # Write report
    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "historical_data_audit_report.md"
    report_path.write_text(report_text)
    log.info("  Report saved to %s", report_path)

    # Also save raw results as JSON for programmatic use
    json_path = REPORT_DIR / "audit_results.json"
    json_data = {
        "timestamp": results.timestamp,
        "overall_score": results.overall_score,
        "raw_csv_count": results.raw_csv_count,
        "raw_date_range": results.raw_date_range,
        "schema_changes": results.schema_changes,
        "missing_days_count": len(results.missing_days),
        "missing_days": results.missing_days,
        "anomaly_count": len(results.anomalies),
        "anomalies": results.anomalies,
        "chain_integrity_issues_count": len(results.chain_integrity_issues),
        "pit_violations_count": len(results.pit_violations),
        "pit_violations": results.pit_violations,
        "yearly_scores": results.yearly_scores,
        "spot_profile": results.spot_profile,
        "feature_plan": results.feature_plan,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    log.info("  Raw results saved to %s", json_path)

    return report_path


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="NIFTY Historical Data Audit")
    parser.add_argument("--year", type=int, default=None,
                        help="Audit only a specific year")
    parser.add_argument("--section", type=int, default=None,
                        help="Run only a specific section (1-9)")
    args = parser.parse_args()

    results = AuditResults(timestamp=datetime.now().isoformat())
    target_year = args.year

    sections = {
        1: lambda: section_1_inventory(results, target_year),
        2: lambda: section_2_quality(results, target_year),
        3: lambda: section_3_chain_integrity(results, target_year),
        4: lambda: section_4_point_in_time(results, target_year),
        5: lambda: section_5_feature_plan(results),
        6: lambda: section_6_normalization_report(results),
        7: lambda: section_7_replay_design(results),
        8: lambda: section_8_scoring(results),
        9: lambda: section_9_report(results),
    }

    if args.section:
        if args.section in sections:
            sections[args.section]()
        else:
            log.error("Invalid section %d (1-9)", args.section)
            sys.exit(1)
    else:
        for section_num in sorted(sections.keys()):
            sections[section_num]()

    log.info("")
    log.info("=" * 60)
    log.info("AUDIT COMPLETE — Overall Score: %.1f / 100", results.overall_score)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
