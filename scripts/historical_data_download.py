#!/usr/bin/env python3
"""
One-time Historical Data Enrichment Pipeline
=============================================

Downloads and stores comprehensive NIFTY historical data from multiple sources:

1. **Spot History** (yfinance)   — ~24 years of daily OHLCV for ^NSEI
2. **NSE F&O Bhav Copies**      — Daily EOD option chain snapshots with real OI,
                                   volume, settlement prices.  Free, no API key.
3. **ICICI Breeze Historical**   — Per-strike intraday OHLCV candles (optional,
                                   needs active session).

Output directory:
    data_store/historical/
        spot/           NIFTY_spot_daily.parquet
        nse_fo/         fo_bhav_YYYY.parquet  (one file per year)
        breeze/         breeze_NIFTY_YYYY.parquet
        merged/         NIFTY_option_chain_historical.parquet

Usage:
    python scripts/historical_data_download.py                   # all sources
    python scripts/historical_data_download.py --spot-only
    python scripts/historical_data_download.py --bhav-only
    python scripts/historical_data_download.py --bhav-only --from-year 2020
    python scripts/historical_data_download.py --merge-only
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so imports work when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hist_data")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
HIST_DIR = Path(_PROJECT_ROOT) / "data_store" / "historical"
SPOT_DIR = HIST_DIR / "spot"
BHAV_DIR = HIST_DIR / "nse_fo"
BHAV_RAW_DIR = BHAV_DIR / "raw"         # raw CSV cache
MERGED_DIR = HIST_DIR / "merged"

for d in (SPOT_DIR, BHAV_RAW_DIR, MERGED_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ===================================================================
#  1.  SPOT HISTORY  (yfinance)
# ===================================================================

def download_spot_history(symbol: str = "NIFTY", max_years: int = 25) -> Path:
    """Download full daily OHLCV for NIFTY via yfinance, save as parquet."""
    import yfinance as yf

    yf_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^NSEFIN"}
    ticker = yf_map.get(symbol.upper(), symbol)

    log.info("Downloading spot history for %s (%s), period=%dy …", symbol, ticker, max_years)
    df = yf.download(ticker, period=f"{max_years}y", interval="1d",
                     auto_adjust=False, progress=False)

    if df is None or df.empty:
        log.error("yfinance returned empty data for %s", ticker)
        return Path()

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df["date"] = pd.to_datetime(df[date_col]).dt.date
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    out = SPOT_DIR / f"{symbol.upper()}_spot_daily.parquet"
    df.to_parquet(out, index=False)
    log.info("Spot history saved: %s  (%d rows, %s → %s)", out.name, len(df),
             df["date"].iloc[0], df["date"].iloc[-1])
    return out


# ===================================================================
#  2.  NSE F&O BHAV COPIES
# ===================================================================

# NSE publishes daily F&O bhavcopy archives.  Two URL formats exist:
#
#   Old (worked through Jul 2024):
#     https://nsearchives.nseindia.com/content/historical/DERIVATIVES/
#           YYYY/MMM/fo{DD}{MON}{YYYY}bhav.csv.zip
#
#   New (Aug 2024 onward):
#     https://nsearchives.nseindia.com/content/fo/
#           BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
#
# Cutover date: August 1, 2024.  We try the appropriate format first.

_NEW_FORMAT_CUTOVER = date(2024, 8, 1)

_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Months that NSE uses in the old-format URL
_MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def _bhav_old_url(d: date) -> str:
    """Old-format bhav URL (works ≤2024)."""
    mon = _MONTH_MAP[d.month]
    day_str = d.strftime("%d")
    return (
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{mon}/fo{day_str}{mon}{d.year}bhav.csv.zip"
    )


def _bhav_new_url(d: date) -> str:
    """New-format bhav URL (2024+)."""
    ds = d.strftime("%Y%m%d")
    return (
        f"https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{ds}_F_0000.csv.zip"
    )


def _get_nse_session():
    """Create a requests session with NSE cookies."""
    import requests
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    # Fetch cookies from the derivatives report page (main page returns 403)
    for url in [
        "https://www.nseindia.com/all-reports-derivatives",
        "https://www.nseindia.com/",
    ]:
        try:
            s.get(url, timeout=10)
            if len(s.cookies) >= 2:
                break
        except Exception:
            continue
    return s


def _download_single_bhav(session, d: date) -> pd.DataFrame | None:
    """Try downloading bhav copy for a single date.  Returns DataFrame or None."""
    import requests

    cache_csv = BHAV_RAW_DIR / f"fo_bhav_{d.isoformat()}.csv"
    if cache_csv.exists():
        try:
            df = pd.read_csv(cache_csv)
            if not df.empty:
                return df
        except Exception:
            pass

    # Try the format most likely to work first based on cutover date
    if d >= _NEW_FORMAT_CUTOVER:
        urls_to_try = [_bhav_new_url(d), _bhav_old_url(d)]
    else:
        urls_to_try = [_bhav_old_url(d), _bhav_new_url(d)]

    for url in urls_to_try:
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
                    if df.empty:
                        continue
                    # Cache raw CSV for re-runs
                    df.to_csv(cache_csv, index=False)
                    return df
        except (zipfile.BadZipFile, Exception):
            continue

    return None


def _normalize_bhav(df: pd.DataFrame, trade_date: date, symbol: str = "NIFTY") -> pd.DataFrame:
    """Normalize a single day's bhav copy into standard columns."""
    # Column names differ between old and new formats; unify them.
    df.columns = [c.strip() for c in df.columns]

    # Old format columns: INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
    #                      OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
    #                      OPEN_INT, CHG_IN_OI, TIMESTAMP
    # New format columns: TradDt, BizDt, Sgmt, Src, FinInstrmTp,
    #                      TckrSymb, XpryDt, StrkPric, OptnTp, etc.

    # Detect which format
    if "INSTRUMENT" in df.columns:
        # Old format
        out = df.rename(columns={
            "SYMBOL": "symbol",
            "INSTRUMENT": "instrument",
            "EXPIRY_DT": "expiry_date",
            "STRIKE_PR": "strike_price",
            "OPTION_TYP": "option_type",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "SETTLE_PR": "settle_price",
            "CONTRACTS": "contracts",
            "OPEN_INT": "open_interest",
            "CHG_IN_OI": "change_in_oi",
        })
    elif "TckrSymb" in df.columns or "FinInstrmTp" in df.columns:
        # New format (Aug 2024+)
        rename = {
            "TckrSymb": "symbol",
            "FinInstrmTp": "instrument",
            "XpryDt": "expiry_date",
            "StrkPric": "strike_price",
            "OptnTp": "option_type",
            "OpnPric": "open",
            "HghPric": "high",
            "LwPric": "low",
            "ClsPric": "close",
            "LastPric": "last_price",
            "SttlmPric": "settle_price",
            "UndrlygPric": "underlying_price",
            "OpnIntrst": "open_interest",
            "ChngInOpnIntrst": "change_in_oi",
            "TtlTradgVol": "contracts",
        }
        out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    else:
        return pd.DataFrame()

    out["trade_date"] = trade_date

    # Filter to target symbol only (exact match)
    sym_upper = symbol.strip().upper()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out[out["symbol"] == sym_upper].copy()

    # Filter to F&O instruments (both old and new codes)
    _VALID_INSTRUMENTS = {
        "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK",  # old format
        "IDO", "IDF", "STO", "STF",                # new format
    }
    if "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype(str).str.strip().str.upper()
        out = out[out["instrument"].isin(_VALID_INSTRUMENTS)].copy()

    # Numeric conversions
    for col in ("strike_price", "open", "high", "low", "close", "settle_price",
                "open_interest", "change_in_oi", "contracts"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Parse expiry date
    if "expiry_date" in out.columns:
        out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce")

    # Normalize option type
    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()
        out.loc[out["option_type"] == "XX", "option_type"] = "FUT"
        # Futures rows have no option_type in new format — fill from instrument
        if "instrument" in out.columns:
            fut_mask = out["instrument"].isin(["FUTIDX", "FUTSTK", "IDF", "STF"])
            out.loc[fut_mask, "option_type"] = out.loc[fut_mask, "option_type"].replace(
                {"NAN": "FUT", "nan": "FUT", "": "FUT"}
            ).fillna("FUT")

    keep_cols = [c for c in (
        "trade_date", "symbol", "instrument", "expiry_date", "strike_price",
        "option_type", "open", "high", "low", "close", "last_price",
        "settle_price", "underlying_price", "contracts",
        "open_interest", "change_in_oi"
    ) if c in out.columns]

    return out[keep_cols].reset_index(drop=True)


def _trading_dates(year: int) -> list[date]:
    """Generate weekday dates for a given year (approximation of trading days)."""
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today() - timedelta(days=1))
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon–Fri
            dates.append(current)
        current += timedelta(days=1)
    return dates


def download_nse_bhav_copies(
    from_year: int = 2012,
    to_year: int | None = None,
    symbol: str = "NIFTY",
) -> list[Path]:
    """Download NSE F&O bhav copies year by year, store as parquet.

    Returns list of saved parquet paths.
    """
    if to_year is None:
        to_year = date.today().year

    saved: list[Path] = []
    session = _get_nse_session()

    for year in range(from_year, to_year + 1):
        out_path = BHAV_DIR / f"fo_bhav_{symbol}_{year}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            log.info("Year %d already cached: %s (%d rows)", year, out_path.name, len(existing))
            saved.append(out_path)
            continue

        dates = _trading_dates(year)
        year_frames: list[pd.DataFrame] = []
        success = 0
        skipped = 0

        log.info("Downloading bhav copies for year %d (%d candidate dates) …", year, len(dates))

        for i, d in enumerate(dates):
            raw = _download_single_bhav(session, d)
            if raw is not None:
                norm = _normalize_bhav(raw, d, symbol=symbol)
                if not norm.empty:
                    year_frames.append(norm)
                    success += 1
            else:
                skipped += 1

            # Progress every 50 dates
            if (i + 1) % 50 == 0:
                log.info("  Year %d: %d/%d dates processed (%d OK, %d skipped)",
                         year, i + 1, len(dates), success, skipped)

            # Polite rate limiting: 0.3s between requests
            time.sleep(0.3)

            # Re-establish session every 100 requests to refresh cookies
            if (i + 1) % 100 == 0:
                session = _get_nse_session()
                time.sleep(1)

        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            year_df.to_parquet(out_path, index=False)
            log.info("Year %d saved: %s (%d rows, %d trading days)",
                     year, out_path.name, len(year_df), success)
            saved.append(out_path)
        else:
            log.warning("Year %d: no data downloaded", year)

    return saved


# ===================================================================
#  3.  MERGE ALL SOURCES  →  unified parquet
# ===================================================================

def merge_historical_database(symbol: str = "NIFTY") -> Path:
    """Merge spot + bhav data into a single analysis-ready parquet.

    The merged file adds spot close price from daily OHLCV to each option row
    so downstream analytics can compute moneyness, delta, etc.
    """
    # Load spot data
    spot_file = SPOT_DIR / f"{symbol}_spot_daily.parquet"
    if not spot_file.exists():
        log.warning("Spot file not found: %s — run --spot-only first", spot_file)
        spot_df = pd.DataFrame()
    else:
        spot_df = pd.read_parquet(spot_file)
        spot_df["date"] = pd.to_datetime(spot_df["date"]).dt.date
        log.info("Spot data loaded: %d rows", len(spot_df))

    # Load all bhav parquets
    bhav_files = sorted(BHAV_DIR.glob(f"fo_bhav_{symbol}_*.parquet"))
    if not bhav_files:
        log.warning("No bhav parquet files found for %s", symbol)
        return Path()

    bhav_frames = []
    for f in bhav_files:
        df = pd.read_parquet(f)
        bhav_frames.append(df)
        log.info("  Loaded %s (%d rows)", f.name, len(df))

    bhav_df = pd.concat(bhav_frames, ignore_index=True)
    log.info("Total bhav rows: %d", len(bhav_df))

    # Join spot close to each option row
    if not spot_df.empty and "trade_date" in bhav_df.columns:
        bhav_df["trade_date"] = pd.to_datetime(bhav_df["trade_date"]).dt.date
        spot_lookup = spot_df.set_index("date")["close"].to_dict()
        bhav_df["spot_close"] = bhav_df["trade_date"].map(spot_lookup)
    else:
        bhav_df["spot_close"] = None

    # Sort chronologically
    bhav_df = bhav_df.sort_values(
        ["trade_date", "expiry_date", "strike_price", "option_type"]
    ).reset_index(drop=True)

    out_path = MERGED_DIR / f"{symbol}_option_chain_historical.parquet"
    bhav_df.to_parquet(out_path, index=False)
    log.info("Merged database saved: %s (%d rows)", out_path.name, len(bhav_df))

    # Print summary stats
    if "trade_date" in bhav_df.columns:
        dates = bhav_df["trade_date"].dropna()
        log.info("  Date range: %s → %s", dates.min(), dates.max())
        log.info("  Unique trading days: %d", dates.nunique())

    if "option_type" in bhav_df.columns:
        for ot in ("CE", "PE", "FUT"):
            count = (bhav_df["option_type"] == ot).sum()
            if count > 0:
                log.info("  %s rows: %d", ot, count)

    return out_path


# ===================================================================
#  MAIN  —  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="One-time historical data download for NIFTY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--spot-only", action="store_true",
                        help="Download only spot history (yfinance)")
    parser.add_argument("--bhav-only", action="store_true",
                        help="Download only NSE F&O bhav copies")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge already-downloaded data")
    parser.add_argument("--from-year", type=int, default=2012,
                        help="First year to download bhav copies (default: 2012)")
    parser.add_argument("--to-year", type=int, default=None,
                        help="Last year to download (default: current year)")
    parser.add_argument("--symbol", default="NIFTY",
                        help="Symbol to download (default: NIFTY)")

    args = parser.parse_args()

    do_all = not (args.spot_only or args.bhav_only or args.merge_only)

    if args.spot_only or do_all:
        log.info("=" * 60)
        log.info("STEP 1: Downloading spot history via yfinance")
        log.info("=" * 60)
        download_spot_history(args.symbol)

    if args.bhav_only or do_all:
        log.info("=" * 60)
        log.info("STEP 2: Downloading NSE F&O bhav copies (%d → %s)",
                 args.from_year, args.to_year or "now")
        log.info("=" * 60)
        download_nse_bhav_copies(
            from_year=args.from_year,
            to_year=args.to_year,
            symbol=args.symbol,
        )

    if args.merge_only or do_all:
        log.info("=" * 60)
        log.info("STEP 3: Merging all sources")
        log.info("=" * 60)
        merge_historical_database(args.symbol)

    log.info("Done.")


if __name__ == "__main__":
    main()
