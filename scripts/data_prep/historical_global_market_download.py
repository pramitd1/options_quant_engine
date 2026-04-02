#!/usr/bin/env python3
"""
Historical Global Market Data Download
=======================================

Downloads and stores long-history daily OHLCV for every cross-asset variable
that the signal engine's global-risk overlay consumes:

  Commodities : Crude Oil (CL=F), Gold (GC=F), Copper (HG=F)
  Volatility  : US VIX (^VIX), India VIX (^INDIAVIX)
  Equities    : S&P 500 (^GSPC), NASDAQ (^IXIC)
  Rates       : US 10Y Yield (^TNX)
  Currencies  : USD/INR (INR=X)
    Macro FX    : US Dollar Index (DX-Y.NYB)
    Lead Index  : GIFT Nifty / configured proxy
  Indices     : NIFTY 50 (^NSEI), BANKNIFTY (^NSEBANK)

For each ticker the script:
  1. Downloads max-available daily history via yfinance
  2. Saves raw OHLCV as Parquet   (one file per ticker)
  3. Computes derived daily features matching engine field names
  4. Saves a combined daily features CSV for backtesting use

Output directory:
    data_store/historical/global_market/
        raw/        <TICKER_LABEL>_daily.parquet   (raw OHLCV)
        features/   global_market_features.parquet  (combined derived features)

Usage:
    python scripts/data_prep/historical_global_market_download.py               # download all
    python scripts/data_prep/historical_global_market_download.py --ticker oil  # single ticker
    python scripts/data_prep/historical_global_market_download.py --list        # show tickers
    python scripts/data_prep/historical_global_market_download.py --features-only  # recompute features from cached raw data
    python scripts/data_prep/historical_global_market_download.py --max-years 10   # limit lookback
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.market_data_policy import GLOBAL_MARKET_TICKERS, IST_TIMEZONE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("global_mkt_hist")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
HIST_DIR = _PROJECT_ROOT / "data_store" / "historical" / "global_market"
RAW_DIR = HIST_DIR / "raw"
FEATURES_DIR = HIST_DIR / "features"

for _d in (RAW_DIR, FEATURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Full ticker map — engine tickers + underlying indices needed for
# realized vol.  Keys are human-readable labels, values are yfinance tickers.
# ---------------------------------------------------------------------------
ALL_TICKERS: dict[str, str] = {
    **GLOBAL_MARKET_TICKERS,
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
}

TICKER_META: dict[str, dict] = {
    "oil":       {"name": "Crude Oil Futures",     "unit": "USD/bbl"},
    "gold":      {"name": "Gold Futures",          "unit": "USD/oz"},
    "copper":    {"name": "Copper Futures",        "unit": "USD/lb"},
    "vix":       {"name": "CBOE VIX",              "unit": "index"},
    "india_vix": {"name": "India VIX",             "unit": "index"},
    "sp500":     {"name": "S&P 500",               "unit": "index"},
    "nasdaq":    {"name": "NASDAQ Composite",      "unit": "index"},
    "us10y":     {"name": "US 10Y Treasury Yield", "unit": "yield_pct"},
    "usdinr":    {"name": "USD/INR Exchange Rate",  "unit": "INR"},
    "dxy":       {"name": "US Dollar Index",        "unit": "index"},
    "gift_nifty": {"name": "GIFT Nifty",            "unit": "index"},
    "nifty50":   {"name": "NIFTY 50",              "unit": "index"},
    "banknifty": {"name": "BANK NIFTY",            "unit": "index"},
}


# ===================================================================
#  1.  RAW DOWNLOAD
# ===================================================================

def download_ticker(label: str, ticker: str, *, max_years: int = 25) -> Path | None:
    """Download daily OHLCV for a single ticker and save as Parquet."""
    meta = TICKER_META.get(label, {})
    log.info("Downloading %s (%s / %s), period=%dy …",
             label, ticker, meta.get("name", ""), max_years)

    try:
        df = yf.download(
            ticker,
            period=f"{max_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        log.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if df is None or df.empty:
        log.warning("No data returned for %s (%s)", label, ticker)
        return None

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(IST_TIMEZONE).dt.date

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    out = RAW_DIR / f"{label}_daily.parquet"
    df.to_parquet(out, index=False)
    log.info("  Saved %s — %d rows (%s → %s)", out.name, len(df),
             df["date"].iloc[0], df["date"].iloc[-1])
    return out


def download_all(*, max_years: int = 25, only: str | None = None) -> dict[str, Path]:
    """Download history for all tickers (or one if ``only`` is specified)."""
    targets = {only: ALL_TICKERS[only]} if only else dict(ALL_TICKERS)
    results: dict[str, Path] = {}
    for label, ticker in targets.items():
        path = download_ticker(label, ticker, max_years=max_years)
        if path:
            results[label] = path
    return results


# ===================================================================
#  2.  FEATURE COMPUTATION
# ===================================================================

def _load_raw(label: str) -> pd.DataFrame:
    """Load a raw Parquet file into a DataFrame."""
    path = RAW_DIR / f"{label}_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _pct_change(series: pd.Series) -> pd.Series:
    """Daily percentage change."""
    return series.pct_change(fill_method=None) * 100.0


def _bp_change(series: pd.Series) -> pd.Series:
    """Daily change in basis points (for ^TNX: 1 index pt ≈ 10 bp)."""
    return series.diff() * 10.0


def _realized_vol(series: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility from log-returns."""
    log_ret = series.apply(math.log).diff()
    return log_ret.rolling(window).std(ddof=0) * math.sqrt(252.0)


def build_features() -> pd.DataFrame:
    """
    Build the combined daily features frame matching the engine's
    ``market_inputs`` field names.

    Each row represents one trading day.  The columns are the same fields
    that ``build_global_market_snapshot()`` produces in real-time, but
    computed over the full historical range.
    """
    log.info("Building combined daily features …")

    # Load all raw histories
    raw: dict[str, pd.DataFrame] = {}
    for label in ALL_TICKERS:
        df = _load_raw(label)
        if df.empty:
            log.warning("  Raw data missing for %s — skipping", label)
        else:
            raw[label] = df

    if not raw:
        log.error("No raw data found. Run download first.")
        return pd.DataFrame()

    # Build a unified date index from the broadest range
    all_dates = set()
    for df in raw.values():
        all_dates.update(df["date"].dt.date)
    date_idx = sorted(all_dates)
    features = pd.DataFrame({"date": pd.to_datetime(date_idx)})

    def _merge_close(label: str, col_name: str):
        if label not in raw:
            return
        df = raw[label][["date", "close"]].rename(columns={"close": col_name})
        nonlocal features
        features = features.merge(df, on="date", how="left")

    # Merge all close prices
    _merge_close("oil", "oil_close")
    _merge_close("gold", "gold_close")
    _merge_close("copper", "copper_close")
    _merge_close("vix", "vix_close")
    _merge_close("india_vix", "india_vix_close")
    _merge_close("sp500", "sp500_close")
    _merge_close("nasdaq", "nasdaq_close")
    _merge_close("us10y", "us10y_close")
    _merge_close("usdinr", "usdinr_close")
    _merge_close("nifty50", "nifty50_close")
    _merge_close("banknifty", "banknifty_close")

    features = features.sort_values("date").reset_index(drop=True)

    # Derived daily change fields (matching engine field names)
    if "oil_close" in features.columns:
        features["oil_change_24h"] = _pct_change(features["oil_close"])
    if "gold_close" in features.columns:
        features["gold_change_24h"] = _pct_change(features["gold_close"])
    if "copper_close" in features.columns:
        features["copper_change_24h"] = _pct_change(features["copper_close"])
    if "vix_close" in features.columns:
        features["vix_change_24h"] = _pct_change(features["vix_close"])
    if "india_vix_close" in features.columns:
        features["india_vix_change_24h"] = _pct_change(features["india_vix_close"])
        features["india_vix_level"] = features["india_vix_close"]
    if "sp500_close" in features.columns:
        features["sp500_change_24h"] = _pct_change(features["sp500_close"])
    if "nasdaq_close" in features.columns:
        features["nasdaq_change_24h"] = _pct_change(features["nasdaq_close"])
    if "us10y_close" in features.columns:
        features["us10y_change_bp"] = _bp_change(features["us10y_close"])
    if "usdinr_close" in features.columns:
        features["usdinr_change_24h"] = _pct_change(features["usdinr_close"])

    # Realized volatility — compute from gap-free raw series (not the
    # merged frame which has NaN on non-Indian trading days).
    def _merge_realized_vol(label: str, prefix: str):
        if label not in raw:
            return
        s = raw[label].sort_values("date").copy()
        s[f"{prefix}_realized_vol_5d"] = _realized_vol(s["close"], 5)
        s[f"{prefix}_realized_vol_30d"] = _realized_vol(s["close"], 30)
        cols = ["date", f"{prefix}_realized_vol_5d", f"{prefix}_realized_vol_30d"]
        nonlocal features
        features = features.merge(s[cols], on="date", how="left")

    _merge_realized_vol("nifty50", "nifty50")
    _merge_realized_vol("banknifty", "banknifty")

    # Save
    out = FEATURES_DIR / "global_market_features.parquet"
    features.to_parquet(out, index=False)

    # Also save CSV for easy inspection
    csv_out = FEATURES_DIR / "global_market_features.csv"
    features.to_csv(csv_out, index=False)

    log.info("Features saved: %s — %d rows, %d columns", out.name,
             len(features), len(features.columns))
    log.info("  Date range: %s → %s",
             features["date"].iloc[0].date(), features["date"].iloc[-1].date())

    # Summary stats
    change_cols = [c for c in features.columns if c.endswith(("_24h", "_bp", "_level", "_5d", "_30d"))]
    if change_cols:
        stats = features[change_cols].describe().round(3)
        log.info("\n%s", stats.to_string())

    return features


# ===================================================================
#  3.  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download historical global market data for backtesting",
    )
    parser.add_argument("--ticker", default=None,
                        help="Download only this ticker label (e.g., oil, vix, usdinr)")
    parser.add_argument("--max-years", type=int, default=25,
                        help="Maximum years of history to request (default: 25)")
    parser.add_argument("--list", action="store_true",
                        help="List available tickers and exit")
    parser.add_argument("--features-only", action="store_true",
                        help="Skip download, only recompute features from cached raw data")
    args = parser.parse_args()

    if args.list:
        print(f"\nGlobal market tickers ({len(ALL_TICKERS)}):")
        print("-" * 65)
        for label, ticker in ALL_TICKERS.items():
            meta = TICKER_META.get(label, {})
            print(f"  {label:<15} {ticker:<15} {meta.get('name', '')}")
        return

    if not args.features_only:
        if args.ticker and args.ticker not in ALL_TICKERS:
            print(f"Unknown ticker label: {args.ticker}")
            print(f"Available: {', '.join(ALL_TICKERS.keys())}")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"Historical Global Market Data Download")
        print(f"{'=' * 60}")
        print(f"Max years : {args.max_years}")
        print(f"Tickers   : {args.ticker or 'ALL (' + str(len(ALL_TICKERS)) + ')'}")
        print(f"Output    : {HIST_DIR}")
        print(f"{'=' * 60}\n")

        results = download_all(max_years=args.max_years, only=args.ticker)

        print(f"\n{'=' * 60}")
        print(f"Download complete: {len(results)}/{len(ALL_TICKERS) if not args.ticker else 1} tickers")
        for label, path in results.items():
            df = pd.read_parquet(path)
            print(f"  {label:<15} {len(df):>6} rows  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
        print(f"{'=' * 60}\n")

    # Always build features after download
    build_features()


if __name__ == "__main__":
    main()
