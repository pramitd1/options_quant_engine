"""
Module: historical_spot_fetcher.py

Purpose:
    Fetch and cache historical OHLC (Open/High/Low/Close) data for underlying assets
    to support technical analysis indicators.

Role in the System:
    Part of the data layer that provides time-series price data for TA features.
    Fetches from broker APIs or public sources, caches locally for replay/backtest.

Key Outputs:
    Pandas DataFrame with OHLCV data (timestamp, open, high, low, close, volume).

Downstream Usage:
    Consumed by features/ta_indicators.py for computing TA signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import requests

from config.market_data_policy import IST_TIMEZONE

logger = logging.getLogger(__name__)

# Cache directory for historical data
HISTORICAL_SPOT_DIR = Path("data_store") / "historical_spot"
HISTORICAL_SPOT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_spot_ohlc(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1D",
    source: str = "YAHOO"
) -> pd.DataFrame:
    """
    Fetch historical OHLC data for a symbol.

    Args:
        symbol: Underlying symbol (e.g., 'NIFTY', '^NSEI')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Time interval ('1D', '1H', etc.)
        source: Data source ('YAHOO' for now, can extend to broker APIs)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    cache_file = HISTORICAL_SPOT_DIR / f"{symbol}_{start_date}_{end_date}_{interval}.parquet"

    # Check cache first
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded cached historical data for {symbol}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # Fetch from source
    if source == "YAHOO":
        df = _fetch_from_yahoo(symbol, start_date, end_date, interval)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not df.empty:
        # Cache the data
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached historical data for {symbol}")

    return df


def _fetch_from_yahoo(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance API.
    Note: This is a basic implementation. In production, consider yfinance library.
    """
    try:
        # Convert dates to Unix timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())

        # Yahoo Finance API URL
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': interval.lower(),
            'events': 'history'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        # Parse CSV response
        df = pd.read_csv(pd.io.common.StringIO(response.text))

        # Clean and format
        df['timestamp'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })

        # Keep only OHLCV columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Convert timezone to IST
        ist_tz = pytz.timezone(IST_TIMEZONE)
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist_tz)

        return df

    except Exception as e:
        logger.error(f"Failed to fetch Yahoo data for {symbol}: {e}")
        return pd.DataFrame()


def get_recent_spot_history(symbol: str, days: int = 30) -> pd.DataFrame:
    """
    Get recent historical data for TA calculations.

    Args:
        symbol: Underlying symbol
        days: Number of days of history to fetch

    Returns:
        DataFrame with recent OHLC data
    """
    end_date = datetime.now(pytz.timezone(IST_TIMEZONE)).strftime('%Y-%m-%d')
    start_date = (datetime.now(pytz.timezone(IST_TIMEZONE)) - timedelta(days=days)).strftime('%Y-%m-%d')

    return fetch_historical_spot_ohlc(symbol, start_date, end_date)