"""
Module: global_market_snapshot.py

Purpose:
    Implement global market snapshot data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

import math
import time as _time_mod

import pandas as pd
import yfinance as yf

from utils.numerics import safe_float as _safe_float

from config.market_data_policy import (
    GLOBAL_MARKET_TICKERS,
    IST_TIMEZONE,
    normalize_symbol_to_yfinance,
)
from config.settings import (
    GLOBAL_MARKET_DATA_ENABLED,
    GLOBAL_MARKET_LOOKBACK_DAYS,
    GLOBAL_MARKET_STALE_DAYS,
)


def _coerce_timestamp(value):
    """
    Purpose:
        Parse flexible timestamp inputs into timezone-aware timestamps.

    Context:
        Function inside the `global market snapshot` module. The module sits in the data layer that ingests, normalizes, and validates market inputs before analytics run.

    Inputs:
        value (Any): Raw value supplied by the caller.

    Returns:
        pd.Timestamp | None: Parsed timestamp or `None` when parsing fails.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    if value is None or value == "":
        return None

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return None

    try:
        return parsed.tz_convert(IST_TIMEZONE)
    except Exception:
        return None


def _symbol_to_yfinance(symbol: str) -> str:
    """
    Purpose:
        Process symbol to yfinance for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    return normalize_symbol_to_yfinance(symbol)


def _download_history(ticker: str, *, lookback_days: int) -> pd.DataFrame:
    """
    Purpose:
        Process download history for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        ticker (str): Input associated with ticker.
        lookback_days (int): Input associated with lookback days.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    history = yf.download(
        ticker,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalize_download_history(history)


def _normalize_download_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Normalize download history into the repository-standard form.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        history (pd.DataFrame): Input associated with history.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [
            col[0] if isinstance(col, tuple) and len(col) > 0 else col
            for col in history.columns
        ]

    history = history.reset_index()
    time_col = "Date" if "Date" in history.columns else "Datetime" if "Datetime" in history.columns else None
    if time_col is None or "Close" not in history.columns:
        return pd.DataFrame()

    history["timestamp"] = pd.to_datetime(history[time_col], errors="coerce", utc=True)
    history["close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history.dropna(subset=["timestamp", "close"]).copy()
    if history.empty:
        return pd.DataFrame()

    history["timestamp"] = history["timestamp"].dt.tz_convert(IST_TIMEZONE)
    return history[["timestamp", "close"]].reset_index(drop=True)


def _extract_batch_history(batch_history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Purpose:
        Extract batch history from the supplied payload.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        batch_history (pd.DataFrame): Input associated with batch history.
        ticker (str): Input associated with ticker.
    
    Returns:
        pd.DataFrame: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if batch_history is None or batch_history.empty:
        return pd.DataFrame()

    if isinstance(batch_history.columns, pd.MultiIndex):
        level_zero = batch_history.columns.get_level_values(0)
        if ticker in level_zero:
            return _normalize_download_history(batch_history[ticker].copy())
    return _normalize_download_history(batch_history.copy())


# TTL cache for yfinance downloads (daily data rarely changes intra-session)
_DOWNLOAD_CACHE_TTL_SECONDS = 300  # 5 minutes
_download_cache: dict | None = None
_download_cache_key: tuple | None = None
_download_cache_time: float = 0.0


def invalidate_download_cache():
    """Clear the yfinance download cache (used by tests and forced refresh)."""
    global _download_cache, _download_cache_key, _download_cache_time
    _download_cache = None
    _download_cache_key = None
    _download_cache_time = 0.0


def _download_histories(tickers: dict[str, str], *, lookback_days: int) -> dict[str, pd.DataFrame]:
    """
    Purpose:
        Process download histories for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        tickers (dict[str, str]): Input associated with tickers.
        lookback_days (int): Input associated with lookback days.
    
    Returns:
        dict[str, pd.DataFrame]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if not tickers:
        return {}

    global _download_cache, _download_cache_key, _download_cache_time
    cache_key = (tuple(sorted(tickers.items())), lookback_days)
    now = _time_mod.monotonic()
    if (
        _download_cache is not None
        and _download_cache_key == cache_key
        and (now - _download_cache_time) < _DOWNLOAD_CACHE_TTL_SECONDS
    ):
        return _download_cache

    unique_tickers = list(dict.fromkeys(tickers.values()))
    batch_history = pd.DataFrame()
    try:
        batch_history = yf.download(
            unique_tickers,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="ticker",
        )
    except Exception:
        batch_history = pd.DataFrame()

    ticker_cache: dict[str, pd.DataFrame] = {}
    histories: dict[str, pd.DataFrame] = {}
    for name, ticker in tickers.items():
        history = ticker_cache.get(ticker)
        if history is None:
            history = _extract_batch_history(batch_history, ticker)
            if history.empty:
                history = _download_history(ticker, lookback_days=lookback_days)
            ticker_cache[ticker] = history
        histories[name] = history

    _download_cache = histories
    _download_cache_key = cache_key
    _download_cache_time = now
    return histories


def _daily_change_pct(history: pd.DataFrame):
    """
    Purpose:
        Process daily change percentage for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        history (pd.DataFrame): Input associated with history.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if history is None or history.empty or len(history) < 2:
        return None

    latest = _safe_float(history.iloc[-1]["close"], None)
    prev = _safe_float(history.iloc[-2]["close"], None)
    if latest in (None, 0) or prev in (None, 0):
        return None

    return ((latest / prev) - 1.0) * 100.0


def _us10y_change_bp(history: pd.DataFrame):
    """
    Purpose:
        Process us10y change basis points for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        history (pd.DataFrame): Input associated with history.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if history is None or history.empty or len(history) < 2:
        return None

    latest = _safe_float(history.iloc[-1]["close"], None)
    prev = _safe_float(history.iloc[-2]["close"], None)
    if latest is None or prev is None:
        return None

    # Yahoo ^TNX is typically 10x the yield percentage, so 1.0 index point ~= 10 bp.
    return (latest - prev) * 10.0


def _realized_volatility(history: pd.DataFrame, window: int):
    """
    Purpose:
        Process realized volatility for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        history (pd.DataFrame): Input associated with history.
        window (int): Input associated with window.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if history is None or history.empty or len(history) < (window + 1):
        return None

    closes = pd.to_numeric(history["close"], errors="coerce").dropna()
    if len(closes) < (window + 1):
        return None

    log_returns = closes.apply(math.log).diff().dropna()
    if len(log_returns) < window:
        return None

    sample = log_returns.tail(window)
    if sample.empty:
        return None

    return float(sample.std(ddof=0) * math.sqrt(252.0))


def _latest_close(history: pd.DataFrame):
    """
    Purpose:
        Return the most recent close from a normalized market history.

    Context:
        Internal helper used when downstream layers need a current level, not
        just a one-day change. India VIX is especially useful as a local
        volatility level for option-valuation heuristics.

    Inputs:
        history (pd.DataFrame): Normalized price history with a `close` column.

    Returns:
        float | None: Latest close value or `None` when unavailable.

    Notes:
        This helper leaves scoring decisions to downstream layers and only
        exposes the raw level in a normalized form.
    """
    if history is None or history.empty:
        return None
    return _safe_float(history.iloc[-1]["close"], None)


def _neutral_market_snapshot(symbol: str, as_of=None, issues=None, warnings=None, provider="YFINANCE"):
    """
    Purpose:
        Process neutral market snapshot for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        as_of (Any): Input associated with as of.
        issues (Any): Input associated with issues.
        warnings (Any): Input associated with warnings.
        provider (Any): Input associated with provider.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    return {
        "symbol": str(symbol or "").upper().strip(),
        "provider": provider,
        "as_of": (_coerce_timestamp(as_of) or pd.Timestamp.now(tz=IST_TIMEZONE)).isoformat(),
        "data_available": False,
        "neutral_fallback": True,
        "issues": issues or [],
        "warnings": warnings or [],
        "stale": True,
        "lookback_days": GLOBAL_MARKET_LOOKBACK_DAYS,
        "market_inputs": {},
    }


def build_global_market_snapshot(symbol: str, *, as_of=None) -> dict:
    """
    Purpose:
        Build the global market snapshot used by downstream components.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        as_of (Any): Input associated with as of.
    
    Returns:
        dict: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if not GLOBAL_MARKET_DATA_ENABLED:
        return _neutral_market_snapshot(
            symbol,
            as_of=as_of,
            warnings=["global_market_data_disabled"],
        )

    as_of_ts = _coerce_timestamp(as_of) or pd.Timestamp.now(tz=IST_TIMEZONE)
    issues = []
    warnings = []
    market_inputs = {}

    underlying_ticker = _symbol_to_yfinance(symbol)
    tickers = dict(GLOBAL_MARKET_TICKERS)
    tickers["underlying"] = underlying_ticker
    histories = _download_histories(tickers, lookback_days=GLOBAL_MARKET_LOOKBACK_DAYS)

    latest_timestamps = []
    for name, history in histories.items():
        if history is None or history.empty:
            warnings.append(f"missing_{name}_history")
            continue
        latest_timestamps.append(history.iloc[-1]["timestamp"])

    if not latest_timestamps:
        return _neutral_market_snapshot(
            symbol,
            as_of=as_of_ts,
            issues=["global_market_history_unavailable"],
            warnings=warnings,
        )

    latest_market_ts = max(latest_timestamps)
    stale_days = max((as_of_ts - latest_market_ts).total_seconds() / (24 * 3600.0), 0.0)
    stale = stale_days > GLOBAL_MARKET_STALE_DAYS
    if stale:
        warnings.append(f"global_market_snapshot_stale:{round(stale_days, 2)}d")

    market_inputs["oil_change_24h"] = _daily_change_pct(histories.get("oil"))
    market_inputs["gold_change_24h"] = _daily_change_pct(histories.get("gold"))
    market_inputs["copper_change_24h"] = _daily_change_pct(histories.get("copper"))
    market_inputs["vix_change_24h"] = _daily_change_pct(histories.get("vix"))
    market_inputs["india_vix_change_24h"] = _daily_change_pct(histories.get("india_vix"))
    market_inputs["india_vix_level"] = _latest_close(histories.get("india_vix"))
    market_inputs["sp500_change_24h"] = _daily_change_pct(histories.get("sp500"))
    market_inputs["nasdaq_change_24h"] = _daily_change_pct(histories.get("nasdaq"))
    market_inputs["us10y_change_bp"] = _us10y_change_bp(histories.get("us10y"))
    market_inputs["usdinr_change_24h"] = _daily_change_pct(histories.get("usdinr"))
    market_inputs["realized_vol_5d"] = _realized_volatility(histories.get("underlying"), 5)
    market_inputs["realized_vol_30d"] = _realized_volatility(histories.get("underlying"), 30)

    if all(value is None for value in market_inputs.values()):
        return _neutral_market_snapshot(
            symbol,
            as_of=as_of_ts,
            issues=["global_market_inputs_empty"],
            warnings=warnings,
        )

    return {
        "symbol": str(symbol or "").upper().strip(),
        "provider": "YFINANCE",
        "as_of": as_of_ts.isoformat(),
        "data_available": not stale,
        "neutral_fallback": stale,
        "issues": issues,
        "warnings": warnings,
        "stale": stale,
        "latest_market_timestamp": latest_market_ts.isoformat(),
        "lookback_days": GLOBAL_MARKET_LOOKBACK_DAYS,
        "market_inputs": market_inputs,
    }
