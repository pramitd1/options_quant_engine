from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf

from config.market_data_policy import IST_TIMEZONE, normalize_symbol_to_yfinance


def coerce_market_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(IST_TIMEZONE)
    return ts.tz_convert(IST_TIMEZONE)


def resolve_research_as_of(as_of=None, *, default=None) -> pd.Timestamp:
    if as_of is not None:
        return coerce_market_timestamp(as_of)
    if default is not None:
        return coerce_market_timestamp(default)
    return pd.Timestamp.now(tz=IST_TIMEZONE)


def fetch_realized_spot_history(symbol: str, *, start_ts, end_ts, interval: str = "5m") -> pd.DataFrame:
    start_ts = coerce_market_timestamp(start_ts)
    end_ts = coerce_market_timestamp(end_ts)

    ticker = yf.Ticker(normalize_symbol_to_yfinance(symbol))
    frame = ticker.history(
        start=start_ts.tz_convert("UTC").to_pydatetime(),
        end=end_ts.tz_convert("UTC").to_pydatetime(),
        interval=interval,
        auto_adjust=False,
    )

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    history = frame.reset_index()
    ts_col = "Datetime" if "Datetime" in history.columns else "Date"
    history["timestamp"] = history[ts_col].map(coerce_market_timestamp)
    history["spot"] = pd.to_numeric(history.get("Close"), errors="coerce")
    history = history.dropna(subset=["timestamp", "spot"])
    return history[["timestamp", "spot"]].reset_index(drop=True)


def slice_realized_spot_history(history: pd.DataFrame, *, signal_timestamp, as_of=None) -> pd.DataFrame:
    signal_ts = coerce_market_timestamp(signal_timestamp)
    end_ts = resolve_research_as_of(as_of, default=signal_ts)
    fetch_end_ts = max(end_ts, signal_ts + pd.Timedelta(days=2))

    if history is None or history.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    return history.loc[
        (history["timestamp"] >= signal_ts) & (history["timestamp"] <= fetch_end_ts)
    ].reset_index(drop=True)


def fetch_realized_spot_path(symbol: str, signal_timestamp, *, as_of=None, interval: str = "5m") -> pd.DataFrame:
    signal_ts = coerce_market_timestamp(signal_timestamp)
    end_ts = resolve_research_as_of(as_of, default=signal_ts)
    fetch_end_ts = max(end_ts, signal_ts + pd.Timedelta(days=2))
    history = fetch_realized_spot_history(
        symbol,
        start_ts=signal_ts - pd.Timedelta(days=1),
        end_ts=fetch_end_ts + pd.Timedelta(days=1),
        interval=interval,
    )
    return slice_realized_spot_history(
        history,
        signal_timestamp=signal_ts,
        as_of=fetch_end_ts,
    )


def build_realized_spot_path_cache(
    rows: Iterable[dict],
    *,
    as_of=None,
    interval: str = "5m",
    fetch_history_fn=fetch_realized_spot_history,
) -> dict[tuple[str, str], pd.DataFrame]:
    row_list = [row for row in rows if row.get("symbol") and row.get("signal_timestamp")]
    if not row_list:
        return {}

    signal_defaults = [coerce_market_timestamp(row["signal_timestamp"]) for row in row_list]
    as_of_ts = resolve_research_as_of(as_of, default=max(signal_defaults))
    grouped_rows: dict[str, list[dict]] = {}
    for row in row_list:
        grouped_rows.setdefault(str(row["symbol"]), []).append(row)

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, symbol_rows in grouped_rows.items():
        signal_times = [coerce_market_timestamp(row["signal_timestamp"]) for row in symbol_rows]
        fetch_start_ts = min(signal_times) - pd.Timedelta(days=1)
        fetch_end_ts = max(max(as_of_ts, signal_ts + pd.Timedelta(days=2)) for signal_ts in signal_times) + pd.Timedelta(days=1)
        history = fetch_history_fn(
            symbol,
            start_ts=fetch_start_ts,
            end_ts=fetch_end_ts,
            interval=interval,
        )
        for row in symbol_rows:
            key = (str(row["symbol"]), str(row["signal_timestamp"]))
            cache[key] = slice_realized_spot_history(
                history,
                signal_timestamp=row["signal_timestamp"],
                as_of=as_of_ts,
            )

    return cache
