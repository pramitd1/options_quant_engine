"""
Module: market_data.py

Purpose:
    Fetch and normalize realized spot paths used to evaluate captured signals after the fact.

Role in the System:
    Part of the research layer that bridges stored signal snapshots with realized underlying price history.

Key Outputs:
    Timestamp-normalized spot histories and per-signal realized spot paths.

Downstream Usage:
    Consumed by the signal evaluator, reporting layer, and parameter-governance workflows.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf

from config.market_data_policy import IST_TIMEZONE, normalize_symbol_to_yfinance


def coerce_market_timestamp(value) -> pd.Timestamp:
    """
    Purpose:
        Normalize timestamp-like inputs into the project's research timezone.

    Context:
        Signal capture, realized market history, and report generation all need to agree on a single timezone before horizons and session boundaries can be computed reliably.

    Inputs:
        value (Any): Timestamp-like value to normalize.

    Returns:
        pd.Timestamp: Timestamp localized or converted to `IST_TIMEZONE`.

    Notes:
        Keeping this conversion centralized prevents subtle differences between live captures, replay runs, and delayed backfills.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(IST_TIMEZONE)
    return ts.tz_convert(IST_TIMEZONE)


def resolve_research_as_of(as_of=None, *, default=None) -> pd.Timestamp:
    """
    Purpose:
        Resolve the timestamp that bounds how much future data research code is allowed to inspect.

    Context:
        Evaluation and reporting workflows often run incrementally, so they need an explicit "known up to" time instead of always looking at the latest available market history.

    Inputs:
        as_of (Any): Explicit lookahead cutoff, when provided.
        default (Any): Fallback timestamp used when `as_of` is absent.

    Returns:
        pd.Timestamp: Resolved evaluation cutoff in `IST_TIMEZONE`.

    Notes:
        If neither input is supplied, the helper falls back to the current time.
    """
    if as_of is not None:
        return coerce_market_timestamp(as_of)
    if default is not None:
        return coerce_market_timestamp(default)
    return pd.Timestamp.now(tz=IST_TIMEZONE)


def fetch_realized_spot_history(symbol: str, *, start_ts, end_ts, interval: str = "5m") -> pd.DataFrame:
    """
    Purpose:
        Download realized underlying prices for the requested symbol and time window.

    Context:
        The signal-evaluation pipeline scores directional quality on the underlying rather than reconstructed option PnL, so it needs a clean realized spot history for each signal window.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        start_ts (Any): Inclusive start timestamp for the requested history.
        end_ts (Any): Exclusive end timestamp for the requested history.
        interval (str): Bar interval passed to the market-data provider.

    Returns:
        pd.DataFrame: Two-column frame with normalized `timestamp` and `spot` values.

    Notes:
        Prices are fetched through `yfinance`, so this is a research convenience path rather than a production market-data feed.
    """
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
    """
    Purpose:
        Trim a fetched history down to the portion relevant for one captured signal.

    Context:
        Evaluation rows should only see prices from the signal timestamp forward, bounded by the current research `as_of` horizon.

    Inputs:
        history (pd.DataFrame): Realized spot history to slice.
        signal_timestamp (Any): Timestamp when the signal was captured.
        as_of (Any): Lookahead cutoff for the returned slice.

    Returns:
        pd.DataFrame: Spot history beginning at the signal timestamp and ending at the allowed lookahead boundary.

    Notes:
        The helper preserves a small two-day buffer so later next-session metrics can be computed from one fetched history block.
    """
    signal_ts = coerce_market_timestamp(signal_timestamp)
    end_ts = resolve_research_as_of(as_of, default=signal_ts)
    fetch_end_ts = max(end_ts, signal_ts + pd.Timedelta(days=2))

    if history is None or history.empty:
        return pd.DataFrame(columns=["timestamp", "spot"])

    return history.loc[
        (history["timestamp"] >= signal_ts) & (history["timestamp"] <= fetch_end_ts)
    ].reset_index(drop=True)


def fetch_realized_spot_path(symbol: str, signal_timestamp, *, as_of=None, interval: str = "5m") -> pd.DataFrame:
    """
    Purpose:
        Fetch the realized spot path needed to evaluate a single captured signal.

    Context:
        This is the single-signal convenience wrapper used by research scripts and backfills that do not already have a shared history cache.

    Inputs:
        symbol (str): Underlying symbol or index identifier.
        signal_timestamp (Any): Timestamp when the signal was captured.
        as_of (Any): Lookahead cutoff for evaluation.
        interval (str): Bar interval passed to the market-data provider.

    Returns:
        pd.DataFrame: Realized spot path aligned to the requested signal timestamp.

    Notes:
        The function fetches extra padding before and after the signal so slicing logic can derive intraday and next-session checkpoints consistently.
    """
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
    """
    Purpose:
        Build a shared realized-spot cache for many signal rows at once.

    Context:
        Batch evaluation is much cheaper when signals in the same symbol reuse one downloaded history block instead of refetching data per row.

    Inputs:
        rows (Iterable[dict]): Signal rows containing at least `symbol` and `signal_timestamp`.
        as_of (Any): Lookahead cutoff applied to every cached path.
        interval (str): Bar interval passed to the market-data provider.
        fetch_history_fn (Any): History-fetch function used to populate the cache, mainly for testing or alternate providers.

    Returns:
        dict[tuple[str, str], pd.DataFrame]: Mapping from `(symbol, signal_timestamp)` to the corresponding realized spot path.

    Notes:
        Grouping by symbol keeps repeated research backfills fast while preserving the exact per-signal slice expected by the evaluator.
    """
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
