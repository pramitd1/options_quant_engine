"""
Module: spot_downloader.py

Purpose:
    Download spot market data for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


IST_TIMEZONE = "Asia/Kolkata"


def normalize_underlying_symbol(symbol: str) -> str:
    """
    Purpose:
        Normalize underlying symbol into the repository-standard representation.
    
    Context:
        Public function in the `spot downloader` module. It forms part of the data workflow exposed by this module.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
    
    Returns:
        str: Value returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    return str(symbol or "").upper().strip()


def _normalize_symbol(symbol: str) -> str:
    """
    Purpose:
        Normalize symbol into the repository-standard form.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    normalized = normalize_underlying_symbol(symbol)
    ticker_map = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^NSEFIN",
    }
    if normalized in ticker_map:
        return ticker_map[normalized]

    if normalized.startswith("^") or "." in normalized:
        return normalized

    # NSE cash equities are typically exposed via the .NS Yahoo Finance suffix.
    return f"{normalized}.NS"


def _to_ist_timestamp(index_value):
    """
    Purpose:
        Convert ist timestamp into the representation expected downstream.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        index_value (Any): Input associated with index value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    ts = pd.Timestamp(index_value)

    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize(IST_TIMEZONE)
        except Exception:
            pass
    else:
        try:
            ts = ts.tz_convert(IST_TIMEZONE)
        except Exception:
            pass

    return ts


def _safe_float(value, default=None):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `spot downloader` module. The module sits in the data layer that ingests, normalizes, and validates market inputs before analytics run.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _compute_lookback_avg_range_pct(daily_hist: pd.DataFrame, completed_days: int = 10):
    """
    Purpose:
        Compute lookback avg range percentage from the supplied inputs.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        daily_hist (pd.DataFrame): Input associated with daily hist.
        completed_days (int): Input associated with completed days.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if daily_hist is None or daily_hist.empty:
        return None

    df = daily_hist.copy()

    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            return None

    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return None

    df["range_pct"] = ((df["High"] - df["Low"]) / df["Close"]) * 100.0

    if len(df) > 1:
        df = df.iloc[:-1]

    if df.empty:
        return None

    tail = df.tail(completed_days)
    if tail.empty:
        return None

    return round(float(tail["range_pct"].mean()), 4)


def get_spot_snapshot(symbol: str) -> dict:
    """
    Returns a richer live spot snapshot for the engine:
    - spot
    - day_open
    - day_high
    - day_low
    - prev_close
    - timestamp
    - lookback_avg_range_pct
    """

    normalized_symbol = normalize_underlying_symbol(symbol)
    ticker = _normalize_symbol(normalized_symbol)
    data = yf.Ticker(ticker)

    intraday = data.history(period="5d", interval="5m", auto_adjust=False)
    daily = data.history(period="20d", interval="1d", auto_adjust=False)

    if intraday is None or intraday.empty:
        raise ValueError("Unable to fetch intraday spot data")

    intraday = intraday.copy()
    intraday.index = [_to_ist_timestamp(x) for x in intraday.index]

    latest_ts = intraday.index[-1]
    latest_row = intraday.iloc[-1]

    today_mask = [idx.date() == latest_ts.date() for idx in intraday.index]
    today_df = intraday.loc[today_mask].copy()

    if today_df.empty:
        day_open = _safe_float(latest_row.get("Open"))
        day_high = _safe_float(latest_row.get("High"))
        day_low = _safe_float(latest_row.get("Low"))
    else:
        day_open = _safe_float(today_df.iloc[0].get("Open"))
        day_high = _safe_float(today_df["High"].max())
        day_low = _safe_float(today_df["Low"].min())

    spot = _safe_float(latest_row.get("Close"))
    if spot is None:
        raise ValueError("Unable to determine latest spot price")

    prev_close = None
    if daily is not None and not daily.empty and "Close" in daily.columns:
        if len(daily) >= 2:
            prev_close = _safe_float(daily.iloc[-2]["Close"])
        elif len(daily) == 1:
            prev_close = _safe_float(daily.iloc[-1]["Close"])

    lookback_avg_range_pct = _compute_lookback_avg_range_pct(daily)

    snapshot = {
        "symbol": normalized_symbol,
        "ticker": ticker,
        "spot": round(spot, 4),
        "day_open": round(day_open, 4) if day_open is not None else None,
        "day_high": round(day_high, 4) if day_high is not None else None,
        "day_low": round(day_low, 4) if day_low is not None else None,
        "prev_close": round(prev_close, 4) if prev_close is not None else None,
        "timestamp": latest_ts.isoformat(),
        "lookback_avg_range_pct": lookback_avg_range_pct,
    }

    snapshot["validation"] = validate_spot_snapshot(snapshot)
    return snapshot


def validate_spot_snapshot(snapshot: dict, replay_mode: bool = False) -> dict:
    """
    Validate spot snapshot completeness and freshness.
    """

    issues = []
    warnings = []

    spot = _safe_float(snapshot.get("spot"), None)
    day_open = _safe_float(snapshot.get("day_open"), None)
    day_high = _safe_float(snapshot.get("day_high"), None)
    day_low = _safe_float(snapshot.get("day_low"), None)
    prev_close = _safe_float(snapshot.get("prev_close"), None)
    ts_raw = snapshot.get("timestamp")

    if spot in (None, 0):
        issues.append("missing_spot")

    if day_high is not None and day_low is not None and day_high < day_low:
        issues.append("invalid_high_low")

    if spot is not None and day_high is not None and spot > day_high + 5:
        warnings.append("spot_above_day_high")

    if spot is not None and day_low is not None and spot < day_low - 5:
        warnings.append("spot_below_day_low")

    if day_open is None:
        warnings.append("missing_day_open")

    if prev_close is None:
        warnings.append("missing_prev_close")

    age_minutes = None
    is_stale = False
    live_trading_valid = True

    if ts_raw:
        try:
            ts = pd.Timestamp(ts_raw)
            now_ts = pd.Timestamp.now(tz=IST_TIMEZONE)
            age_minutes = round((now_ts - ts).total_seconds() / 60.0, 2)

            market_open = now_ts.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now_ts.replace(hour=15, minute=30, second=0, microsecond=0)
            in_market_hours = market_open <= now_ts <= market_close

            stale_limit = 20 if in_market_hours else 180
            is_stale = age_minutes > stale_limit

            if is_stale:
                live_trading_valid = False
                if replay_mode:
                    warnings.append(f"replay_stale_spot_snapshot_{age_minutes}m")
                else:
                    issues.append(f"stale_spot_snapshot_{age_minutes}m")
        except Exception:
            warnings.append("timestamp_parse_failed")
    else:
        warnings.append("missing_timestamp")
        if replay_mode:
            live_trading_valid = False

    replay_analysis_valid = len([issue for issue in issues if not str(issue).startswith("stale_spot_snapshot_")]) == 0
    is_valid = replay_analysis_valid if replay_mode else len(issues) == 0

    return {
        "is_valid": is_valid,
        "live_trading_valid": live_trading_valid and len(issues) == 0,
        "replay_analysis_valid": replay_analysis_valid,
        "validation_mode": "REPLAY" if replay_mode else "LIVE",
        "is_stale": is_stale,
        "age_minutes": age_minutes,
        "issues": issues,
        "warnings": warnings,
    }


def get_spot_price(symbol: str):
    """
    Backward-compatible wrapper.
    """
    snapshot = get_spot_snapshot(symbol)
    return float(snapshot["spot"])


def save_spot_snapshot(snapshot: dict, output_dir: str = "debug_samples"):
    """
    Save one live spot snapshot for inspection.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = snapshot.get("symbol", "UNKNOWN")
    ts = snapshot.get("timestamp", "").replace(":", "-")
    filename = out_dir / f"{symbol}_spot_snapshot_{ts}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    return str(filename)
