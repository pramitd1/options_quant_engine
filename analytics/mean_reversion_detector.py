from __future__ import annotations

import pandas as pd

from data.historical_spot_fetcher import get_recent_spot_history


def _empty_mean_reversion(signal: str, *, reason: str | None = None) -> dict:
    payload = {
        "mean_reversion_signal": signal,
        "mean_reversion_zscore": 0.0,
        "mean_reversion_strength": 0.0,
        "mean_reversion_distance_pct": 0.0,
    }
    if reason:
        payload["mean_reversion_reason"] = reason
    return payload


def _prepare_history_frame(hist_df: pd.DataFrame | None, *, as_of=None) -> pd.DataFrame:
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()

    df = hist_df.copy()
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    if as_of is not None and "timestamp" in df.columns:
        try:
            history_ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            as_of_ts = pd.Timestamp(as_of)
            if as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize("Asia/Kolkata")
            else:
                as_of_ts = as_of_ts.tz_convert("Asia/Kolkata")
            df = df[history_ts <= as_of_ts.tz_convert("UTC")]
        except Exception:
            return pd.DataFrame()

    return df.dropna(subset=["close"]) if "close" in df.columns else pd.DataFrame()


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compute_mean_reversion_features(
    prices: list[float] | pd.Series,
    *,
    lookback: int = 20,
    zscore_threshold: float = 1.5,
) -> dict:
    """Compute simple mean reversion indicators for a price series."""
    series = pd.Series(prices).dropna().astype(float)
    if series.empty:
        return _empty_mean_reversion("INSUFFICIENT_DATA")

    lookback = min(max(2, lookback), len(series))
    window = series.iloc[-lookback:]
    returns = window.pct_change().dropna()
    if returns.empty:
        return _empty_mean_reversion("INSUFFICIENT_DATA")

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=0))
    latest_return = float(returns.iloc[-1])
    zscore = 0.0 if std_return <= 1e-12 else (latest_return - mean_return) / std_return
    price_std = float(window.std(ddof=0))
    distance_pct = 0.0
    if price_std > 1e-12:
        distance_pct = float((window.iloc[-1] - window.mean()) / price_std)

    signal = "TREND_CONTINUATION" if abs(zscore) <= zscore_threshold else "MEAN_REVERSION"
    strength = _clip(abs(zscore) * 12.5, 0.0, 100.0)

    return {
        "mean_reversion_signal": signal,
        "mean_reversion_zscore": round(zscore, 3),
        "mean_reversion_strength": round(strength, 2),
        "mean_reversion_distance_pct": round(distance_pct, 3),
    }


def get_mean_reversion_features_for_trade(
    symbol: str,
    current_spot: float,
    *,
    days_history: int = 30,
    history_df: pd.DataFrame | None = None,
    as_of=None,
    allow_live_history: bool = True,
) -> dict:
    """Fetch historical spot prices and compute mean reversion features for a trade."""
    try:
        if history_df is None:
            if not allow_live_history:
                return _empty_mean_reversion(
                    "INSUFFICIENT_DATA",
                    reason="mean_reversion_history_not_supplied_for_historical_mode",
                )
            history_df = get_recent_spot_history(symbol, days_history)
        hist_df = _prepare_history_frame(history_df, as_of=as_of)
        if hist_df is None or hist_df.empty or "close" not in hist_df.columns:
            return _empty_mean_reversion("INSUFFICIENT_DATA")
        prices = hist_df["close"].dropna().astype(float).tolist()
        return compute_mean_reversion_features(prices, lookback=min(20, len(prices)))
    except Exception:
        return _empty_mean_reversion("ERROR")


def detect_mean_reversion_opportunity(prices: list[float] | pd.Series, *, threshold: float = 1.8) -> bool:
    features = compute_mean_reversion_features(prices)
    return features["mean_reversion_signal"] == "MEAN_REVERSION" and abs(features["mean_reversion_zscore"]) >= threshold
