from __future__ import annotations

import pandas as pd

from data.historical_spot_fetcher import get_recent_spot_history


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
        return {
            "mean_reversion_signal": "INSUFFICIENT_DATA",
            "mean_reversion_zscore": 0.0,
            "mean_reversion_strength": 0.0,
            "mean_reversion_distance_pct": 0.0,
        }

    lookback = min(max(2, lookback), len(series))
    window = series.iloc[-lookback:]
    returns = window.pct_change().dropna()
    if returns.empty:
        return {
            "mean_reversion_signal": "INSUFFICIENT_DATA",
            "mean_reversion_zscore": 0.0,
            "mean_reversion_strength": 0.0,
            "mean_reversion_distance_pct": 0.0,
        }

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


def get_mean_reversion_features_for_trade(symbol: str, current_spot: float, *, days_history: int = 30) -> dict:
    """Fetch historical spot prices and compute mean reversion features for a trade."""
    try:
        hist_df = get_recent_spot_history(symbol, days_history)
        if hist_df is None or hist_df.empty or "close" not in hist_df.columns:
            return {
                "mean_reversion_signal": "INSUFFICIENT_DATA",
                "mean_reversion_zscore": 0.0,
                "mean_reversion_strength": 0.0,
                "mean_reversion_distance_pct": 0.0,
            }
        prices = hist_df["close"].dropna().astype(float).tolist()
        return compute_mean_reversion_features(prices, lookback=min(20, len(prices)))
    except Exception:
        return {
            "mean_reversion_signal": "ERROR",
            "mean_reversion_zscore": 0.0,
            "mean_reversion_strength": 0.0,
            "mean_reversion_distance_pct": 0.0,
        }


def detect_mean_reversion_opportunity(prices: list[float] | pd.Series, *, threshold: float = 1.8) -> bool:
    features = compute_mean_reversion_features(prices)
    return features["mean_reversion_signal"] == "MEAN_REVERSION" and abs(features["mean_reversion_zscore"]) >= threshold
