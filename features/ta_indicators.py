"""
Module: ta_indicators.py

Purpose:
    Compute technical analysis indicators from historical spot price data
    to generate trading signals for the options quant engine.

Role in the System:
    Part of the features layer that provides TA-based signals for signal aggregation.
    Computes indicators like RSI, MACD, moving averages from OHLC data.

Key Outputs:
    Dictionary with TA signals: direction, confidence, regime labels.

Downstream Usage:
    Consumed by signal evaluation for combining with quant signals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.analytics_feature_policy import (
    TechnicalAnalysisPolicyConfig,
    get_technical_analysis_policy_config,
)
from data.historical_spot_fetcher import get_recent_spot_history

logger = logging.getLogger(__name__)


def _no_ta_signal(regime: str, *, warning: str | None = None) -> dict:
    payload = {
        "ta_direction": "NO_SIGNAL",
        "ta_confidence": 0.0,
        "ta_regime": regime,
        "indicators": {},
    }
    if warning:
        payload["ta_warning"] = warning
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
        except Exception as exc:
            logger.warning("Unable to filter TA history as_of=%r: %s", as_of, exc)

    return df.dropna(subset=["close"]) if "close" in df.columns else pd.DataFrame()


def build_ta_features(
    symbol: str,
    current_spot: float,
    days_history: int | None = None,
    *,
    history_df: pd.DataFrame | None = None,
    as_of=None,
    allow_live_history: bool = True,
) -> dict:
    """
    Build technical analysis features for signal generation.

    Args:
        symbol: Underlying symbol (e.g., 'NIFTY')
        current_spot: Current spot price
        days_history: Days of historical data to use

    Returns:
        Dict with TA signals and metadata
    """
    try:
        cfg = get_technical_analysis_policy_config()
        lookback_days = int(days_history or cfg.default_history_days)
        if history_df is None:
            if not allow_live_history:
                return _no_ta_signal(
                    "point_in_time_unavailable",
                    warning="ta_history_not_supplied_for_historical_mode",
                )
            history_df = get_recent_spot_history(symbol, lookback_days)

        hist_df = _prepare_history_frame(history_df, as_of=as_of)

        if hist_df.empty or len(hist_df) < cfg.minimum_history_rows:
            return _no_ta_signal("insufficient_data")

        # Compute indicators
        indicators = _compute_ta_indicators(hist_df, current_spot, cfg)

        # Generate signals
        direction, confidence, regime = _generate_ta_signals(indicators, cfg)

        return {
            "ta_direction": direction,
            "ta_confidence": confidence,
            "ta_regime": regime,
            "indicators": indicators,
        }

    except Exception as e:
        logger.error(f"Failed to build TA features for {symbol}: {e}")
        return _no_ta_signal("error")


def _valid_window(value: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, parsed)


def _close_price_series(hist_df: pd.DataFrame) -> pd.Series:
    """Return numeric close prices while preserving the original index."""
    if "close" not in hist_df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(hist_df["close"], errors="coerce").dropna()


def _return_bps(current_spot: float, base_price: float) -> float | None:
    try:
        spot = float(current_spot)
        base = float(base_price)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(spot) or not np.isfinite(base) or base == 0.0:
        return None
    return ((spot / base) - 1.0) * 10000.0


def _compute_ta_indicator_series(
    hist_df: pd.DataFrame,
    cfg: TechnicalAnalysisPolicyConfig | None = None,
) -> dict[str, pd.Series]:
    """Compute technical indicator series for charts and latest-value extraction."""
    cfg = cfg or get_technical_analysis_policy_config()
    close_prices = _close_price_series(hist_df)
    if close_prices.empty:
        return {}

    series: dict[str, pd.Series] = {}
    sma_fast_window = _valid_window(cfg.sma_fast_window)
    sma_slow_window = _valid_window(cfg.sma_slow_window)
    ema_fast_span = _valid_window(cfg.ema_fast_span)
    ema_slow_span = _valid_window(cfg.ema_slow_span)
    macd_signal_span = _valid_window(cfg.macd_signal_span)
    rsi_window = _valid_window(cfg.rsi_window)
    bollinger_window = _valid_window(cfg.bollinger_window)

    if len(close_prices) >= sma_fast_window:
        series[f"sma_{sma_fast_window}"] = close_prices.rolling(window=sma_fast_window).mean()
    if len(close_prices) >= sma_slow_window:
        series[f"sma_{sma_slow_window}"] = close_prices.rolling(window=sma_slow_window).mean()

    if len(close_prices) >= ema_fast_span:
        series[f"ema_{ema_fast_span}"] = close_prices.ewm(span=ema_fast_span).mean()
    if len(close_prices) >= ema_slow_span:
        series[f"ema_{ema_slow_span}"] = close_prices.ewm(span=ema_slow_span).mean()

    if len(close_prices) >= max(ema_fast_span, ema_slow_span):
        ema_fast = close_prices.ewm(span=ema_fast_span).mean()
        ema_slow = close_prices.ewm(span=ema_slow_span).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal_span).mean()
        series["macd_line"] = macd_line
        series["macd_signal"] = signal_line
        series["macd_histogram"] = macd_line - signal_line

    if len(close_prices) >= rsi_window:
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_window).mean()
        rs = gain / loss.mask(loss == 0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((loss == 0.0) & (gain <= 0.0), 50.0)
        series["rsi"] = rsi

    if len(close_prices) >= bollinger_window:
        sma = close_prices.rolling(window=bollinger_window).mean()
        std = close_prices.rolling(window=bollinger_window).std()
        series["bb_lower"] = sma - cfg.bollinger_std_mult * std
        series["bb_upper"] = sma + cfg.bollinger_std_mult * std
        series["bb_sma"] = sma

    return series


def _compute_ta_indicators(
    hist_df: pd.DataFrame,
    current_spot: float,
    cfg: TechnicalAnalysisPolicyConfig | None = None,
) -> dict:
    """Compute technical indicators manually."""
    indicators = {}
    cfg = cfg or get_technical_analysis_policy_config()

    # Use close prices for indicators
    close_prices = _close_price_series(hist_df)
    if close_prices.empty:
        return indicators

    try:
        sma_fast_window = _valid_window(cfg.sma_fast_window)
        for key, values in _compute_ta_indicator_series(hist_df, cfg).items():
            latest = values.iloc[-1]
            if pd.notna(latest):
                indicators[key] = float(latest)

        if len(close_prices) >= 20:
            ret_20d = _return_bps(current_spot, close_prices.iloc[-20])
            if ret_20d is not None:
                indicators["ret_20d_bps"] = ret_20d
        if sma_fast_window != 20 and len(close_prices) >= sma_fast_window:
            ret_fast = _return_bps(current_spot, close_prices.iloc[-sma_fast_window])
            if ret_fast is not None:
                indicators[f"ret_{sma_fast_window}d_bps"] = ret_fast

    except Exception as e:
        logger.warning(f"Error computing TA indicators: {e}")

    return indicators


def _generate_ta_signals(
    indicators: dict,
    cfg: TechnicalAnalysisPolicyConfig | None = None,
) -> tuple[str, float, str]:
    """Generate trading signals from indicators."""
    cfg = cfg or get_technical_analysis_policy_config()
    direction = "NO_SIGNAL"
    confidence = 0.0
    regime = "neutral"

    signals = []
    sma_fast_key = f"sma_{_valid_window(cfg.sma_fast_window)}"
    sma_slow_key = f"sma_{_valid_window(cfg.sma_slow_window)}"

    # Moving Average signals
    if indicators.get(sma_fast_key) is not None and indicators.get(sma_slow_key) is not None:
        sma_fast = indicators[sma_fast_key]
        sma_slow = indicators[sma_slow_key]

        if sma_fast > sma_slow:
            signals.append(("CALL", cfg.trend_signal_confidence, "bullish_trend"))
        elif sma_fast < sma_slow:
            signals.append(("PUT", cfg.trend_signal_confidence, "bearish_trend"))

    # MACD signals
    if indicators.get('macd_histogram') is not None:
        macd_hist = indicators['macd_histogram']

        if macd_hist > 0:
            signals.append(("CALL", cfg.macd_signal_confidence, "macd_bullish"))
        elif macd_hist < 0:
            signals.append(("PUT", cfg.macd_signal_confidence, "macd_bearish"))

    # RSI signals
    if indicators.get('rsi') is not None:
        rsi = indicators['rsi']

        if rsi > cfg.rsi_overbought:
            signals.append(("PUT", cfg.rsi_signal_confidence, "overbought"))
        elif rsi < cfg.rsi_oversold:
            signals.append(("CALL", cfg.rsi_signal_confidence, "oversold"))

    # Aggregate signals
    if signals:
        # Simple majority vote with average confidence
        call_signals = [s for s in signals if s[0] == "CALL"]
        put_signals = [s for s in signals if s[0] == "PUT"]

        if len(call_signals) > len(put_signals):
            direction = "CALL"
            confidence = sum(s[1] for s in call_signals) / len(call_signals)
            regime = call_signals[0][2] if call_signals else "bullish"
        elif len(put_signals) > len(call_signals):
            direction = "PUT"
            confidence = sum(s[1] for s in put_signals) / len(put_signals)
            regime = put_signals[0][2] if put_signals else "bearish"
        else:
            direction = "NO_SIGNAL"
            confidence = 0.0
            regime = "mixed_signals"

    return direction, float(confidence), regime


# Helper function to get TA features for a trade
def get_ta_features_for_trade(
    symbol: str,
    spot_price: float,
    *,
    history_df: pd.DataFrame | None = None,
    as_of=None,
    allow_live_history: bool = True,
) -> dict:
    """
    Convenience function to get TA features for integration with trade evaluation.

    Args:
        symbol: Underlying symbol
        spot_price: Current spot price

    Returns:
        TA features dict ready for trade evaluation
    """
    return build_ta_features(
        symbol,
        spot_price,
        history_df=history_df,
        as_of=as_of,
        allow_live_history=allow_live_history,
    )
