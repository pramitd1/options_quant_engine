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
from typing import Optional

import numpy as np
import pandas as pd

from data.historical_spot_fetcher import get_recent_spot_history

logger = logging.getLogger(__name__)


def build_ta_features(
    symbol: str,
    current_spot: float,
    days_history: int = 30
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
        # Fetch historical data
        hist_df = get_recent_spot_history(symbol, days_history)

        if hist_df.empty or len(hist_df) < 14:  # Need minimum data for indicators
            return {
                "ta_direction": "NO_SIGNAL",
                "ta_confidence": 0.0,
                "ta_regime": "insufficient_data",
                "indicators": {}
            }

        # Compute indicators
        indicators = _compute_ta_indicators(hist_df, current_spot)

        # Generate signals
        direction, confidence, regime = _generate_ta_signals(indicators)

        return {
            "ta_direction": direction,
            "ta_confidence": confidence,
            "ta_regime": regime,
            "indicators": indicators
        }

    except Exception as e:
        logger.error(f"Failed to build TA features for {symbol}: {e}")
        return {
            "ta_direction": "NO_SIGNAL",
            "ta_confidence": 0.0,
            "ta_regime": "error",
            "indicators": {}
        }


def _compute_ta_indicators(hist_df: pd.DataFrame, current_spot: float) -> dict:
    """Compute technical indicators manually."""
    indicators = {}

    # Ensure we have enough data
    if len(hist_df) < 30:
        return indicators

    # Use close prices for indicators
    close_prices = hist_df['close']

    try:
        # Simple Moving Averages
        if len(close_prices) >= 20:
            indicators['sma_20'] = close_prices.rolling(window=20).mean().iloc[-1]
        if len(close_prices) >= 50:
            indicators['sma_50'] = close_prices.rolling(window=50).mean().iloc[-1]

        # Exponential Moving Averages
        if len(close_prices) >= 12:
            indicators['ema_12'] = close_prices.ewm(span=12).mean().iloc[-1]
        if len(close_prices) >= 26:
            indicators['ema_26'] = close_prices.ewm(span=26).mean().iloc[-1]

        # MACD
        if len(close_prices) >= 26:
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd_line'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = (macd_line - signal_line).iloc[-1]

        # RSI
        if len(close_prices) >= 14:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]

        # Bollinger Bands
        if len(close_prices) >= 20:
            sma = close_prices.rolling(window=20).mean()
            std = close_prices.rolling(window=20).std()
            indicators['bb_lower'] = (sma - 2 * std).iloc[-1]
            indicators['bb_upper'] = (sma + 2 * std).iloc[-1]
            indicators['bb_sma'] = sma.iloc[-1]

    except Exception as e:
        logger.warning(f"Error computing TA indicators: {e}")

    return indicators


def _generate_ta_signals(indicators: dict) -> tuple[str, float, str]:
    """Generate trading signals from indicators."""
    direction = "NO_SIGNAL"
    confidence = 0.0
    regime = "neutral"

    signals = []

    # Moving Average signals
    if indicators.get('sma_20') is not None and indicators.get('sma_50') is not None:
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']

        if sma_20 > sma_50:
            signals.append(("CALL", 0.6, "bullish_trend"))
        elif sma_20 < sma_50:
            signals.append(("PUT", 0.6, "bearish_trend"))

    # MACD signals
    if indicators.get('macd_histogram') is not None:
        macd_hist = indicators['macd_histogram']

        if macd_hist > 0:
            signals.append(("CALL", 0.5, "macd_bullish"))
        elif macd_hist < 0:
            signals.append(("PUT", 0.5, "macd_bearish"))

    # RSI signals
    if indicators.get('rsi') is not None:
        rsi = indicators['rsi']

        if rsi > 70:
            signals.append(("PUT", 0.7, "overbought"))
        elif rsi < 30:
            signals.append(("CALL", 0.7, "oversold"))

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
def get_ta_features_for_trade(symbol: str, spot_price: float) -> dict:
    """
    Convenience function to get TA features for integration with trade evaluation.

    Args:
        symbol: Underlying symbol
        spot_price: Current spot price

    Returns:
        TA features dict ready for trade evaluation
    """
    return build_ta_features(symbol, spot_price)
