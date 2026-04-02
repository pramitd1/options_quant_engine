"""
Module: market_data_policy.py

Purpose:
    Define the thresholds, weights, and policy getters used by market data.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""
from __future__ import annotations

import os


IST_TIMEZONE = "Asia/Kolkata"
DXY_TICKER = os.getenv("OQE_DXY_TICKER", "DX-Y.NYB")
GIFT_NIFTY_TICKER = os.getenv("OQE_GIFT_NIFTY_TICKER", "^NSEI")
GIFT_NIFTY_PROXY_IN_USE = GIFT_NIFTY_TICKER == "^NSEI"
GLOBAL_MARKET_TICKERS = {
    "oil": "CL=F",
    "gold": "GC=F",
    "copper": "HG=F",
    "vix": "^VIX",
    "india_vix": "^INDIAVIX",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "us10y": "^TNX",
    "usdinr": "INR=X",
    "dxy": DXY_TICKER,
    "gift_nifty": GIFT_NIFTY_TICKER,
}
YFINANCE_UNDERLYING_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^NSEFIN",
}


def normalize_symbol_to_yfinance(symbol: str) -> str:
    """
    Purpose:
        Normalize symbol to yfinance into the repository-standard representation.
    
    Context:
        Public function in the `market data policy` module. It forms part of the config workflow exposed by this module.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
    
    Returns:
        str: Value returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    normalized = str(symbol or "").upper().strip()
    if normalized in YFINANCE_UNDERLYING_SYMBOL_MAP:
        return YFINANCE_UNDERLYING_SYMBOL_MAP[normalized]
    if normalized.startswith("^") or "." in normalized:
        return normalized
    return f"{normalized}.NS"
