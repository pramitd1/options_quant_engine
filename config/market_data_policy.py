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
GIFT_NIFTY_SOURCE = os.getenv("OQE_GIFT_NIFTY_SOURCE", "ICICI").strip().upper() or "ICICI"
GIFT_NIFTY_TICKER = os.getenv("OQE_GIFT_NIFTY_TICKER", "").strip()
GIFT_NIFTY_PROXY_IN_USE = GIFT_NIFTY_SOURCE == "YFINANCE" and GIFT_NIFTY_TICKER == "^NSEI"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


GIFT_NIFTY_ICICI_CACHE_TTL_SECONDS = _env_int("OQE_GIFT_NIFTY_ICICI_CACHE_TTL_SECONDS", 60)


def _parse_gift_nifty_icici_candidates(raw: str | None) -> tuple[dict[str, str], ...]:
    """
    Parse semicolon-separated Breeze quote candidates.

    Format:
        EXCHANGE:STOCK_CODE[:PRODUCT_TYPE]

    Examples:
        NDX:NIFTY
        NDX:NIFTY:futures
        NDX:GIFTNIFTY
    """
    default = (
        {"exchange_code": "NDX", "stock_code": "NIFTY", "product_type": ""},
        {"exchange_code": "NDX", "stock_code": "GIFTNIFTY", "product_type": ""},
        {"exchange_code": "NDX", "stock_code": "GIFT NIFTY", "product_type": ""},
        {"exchange_code": "NDX", "stock_code": "NIFTY", "product_type": "futures"},
        {"exchange_code": "NDX", "stock_code": "GIFTNIFTY", "product_type": "futures"},
    )
    if raw is None or not str(raw).strip():
        return default

    raw = str(raw).strip().strip('"').strip("'")
    parsed: list[dict[str, str]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 2:
            continue
        parsed.append(
            {
                "exchange_code": parts[0].upper(),
                "stock_code": parts[1],
                "product_type": parts[2].lower() if len(parts) > 2 else "",
            }
        )
    return tuple(parsed) or default


GIFT_NIFTY_ICICI_CANDIDATES = _parse_gift_nifty_icici_candidates(
    os.getenv("OQE_GIFT_NIFTY_ICICI_CANDIDATES")
)
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
}
if GIFT_NIFTY_SOURCE == "YFINANCE" and GIFT_NIFTY_TICKER:
    GLOBAL_MARKET_TICKERS["gift_nifty"] = GIFT_NIFTY_TICKER
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
