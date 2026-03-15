from __future__ import annotations


IST_TIMEZONE = "Asia/Kolkata"
GLOBAL_MARKET_TICKERS = {
    "oil": "CL=F",
    "gold": "GC=F",
    "copper": "HG=F",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "us10y": "^TNX",
    "usdinr": "INR=X",
}
YFINANCE_UNDERLYING_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^NSEFIN",
}


def normalize_symbol_to_yfinance(symbol: str) -> str:
    normalized = str(symbol or "").upper().strip()
    if normalized in YFINANCE_UNDERLYING_SYMBOL_MAP:
        return YFINANCE_UNDERLYING_SYMBOL_MAP[normalized]
    if normalized.startswith("^") or "." in normalized:
        return normalized
    return f"{normalized}.NS"
