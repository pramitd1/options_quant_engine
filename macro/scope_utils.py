"""
Shared scope normalization and matching utilities for macro/news layers.
"""

from __future__ import annotations


INDEX_SCOPES = {"INDEX", "INDICES", "NIFTY", "BANKNIFTY", "FINNIFTY"}
INDEX_KEYWORD_MAP = {
    "NIFTY": ["nifty", "nifty 50", "india equities", "indian equities"],
    "BANKNIFTY": ["bank nifty", "banknifty", "bank index", "financials", "banks"],
    "FINNIFTY": ["fin nifty", "finnifty", "financial services"],
}


def normalize_scope(value) -> list[str]:
    if value is None:
        return ["ALL"]

    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, list):
        parts = value
    else:
        return ["ALL"]

    cleaned = []
    for part in parts:
        text = str(part).strip().upper()
        if text:
            cleaned.append(text)

    return cleaned or ["ALL"]


def symbol_scope_matches(symbol: str, scopes: list[str]) -> bool:
    symbol_upper = str(symbol or "").strip().upper()

    if "ALL" in scopes:
        return True

    if symbol_upper in scopes:
        return True

    if symbol_upper in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        return any(scope in INDEX_SCOPES for scope in scopes)

    if "STOCK" in scopes and symbol_upper not in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        return True

    return False


def headline_mentions_symbol(symbol: str, headline: str) -> bool:
    symbol_upper = str(symbol or "").strip().upper()
    text = str(headline or "").strip().lower()

    if not symbol_upper or not text:
        return True

    if symbol_upper in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
        keywords = INDEX_KEYWORD_MAP.get(symbol_upper, [])
        return any(keyword in text for keyword in keywords)

    simple_aliases = {
        symbol_upper.lower(),
        f"{symbol_upper.lower()}.ns",
        f"nse:{symbol_upper.lower()}",
    }
    return any(alias in text for alias in simple_aliases)
