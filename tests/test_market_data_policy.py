from __future__ import annotations

from config.market_data_policy import normalize_symbol_to_yfinance


def test_normalize_symbol_to_yfinance_nifty():
    assert normalize_symbol_to_yfinance("NIFTY") == "^NSEI"
    assert normalize_symbol_to_yfinance("nifty") == "^NSEI"


def test_normalize_symbol_to_yfinance_banknifty():
    assert normalize_symbol_to_yfinance("BANKNIFTY") == "^NSEBANK"
    assert normalize_symbol_to_yfinance("banknifty") == "^NSEBANK"


def test_normalize_symbol_to_yfinance_unknown_nse_symbol():
    assert normalize_symbol_to_yfinance("RELIANCE") == "RELIANCE.NS"
