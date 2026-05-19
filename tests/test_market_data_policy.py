from __future__ import annotations

from config.market_data_policy import (
    _parse_gift_nifty_icici_candidates,
    normalize_symbol_to_yfinance,
)


def test_normalize_symbol_to_yfinance_nifty():
    assert normalize_symbol_to_yfinance("NIFTY") == "^NSEI"
    assert normalize_symbol_to_yfinance("nifty") == "^NSEI"


def test_normalize_symbol_to_yfinance_banknifty():
    assert normalize_symbol_to_yfinance("BANKNIFTY") == "^NSEBANK"
    assert normalize_symbol_to_yfinance("banknifty") == "^NSEBANK"


def test_normalize_symbol_to_yfinance_unknown_nse_symbol():
    assert normalize_symbol_to_yfinance("RELIANCE") == "RELIANCE.NS"


def test_parse_quoted_gift_nifty_icici_candidates():
    candidates = _parse_gift_nifty_icici_candidates('"NDX:NIFTY;NDX:GIFT NIFTY:futures"')

    assert candidates[0] == {
        "exchange_code": "NDX",
        "stock_code": "NIFTY",
        "product_type": "",
    }
    assert candidates[1] == {
        "exchange_code": "NDX",
        "stock_code": "GIFT NIFTY",
        "product_type": "futures",
    }
