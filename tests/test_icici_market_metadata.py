from __future__ import annotations

from datetime import datetime

import pandas as pd

import data.icici_breeze_option_chain as icici_module
from data.icici_breeze_option_chain import ICICIBreezeOptionChain
from data.icici_market_metadata import ICICIMarketMetadataResolver


def test_icici_market_metadata_resolver_caches_per_symbol():
    load_calls = []

    resolver = ICICIMarketMetadataResolver(
        load_security_master=lambda: load_calls.append("load") or pd.DataFrame([{"symbol": "NIFTY"}]),
        normalize_master_columns=lambda frame: frame,
        match_symbol_in_master=lambda frame, symbol: frame,
        filter_option_rows_from_master=lambda frame: frame,
        extract_expiry_from_master=lambda row: "2026-03-26T06:00:00.000Z",
        extract_request_symbols_from_master=lambda frame, symbol: [symbol, "NIFTY"],
        normalize_symbol=lambda symbol: str(symbol).upper().strip(),
    )

    first = resolver.resolve("nifty")
    second = resolver.resolve("NIFTY")

    assert load_calls == ["load"]
    assert first == second
    assert first["request_symbols"][0] == "NIFTY"


def test_icici_dynamic_expiry_keeps_today_until_session_close(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def utcnow(cls):
            return cls(2026, 5, 19, 5, 0, 0)  # Tuesday, 10:30 IST.

    monkeypatch.setattr(icici_module, "datetime", FrozenDateTime)
    loader = ICICIBreezeOptionChain.__new__(ICICIBreezeOptionChain)

    candidates = loader._generate_dynamic_expiries("NIFTY", count=1)

    assert candidates[0] == "2026-05-19T06:00:00.000Z"


def test_icici_dynamic_expiry_skips_today_after_session_close(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def utcnow(cls):
            return cls(2026, 5, 19, 10, 5, 0)  # Tuesday, after 15:30 IST close.

    monkeypatch.setattr(icici_module, "datetime", FrozenDateTime)
    loader = ICICIBreezeOptionChain.__new__(ICICIBreezeOptionChain)

    candidates = loader._generate_dynamic_expiries("NIFTY", count=1)

    assert "2026-05-19T06:00:00.000Z" not in candidates
    assert candidates[0] == "2026-05-26T06:00:00.000Z"
