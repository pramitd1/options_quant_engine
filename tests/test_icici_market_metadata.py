from __future__ import annotations

import pandas as pd

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
