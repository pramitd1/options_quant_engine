from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd


class ICICIMarketMetadataResolver:
    def __init__(
        self,
        *,
        load_security_master: Callable[[], pd.DataFrame],
        normalize_master_columns: Callable[[pd.DataFrame], pd.DataFrame],
        match_symbol_in_master: Callable[[pd.DataFrame, str], pd.DataFrame],
        filter_option_rows_from_master: Callable[[pd.DataFrame], pd.DataFrame],
        extract_expiry_from_master: Callable[[dict], str | None],
        extract_request_symbols_from_master: Callable[[pd.DataFrame, str], list[str]],
        normalize_symbol: Callable[[str], str],
        logger: Callable[..., None] | None = None,
    ) -> None:
        self._load_security_master = load_security_master
        self._normalize_master_columns = normalize_master_columns
        self._match_symbol_in_master = match_symbol_in_master
        self._filter_option_rows_from_master = filter_option_rows_from_master
        self._extract_expiry_from_master = extract_expiry_from_master
        self._extract_request_symbols_from_master = extract_request_symbols_from_master
        self._normalize_symbol = normalize_symbol
        self._log = logger or (lambda *args: None)
        self._metadata_cache: dict[str, dict[str, list[str]]] = {}

    def resolve(self, symbol: str) -> dict[str, list[str]]:
        normalized_symbol = self._normalize_symbol(symbol)
        cached = self._metadata_cache.get(normalized_symbol)
        if cached is not None:
            return {
                "expiries": list(cached.get("expiries", [])),
                "request_symbols": list(cached.get("request_symbols", [])),
            }

        master_df = self._load_security_master()
        if master_df.empty:
            metadata = {
                "expiries": [],
                "request_symbols": [normalized_symbol],
            }
            self._metadata_cache[normalized_symbol] = metadata
            return {
                "expiries": list(metadata["expiries"]),
                "request_symbols": list(metadata["request_symbols"]),
            }

        normalized = self._normalize_master_columns(master_df)
        symbol_rows = self._match_symbol_in_master(normalized, normalized_symbol)
        option_rows = self._filter_option_rows_from_master(symbol_rows)

        expiries = []
        for row in option_rows.to_dict(orient="records"):
            expiry = self._extract_expiry_from_master(row)
            if expiry and expiry not in expiries:
                expiries.append(expiry)

        expiries.sort(key=lambda value: datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.000Z"))
        request_symbols = self._extract_request_symbols_from_master(option_rows, normalized_symbol)
        metadata = {
            "expiries": expiries,
            "request_symbols": request_symbols,
        }
        self._metadata_cache[normalized_symbol] = metadata
        self._log("icici_master_expiry_candidates", f"symbol={normalized_symbol}", f"candidates={expiries[:10]}")
        self._log("icici_master_request_symbols", f"symbol={normalized_symbol}", f"candidates={request_symbols[:10]}")
        return {
            "expiries": list(expiries),
            "request_symbols": list(request_symbols),
        }
