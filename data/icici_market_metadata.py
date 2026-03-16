"""
Module: icici_market_metadata.py

Purpose:
    Implement icici market metadata data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd


class ICICIMarketMetadataResolver:
    """
    Purpose:
        Represent ICICIMarketMetadataResolver within the repository.
    
    Context:
        Used within the `icici market metadata` module. The class participates in the module's role within the trading system.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        The class groups behavior and state that need to stay explicit for maintainability and auditability.
    """
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
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `ICICIMarketMetadataResolver` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            load_security_master (Callable[[], pd.DataFrame]): Input associated with load security master.
            normalize_master_columns (Callable[[pd.DataFrame], pd.DataFrame]): Input associated with normalize master columns.
            match_symbol_in_master (Callable[[pd.DataFrame, str], pd.DataFrame]): Input associated with match symbol in master.
            filter_option_rows_from_master (Callable[[pd.DataFrame], pd.DataFrame]): Input associated with filter option rows from master.
            extract_expiry_from_master (Callable[[dict], str | None]): Input associated with extract expiry from master.
            extract_request_symbols_from_master (Callable[[pd.DataFrame, str], list[str]]): Input associated with extract request symbols from master.
            normalize_symbol (Callable[[str], str]): Input associated with normalize symbol.
            logger (Callable[..., None] | None): Input associated with logger.
        
        Returns:
            None: The function operates through side effects.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
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
        """
        Purpose:
            Process resolve for downstream use.
        
        Context:
            Method on `ICICIMarketMetadataResolver` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (str): Underlying symbol or index identifier.
        
        Returns:
            dict[str, list[str]]: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
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
