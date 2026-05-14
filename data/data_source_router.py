"""
Module: data_source_router.py

Purpose:
    Route spot and option-chain requests to the configured market-data provider.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from config.settings import ICICI_DEBUG, NSE_DEBUG
from data.zerodha_option_chain import ZerodhaOptionChain
from data.nse_option_chain_downloader import NSEOptionChainDownloader
from data.icici_breeze_option_chain import ICICIBreezeOptionChain
from data.provider_normalization import normalize_live_option_chain


LoaderFactory = Callable[[], Any]
_LOG = logging.getLogger(__name__)


def _build_loader_factories() -> dict[str, LoaderFactory]:
    """
    Purpose:
        Build the loader factories used by downstream components.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict[str, LoaderFactory]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    return {
        "ZERODHA": ZerodhaOptionChain,
        "NSE": lambda: NSEOptionChainDownloader(debug=NSE_DEBUG),
        "ICICI": lambda: ICICIBreezeOptionChain(debug=ICICI_DEBUG),
    }


def _build_fetch_method_names() -> dict[str, str]:
    """
    Purpose:
        Build the fetch method names used by downstream components.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict[str, str]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    return {
        "ZERODHA": "build_option_chain",
        "NSE": "fetch_option_chain",
        "ICICI": "fetch_option_chain",
    }


class DataSourceRouter:
    """
    Routes option-chain requests to the selected data source.
    """

    def __init__(self, source: str):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `DataSourceRouter` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            source (str): Data-source label associated with the current snapshot.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self.source = source.upper().strip()
        self.loader = None
        self.last_validation = None
        loader_factories = _build_loader_factories()
        if self.source not in loader_factories:
            raise ValueError(
                "Unsupported data source. Use 'ZERODHA', 'NSE', or 'ICICI'."
            )
        self.loader = loader_factories[self.source]()

    def get_option_chain(self, symbol: str):
        """
        Fetch option-chain data for the selected source.
        """
        from data.option_chain_validation import validate_option_chain

        fetch_method_name = _build_fetch_method_names()[self.source]
        raw_chain = getattr(self.loader, fetch_method_name)(symbol)
        normalized_chain = normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)

        validation = validate_option_chain(normalized_chain)
        self.last_validation = validation
        provider_health = validation.get("provider_health", {})
        primary_valid = bool(validation.get("is_valid"))
        summary_status = str(provider_health.get("summary_status") or "").upper().strip()
        if not summary_status:
            summary_status = "INVALID" if not primary_valid else "GOOD"

        if not primary_valid or summary_status in ("WEAK", "CAUTION"):
            _LOG.warning(
                "Selected data source %s returned option-chain quality %s "
                "(is_valid=%s, issues=%s, warnings=%s, blocking_reasons=%s). "
                "Keeping the user-selected source; no fallback provider will be used.",
                self.source,
                summary_status,
                primary_valid,
                validation.get("issues", []),
                validation.get("warnings", []),
                provider_health.get("trade_blocking_reasons", []),
            )

        return normalized_chain

    def get_expiry_candidates(self) -> list:
        """Return the expiry candidates resolved during the last fetch, if available."""
        return getattr(self.loader, "_last_expiry_candidates", [])

    def close(self):
        """
        Cleanly close any underlying resources.
        """

        if self.loader is not None and hasattr(self.loader, "close"):
            try:
                self.loader.close()
            except Exception:
                pass
