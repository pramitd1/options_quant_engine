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

from typing import Any, Callable

from config.settings import ICICI_DEBUG, NSE_DEBUG
from data.zerodha_option_chain import ZerodhaOptionChain
from data.nse_option_chain_downloader import NSEOptionChainDownloader
from data.icici_breeze_option_chain import ICICIBreezeOptionChain
from data.provider_normalization import normalize_live_option_chain


LoaderFactory = Callable[[], Any]


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
        loader_factories = _build_loader_factories()
        if self.source not in loader_factories:
            raise ValueError(
                "Unsupported data source. Use 'ZERODHA', 'NSE', or 'ICICI'."
            )
        self.loader = loader_factories[self.source]()

    def get_option_chain(self, symbol: str):
        """
        Fetch option-chain data for the selected source, with automatic fallback on weak data.
        """
        from data.option_chain_validation import validate_option_chain
        
        # Try primary source first
        fetch_method_name = _build_fetch_method_names()[self.source]
        raw_chain = getattr(self.loader, fetch_method_name)(symbol)
        normalized_chain = normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)
        
        # Validate the data quality
        validation = validate_option_chain(normalized_chain)
        provider_health = validation.get("provider_health", {})
        summary_status = provider_health.get("summary_status", "GOOD")
        
        # If primary source has weak data, try fallback sources
        if summary_status in ("WEAK", "CAUTION"):
            fallback_sources = [s for s in ["ICICI", "NSE", "ZERODHA"] if s != self.source]
            for fallback_source in fallback_sources:
                try:
                    # Create temporary loader for fallback
                    loader_factories = _build_loader_factories()
                    fallback_loader = loader_factories[fallback_source]()
                    fallback_fetch_method = _build_fetch_method_names()[fallback_source]
                    fallback_raw_chain = getattr(fallback_loader, fallback_fetch_method)(symbol)
                    fallback_normalized = normalize_live_option_chain(fallback_raw_chain, source=fallback_source, symbol=symbol)
                    
                    # Validate fallback data
                    fallback_validation = validate_option_chain(fallback_normalized)
                    fallback_health = fallback_validation.get("provider_health", {})
                    fallback_status = fallback_health.get("summary_status", "GOOD")
                    
                    # Use fallback if it's better
                    if fallback_status == "GOOD" or (fallback_status == "CAUTION" and summary_status == "WEAK"):
                        print(f"DataSourceRouter: Falling back from {self.source} ({summary_status}) to {fallback_source} ({fallback_status})")
                        return fallback_normalized
                        
                    # Clean up fallback loader
                    if hasattr(fallback_loader, "close"):
                        fallback_loader.close()
                        
                except Exception as e:
                    print(f"DataSourceRouter: Fallback to {fallback_source} failed: {e}")
                    continue
        
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
