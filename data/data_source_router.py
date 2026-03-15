"""
Data Source Router

Routes option-chain requests to Zerodha, NSE, or ICICI.
Keeps a common interface for the rest of the engine.
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
    return {
        "ZERODHA": ZerodhaOptionChain,
        "NSE": lambda: NSEOptionChainDownloader(debug=NSE_DEBUG),
        "ICICI": lambda: ICICIBreezeOptionChain(debug=ICICI_DEBUG),
    }


def _build_fetch_method_names() -> dict[str, str]:
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
        Fetch option-chain data for the selected source.
        """
        fetch_method_name = _build_fetch_method_names()[self.source]
        raw_chain = getattr(self.loader, fetch_method_name)(symbol)
        return normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)

    def close(self):
        """
        Cleanly close any underlying resources.
        """

        if self.loader is not None and hasattr(self.loader, "close"):
            try:
                self.loader.close()
            except Exception:
                pass
