"""
Data Source Router

Routes option-chain requests to Zerodha, NSE, or ICICI.
Keeps a common interface for the rest of the engine.
"""

from config.settings import ICICI_DEBUG, NSE_DEBUG
from data.zerodha_option_chain import ZerodhaOptionChain
from data.nse_option_chain_downloader import NSEOptionChainDownloader
from data.icici_breeze_option_chain import ICICIBreezeOptionChain
from data.provider_normalization import normalize_live_option_chain


class DataSourceRouter:
    """
    Routes option-chain requests to the selected data source.
    """

    def __init__(self, source: str):
        self.source = source.upper().strip()
        self.loader = None

        if self.source == "ZERODHA":
            self.loader = ZerodhaOptionChain()

        elif self.source == "NSE":
            self.loader = NSEOptionChainDownloader(debug=NSE_DEBUG)

        elif self.source == "ICICI":
            self.loader = ICICIBreezeOptionChain(debug=ICICI_DEBUG)

        else:
            raise ValueError(
                "Unsupported data source. Use 'ZERODHA', 'NSE', or 'ICICI'."
            )

    def get_option_chain(self, symbol: str):
        """
        Fetch option-chain data for the selected source.
        """

        if self.source == "ZERODHA":
            raw_chain = self.loader.build_option_chain(symbol)
            return normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)

        if self.source == "NSE":
            raw_chain = self.loader.fetch_option_chain(symbol)
            return normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)

        if self.source == "ICICI":
            raw_chain = self.loader.fetch_option_chain(symbol)
            return normalize_live_option_chain(raw_chain, source=self.source, symbol=symbol)

        raise ValueError("Invalid source selected")

    def close(self):
        """
        Cleanly close any underlying resources.
        """

        if self.loader is not None and hasattr(self.loader, "close"):
            try:
                self.loader.close()
            except Exception:
                pass
