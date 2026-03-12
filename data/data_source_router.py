"""
Data Source Router

Routes option-chain requests to Zerodha or NSE.
Keeps a common interface for the rest of the engine.
"""

from data.zerodha_option_chain import ZerodhaOptionChain
from data.nse_option_chain_downloader import NSEOptionChainDownloader


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
            self.loader = NSEOptionChainDownloader()

        else:
            raise ValueError(
                "Unsupported data source. Use 'ZERODHA' or 'NSE'."
            )

    def get_option_chain(self, symbol: str):
        """
        Fetch option-chain data for the selected source.
        """

        if self.source == "ZERODHA":
            return self.loader.build_option_chain(symbol)

        if self.source == "NSE":
            return self.loader.fetch_option_chain(symbol)

        raise ValueError("Invalid source selected")

    def close(self):
        """
        Cleanly close any underlying resources.
        Useful for Playwright-based NSE loader.
        """

        if self.loader is not None and hasattr(self.loader, "close"):
            try:
                self.loader.close()
            except Exception:
                pass