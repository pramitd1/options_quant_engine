"""
Module: zerodha_option_chain.py

Purpose:
    Implement zerodha option chain data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""
from kiteconnect import KiteConnect
import pandas as pd

from config.settings import QUOTE_BATCH_SIZE, get_zerodha_runtime_config


class ZerodhaOptionChain:
    """
    Builds a live option chain from Zerodha instruments + quotes.
    Uses credentials from config/settings.py.
    """

    def __init__(self):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `ZerodhaOptionChain` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        creds = get_zerodha_runtime_config()

        api_key = creds["api_key"]
        access_token = creds["access_token"]

        if str(api_key).startswith("YOUR_"):
            raise ValueError("ZERODHA_API_KEY is not configured")

        if str(access_token).startswith("YOUR_"):
            raise ValueError("ZERODHA_ACCESS_TOKEN is not configured")

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

        print("Connected to Zerodha")
        self.instruments_df = None

    def load_instruments(self):
        """
        Purpose:
            Process load instruments for downstream use.
        
        Context:
            Method on `ZerodhaOptionChain` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        if self.instruments_df is None:
            instruments = self.kite.instruments("NFO")
            self.instruments_df = pd.DataFrame(instruments)

        return self.instruments_df

    def chunk_list(self, data, size=QUOTE_BATCH_SIZE):
        """
        Purpose:
            Process chunk list for downstream use.
        
        Context:
            Method on `ZerodhaOptionChain` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            data (Any): Input associated with data.
            size (Any): Input associated with size.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        for i in range(0, len(data), size):
            yield data[i:i + size]

    def build_option_chain(self, symbol="NIFTY"):
        """
        Purpose:
            Build the option chain used by downstream components.
        
        Context:
            Method on `ZerodhaOptionChain` within the data layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            symbol (Any): Underlying symbol or index identifier.
        
        Returns:
            Any: Computed value returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        instruments = self.load_instruments()

        options = instruments[
            (instruments["name"] == symbol) &
            (instruments["segment"] == "NFO-OPT")
        ].copy()

        if options.empty:
            return pd.DataFrame()

        tokens = options["instrument_token"].tolist()
        rows = []

        for token_batch in self.chunk_list(tokens):
            quotes = self.kite.quote(token_batch)

            for token in token_batch:
                if token not in quotes:
                    continue

                q = quotes[token]
                inst = options[
                    options["instrument_token"] == token
                ].iloc[0]

                strike = inst["strike"]
                expiry = str(inst.get("expiry", "NEAR"))

                last_price = q.get("last_price", 0)
                oi = q.get("oi", 0)
                volume = q.get("volume", 0)

                rows.append({
                    "strikePrice": strike,
                    "OPTION_TYP": inst["instrument_type"],
                    "lastPrice": last_price,
                    "openInterest": oi,
                    "totalTradedVolume": volume,
                    "changeinOI": 0,
                    "impliedVolatility": 0,
                    "IV": 0,
                    "VOLUME": volume,
                    "OPEN_INT": oi,
                    "STRIKE_PR": strike,
                    "LAST_PRICE": last_price,
                    "EXPIRY_DT": expiry,
                })

        return pd.DataFrame(rows)
