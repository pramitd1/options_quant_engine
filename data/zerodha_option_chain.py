from kiteconnect import KiteConnect
import pandas as pd

from config.settings import API_KEY, ACCESS_TOKEN, QUOTE_BATCH_SIZE


class ZerodhaOptionChain:
    """
    Builds a live option chain from Zerodha instruments + quotes.
    Uses credentials from config/settings.py.
    """

    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY)
        self.kite.set_access_token(ACCESS_TOKEN)

        print("Connected to Zerodha")
        self.instruments_df = None

    def load_instruments(self):
        if self.instruments_df is None:
            instruments = self.kite.instruments("NFO")
            self.instruments_df = pd.DataFrame(instruments)

        return self.instruments_df

    def chunk_list(self, data, size=QUOTE_BATCH_SIZE):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    def build_option_chain(self, symbol="NIFTY"):
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