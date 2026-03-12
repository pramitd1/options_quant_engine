"""
NSE Option Chain Downloader
"""

import time
from urllib.parse import quote

import pandas as pd
import requests


class NSEOptionChainDownloader:
    BASE_URL = "https://www.nseindia.com"
    OPTION_CHAIN_PAGE = "https://www.nseindia.com/option-chain"

    CONTRACT_INFO_URL = (
        "https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
    )

    INDEX_DATA_URL = (
        "https://www.nseindia.com/api/option-chain-v3"
        "?type=Indices&symbol={symbol}&expiry={expiry}"
    )

    STOCK_DATA_URL = (
        "https://www.nseindia.com/api/option-chain-v3"
        "?type=Equity&symbol={symbol}&expiry={expiry}"
    )

    INDEX_SYMBOLS = {
        "NIFTY",
        "BANKNIFTY",
        "FINNIFTY",
        "MIDCPNIFTY",
        "NIFTYNXT50",
    }

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "accept-language": "en,gu;q=0.9,hi;q=0.8",
            "accept-encoding": "gzip, deflate, br",
            "accept": "application/json, text/plain, */*",
            "referer": self.OPTION_CHAIN_PAGE,
            "connection": "keep-alive",
        }
        self.cookies = {}
        self._bootstrap_session()

    def _bootstrap_session(self):
        self.session.close()
        self.session = requests.Session()

        response = self.session.get(
            self.OPTION_CHAIN_PAGE,
            headers=self.headers,
            timeout=10
        )
        self.cookies = dict(response.cookies)

    def _is_index(self, symbol: str) -> bool:
        return symbol.upper().strip() in self.INDEX_SYMBOLS

    def _safe_json(self, response: requests.Response) -> dict:
        try:
            return response.json()
        except Exception:
            return {}

    def _extract_expiry_dates(self, data: dict) -> list:
        if "expiryDates" in data:
            return data["expiryDates"]

        if "records" in data and "expiryDates" in data["records"]:
            return data["records"]["expiryDates"]

        return []

    def _get_expiry_dates(self, symbol: str) -> list:
        url = self.CONTRACT_INFO_URL.format(symbol=symbol)

        for _ in range(3):
            try:
                response = self.session.get(
                    url,
                    headers=self.headers,
                    cookies=self.cookies,
                    timeout=10
                )

                if response.status_code == 401:
                    self._bootstrap_session()
                    continue

                data = self._safe_json(response)
                expiries = self._extract_expiry_dates(data)

                if expiries:
                    return expiries

            except Exception:
                pass

            time.sleep(1)
            self._bootstrap_session()

        return []

    def _extract_chain_rows(self, data: dict) -> list:
        if "data" in data and isinstance(data["data"], list):
            return data["data"]

        if "records" in data and "data" in data["records"]:
            return data["records"]["data"]

        if "filtered" in data and "data" in data["filtered"]:
            return data["filtered"]["data"]

        return []

    def _fetch_chain_for_expiry(self, symbol: str, expiry: str) -> pd.DataFrame:
        expiry_q = quote(expiry)

        if self._is_index(symbol):
            url = self.INDEX_DATA_URL.format(symbol=symbol, expiry=expiry_q)
        else:
            url = self.STOCK_DATA_URL.format(symbol=symbol, expiry=expiry_q)

        for _ in range(3):
            try:
                response = self.session.get(
                    url,
                    headers=self.headers,
                    cookies=self.cookies,
                    timeout=10
                )

                if response.status_code == 401:
                    self._bootstrap_session()
                    continue

                data = self._safe_json(response)
                items = self._extract_chain_rows(data)

                rows = []

                for item in items:
                    strike = item.get("strikePrice")

                    ce = item.get("CE")
                    if ce:
                        rows.append({
                            "strikePrice": strike,
                            "OPTION_TYP": "CE",
                            "lastPrice": ce.get("lastPrice", 0),
                            "openInterest": ce.get("openInterest", 0),
                            "changeinOI": ce.get("changeinOpenInterest", 0),
                            "impliedVolatility": ce.get("impliedVolatility", 0),
                            "totalTradedVolume": ce.get("totalTradedVolume", 0),
                            "IV": ce.get("impliedVolatility", 0),
                            "VOLUME": ce.get("totalTradedVolume", 0),
                            "OPEN_INT": ce.get("openInterest", 0),
                            "STRIKE_PR": strike,
                            "LAST_PRICE": ce.get("lastPrice", 0),
                            "EXPIRY_DT": expiry,
                        })

                    pe = item.get("PE")
                    if pe:
                        rows.append({
                            "strikePrice": strike,
                            "OPTION_TYP": "PE",
                            "lastPrice": pe.get("lastPrice", 0),
                            "openInterest": pe.get("openInterest", 0),
                            "changeinOI": pe.get("changeinOpenInterest", 0),
                            "impliedVolatility": pe.get("impliedVolatility", 0),
                            "totalTradedVolume": pe.get("totalTradedVolume", 0),
                            "IV": pe.get("impliedVolatility", 0),
                            "VOLUME": pe.get("totalTradedVolume", 0),
                            "OPEN_INT": pe.get("openInterest", 0),
                            "STRIKE_PR": strike,
                            "LAST_PRICE": pe.get("lastPrice", 0),
                            "EXPIRY_DT": expiry,
                        })

                df = pd.DataFrame(rows)

                if not df.empty:
                    return df

            except Exception:
                pass

            time.sleep(1)
            self._bootstrap_session()

        return pd.DataFrame()

    def fetch_option_chain(self, symbol="NIFTY") -> pd.DataFrame:
        symbol = symbol.upper().strip()

        expiries = self._get_expiry_dates(symbol)

        if not expiries:
            print("Option chain download error: could not fetch expiry dates")
            return pd.DataFrame()

        nearest_expiry = expiries[0]
        df = self._fetch_chain_for_expiry(symbol, nearest_expiry)

        if df.empty:
            print("Option chain download error: could not fetch option chain data")
            return pd.DataFrame()

        return df

    def stream_option_chain(self, symbol="NIFTY", interval=30):
        while True:
            df = self.fetch_option_chain(symbol)

            if not df.empty:
                print("Option chain rows:", len(df))
                yield df
            else:
                print("Option chain unavailable")

            time.sleep(interval)

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass