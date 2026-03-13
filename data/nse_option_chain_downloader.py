"""
NSE Option Chain Downloader

Hardened version for NSE's silent anti-bot behavior:
- handles HTTP 200 with empty JSON {}
- aggressively refreshes cookies/session
- uses legacy endpoint directly
"""

import json
import random
import time
from datetime import datetime

import pandas as pd
import requests


class NSEOptionChainDownloader:
    HOME_PAGE = "https://www.nseindia.com/"
    OPTION_CHAIN_PAGE = "https://www.nseindia.com/option-chain"
    ALT_OPTION_CHAIN_PAGE = "https://www.nseindia.com/market-data/option-chain"

    LEGACY_INDEX_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    LEGACY_STOCK_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    INDEX_SYMBOLS = {
        "NIFTY",
        "BANKNIFTY",
        "FINNIFTY",
        "MIDCPNIFTY",
        "NIFTYNXT50",
    }

    def __init__(self, debug=False):
        self.debug = debug
        self.session = None
        self.last_error = None
        self._new_session()

    def _log(self, *args):
        if self.debug:
            print("[NSE DEBUG]", *args)

    def _base_headers(self, referer=None):
        return {
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": referer or self.OPTION_CHAIN_PAGE,
            "origin": "https://www.nseindia.com",
            "connection": "keep-alive",
        }

    def _new_session(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass

        self.session = requests.Session()
        self._bootstrap_session()

    def _bootstrap_session(self):
        warmup_urls = [
            self.HOME_PAGE,
            self.OPTION_CHAIN_PAGE,
            self.ALT_OPTION_CHAIN_PAGE,
        ]

        for url in warmup_urls:
            try:
                resp = self.session.get(
                    url,
                    headers=self._base_headers(referer=self.HOME_PAGE),
                    timeout=10
                )
                self._log("bootstrap", url, "status=", resp.status_code)
                time.sleep(0.4)
            except Exception as e:
                self._log("bootstrap_failed", url, str(e))

    def _is_index(self, symbol: str) -> bool:
        return symbol.upper().strip() in self.INDEX_SYMBOLS

    def _request_json_once(self, url: str, referer: str):
        try:
            response = self.session.get(
                url,
                headers=self._base_headers(referer=referer),
                timeout=12
            )

            content_type = response.headers.get("content-type", "")
            text_preview = response.text[:250].replace("\n", " ").replace("\r", " ")

            self._log(
                f"url={url}",
                f"status={response.status_code}",
                f"content_type={content_type}",
                f"preview={text_preview}"
            )

            if response.status_code != 200:
                return None, f"http_{response.status_code}"

            try:
                data = response.json()
            except Exception as e:
                return None, f"json_decode_error:{e}"

            if not isinstance(data, dict):
                return None, "response_not_dict"

            if not data:
                return None, "empty_json_dict"

            return data, None

        except Exception as e:
            return None, str(e)

    def _request_json(self, url: str):
        referers = [
            self.OPTION_CHAIN_PAGE,
            self.ALT_OPTION_CHAIN_PAGE,
            self.HOME_PAGE,
        ]

        last_error = None

        for attempt in range(1, 7):
            referer = referers[(attempt - 1) % len(referers)]

            data, err = self._request_json_once(url, referer=referer)
            if data:
                self.last_error = None
                self._log("json_keys", list(data.keys())[:10])
                return data

            last_error = err
            self._log(f"attempt={attempt}", f"error={err}", "refreshing session")

            self._new_session()
            time.sleep(0.8 + random.uniform(0.2, 0.8))

        self.last_error = last_error
        self._log("request_failed", f"url={url}", f"last_error={last_error}")
        return {}

    def _get_legacy_chain_json(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()

        if self._is_index(symbol):
            url = self.LEGACY_INDEX_URL.format(symbol=symbol)
        else:
            url = self.LEGACY_STOCK_URL.format(symbol=symbol)

        return self._request_json(url)

    def _extract_rows(self, data: dict) -> list:
        if not isinstance(data, dict):
            return []

        if isinstance(data.get("records"), dict) and isinstance(data["records"].get("data"), list):
            return data["records"]["data"]

        if isinstance(data.get("filtered"), dict) and isinstance(data["filtered"].get("data"), list):
            return data["filtered"]["data"]

        if isinstance(data.get("data"), list):
            return data["data"]

        self._log("row_extraction_failed", f"top_level_keys={list(data.keys())[:15]}")
        try:
            self._log("response_json_preview", json.dumps(data)[:500])
        except Exception:
            pass

        return []

    def _extract_expiry_from_item(self, item: dict):
        expiry = item.get("expiryDate")
        if expiry:
            return expiry

        ce = item.get("CE", {})
        pe = item.get("PE", {})

        if isinstance(ce, dict) and ce.get("expiryDate"):
            return ce.get("expiryDate")

        if isinstance(pe, dict) and pe.get("expiryDate"):
            return pe.get("expiryDate")

        return None

    def _extract_nearest_expiry(self, items: list):
        expiries = self._extract_ordered_expiries(items)
        return expiries[0] if expiries else None

    def _extract_ordered_expiries(self, items: list):
        expiries = []

        for item in items:
            if not isinstance(item, dict):
                continue

            expiry = self._extract_expiry_from_item(item)
            if expiry and expiry not in expiries:
                expiries.append(expiry)

        sortable = []
        unsortable = []

        for expiry in expiries:
            try:
                sortable.append((datetime.strptime(expiry, "%d-%b-%Y"), expiry))
            except Exception:
                unsortable.append(expiry)

        sortable.sort(key=lambda item: item[0])
        ordered = [expiry for _, expiry in sortable] + unsortable
        self._log("detected_expiries", ordered[:10])
        return ordered

    def fetch_available_expiries(self, symbol="NIFTY") -> list[str]:
        symbol = symbol.upper().strip()
        data = self._get_legacy_chain_json(symbol)

        if not data:
            self._log("available_expiries_failed", symbol, self.last_error)
            return []

        items = self._extract_rows(data)
        if not items:
            self._log("available_expiries_no_rows", symbol)
            return []

        return self._extract_ordered_expiries(items)

    def _safe_float(self, value, default=0.0):
        try:
            if value in (None, ""):
                return default
            return float(value)
        except Exception:
            return default

    def _validate_dataframe_quality(self, df: pd.DataFrame):
        if df is None or df.empty:
            return {
                "is_valid": False,
                "row_count": 0,
                "ce_rows": 0,
                "pe_rows": 0,
                "priced_rows": 0,
            }

        ce_rows = int((df["OPTION_TYP"] == "CE").sum()) if "OPTION_TYP" in df.columns else 0
        pe_rows = int((df["OPTION_TYP"] == "PE").sum()) if "OPTION_TYP" in df.columns else 0
        priced_rows = int((pd.to_numeric(df["lastPrice"], errors="coerce") > 0).sum()) if "lastPrice" in df.columns else 0
        row_count = len(df)

        return {
            "is_valid": row_count >= 20 and ce_rows > 0 and pe_rows > 0 and priced_rows > 0,
            "row_count": row_count,
            "ce_rows": ce_rows,
            "pe_rows": pe_rows,
            "priced_rows": priced_rows,
        }

    def _rows_to_df(self, items: list, expiry_filter=None) -> pd.DataFrame:
        rows = []

        for item in items:
            if not isinstance(item, dict):
                continue

            strike = item.get("strikePrice")
            item_expiry = self._extract_expiry_from_item(item)

            if expiry_filter is not None and item_expiry != expiry_filter:
                continue

            ce = item.get("CE")
            if isinstance(ce, dict):
                rows.append({
                    "strikePrice": strike,
                    "OPTION_TYP": "CE",
                    "lastPrice": self._safe_float(ce.get("lastPrice", 0), 0.0),
                    "openInterest": self._safe_float(ce.get("openInterest", 0), 0.0),
                    "changeinOI": self._safe_float(ce.get("changeinOpenInterest", 0), 0.0),
                    "impliedVolatility": self._safe_float(ce.get("impliedVolatility", 0), 0.0),
                    "totalTradedVolume": self._safe_float(ce.get("totalTradedVolume", 0), 0.0),
                    "IV": self._safe_float(ce.get("impliedVolatility", 0), 0.0),
                    "VOLUME": self._safe_float(ce.get("totalTradedVolume", 0), 0.0),
                    "OPEN_INT": self._safe_float(ce.get("openInterest", 0), 0.0),
                    "STRIKE_PR": strike,
                    "LAST_PRICE": self._safe_float(ce.get("lastPrice", 0), 0.0),
                    "EXPIRY_DT": ce.get("expiryDate", item_expiry),
                })

            pe = item.get("PE")
            if isinstance(pe, dict):
                rows.append({
                    "strikePrice": strike,
                    "OPTION_TYP": "PE",
                    "lastPrice": self._safe_float(pe.get("lastPrice", 0), 0.0),
                    "openInterest": self._safe_float(pe.get("openInterest", 0), 0.0),
                    "changeinOI": self._safe_float(pe.get("changeinOpenInterest", 0), 0.0),
                    "impliedVolatility": self._safe_float(pe.get("impliedVolatility", 0), 0.0),
                    "totalTradedVolume": self._safe_float(pe.get("totalTradedVolume", 0), 0.0),
                    "IV": self._safe_float(pe.get("impliedVolatility", 0), 0.0),
                    "VOLUME": self._safe_float(pe.get("totalTradedVolume", 0), 0.0),
                    "OPEN_INT": self._safe_float(pe.get("openInterest", 0), 0.0),
                    "STRIKE_PR": strike,
                    "LAST_PRICE": self._safe_float(pe.get("lastPrice", 0), 0.0),
                    "EXPIRY_DT": pe.get("expiryDate", item_expiry),
                })

        df = pd.DataFrame(rows)
        self._log("rows_to_df", f"expiry_filter={expiry_filter}", f"row_count={len(df)}")
        return df

    def fetch_option_chain(self, symbol="NIFTY") -> pd.DataFrame:
        symbol = symbol.upper().strip()

        data = self._get_legacy_chain_json(symbol)
        if not data:
            detail = f" ({self.last_error})" if self.last_error else ""
            print(f"Option chain download error: empty or blocked NSE response{detail}")
            return pd.DataFrame()

        items = self._extract_rows(data)
        self._log("raw_item_count", len(items))

        if not items:
            print("Option chain download error: could not fetch option chain rows from NSE payload")
            return pd.DataFrame()

        nearest_expiry = self._extract_nearest_expiry(items)

        if nearest_expiry is not None:
            df = self._rows_to_df(items, expiry_filter=nearest_expiry)
            quality = self._validate_dataframe_quality(df)
            self._log("nearest_expiry_quality", nearest_expiry, quality)
            if quality["is_valid"]:
                return df

        df = self._rows_to_df(items, expiry_filter=None)
        quality = self._validate_dataframe_quality(df)
        self._log("unfiltered_quality", quality)
        if quality["is_valid"] or not df.empty:
            return df

        print("Option chain download error: could not build option chain dataframe")
        return pd.DataFrame()

    def close(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
