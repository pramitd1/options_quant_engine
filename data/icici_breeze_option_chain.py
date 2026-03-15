"""
ICICI Breeze Option Chain Loader

SDK-based version only.
- Uses breeze.get_option_chain_quotes(...)
- Tries configured expiry candidates one by one
- Normalizes data into the engine's standard option-chain format
- Protects against the local `config` package colliding with Breeze SDK imports
"""

import importlib
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config.settings import (
    ICICI_SYMBOL_EXPIRY_CANDIDATES,
    ICICI_DEFAULT_EXPIRY_DATE,
    ICICI_DEBUG,
    get_icici_runtime_config,
)
from data.icici_market_metadata import ICICIMarketMetadataResolver


def _load_breeze_connect_class():
    """
    Lazily import BreezeConnect while shielding the import from this project's
    local `config` package collision.
    """
    project_root = str(Path(__file__).resolve().parents[1])

    original_sys_path = list(sys.path)
    original_config_module = sys.modules.get("config")

    try:
        filtered_path = []
        for p in sys.path:
            normalized = str(Path(p).resolve()) if p not in ("", ".") else project_root
            if normalized == project_root:
                continue
            filtered_path.append(p)

        sys.path = filtered_path

        if "config" in sys.modules:
            del sys.modules["config"]

        breeze_module = importlib.import_module("breeze_connect")
        return getattr(breeze_module, "BreezeConnect", None)

    finally:
        sys.path = original_sys_path

        if original_config_module is not None:
            sys.modules["config"] = original_config_module


class ICICIBreezeOptionChain:
    SECURITY_MASTER_URL = "https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.zip"
    SECURITY_MASTER_MEMBER = "FONSEScripMaster.txt"
    INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

    def __init__(self, debug=None):
        self.debug = ICICI_DEBUG if debug is None else debug
        self.breeze = None
        self._market_metadata_resolver = ICICIMarketMetadataResolver(
            load_security_master=self._load_security_master,
            normalize_master_columns=self._normalize_master_columns,
            match_symbol_in_master=self._match_symbol_in_master,
            filter_option_rows_from_master=self._filter_option_rows_from_master,
            extract_expiry_from_master=self._extract_expiry_from_master,
            extract_request_symbols_from_master=self._extract_request_symbols_from_master,
            normalize_symbol=self._normalize_symbol,
            logger=self._log,
        )
        self._init_client()

    def _log(self, *args):
        if self.debug:
            print("[ICICI DEBUG]", *args)

    def _format_expiry(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT06:00:00.000Z")

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = str(symbol or "").upper().strip()
        for prefix in ("NSE:", "NFO:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        for suffix in (".NS", ".BO"):
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        return normalized.strip()

    def _symbol_aliases(self, symbol: str) -> list[str]:
        normalized = self._normalize_symbol(symbol)
        aliases = [normalized]

        if "&" in normalized:
            aliases.append(normalized.replace("&", "AND"))
            aliases.append(normalized.replace("&", ""))

        if "-" in normalized:
            aliases.append(normalized.replace("-", ""))

        cleaned = []
        for alias in aliases:
            alias = alias.strip()
            if alias and alias not in cleaned:
                cleaned.append(alias)

        return cleaned

    def _is_index_symbol(self, symbol: str) -> bool:
        return self._normalize_symbol(symbol) in self.INDEX_SYMBOLS

    def _last_weekday_of_month(self, year: int, month: int, weekday: int) -> datetime:
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        candidate = next_month - timedelta(days=1)
        while candidate.weekday() != weekday:
            candidate -= timedelta(days=1)

        return candidate

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _black_scholes_price(self, spot, strike, t, sigma, option_type):
        spot = max(float(spot), 1e-6)
        strike = max(float(strike), 1e-6)
        sigma = max(float(sigma), 1e-6)
        t = max(float(t), 1e-6)

        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)

        if option_type == "CE":
            return spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)

        return strike * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)

    def _estimate_implied_volatility(self, price, spot, strike, expiry_date, option_type):
        price = self._safe_float(price, None)
        spot = self._safe_float(spot, None)
        strike = self._safe_float(strike, None)

        if price in (None, 0) or spot in (None, 0) or strike in (None, 0):
            return None

        intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
        if price <= intrinsic:
            return None

        expiry_ts = pd.to_datetime(expiry_date, errors="coerce", dayfirst=True)
        if pd.isna(expiry_ts):
            return None

        now_ts = pd.Timestamp.utcnow()
        if expiry_ts.tzinfo is None:
            expiry_ts = expiry_ts.tz_localize("UTC")

        t = max((expiry_ts - now_ts).total_seconds() / (365.0 * 24 * 3600), 1e-6)

        low_sigma = 0.01
        high_sigma = 5.0

        try:
            low_price = self._black_scholes_price(spot, strike, t, low_sigma, option_type)
            high_price = self._black_scholes_price(spot, strike, t, high_sigma, option_type)
        except Exception:
            return None

        if price < low_price or price > high_price:
            return None

        for _ in range(60):
            mid_sigma = (low_sigma + high_sigma) / 2.0
            model_price = self._black_scholes_price(spot, strike, t, mid_sigma, option_type)

            if abs(model_price - price) < 1e-3:
                return round(mid_sigma * 100.0, 2)

            if model_price > price:
                high_sigma = mid_sigma
            else:
                low_sigma = mid_sigma

        return round(((low_sigma + high_sigma) / 2.0) * 100.0, 2)

    def _security_master_cache_path(self) -> Path:
        project_root = Path(__file__).resolve().parents[1]
        cache_dir = project_root / "data_store" / "icici_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "SecurityMaster.zip"

    def _read_master_dataframe(self, raw_bytes: bytes) -> pd.DataFrame:
        if not raw_bytes:
            return pd.DataFrame()
        if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            return pd.DataFrame()

        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
                members = archive.namelist()
                target = self.SECURITY_MASTER_MEMBER
                if target not in members:
                    lower_map = {name.lower(): name for name in members}
                    target = lower_map.get(self.SECURITY_MASTER_MEMBER.lower())

                if not target:
                    self._log("security_master_member_missing", members[:10])
                    return pd.DataFrame()

                blob = archive.read(target)
                df = pd.read_csv(io.BytesIO(blob), sep=",")
                self._log("security_master_member_loaded", target, f"rows={len(df)}")
                return df
        except Exception as e:
            self._log("security_master_parse_failed", str(e))
            return pd.DataFrame()

    def _load_security_master(self) -> pd.DataFrame:
        cache_path = self._security_master_cache_path()

        if cache_path.exists():
            modified = datetime.fromtimestamp(cache_path.stat().st_mtime).date()
            if modified == datetime.utcnow().date():
                try:
                    df = self._read_master_dataframe(cache_path.read_bytes())
                    if not df.empty:
                        self._log("security_master_cache_hit", str(cache_path), f"rows={len(df)}")
                        return df
                except Exception as e:
                    self._log("security_master_cache_read_failed", str(e))

        try:
            response = requests.get(self.SECURITY_MASTER_URL, timeout=20)
            response.raise_for_status()
            cache_path.write_bytes(response.content)
            df = self._read_master_dataframe(response.content)
            self._log("security_master_downloaded", str(cache_path), f"rows={len(df)}")
            return df
        except Exception as e:
            self._log("security_master_download_failed", str(e))

        if cache_path.exists():
            try:
                df = self._read_master_dataframe(cache_path.read_bytes())
                if not df.empty:
                    self._log("security_master_stale_cache_used", str(cache_path), f"rows={len(df)}")
                    return df
            except Exception as e:
                self._log("security_master_stale_cache_failed", str(e))

        return pd.DataFrame()

    def _normalize_master_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_")
            for col in normalized.columns
        ]
        return normalized

    def _extract_expiry_from_master(self, row) -> Optional[str]:
        for col in [
            "expiry_date", "expiry", "expirydate", "exp_date", "expiry_dt",
            "maturity_date", "contract_expiry", "securityexpirydate"
        ]:
            value = row.get(col)
            if value in (None, "", "nan"):
                continue

            parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
            if pd.isna(parsed):
                continue

            return self._format_expiry(parsed.to_pydatetime())

        return None

    def _match_symbol_in_master(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        aliases = self._symbol_aliases(symbol)
        symbol_columns = [
            "exchangecode", "exchange_code", "stock_code", "stockcode", "symbol", "underlying", "underlying_name",
            "underlying_symbol", "exchange_stock_code", "short_name", "stock_name",
            "shortname", "assetname", "companyname"
        ]

        mask = pd.Series(False, index=df.index)
        for col in symbol_columns:
            if col not in df.columns:
                continue
            values = df[col].astype(str).str.upper().str.strip()
            for alias in aliases:
                mask = mask | values.eq(alias)

        if mask.any():
            return df.loc[mask].copy()

        # Fallback for security master variants that embed the underlying symbol
        # in a longer display name.
        fuzzy_columns = [
            "underlying_name", "stock_name", "companyname", "short_name", "assetname"
        ]
        for col in fuzzy_columns:
            if col not in df.columns:
                continue
            values = df[col].astype(str).str.upper().str.strip()
            for alias in aliases:
                if len(alias) >= 4:
                    mask = mask | values.str.contains(alias, regex=False, na=False)

        return df.loc[mask].copy()

    def _filter_option_rows_from_master(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df.copy()

        if "exchange_code" in filtered.columns:
            filtered = filtered[
                filtered["exchange_code"].astype(str).str.upper().str.strip().eq("NFO")
            ]

        product_columns = ["product_type", "instrument_type", "segment", "series", "instrumentname"]
        option_mask = pd.Series(False, index=filtered.index)
        for col in product_columns:
            if col not in filtered.columns:
                continue
            values = filtered[col].astype(str).str.upper()
            option_mask = option_mask | values.str.contains("OPT|OPTION", regex=True, na=False)

        if option_mask.any():
            filtered = filtered.loc[option_mask].copy()

        return filtered

    def _generate_dynamic_expiries(self, symbol: str, count: int = 6):
        now_utc = datetime.utcnow()
        expiries = []

        if self._is_index_symbol(symbol):
            target_weekday = 1  # Tuesday
            current_date = now_utc.date()

            while len(expiries) < count:
                days_ahead = (target_weekday - current_date.weekday()) % 7
                if days_ahead == 0 and now_utc.hour >= 6:
                    days_ahead = 7

                expiry_date = current_date + timedelta(days=days_ahead)
                expiries.append(
                    self._format_expiry(datetime.combine(expiry_date, datetime.min.time()))
                )
                current_date = expiry_date + timedelta(days=1)

            return expiries

        # Individual stock options are monthly contracts; use the last Tuesday
        # of the month as the fallback expiry schedule.
        month_cursor = datetime(now_utc.year, now_utc.month, 1)

        while len(expiries) < count:
            expiry_date = self._last_weekday_of_month(
                year=month_cursor.year,
                month=month_cursor.month,
                weekday=1,  # Tuesday
            )

            if expiry_date > now_utc:
                expiries.append(self._format_expiry(expiry_date))

            if month_cursor.month == 12:
                month_cursor = datetime(month_cursor.year + 1, 1, 1)
            else:
                month_cursor = datetime(month_cursor.year, month_cursor.month + 1, 1)

        return expiries

    def _extract_request_symbols_from_master(self, df: pd.DataFrame, input_symbol: str) -> list[str]:
        candidates = []

        for symbol in self._symbol_aliases(input_symbol):
            if symbol not in candidates:
                candidates.append(symbol)

        code_columns = [
            "exchangecode",
            "exchange_code",
            "underlying_symbol",
            "underlying",
            "exchange_stock_code",
            "stock_code",
            "stockcode",
            "symbol",
            "shortname",
            "short_name",
        ]

        for col in code_columns:
            if col not in df.columns:
                continue

            for value in df[col].dropna().astype(str).tolist():
                normalized = self._normalize_symbol(value)
                if not normalized or " " in normalized:
                    continue
                if normalized not in candidates:
                    candidates.append(normalized)

        return candidates

    def _fetch_market_metadata(self, symbol: str):
        metadata = self._market_metadata_resolver.resolve(symbol)
        if metadata.get("request_symbols"):
            return metadata
        return {
            "expiries": list(metadata.get("expiries", [])),
            "request_symbols": self._symbol_aliases(symbol),
        }

    def _init_client(self):
        BreezeConnect = _load_breeze_connect_class()
        creds = get_icici_runtime_config()

        if BreezeConnect is None:
            raise ImportError(
                "breeze-connect is not installed correctly. Run: pip install --upgrade breeze-connect"
            )

        if not creds["api_key"] or str(creds["api_key"]).startswith("YOUR_"):
            raise ValueError("ICICI_BREEZE_API_KEY is not configured in settings.py")

        if not creds["secret_key"] or str(creds["secret_key"]).startswith("YOUR_"):
            raise ValueError("ICICI_BREEZE_SECRET_KEY is not configured in settings.py")

        if not creds["session_token"] or str(creds["session_token"]).startswith("YOUR_"):
            raise ValueError("ICICI_BREEZE_SESSION_TOKEN is not configured in settings.py")

        self.breeze = BreezeConnect(api_key=creds["api_key"])
        self.breeze.generate_session(
            api_secret=creds["secret_key"],
            session_token=str(creds["session_token"])
        )
        self._log("Breeze session initialized")

    def _resolve_expiry_candidates(self, symbol: str):
        symbol = self._normalize_symbol(symbol)
        cleaned = []
        metadata = self._fetch_market_metadata(symbol)
        market_candidates = metadata.get("expiries", [])
        manual_candidates = ICICI_SYMBOL_EXPIRY_CANDIDATES.get(symbol, [])

        for expiry in market_candidates + manual_candidates:
            if expiry and expiry not in cleaned:
                cleaned.append(expiry)

        if ICICI_DEFAULT_EXPIRY_DATE and ICICI_DEFAULT_EXPIRY_DATE not in cleaned:
            cleaned.append(ICICI_DEFAULT_EXPIRY_DATE)

        for expiry in self._generate_dynamic_expiries(symbol):
            if expiry not in cleaned:
                cleaned.append(expiry)

        self._log("resolved_expiry_candidates", f"symbol={symbol}", f"candidates={cleaned}")
        return cleaned

    def _resolve_request_symbols(self, symbol: str) -> list[str]:
        symbol = self._normalize_symbol(symbol)
        metadata = self._fetch_market_metadata(symbol)
        request_symbols = metadata.get("request_symbols", [])
        cleaned = []

        for candidate in request_symbols + self._symbol_aliases(symbol):
            if candidate and candidate not in cleaned:
                cleaned.append(candidate)

        self._log("resolved_request_symbols", f"symbol={symbol}", f"candidates={cleaned}")
        return cleaned

    def _preview_response(self, response, label):
        if not self.debug:
            return

        try:
            if isinstance(response, dict):
                keys = list(response.keys())
                preview = json.dumps(response)[:800]
                self._log(f"{label}_keys", keys)
                self._log(f"{label}_preview", preview)
            else:
                self._log(f"{label}_type", type(response).__name__)
                self._log(f"{label}_preview", str(response)[:800])
        except Exception as e:
            self._log(f"{label}_preview_failed", str(e))

    def _extract_success_rows(self, response, label="resp"):
        self._preview_response(response, label)

        if response is None:
            return []

        if isinstance(response, dict):
            success = response.get("Success")
            if isinstance(success, list):
                return success

            if isinstance(response.get("success"), list):
                return response.get("success")

            for key in ["Error", "error", "Status", "status", "message", "Message"]:
                if key in response:
                    self._log(f"{label}_{key}", response.get(key))

        return []

    def _normalize_side(self, side: str) -> str:
        side = str(side).strip().lower()
        if side == "call":
            return "CE"
        if side == "put":
            return "PE"
        if side == "ce":
            return "CE"
        if side == "pe":
            return "PE"
        return side.upper()

    def _safe_float(self, value, default=0.0):
        try:
            if value in [None, ""]:
                return default
            return float(value)
        except Exception:
            return default

    def _normalize_rows(self, rows):
        normalized = []

        for row in rows:
            try:
                strike = row.get("strike_price", row.get("strike"))
                option_typ = self._normalize_side(
                    row.get("right", row.get("option_type", ""))
                )

                if strike in [None, ""] or option_typ not in ["CE", "PE"]:
                    continue

                ltp = row.get("ltp", row.get("last_traded_price", row.get("lastPrice", 0)))
                oi = row.get("open_interest", row.get("openInterest", 0))
                chg_oi = row.get("chnge_oi", row.get("changeinOI", row.get("change_in_oi", 0)))
                iv = row.get("implied_volatility", row.get("iv", row.get("impliedVolatility", 0)))
                volume = row.get(
                    "total_quantity_traded",
                    row.get("total_traded_volume", row.get("totalTradedVolume", 0))
                )
                expiry_dt = row.get("expiry_date", row.get("expiryDate"))
                spot_price = row.get("spot_price", row.get("underlying_value", row.get("spotPrice")))

                strike_val = self._safe_float(strike, None)
                if strike_val is None:
                    continue

                ltp_val = self._safe_float(ltp, 0.0)
                oi_val = self._safe_float(oi, 0.0)
                chg_oi_val = self._safe_float(chg_oi, 0.0)
                iv_val = self._safe_float(iv, 0.0)
                volume_val = self._safe_float(volume, 0.0)
                spot_val = self._safe_float(spot_price, None)

                if iv_val <= 0:
                    estimated_iv = self._estimate_implied_volatility(
                        price=ltp_val,
                        spot=spot_val,
                        strike=strike_val,
                        expiry_date=expiry_dt,
                        option_type=option_typ,
                    )
                    if estimated_iv is not None:
                        iv_val = estimated_iv

                normalized.append({
                    "strikePrice": strike_val,
                    "OPTION_TYP": option_typ,
                    "lastPrice": ltp_val,
                    "openInterest": oi_val,
                    "changeinOI": chg_oi_val,
                    "impliedVolatility": iv_val,
                    "totalTradedVolume": volume_val,
                    "IV": iv_val,
                    "VOLUME": volume_val,
                    "OPEN_INT": oi_val,
                    "STRIKE_PR": strike_val,
                    "LAST_PRICE": ltp_val,
                    "EXPIRY_DT": expiry_dt,
                })
            except Exception as e:
                self._log("row_normalization_failed", str(e), row)

        df = pd.DataFrame(normalized)
        if not df.empty:
            df = df.dropna(subset=["strikePrice"])
            df = df.sort_values(["strikePrice", "OPTION_TYP"]).reset_index(drop=True)
        return df

    def _fetch_for_expiry(self, symbol: str, expiry_date: str, request_symbols: list[str]):
        for request_symbol in request_symbols:
            self._log(
                "fetching option chain",
                f"symbol={symbol}",
                f"request_symbol={request_symbol}",
                f"expiry={expiry_date}",
            )

            call_resp = self.breeze.get_option_chain_quotes(
                stock_code=request_symbol,
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_date,
                right="call",
                strike_price="",
            )

            put_resp = self.breeze.get_option_chain_quotes(
                stock_code=request_symbol,
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_date,
                right="put",
                strike_price="",
            )

            call_rows = self._extract_success_rows(call_resp, label=f"call_{request_symbol}_{expiry_date}")
            put_rows = self._extract_success_rows(put_resp, label=f"put_{request_symbol}_{expiry_date}")

            self._log(
                "expiry_result",
                expiry_date,
                f"request_symbol={request_symbol}",
                f"call_rows={len(call_rows)}",
                f"put_rows={len(put_rows)}"
            )

            all_rows = call_rows + put_rows
            if not all_rows:
                continue

            df = self._normalize_rows(all_rows)
            if df.empty:
                self._log(
                    "normalization_produced_empty_df",
                    f"request_symbol={request_symbol}",
                    f"expiry={expiry_date}",
                )
                continue

            self._log("normalized_rows", len(df), f"request_symbol={request_symbol}", f"expiry={expiry_date}")
            return df

        return pd.DataFrame()

    def fetch_option_chain(self, symbol="NIFTY"):
        symbol = self._normalize_symbol(symbol)
        expiry_candidates = self._resolve_expiry_candidates(symbol)
        request_symbols = self._resolve_request_symbols(symbol)

        last_errors = []

        for expiry_date in expiry_candidates:
            try:
                df = self._fetch_for_expiry(
                    symbol=symbol,
                    expiry_date=expiry_date,
                    request_symbols=request_symbols,
                )
                if df is not None and not df.empty:
                    self._log("selected_expiry", expiry_date, f"rows={len(df)}")
                    return df
                last_errors.append(f"{expiry_date}:no_data")
            except Exception as e:
                self._log("expiry_fetch_exception", expiry_date, str(e))
                last_errors.append(f"{expiry_date}:{e}")

        self._log("all_expiry_attempts_failed", last_errors)
        print("Option chain download error: ICICI returned no option chain rows for any attempted expiry")
        return pd.DataFrame()

    def close(self):
        return
