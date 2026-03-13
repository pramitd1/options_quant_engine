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

    def __init__(self, debug=None):
        self.debug = ICICI_DEBUG if debug is None else debug
        self.breeze = None
        self._init_client()

    def _log(self, *args):
        if self.debug:
            print("[ICICI DEBUG]", *args)

    def _format_expiry(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT06:00:00.000Z")

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
        symbol = symbol.upper().strip()
        symbol_columns = [
            "stock_code", "stockcode", "symbol", "underlying", "underlying_name",
            "underlying_symbol", "exchange_stock_code", "short_name", "stock_name",
            "shortname", "assetname", "companyname"
        ]

        mask = pd.Series(False, index=df.index)
        for col in symbol_columns:
            if col not in df.columns:
                continue
            values = df[col].astype(str).str.upper().str.strip()
            mask = mask | values.eq(symbol)

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
        weekday_map = {
            "NIFTY": 1,
            "BANKNIFTY": 1,
            "FINNIFTY": 1,
        }
        target_weekday = weekday_map.get(symbol.upper().strip(), 3)
        now_utc = datetime.utcnow()
        current_date = now_utc.date()
        expiries = []

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

    def _fetch_market_expiries(self, symbol: str):
        master_df = self._load_security_master()
        if master_df.empty:
            return []

        normalized = self._normalize_master_columns(master_df)
        symbol_rows = self._match_symbol_in_master(normalized, symbol)
        option_rows = self._filter_option_rows_from_master(symbol_rows)

        expiries = []
        for _, row in option_rows.iterrows():
            expiry = self._extract_expiry_from_master(row)
            if expiry and expiry not in expiries:
                expiries.append(expiry)

        expiries.sort(key=lambda value: datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.000Z"))
        self._log("icici_master_expiry_candidates", f"symbol={symbol}", f"candidates={expiries[:10]}")
        return expiries

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
        symbol = symbol.upper().strip()
        cleaned = []
        market_candidates = self._fetch_market_expiries(symbol)
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

    def _fetch_for_expiry(self, symbol: str, expiry_date: str):
        self._log("fetching option chain", f"symbol={symbol}", f"expiry={expiry_date}")

        call_resp = self.breeze.get_option_chain_quotes(
            stock_code=symbol,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
            right="call",
            strike_price="",
        )

        put_resp = self.breeze.get_option_chain_quotes(
            stock_code=symbol,
            exchange_code="NFO",
            product_type="options",
            expiry_date=expiry_date,
            right="put",
            strike_price="",
        )

        call_rows = self._extract_success_rows(call_resp, label=f"call_{expiry_date}")
        put_rows = self._extract_success_rows(put_resp, label=f"put_{expiry_date}")

        self._log(
            "expiry_result",
            expiry_date,
            f"call_rows={len(call_rows)}",
            f"put_rows={len(put_rows)}"
        )

        all_rows = call_rows + put_rows
        if not all_rows:
            return pd.DataFrame()

        df = self._normalize_rows(all_rows)
        if df.empty:
            self._log("normalization_produced_empty_df", f"expiry={expiry_date}")
        self._log("normalized_rows", len(df), f"expiry={expiry_date}")
        return df

    def fetch_option_chain(self, symbol="NIFTY"):
        symbol = symbol.upper().strip()
        expiry_candidates = self._resolve_expiry_candidates(symbol)

        last_errors = []

        for expiry_date in expiry_candidates:
            try:
                df = self._fetch_for_expiry(symbol=symbol, expiry_date=expiry_date)
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
