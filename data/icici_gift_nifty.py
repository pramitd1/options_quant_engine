"""
ICICI Breeze adapter for live GIFT NIFTY lead-index quotes.

The engine keeps broad cross-asset history in yfinance, but GIFT NIFTY should
not default to a NIFTY spot proxy. This module fetches GIFT NIFTY through
ICICI Breeze when enabled and degrades to an explicit unavailable state when
the quote cannot be obtained.
"""

from __future__ import annotations

import time as _time_mod
from typing import Any

from config.market_data_policy import (
    GIFT_NIFTY_ICICI_CACHE_TTL_SECONDS,
    GIFT_NIFTY_ICICI_CANDIDATES,
    GIFT_NIFTY_SOURCE,
)
from config.settings import get_icici_runtime_config
from data.icici_breeze_option_chain import _load_breeze_connect_class


_quote_cache: dict | None = None
_quote_cache_time: float = 0.0
_client_cache = None


def invalidate_icici_gift_nifty_cache():
    """Clear ICICI GIFT NIFTY quote and Breeze-client caches."""
    global _quote_cache, _quote_cache_time, _client_cache
    _quote_cache = None
    _quote_cache_time = 0.0
    _client_cache = None


def _safe_number(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
            if not value or value.upper() in {"NA", "N/A", "NONE", "NULL"}:
                return default
        return float(value)
    except Exception:
        return default


def _first_present(row: dict, keys: tuple[str, ...]):
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row:
            return row.get(key)
        lowered_value = lowered.get(key.lower())
        if lowered_value is not None:
            return lowered_value
    return None


def _success_rows(response: Any) -> list[dict]:
    if not isinstance(response, dict):
        return []

    payload = (
        response.get("Success")
        or response.get("success")
        or response.get("data")
        or response.get("Data")
    )
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _response_error(response: Any) -> str | None:
    if not isinstance(response, dict):
        return "non_dict_response"
    for key in ("Error", "error", "Message", "message", "Status", "status"):
        value = response.get(key)
        if value in (None, "", 200, "200", "Success", "success"):
            continue
        if isinstance(value, str) and value.strip().upper() in {"OK", "SUCCESS"}:
            continue
        return f"{key}:{value}"
    return None


def _normalize_quote_row(row: dict) -> dict:
    level = _safe_number(
        _first_present(
            row,
            (
                "ltp",
                "last",
                "last_price",
                "lastPrice",
                "last_traded_price",
                "close",
                "price",
            ),
        ),
        None,
    )
    previous_close = _safe_number(
        _first_present(
            row,
            (
                "previous_close",
                "previousClose",
                "prev_close",
                "prevClose",
                "previous_close_price",
                "previousClosePrice",
                "close_price",
            ),
        ),
        None,
    )
    change_pct = _safe_number(
        _first_present(
            row,
            (
                "change_percent",
                "change_percentage",
                "percent_change",
                "percentage_change",
                "pct_change",
                "change_pct",
            ),
        ),
        None,
    )
    if change_pct is None and level not in (None, 0) and previous_close not in (None, 0):
        change_pct = ((level / previous_close) - 1.0) * 100.0

    timestamp = _first_present(
        row,
        (
            "ltt",
            "last_trade_time",
            "last_traded_time",
            "datetime",
            "timestamp",
            "time",
        ),
    )
    return {
        "gift_nifty_change_24h": change_pct,
        "gift_nifty_level": level,
        "gift_nifty_previous_close": previous_close,
        "gift_nifty_quote_timestamp": timestamp,
    }


def _build_breeze_client():
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    creds = get_icici_runtime_config()
    api_key = str(creds.get("api_key") or "").strip()
    secret_key = str(creds.get("secret_key") or "").strip()
    session_token = str(creds.get("session_token") or "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError("ICICI_BREEZE_API_KEY is not configured")
    if not secret_key or secret_key.startswith("YOUR_"):
        raise ValueError("ICICI_BREEZE_SECRET_KEY is not configured")
    if not session_token or session_token.startswith("YOUR_"):
        raise ValueError("ICICI_BREEZE_SESSION_TOKEN is not configured")

    BreezeConnect = _load_breeze_connect_class()
    if BreezeConnect is None:
        raise ImportError("breeze-connect is not installed correctly")

    client = BreezeConnect(api_key=api_key)
    client.generate_session(api_secret=secret_key, session_token=session_token)
    _client_cache = client
    return client


def _query_candidate(client, candidate: dict[str, str]) -> dict:
    response = client.get_quotes(
        stock_code=str(candidate.get("stock_code") or "").strip(),
        exchange_code=str(candidate.get("exchange_code") or "").strip(),
        expiry_date="",
        product_type=str(candidate.get("product_type") or "").strip(),
        right="",
        strike_price="",
    )
    rows = _success_rows(response)
    if not rows:
        return {
            "available": False,
            "error": _response_error(response) or "empty_success_payload",
            "response": response,
        }
    normalized = _normalize_quote_row(rows[0])
    if normalized.get("gift_nifty_change_24h") is None:
        return {
            "available": False,
            "error": "missing_change_pct_or_previous_close",
            "response": response,
        }
    return {"available": True, **normalized, "response": response}


def fetch_icici_gift_nifty_snapshot(*, force: bool = False) -> dict:
    """
    Fetch a live GIFT NIFTY quote from ICICI Breeze.

    Returns a serializable dictionary. Failures are non-raising so global-risk
    ingestion can continue with the rest of the macro basket.
    """
    global _quote_cache, _quote_cache_time

    if GIFT_NIFTY_SOURCE != "ICICI":
        return {
            "available": False,
            "provider": "ICICI",
            "source": "DISABLED",
            "issues": [],
            "warnings": [f"gift_nifty_icici_disabled:{GIFT_NIFTY_SOURCE}"],
        }

    now = _time_mod.monotonic()
    ttl = max(int(GIFT_NIFTY_ICICI_CACHE_TTL_SECONDS or 0), 0)
    if not force and _quote_cache is not None and (now - _quote_cache_time) < ttl:
        return dict(_quote_cache)

    candidates = tuple(GIFT_NIFTY_ICICI_CANDIDATES or ())
    if not candidates:
        return {
            "available": False,
            "provider": "ICICI",
            "source": "ICICI",
            "issues": ["gift_nifty_icici_candidates_empty"],
            "warnings": [],
        }

    try:
        client = _build_breeze_client()
    except Exception as exc:
        result = {
            "available": False,
            "provider": "ICICI",
            "source": "ICICI",
            "issues": [f"gift_nifty_icici_client_unavailable:{exc}"],
            "warnings": [],
        }
        _quote_cache = result
        _quote_cache_time = now
        return dict(result)

    attempts = []
    for candidate in candidates:
        exchange_code = str(candidate.get("exchange_code") or "").strip().upper()
        stock_code = str(candidate.get("stock_code") or "").strip()
        product_type = str(candidate.get("product_type") or "").strip().lower()
        if not exchange_code or not stock_code:
            continue
        try:
            quote = _query_candidate(
                client,
                {
                    "exchange_code": exchange_code,
                    "stock_code": stock_code,
                    "product_type": product_type,
                },
            )
        except Exception as exc:
            attempts.append(f"{exchange_code}:{stock_code}:{product_type or '-'}:{exc}")
            continue

        if quote.get("available"):
            result = {
                "available": True,
                "provider": "ICICI",
                "source": "ICICI",
                "gift_nifty_source": "ICICI",
                "gift_nifty_exchange_code": exchange_code,
                "gift_nifty_stock_code": stock_code,
                "gift_nifty_product_type": product_type,
                "gift_nifty_change_24h": quote.get("gift_nifty_change_24h"),
                "gift_nifty_level": quote.get("gift_nifty_level"),
                "gift_nifty_previous_close": quote.get("gift_nifty_previous_close"),
                "gift_nifty_quote_timestamp": quote.get("gift_nifty_quote_timestamp"),
                "issues": [],
                "warnings": [],
            }
            _quote_cache = result
            _quote_cache_time = now
            return dict(result)
        attempts.append(f"{exchange_code}:{stock_code}:{product_type or '-'}:{quote.get('error')}")

    result = {
        "available": False,
        "provider": "ICICI",
        "source": "ICICI",
        "issues": ["gift_nifty_icici_quote_unavailable"],
        "warnings": [f"gift_nifty_icici_attempts:{';'.join(attempts[:5])}"],
    }
    _quote_cache = result
    _quote_cache_time = now
    return dict(result)
