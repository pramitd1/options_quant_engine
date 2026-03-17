"""
Historical Snapshot API
========================
Provides deterministic, point-in-time market snapshots from the NSE bhav-copy
historical database for backtesting and signal replay.

Key function:
    replay_historical_snapshot(trade_date, symbol) → dict

Returns data in the format expected by `run_preloaded_engine_snapshot()`:
    - spot_snapshot: dict with spot, day_open, day_high, day_low, prev_close, timestamp
    - option_chain: DataFrame in backtest schema (timestamp, spot, strikePrice, OPTION_TYP,
      lastPrice, openInterest, changeinOI, impliedVolatility, totalTradedVolume, expiry_days,
      EXPIRY_DT)
    - global_market_snapshot: dict from precomputed features parquet (or neutral fallback)
    - macro_event_state: dict from historical macro events JSON
    - quality_score: float (0-100 day quality)
    - metadata: provenance metadata
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, RISK_FREE_RATE
from data.historical_chain_normalizer import SPOT_FILE, load_normalized_day
from utils.math_helpers import norm_cdf as _norm_cdf

log = logging.getLogger(__name__)

_DATA = Path(DATA_DIR)
GLOBAL_FEATURES_FILE = _DATA / "historical" / "global_market" / "features" / "global_market_features.parquet"
MACRO_EVENTS_FILE = _DATA / "historical" / "macro_events" / "india_macro_events_historical.json"

# Cache to avoid repeated disk reads
_spot_cache: pd.DataFrame | None = None
_global_features_cache: pd.DataFrame | None = None
_macro_events_cache: list[dict] | None = None


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def replay_historical_snapshot(
    trade_date: date | str,
    symbol: str = "NIFTY",
    *,
    compute_iv: bool = True,
    include_global_market: bool = True,
    include_macro_events: bool = True,
) -> dict:
    """Build a deterministic, point-in-time snapshot for backtesting.

    Parameters
    ----------
    trade_date : date or str
        Target date (YYYY-MM-DD).
    symbol : str
        Target symbol (default NIFTY).
    compute_iv : bool
        Whether to compute implied volatility via Newton-Raphson.
    include_global_market : bool
        Whether to include historical global market snapshot.
    include_macro_events : bool
        Whether to include macro event state.

    Returns
    -------
    dict with keys:
        ok: bool
        spot_snapshot: dict
        option_chain: pd.DataFrame (backtest schema)
        global_market_snapshot: dict
        macro_event_state: dict
        quality_score: float
        metadata: dict
    """
    if isinstance(trade_date, str):
        trade_date = pd.to_datetime(trade_date).date()

    # 1. Load normalized chain for this date
    chain = load_normalized_day(trade_date, symbol=symbol, options_only=True)

    if chain is None or chain.empty:
        return {
            "ok": False,
            "error": f"No option chain data for {symbol} on {trade_date}",
            "spot_snapshot": None,
            "option_chain": pd.DataFrame(),
            "global_market_snapshot": None,
            "macro_event_state": None,
            "quality_score": 0.0,
            "metadata": {"trade_date": str(trade_date), "symbol": symbol},
        }

    # 2. Build spot snapshot
    spot_data = _get_spot_data(trade_date, symbol)
    if spot_data is None:
        # Derive from chain underlying_price
        underlying = chain["underlying_price"].dropna()
        spot_val = float(underlying.iloc[0]) if len(underlying) > 0 else float(chain["strike_price"].median())
        spot_data = {"open": spot_val, "high": spot_val, "low": spot_val,
                     "close": spot_val, "volume": 0}

    prev_close = _get_prev_close(trade_date, symbol)

    spot_snapshot = {
        "symbol": symbol.upper(),
        "spot": float(spot_data["close"]),
        "day_open": float(spot_data["open"]),
        "day_high": float(spot_data["high"]),
        "day_low": float(spot_data["low"]),
        "prev_close": prev_close,
        "timestamp": f"{trade_date}T15:30:00+05:30",
        "lookback_avg_range_pct": None,
    }

    # 3. Convert chain to backtest schema
    option_chain = _to_backtest_schema(chain, float(spot_data["close"]), compute_iv=compute_iv)

    # 4. Global market snapshot
    global_market_snapshot = None
    if include_global_market:
        global_market_snapshot = _build_historical_global_market_snapshot(
            trade_date, symbol
        )

    # 5. Macro event state
    macro_event_state = None
    if include_macro_events:
        macro_event_state = _build_historical_macro_event_state(
            trade_date, symbol
        )

    # 6. Quality score
    quality_score = _compute_day_quality(chain, spot_data)

    return {
        "ok": True,
        "spot_snapshot": spot_snapshot,
        "option_chain": option_chain,
        "global_market_snapshot": global_market_snapshot,
        "macro_event_state": macro_event_state,
        "quality_score": quality_score,
        "metadata": {
            "trade_date": str(trade_date),
            "symbol": symbol,
            "data_source": "NSE_BHAV_COPY",
            "granularity": "EOD",
            "chain_rows": len(option_chain),
            "unique_strikes": int(option_chain["strikePrice"].nunique()) if len(option_chain) > 0 else 0,
            "unique_expiries": int(option_chain["EXPIRY_DT"].nunique()) if len(option_chain) > 0 else 0,
        },
    }


def get_available_dates(
    symbol: str = "NIFTY",
    from_year: int = 2012,
    to_year: int | None = None,
) -> list[date]:
    """Return list of all trading dates available in the historical database."""
    from data.historical_chain_normalizer import NSE_FO_DIR

    if to_year is None:
        to_year = date.today().year

    all_dates = set()
    for year in range(from_year, to_year + 1):
        pf = NSE_FO_DIR / f"fo_bhav_{symbol}_{year}.parquet"
        if pf.exists():
            df = pd.read_parquet(pf, columns=["trade_date"])
            dates = pd.to_datetime(df["trade_date"]).dt.date.unique()
            all_dates.update(dates)

    return sorted(all_dates)


# ---------------------------------------------------------------
# Spot data helpers
# ---------------------------------------------------------------

def _load_spot_df() -> pd.DataFrame:
    global _spot_cache
    if _spot_cache is not None:
        return _spot_cache
    if not SPOT_FILE.exists():
        return pd.DataFrame()
    _spot_cache = pd.read_parquet(SPOT_FILE)
    _spot_cache["date"] = pd.to_datetime(_spot_cache["date"]).dt.date
    return _spot_cache


def _get_spot_data(trade_date: date, symbol: str) -> dict | None:
    spot = _load_spot_df()
    if spot.empty:
        return None
    row = spot[spot["date"] == trade_date]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": int(r.get("volume", 0)),
    }


def _get_prev_close(trade_date: date, symbol: str) -> float:
    spot = _load_spot_df()
    if spot.empty:
        return 0.0
    prev = spot[spot["date"] < trade_date]
    if prev.empty:
        return 0.0
    return float(prev.iloc[-1]["close"])


# ---------------------------------------------------------------
# Backtest schema conversion + IV computation
# ---------------------------------------------------------------

def _bs_price(spot: float, strike: float, t: float, sigma: float,
              option_type: str, r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _bs_vega(spot: float, strike: float, t: float, sigma: float,
             r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return spot * math.sqrt(t) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


def _implied_vol_newton(market_price: float, spot: float, strike: float,
                        t: float, option_type: str, r: float = RISK_FREE_RATE,
                        tol: float = 1e-6, max_iter: int = 50) -> float:
    intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    if market_price <= intrinsic or t <= 0:
        return 0.0
    sigma = 0.20
    for _ in range(max_iter):
        price = _bs_price(spot, strike, t, sigma, option_type, r)
        vega = _bs_vega(spot, strike, t, sigma, r)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.001
        if sigma > 5.0:
            return 500.0
        if abs(price - market_price) < tol:
            break
    return round(sigma * 100.0, 2) if sigma <= 5.0 else 500.0


def _to_backtest_schema(chain: pd.DataFrame, spot: float,
                        compute_iv: bool = True) -> pd.DataFrame:
    """Convert normalized chain to the schema expected by intraday_backtester."""
    df = chain.copy()

    # Ensure datetime types
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    # Compute expiry_days
    df["expiry_days"] = (df["expiry_date"] - df["trade_date"]).dt.days.clip(lower=0)

    # Use underlying_price if available, else use spot parameter
    df["_spot"] = df["underlying_price"].fillna(spot)

    # Compute IV if requested
    if compute_iv:
        def _compute_iv(row):
            price = float(row["close"]) if row["close"] > 0 else float(row.get("last_price", 0))
            if price <= 0:
                return 0.0
            return _implied_vol_newton(
                market_price=price,
                spot=float(row["_spot"]),
                strike=float(row["strike_price"]),
                t=max(float(row["expiry_days"]) / 365.0, 1e-6),
                option_type=row["option_type"],
            )
        df["impliedVolatility"] = df.apply(_compute_iv, axis=1)
    else:
        df["impliedVolatility"] = 0.0

    # Build backtest schema
    out = pd.DataFrame({
        "timestamp": df["trade_date"],
        "spot": df["_spot"].round(2),
        "strikePrice": df["strike_price"].astype(int),
        "OPTION_TYP": df["option_type"],
        "lastPrice": df["close"].round(2),
        "openInterest": df["open_interest"].fillna(0).astype(int),
        "changeinOI": df["change_in_oi"].fillna(0).astype(int),
        "impliedVolatility": df["impliedVolatility"],
        "totalTradedVolume": df["contracts"].fillna(0).astype(int),
        "expiry_days": df["expiry_days"].astype(int),
        "EXPIRY_DT": df["expiry_date"].dt.strftime("%Y-%m-%d"),
    })

    # Add aliases expected by downstream
    out["STRIKE_PR"] = out["strikePrice"]
    out["OPEN_INT"] = out["openInterest"]
    out["LAST_PRICE"] = out["lastPrice"]
    out["VOLUME"] = out["totalTradedVolume"]
    out["IV"] = out["impliedVolatility"]

    return out.sort_values(["EXPIRY_DT", "OPTION_TYP", "strikePrice"]).reset_index(drop=True)


# ---------------------------------------------------------------
# Historical global market snapshot
# ---------------------------------------------------------------

def _load_global_features() -> pd.DataFrame:
    global _global_features_cache
    if _global_features_cache is not None:
        return _global_features_cache
    if not GLOBAL_FEATURES_FILE.exists():
        return pd.DataFrame()
    _global_features_cache = pd.read_parquet(GLOBAL_FEATURES_FILE)
    _global_features_cache["date"] = pd.to_datetime(
        _global_features_cache["date"]
    ).dt.date
    return _global_features_cache


def _build_historical_global_market_snapshot(
    trade_date: date, symbol: str
) -> dict:
    """Build a global market snapshot from precomputed historical features."""
    gf = _load_global_features()
    if gf.empty:
        return _neutral_global_snapshot(symbol, trade_date)

    row = gf[gf["date"] == trade_date]
    if row.empty:
        # Try previous trading day
        prev = gf[gf["date"] < trade_date]
        if prev.empty:
            return _neutral_global_snapshot(symbol, trade_date)
        row = prev.tail(1)

    r = row.iloc[0]

    market_inputs = {}
    for col in [
        "oil_change_24h", "gold_change_24h", "copper_change_24h",
        "vix_change_24h", "india_vix_change_24h", "india_vix_level",
        "sp500_change_24h", "nasdaq_change_24h", "us10y_change_bp",
        "usdinr_change_24h",
    ]:
        val = r.get(col)
        market_inputs[col] = float(val) if pd.notna(val) else None

    # Realized vol from features
    for col in ["nifty50_realized_vol_5d", "nifty50_realized_vol_30d"]:
        val = r.get(col)
        if pd.notna(val):
            if col.endswith("_5d"):
                market_inputs["realized_vol_5d"] = float(val)
            elif col.endswith("_30d"):
                market_inputs["realized_vol_30d"] = float(val)

    return {
        "symbol": symbol.upper(),
        "provider": "HISTORICAL_FEATURES",
        "as_of": f"{trade_date}T15:30:00+05:30",
        "data_available": True,
        "neutral_fallback": False,
        "issues": [],
        "warnings": [],
        "stale": False,
        "lookback_days": 5,
        "market_inputs": market_inputs,
    }


def _neutral_global_snapshot(symbol: str, trade_date: date) -> dict:
    return {
        "symbol": symbol.upper(),
        "provider": "HISTORICAL_NEUTRAL",
        "as_of": f"{trade_date}T15:30:00+05:30",
        "data_available": False,
        "neutral_fallback": True,
        "issues": [],
        "warnings": ["historical_global_market_data_unavailable"],
        "stale": False,
        "lookback_days": None,
        "market_inputs": {},
    }


# ---------------------------------------------------------------
# Historical macro event state
# ---------------------------------------------------------------

def _load_macro_events() -> list[dict]:
    global _macro_events_cache
    if _macro_events_cache is not None:
        return _macro_events_cache
    if not MACRO_EVENTS_FILE.exists():
        return []
    with open(MACRO_EVENTS_FILE) as f:
        data = json.load(f)
    # The JSON may be a wrapper dict with an "events" key, or a flat list
    if isinstance(data, dict):
        _macro_events_cache = data.get("events", [])
    else:
        _macro_events_cache = data
    return _macro_events_cache


def _build_historical_macro_event_state(
    trade_date: date, symbol: str
) -> dict:
    """Build macro event state from historical event schedule."""
    events = _load_macro_events()
    if not events:
        return _neutral_macro_state(trade_date)

    trade_date_str = str(trade_date)

    # Find events within ±2 days window
    pre_window = trade_date - timedelta(days=2)
    post_window = trade_date + timedelta(days=2)

    active_events = []
    upcoming_events = []
    recent_events = []

    for evt in events:
        # Events use "timestamp" field (ISO format with timezone)
        evt_ts = evt.get("timestamp") or evt.get("date", "")
        try:
            evt_date = pd.to_datetime(evt_ts).date()
        except Exception:
            continue

        if evt_date == trade_date:
            active_events.append(evt)
        elif trade_date < evt_date <= post_window:
            upcoming_events.append(evt)
        elif pre_window <= evt_date < trade_date:
            recent_events.append(evt)

    if not active_events and not upcoming_events and not recent_events:
        return _neutral_macro_state(trade_date)

    # Determine event window status
    if active_events:
        window_status = "DURING_EVENT"
        risk_score = max(
            _event_risk_score(evt) for evt in active_events
        )
    elif upcoming_events:
        window_status = "PRE_EVENT"
        risk_score = max(
            _event_risk_score(evt) * 0.6 for evt in upcoming_events
        )
    else:
        window_status = "POST_EVENT"
        risk_score = max(
            _event_risk_score(evt) * 0.3 for evt in recent_events
        )

    _evt_name = lambda e: e.get("name") or e.get("event", "Unknown")

    next_event = upcoming_events[0] if upcoming_events else (active_events[0] if active_events else None)

    return {
        "enabled": True,
        "as_of": f"{trade_date}T15:30:00+05:30",
        "event_window_status": window_status,
        "macro_event_risk_score": round(risk_score, 2),
        "event_lockdown_flag": risk_score >= 0.7,
        "next_event_name": _evt_name(next_event) if next_event else None,
        "next_event_date": (next_event.get("timestamp") or next_event.get("date")) if next_event else None,
        "active_event_name": _evt_name(active_events[0]) if active_events else None,
        "event_count_in_window": len(active_events) + len(upcoming_events),
        "events": {
            "active": [_evt_name(e) for e in active_events],
            "upcoming": [_evt_name(e) for e in upcoming_events],
            "recent": [_evt_name(e) for e in recent_events],
        },
    }


def _neutral_macro_state(trade_date: date) -> dict:
    return {
        "enabled": True,
        "as_of": f"{trade_date}T15:30:00+05:30",
        "event_window_status": "NO_EVENT",
        "macro_event_risk_score": 0.0,
        "event_lockdown_flag": False,
        "next_event_name": None,
        "next_event_date": None,
        "active_event_name": None,
        "event_count_in_window": 0,
        "events": {"active": [], "upcoming": [], "recent": []},
    }


def _event_risk_score(evt: dict) -> float:
    """Assign 0-1 risk score based on event type."""
    event_name = (evt.get("name") or evt.get("event", "")).lower()
    # Also use severity field if available
    severity = evt.get("severity", "").upper()
    if severity == "CRITICAL":
        return 1.0

    scores = {
        "rbi": 0.9, "mpc": 0.9, "monetary": 0.9,
        "budget": 1.0, "union budget": 1.0,
        "gdp": 0.7,
        "cpi": 0.6, "inflation": 0.6,
        "iip": 0.4,
        "wpi": 0.4,
        "pmi": 0.5,
        "trade": 0.3,
        "election": 0.8,
    }
    for keyword, score in scores.items():
        if keyword in event_name:
            return score
    return 0.3


# ---------------------------------------------------------------
# Day quality
# ---------------------------------------------------------------

def _compute_day_quality(chain: pd.DataFrame, spot_data: dict | None) -> float:
    score = 100.0
    if len(chain) < 20:
        score -= 30
    elif len(chain) < 50:
        score -= 15

    ce = (chain["option_type"] == "CE").sum()
    pe = (chain["option_type"] == "PE").sum()
    if ce == 0 or pe == 0:
        score -= 25

    if "open_interest" in chain.columns:
        zero_oi = (chain["open_interest"] == 0).mean()
        if zero_oi > 0.8:
            score -= 15
        elif zero_oi > 0.5:
            score -= 8

    if "close" in chain.columns:
        zero_close = (chain["close"] == 0).mean()
        if zero_close > 0.5:
            score -= 20

    return max(0.0, score)
