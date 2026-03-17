"""
Adapter that reads the real NSE bhav-copy historical database and converts it
into the column schema expected by the backtest / tuning pipeline.

The merged parquet lives at:
    data_store/historical/merged/NIFTY_option_chain_historical.parquet

This module:
 1. Loads the parquet, filters to the requested symbol + year window
 2. Computes implied volatility via Newton-Raphson BS-inversion
 3. Renames columns to match the backtest contract:
      timestamp, spot, strikePrice, OPTION_TYP, lastPrice,
      openInterest, changeinOI, impliedVolatility,
      totalTradedVolume, expiry_days
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DATA_DIR, RISK_FREE_RATE
from utils.math_helpers import norm_cdf as _norm_cdf


HISTORICAL_DIR = Path(DATA_DIR) / "historical" / "merged"


# ------------------------------------------------------------------
# Black-Scholes helpers (vectorised where possible, row-level fallback)
# ------------------------------------------------------------------

def _bs_price(spot: float, strike: float, t: float, sigma: float,
              option_type: str, r: float = RISK_FREE_RATE) -> float:
    """Standard Black-Scholes European price (no dividend)."""
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
    return spot * math.sqrt(t) * _pdf(d1)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _implied_vol_newton(market_price: float, spot: float, strike: float,
                        t: float, option_type: str, r: float = RISK_FREE_RATE,
                        tol: float = 1e-6, max_iter: int = 50) -> float:
    """Newton-Raphson implied-vol solver.  Returns annualised IV as a
    *percentage* (e.g. 18.0 for 18%)."""
    intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    if market_price <= intrinsic or t <= 0:
        return 0.0

    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        price = _bs_price(spot, strike, t, sigma, option_type, r)
        vega = _bs_vega(spot, strike, t, sigma, r)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.001
        if sigma > 5.0:          # cap at 500%
            return 500.0
        if abs(price - market_price) < tol:
            break
    if sigma > 5.0:
        return 500.0
    return round(sigma * 100.0, 2)   # percentage


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_historical_option_chain(
    symbol: str = "NIFTY",
    years: Optional[int] = None,
) -> pd.DataFrame:
    """Load the real historical option-chain parquet and return a DataFrame
    in the same column schema that the backtest pipeline expects.

    Parameters
    ----------
    symbol : str
        Underlying symbol (default ``"NIFTY"``).
    years : int or None
        If given, only keep the most recent *years* of data.

    Returns
    -------
    pd.DataFrame   with columns:
        timestamp, spot, strikePrice, OPTION_TYP, lastPrice, openInterest,
        changeinOI, impliedVolatility, totalTradedVolume, expiry_days,
        EXPIRY_DT
    """
    parquet_path = HISTORICAL_DIR / f"{symbol.upper()}_option_chain_historical.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Historical database not found at {parquet_path}.  "
            "Run  scripts/historical_data_download.py  first."
        )

    df = pd.read_parquet(parquet_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # --- filter to CE / PE only (drop futures rows) ---
    df = df[df["option_type"].isin(["CE", "PE"])].copy()

    # --- year window ---
    if years is not None and years > 0:
        cutoff = df["trade_date"].max() - pd.DateOffset(years=years)
        df = df[df["trade_date"] >= cutoff].copy()

    # --- resolve spot: prefer spot_close, fall back to underlying_price ---
    if "spot_close" in df.columns and "underlying_price" in df.columns:
        df["_spot"] = df["spot_close"].fillna(df["underlying_price"])
    elif "spot_close" in df.columns:
        df["_spot"] = df["spot_close"]
    elif "underlying_price" in df.columns:
        df["_spot"] = df["underlying_price"]
    else:
        df["_spot"] = df["strike_price"]  # last resort

    # --- expiry_days ---
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["expiry_days"] = (df["expiry_date"] - df["trade_date"]).dt.days.clip(lower=0)

    # --- compute implied volatility ---
    df["impliedVolatility"] = df.apply(
        lambda r: _implied_vol_newton(
            market_price=float(r["close"]) if r["close"] > 0 else float(r.get("last_price", 0)),
            spot=float(r["_spot"]),
            strike=float(r["strike_price"]),
            t=max(float(r["expiry_days"]) / 365.0, 1e-6),
            option_type=r["option_type"],
        ),
        axis=1,
    )

    # --- rename to backtest schema ---
    out = pd.DataFrame({
        "timestamp":          df["trade_date"],
        "spot":               df["_spot"].round(2),
        "strikePrice":        df["strike_price"].astype(int),
        "OPTION_TYP":         df["option_type"],
        "lastPrice":          df["close"].round(2),
        "openInterest":       df["open_interest"].astype(int),
        "changeinOI":         df["change_in_oi"].fillna(0).astype(int),
        "impliedVolatility":  df["impliedVolatility"],
        "totalTradedVolume":  df["contracts"].fillna(0).astype(int),
        "expiry_days":        df["expiry_days"].astype(int),
        "EXPIRY_DT":          df["expiry_date"].dt.strftime("%Y-%m-%d"),
    })

    # duplicate aliases expected by _normalize_columns / downstream
    out["STRIKE_PR"]  = out["strikePrice"]
    out["OPEN_INT"]   = out["openInterest"]
    out["LAST_PRICE"] = out["lastPrice"]
    out["VOLUME"]     = out["totalTradedVolume"]
    out["IV"]         = out["impliedVolatility"]

    out = out.sort_values("timestamp").reset_index(drop=True)
    return out
