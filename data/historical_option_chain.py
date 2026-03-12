"""
Historical Option Chain Loader / Builder

Adds:
- cache loading
- auto synthetic build
- expiry-aware synthetic chain
- optional real IV surface integration
"""

from pathlib import Path
from math import sqrt, log
import math
import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_DIR,
    BACKTEST_STRIKE_STEP,
    BACKTEST_STRIKE_RANGE,
    BACKTEST_DEFAULT_IV
)
from data.historical_iv_surface import (
    load_historical_iv_surface,
    get_surface_iv
)


YF_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}


def _candidate_paths(symbol: str, years: int):
    symbol = symbol.upper().strip()
    base = Path(DATA_DIR)

    return [
        base / f"{symbol}_historical_option_chain.csv",
        base / symbol / "historical_option_chain.csv",
        base / symbol / f"{symbol}_{years}y_option_chain.csv",
    ]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "STRIKE_PR": "strikePrice",
        "OPEN_INT": "openInterest",
        "LAST_PRICE": "lastPrice",
        "VOLUME": "totalTradedVolume",
        "IV": "impliedVolatility",
        "TIMESTAMP": "timestamp",
        "OPTION_TYPE": "OPTION_TYP",
        "OPTIONTYPE": "OPTION_TYP",
        "type": "OPTION_TYP",
        "spotPrice": "spot",
        "Spot": "spot",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    return df


def _validate_dataframe(df: pd.DataFrame):
    required = ["timestamp", "strikePrice", "OPTION_TYP", "lastPrice", "openInterest"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Historical option chain missing required columns: {missing}")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes_price(spot, strike, t, sigma, option_type):
    spot = max(float(spot), 1e-6)
    strike = max(float(strike), 1e-6)
    sigma = max(float(sigma), 1e-6)
    t = max(float(t), 1e-6)

    d1 = (log(spot / strike) + 0.5 * sigma * sigma * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)

    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _symbol_to_yfinance(symbol: str) -> str:
    return YF_SYMBOL_MAP.get(symbol.upper().strip(), symbol)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) and len(col) > 0 else col
            for col in df.columns
        ]
    return df


def _download_spot_history(symbol: str, years: int) -> pd.DataFrame:
    yf_symbol = _symbol_to_yfinance(symbol)

    df = yf.download(
        yf_symbol,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"Could not download historical spot data for {symbol}")

    df = _flatten_yfinance_columns(df).reset_index()

    if "Date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Datetime"])
    else:
        raise ValueError("Downloaded spot history missing Date/Datetime column")

    df["spot"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["spot"])

    return df[["timestamp", "spot"]].copy()


def _days_to_next_expiry(timestamp) -> int:
    """
    Approx weekly expiry-aware proxy: next Thursday.
    """
    ts = pd.Timestamp(timestamp)
    weekday = ts.weekday()  # Monday=0
    target = 3  # Thursday
    days = (target - weekday) % 7
    return 7 if days == 0 else days


def _build_synthetic_snapshot(
    timestamp,
    spot,
    strike_step,
    strike_range,
    default_iv,
    iv_surface_df=None
):
    spot = float(spot)
    rows = []

    atm = round(spot / strike_step) * strike_step
    strikes = [atm + i * strike_step for i in range(-strike_range, strike_range + 1)]

    expiry_days = _days_to_next_expiry(timestamp)
    t = max(expiry_days / 365.0, 1e-6)

    for strike in strikes:
        distance = abs(strike - spot)
        oi_base = max(1000, int(250000 / (1 + distance / strike_step)))
        volume_base = max(100, int(50000 / (1 + distance / strike_step)))

        for opt_type in ["CE", "PE"]:
            iv = get_surface_iv(
                iv_surface_df=iv_surface_df,
                timestamp=timestamp,
                strike=strike,
                option_type=opt_type,
                default_iv=default_iv
            )
            sigma = iv / 100.0

            price = _black_scholes_price(
                spot=spot,
                strike=strike,
                t=t,
                sigma=sigma,
                option_type=opt_type
            )

            intrinsic = max(spot - strike, 0) if opt_type == "CE" else max(strike - spot, 0)
            last_price = max(price, intrinsic + 0.5)

            rows.append({
                "timestamp": pd.to_datetime(timestamp),
                "spot": round(spot, 2),
                "strikePrice": int(strike),
                "OPTION_TYP": opt_type,
                "lastPrice": round(float(last_price), 2),
                "openInterest": int(oi_base),
                "changeinOI": int(oi_base * 0.03),
                "impliedVolatility": float(iv),
                "totalTradedVolume": int(volume_base),
                "expiry_days": int(expiry_days),
                "STRIKE_PR": int(strike),
                "OPEN_INT": int(oi_base),
                "LAST_PRICE": round(float(last_price), 2),
                "VOLUME": int(volume_base),
                "IV": float(iv),
                "EXPIRY_DT": f"T+{expiry_days}"
            })

    return rows


def _build_historical_option_chain(symbol: str, years: int) -> pd.DataFrame:
    spot_df = _download_spot_history(symbol, years)
    iv_surface_df = load_historical_iv_surface(symbol, years)

    all_rows = []

    for _, row in spot_df.iterrows():
        snapshot_rows = _build_synthetic_snapshot(
            timestamp=row["timestamp"],
            spot=float(row["spot"]),
            strike_step=BACKTEST_STRIKE_STEP,
            strike_range=BACKTEST_STRIKE_RANGE,
            default_iv=BACKTEST_DEFAULT_IV,
            iv_surface_df=iv_surface_df
        )
        all_rows.extend(snapshot_rows)

    historical_df = pd.DataFrame(all_rows)
    historical_df = _normalize_columns(historical_df)
    _validate_dataframe(historical_df)

    historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"], errors="coerce")
    historical_df = historical_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return historical_df


def _cache_path(symbol: str, years: int) -> Path:
    symbol_dir = Path(DATA_DIR) / symbol.upper().strip()
    symbol_dir.mkdir(parents=True, exist_ok=True)
    return symbol_dir / f"{symbol.upper().strip()}_{years}y_option_chain.csv"


def load_option_chain(symbol: str, years: int = 1) -> pd.DataFrame:
    for path in _candidate_paths(symbol, years):
        if path.exists() and path.is_file():
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            _validate_dataframe(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
            return df

    print(f"No cached historical option chain found for {symbol}.")
    print("Building synthetic historical option chain automatically...")

    historical_df = _build_historical_option_chain(symbol, years)

    cache_file = _cache_path(symbol, years)
    historical_df.to_csv(cache_file, index=False)

    print(f"Historical option chain cached at: {cache_file}")
    return historical_df