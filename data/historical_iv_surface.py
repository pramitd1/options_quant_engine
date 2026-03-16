"""
Module: historical_iv_surface.py

Purpose:
    Implement historical IV surface data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from pathlib import Path
import pandas as pd

from config.settings import IV_SURFACE_DIR


def _candidate_paths(symbol: str, years: int):
    """
    Purpose:
        Process candidate paths for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        symbol (str): Underlying symbol or index identifier.
        years (int): Input associated with years.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    symbol = symbol.upper().strip()
    base = Path(IV_SURFACE_DIR)
    return [
        base / f"{symbol}_iv_surface.csv",
        base / symbol / "iv_surface.csv",
        base / symbol / f"{symbol}_{years}y_iv_surface.csv",
    ]


def load_historical_iv_surface(symbol: str, years: int = 1):
    """
    Expected columns at minimum:
    - timestamp
    - strike
    - option_type
    - implied_volatility
    - expiry_days
    """
    for path in _candidate_paths(symbol, years):
        if path.exists() and path.is_file():
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"])
            return df.reset_index(drop=True)

    return None


def get_surface_iv(iv_surface_df, timestamp, strike, option_type, default_iv):
    """
    Returns real IV if available for that timestamp/strike/type, else default.
    """
    if iv_surface_df is None or iv_surface_df.empty:
        return float(default_iv)

    rows = iv_surface_df[
        (iv_surface_df["timestamp"] == pd.to_datetime(timestamp)) &
        (iv_surface_df["strike"] == strike) &
        (iv_surface_df["option_type"] == option_type)
    ]

    if rows.empty:
        return float(default_iv)

    return float(rows.iloc[0]["implied_volatility"])
