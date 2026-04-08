"""
Module: gamma_walls.py

Purpose:
    Compute gamma walls analytics used by downstream signal and risk layers.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""
from __future__ import annotations

import pandas as pd


def _resolve_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    strike_col = "STRIKE_PR" if "STRIKE_PR" in df.columns else ("strikePrice" if "strikePrice" in df.columns else None)
    oi_col = "OPEN_INT" if "OPEN_INT" in df.columns else ("openInterest" if "openInterest" in df.columns else None)
    type_col = "OPTION_TYP" if "OPTION_TYP" in df.columns else None
    return strike_col, oi_col, type_col


def _signed_option_type(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.upper().str.strip()
    signed = normalized.map({"CE": 1.0, "PE": -1.0})
    if signed.isna().any():
        unknown = sorted(set(normalized[signed.isna()].tolist()))
        raise ValueError(f"Unknown OPTION_TYP values in gamma_walls: {unknown}")
    return signed


def _gamma_exposure_by_strike(df: pd.DataFrame) -> pd.Series:
    strike_col, oi_col, type_col = _resolve_columns(df)
    if strike_col is None or oi_col is None or type_col is None:
        return pd.Series(dtype=float)

    work = df.copy()
    work[strike_col] = pd.to_numeric(work[strike_col], errors="coerce")
    work[oi_col] = pd.to_numeric(work[oi_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[strike_col])
    if work.empty:
        return pd.Series(dtype=float)

    if "GAMMA" in work.columns:
        gamma = pd.to_numeric(work["GAMMA"], errors="coerce").fillna(0.0)
    else:
        # Fallback proxy when explicit gamma is unavailable.
        spot_proxy = float(work[strike_col].median()) if work[strike_col].notna().any() else 0.0
        if spot_proxy <= 0:
            return pd.Series(dtype=float)
        distance = (work[strike_col] - spot_proxy).abs() / max(spot_proxy, 1e-6)
        gamma = 1.0 / (1.0 + distance)

    signed = _signed_option_type(work[type_col])
    work["_signed_gamma_exposure"] = gamma * work[oi_col] * signed
    return work.groupby(strike_col)["_signed_gamma_exposure"].sum()


def _oi_by_strike(df: pd.DataFrame) -> pd.Series:
    strike_col, oi_col, _ = _resolve_columns(df)
    if strike_col is None or oi_col is None:
        return pd.Series(dtype=float)

    work = df.copy()
    work[strike_col] = pd.to_numeric(work[strike_col], errors="coerce")
    work[oi_col] = pd.to_numeric(work[oi_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[strike_col])
    if work.empty:
        return pd.Series(dtype=float)
    return work.groupby(strike_col)[oi_col].sum()


def detect_gamma_walls(option_chain, top_n=3):
    """
    Detect strikes with highest open interest.

    These act as gamma walls (strong support/resistance).
    """

    if option_chain is None or option_chain.empty:
        return []

    gamma_exposure = _gamma_exposure_by_strike(option_chain)
    if not gamma_exposure.empty and gamma_exposure.abs().sum() > 0:
        walls = gamma_exposure.abs().sort_values(ascending=False).head(top_n)
        return [float(x) for x in walls.index.tolist()]

    oi_by_strike = _oi_by_strike(option_chain)
    if oi_by_strike.empty:
        return []
    walls = oi_by_strike.sort_values(ascending=False).head(top_n)

    return [float(x) for x in walls.index.tolist()]


def classify_walls(option_chain):
    """
    Classify support and resistance walls.
    """

    if option_chain is None or option_chain.empty:
        return {}

    gamma_exposure = _gamma_exposure_by_strike(option_chain)
    if not gamma_exposure.empty and gamma_exposure.abs().sum() > 0:
        positive = gamma_exposure[gamma_exposure > 0]
        negative = gamma_exposure[gamma_exposure < 0]

        support = float(negative.idxmin()) if not negative.empty else None
        resistance = float(positive.idxmax()) if not positive.empty else None

        # Graceful fallback if one side is structurally absent.
        if support is None or resistance is None:
            oi_result = _classify_walls_from_oi(option_chain)
            if support is None:
                support = oi_result.get("support_wall")
            if resistance is None:
                resistance = oi_result.get("resistance_wall")

        return {
            "support_wall": support,
            "resistance_wall": resistance,
        }

    return _classify_walls_from_oi(option_chain)


def _classify_walls_from_oi(option_chain):
    """Fallback classification using OI concentration when gamma is unavailable."""
    strike_col, oi_col, type_col = _resolve_columns(option_chain)
    if strike_col is None or oi_col is None or type_col is None:
        return {}

    df = option_chain.copy()
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=[strike_col])
    if df.empty:
        return {}

    call_oi = df[df[type_col].astype(str).str.upper() == "CE"].groupby(strike_col)[oi_col].sum()
    put_oi = df[df[type_col].astype(str).str.upper() == "PE"].groupby(strike_col)[oi_col].sum()

    if call_oi.empty or put_oi.empty:
        return {}

    resistance = float(call_oi.idxmax())
    support = float(put_oi.idxmax())

    return {
        "support_wall": support,
        "resistance_wall": resistance,
    }