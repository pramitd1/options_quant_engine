"""
Module: provider_normalization.py

Purpose:
    Implement provider normalization data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

import pandas as pd


CANONICAL_NUMERIC_COLUMNS = [
    "strikePrice",
    "lastPrice",
    "openInterest",
    "changeinOI",
    "impliedVolatility",
    "totalTradedVolume",
    "STRIKE_PR",
    "LAST_PRICE",
    "OPEN_INT",
    "IV",
    "VOLUME",
]


def _copy_alias(df: pd.DataFrame, canonical_name: str, alias_name: str) -> None:
    """
    Purpose:
        Process copy alias for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        df (pd.DataFrame): Input associated with df.
        canonical_name (str): Human-readable name for canonical.
        alias_name (str): Human-readable name for alias.
    
    Returns:
        None: The function operates through side effects.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if canonical_name in df.columns and alias_name not in df.columns:
        df[alias_name] = df[canonical_name]
    elif alias_name in df.columns and canonical_name not in df.columns:
        df[canonical_name] = df[alias_name]
    elif canonical_name in df.columns and alias_name in df.columns:
        df[canonical_name] = df[canonical_name].fillna(df[alias_name])
        df[alias_name] = df[alias_name].fillna(df[canonical_name])


def normalize_live_option_chain(option_chain, *, source: str, symbol: str):
    """
    Purpose:
        Normalize live option chain into the repository-standard representation.
    
    Context:
        Public function in the `provider normalization` module. It forms part of the data workflow exposed by this module.
    
    Inputs:
        option_chain (Any): Option-chain snapshot used for scoring or signal generation.
        source (str): Market-data source label.
        symbol (str): Underlying symbol or index identifier.
    
    Returns:
        Any: Value returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    if option_chain is None or getattr(option_chain, "empty", True):
        return pd.DataFrame()

    df = option_chain.copy()

    alias_pairs = [
        ("strikePrice", "STRIKE_PR"),
        ("lastPrice", "LAST_PRICE"),
        ("openInterest", "OPEN_INT"),
        ("impliedVolatility", "IV"),
        ("totalTradedVolume", "VOLUME"),
    ]
    for canonical_name, alias_name in alias_pairs:
        _copy_alias(df, canonical_name, alias_name)

    if "OPTION_TYP" in df.columns:
        df["OPTION_TYP"] = df["OPTION_TYP"].astype(str).str.upper().str.strip()

    if "EXPIRY_DT" in df.columns:
        df["EXPIRY_DT"] = df["EXPIRY_DT"].astype(str).str.strip()
        df.loc[df["EXPIRY_DT"].isin(["", "nan", "None"]), "EXPIRY_DT"] = pd.NA

    for column in CANONICAL_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "strikePrice" in df.columns:
        df = df.dropna(subset=["strikePrice"])

    dedupe_subset = [col for col in ["strikePrice", "OPTION_TYP", "EXPIRY_DT"] if col in df.columns]
    if dedupe_subset:
        df = df.drop_duplicates(subset=dedupe_subset, keep="last")

    sort_columns = [col for col in ["EXPIRY_DT", "strikePrice", "OPTION_TYP"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["source"] = str(source or "").upper().strip()
    df["underlying_symbol"] = str(symbol or "").upper().strip()

    return df
