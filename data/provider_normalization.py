"""
Normalize live provider option-chain outputs into a consistent engine contract.
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
    if canonical_name in df.columns and alias_name not in df.columns:
        df[alias_name] = df[canonical_name]
    elif alias_name in df.columns and canonical_name not in df.columns:
        df[canonical_name] = df[alias_name]
    elif canonical_name in df.columns and alias_name in df.columns:
        df[canonical_name] = df[canonical_name].fillna(df[alias_name])
        df[alias_name] = df[alias_name].fillna(df[canonical_name])


def normalize_live_option_chain(option_chain, *, source: str, symbol: str):
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
