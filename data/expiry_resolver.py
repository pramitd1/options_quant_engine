"""
Module: expiry_resolver.py

Purpose:
    Implement expiry resolver data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


KNOWN_EXPIRY_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.000Z",
    "%d-%b-%Y",
    "%Y-%m-%d",
)


def normalize_expiry_value(value) -> str | None:
    """
    Purpose:
        Normalize expiry value into the repository-standard representation.
    
    Context:
        Public function in the `expiry resolver` module. It forms part of the data workflow exposed by this module.
    
    Inputs:
        value (Any): Raw value supplied by the caller.
    
    Returns:
        str | None: Value returned by the current workflow step.
    
    Notes:
        Outputs are designed to remain serializable and reusable across live, replay, research, and tuning workflows.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    return text


def _parse_expiry(value: str):
    """
    Purpose:
        Process parse expiry for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (str): Input associated with value.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    for fmt in KNOWN_EXPIRY_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None

    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def ordered_expiries(option_chain) -> list[str]:
    """
    Purpose:
        Process ordered expiries for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
    
    Returns:
        list[str]: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if option_chain is None or len(option_chain) == 0 or "EXPIRY_DT" not in option_chain.columns:
        return []

    raw_values = [
        normalize_expiry_value(value)
        for value in option_chain["EXPIRY_DT"].dropna().astype(str).tolist()
    ]

    expiries = []
    for value in raw_values:
        if value and value not in expiries:
            expiries.append(value)

    sortable = []
    unsortable = []

    for expiry in expiries:
        parsed = _parse_expiry(expiry)
        if parsed is None:
            unsortable.append(expiry)
        else:
            sortable.append((parsed, expiry))

    sortable.sort(key=lambda item: item[0])
    return [expiry for _, expiry in sortable] + unsortable


def resolve_selected_expiry(option_chain) -> str | None:
    """
    Purpose:
        Resolve selected expiry needed by downstream logic.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
    
    Returns:
        str | None: Computed value returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    expiries = ordered_expiries(option_chain)
    return expiries[0] if expiries else None


def filter_option_chain_by_expiry(option_chain, selected_expiry: str | None):
    """
    Purpose:
        Process filter option chain by expiry for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        selected_expiry (str | None): Expiry associated with the contract referenced by the signal.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    if (
        option_chain is None
        or len(option_chain) == 0
        or not selected_expiry
        or "EXPIRY_DT" not in option_chain.columns
    ):
        return option_chain

    expiry_values = option_chain["EXPIRY_DT"].astype(str).str.strip()
    filtered = option_chain.loc[expiry_values.eq(str(selected_expiry).strip())].copy()
    return filtered if not filtered.empty else option_chain
