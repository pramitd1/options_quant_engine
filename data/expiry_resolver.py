"""
Expiry resolver utilities.

Provides a single place to:
- normalize expiry labels coming from different providers
- sort expiries when possible
- pick the selected/front expiry for the engine
- optionally filter a mixed-expiry chain down to one expiry
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
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    return text


def _parse_expiry(value: str):
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
    expiries = ordered_expiries(option_chain)
    return expiries[0] if expiries else None


def filter_option_chain_by_expiry(option_chain, selected_expiry: str | None):
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
