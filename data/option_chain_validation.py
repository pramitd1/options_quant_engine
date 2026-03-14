"""
Reusable option-chain validation helpers for live engine entry points.
"""

from __future__ import annotations

import pandas as pd

from data.expiry_resolver import ordered_expiries


COLUMN_ALIASES = {
    "strikePrice": ["STRIKE_PR"],
    "OPTION_TYP": ["OPTION_TYPE", "optionType", "type"],
    "lastPrice": ["LAST_PRICE"],
    "openInterest": ["OPEN_INT"],
    "impliedVolatility": ["IV"],
}


def _resolve_column_name(option_chain, canonical_name: str) -> str | None:
    candidates = [canonical_name] + COLUMN_ALIASES.get(canonical_name, [])
    for column in candidates:
        if column in option_chain.columns:
            return column
    return None


def _series_or_empty(option_chain, canonical_name: str) -> pd.Series:
    column = _resolve_column_name(option_chain, canonical_name)
    if column is None:
        return pd.Series(dtype="object")
    return option_chain[column]


def validate_option_chain(option_chain):
    issues = []
    warnings = []

    if option_chain is None:
        issues.append("option_chain_none")
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "row_count": 0,
            "ce_rows": 0,
            "pe_rows": 0,
            "priced_rows": 0,
            "strike_count": 0,
            "paired_strike_count": 0,
            "paired_strike_ratio": 0.0,
        }

    if option_chain.empty:
        issues.append("option_chain_empty")
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "row_count": 0,
            "ce_rows": 0,
            "pe_rows": 0,
            "priced_rows": 0,
            "strike_count": 0,
            "paired_strike_count": 0,
            "paired_strike_ratio": 0.0,
        }

    required_cols = ["strikePrice", "OPTION_TYP", "lastPrice"]
    missing_cols = [
        canonical_name
        for canonical_name in required_cols
        if _resolve_column_name(option_chain, canonical_name) is None
    ]
    if missing_cols:
        issues.append(f"missing_columns:{','.join(missing_cols)}")

    row_count = len(option_chain)
    option_type_series = _series_or_empty(option_chain, "OPTION_TYP").astype(str).str.upper().str.strip()
    strike_series = pd.to_numeric(_series_or_empty(option_chain, "strikePrice"), errors="coerce")
    last_price_series = pd.to_numeric(_series_or_empty(option_chain, "lastPrice"), errors="coerce")
    iv_series = pd.to_numeric(_series_or_empty(option_chain, "impliedVolatility"), errors="coerce")

    ce_rows = int(option_type_series.eq("CE").sum()) if not option_type_series.empty else 0
    pe_rows = int(option_type_series.eq("PE").sum()) if not option_type_series.empty else 0
    priced_rows = int(last_price_series.gt(0).sum()) if not last_price_series.empty else 0
    iv_rows = int(iv_series.gt(0).sum()) if not iv_series.empty else 0
    strike_count = int(strike_series.dropna().nunique()) if not strike_series.empty else 0

    unknown_option_type_rows = int((~option_type_series.isin(["CE", "PE"])).sum()) if not option_type_series.empty else 0
    if unknown_option_type_rows > 0:
        warnings.append(f"unknown_option_type_rows:{unknown_option_type_rows}")

    duplicate_row_count = int(option_chain.duplicated().sum())
    duplicate_ratio = round(duplicate_row_count / max(row_count, 1), 4)
    if duplicate_row_count > 0:
        warnings.append(f"duplicate_rows:{duplicate_row_count}")

    paired_strike_count = 0
    paired_strike_ratio = 0.0
    if not strike_series.empty and not option_type_series.empty:
        strike_type_df = pd.DataFrame({
            "strike": strike_series,
            "option_type": option_type_series,
        }).dropna(subset=["strike"])

        if not strike_type_df.empty:
            type_counts = strike_type_df.groupby("strike")["option_type"].nunique()
            paired_strike_count = int(type_counts.ge(2).sum())
            paired_strike_ratio = round(paired_strike_count / max(strike_count, 1), 4)
            if strike_count >= 4 and paired_strike_ratio < 0.5:
                warnings.append(f"low_paired_strike_ratio:{paired_strike_count}/{strike_count}")

    selected_expiry = None
    expiry_count = 0
    expiry_missing_rows = 0
    try:
        expiries = ordered_expiries(option_chain)
        expiry_count = len(expiries)
        selected_expiry = expiries[0] if expiries else None
        if expiry_count > 1:
            warnings.append(f"multiple_expiries_detected:{expiry_count}")
    except Exception:
        warnings.append("expiry_summary_failed")

    try:
        expiry_series = option_chain.get("EXPIRY_DT")
        if expiry_series is not None:
            expiry_missing_rows = int(pd.Series(expiry_series).isna().sum())
            if expiry_missing_rows > 0:
                warnings.append(f"missing_expiry_rows:{expiry_missing_rows}")
    except Exception:
        warnings.append("expiry_missing_row_count_failed")

    if row_count < 20:
        issues.append(f"too_few_rows:{row_count}")
    if ce_rows == 0:
        issues.append("no_ce_rows")
    if pe_rows == 0:
        issues.append("no_pe_rows")
    if priced_rows == 0:
        issues.append("no_priced_rows")
    if strike_count < 6:
        warnings.append(f"low_strike_count:{strike_count}")
    if priced_rows > 0 and priced_rows < max(10, int(0.2 * row_count)):
        warnings.append(f"low_priced_row_ratio:{priced_rows}/{row_count}")
    if row_count > 0 and iv_rows == 0:
        warnings.append("no_positive_iv_rows")

    source = None
    if "source" in option_chain.columns:
        try:
            non_null_sources = option_chain["source"].dropna().astype(str).str.upper().str.strip().unique().tolist()
            source = non_null_sources[0] if non_null_sources else None
        except Exception:
            source = None

    priced_ratio = round(priced_rows / max(row_count, 1), 4)
    iv_ratio = round(iv_rows / max(row_count, 1), 4)
    provider_health = {
        "source": source,
        "row_health": "GOOD" if row_count >= 120 else "THIN" if row_count >= 60 else "WEAK",
        "pricing_health": "GOOD" if priced_ratio >= 0.55 else "CAUTION" if priced_ratio >= 0.35 else "WEAK",
        "pairing_health": "GOOD" if paired_strike_ratio >= 0.75 else "CAUTION" if paired_strike_ratio >= 0.5 else "WEAK",
        "iv_health": "GOOD" if iv_ratio >= 0.4 else "CAUTION" if iv_ratio >= 0.15 else "WEAK",
        "duplicate_health": "GOOD" if duplicate_ratio == 0 else "CAUTION" if duplicate_ratio <= 0.05 else "WEAK",
        "summary_status": "GOOD",
    }
    if "WEAK" in provider_health.values():
        provider_health["summary_status"] = "WEAK"
    elif "CAUTION" in provider_health.values():
        provider_health["summary_status"] = "CAUTION"

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "row_count": row_count,
        "ce_rows": ce_rows,
        "pe_rows": pe_rows,
        "priced_rows": priced_rows,
        "priced_ratio": priced_ratio,
        "iv_rows": iv_rows,
        "iv_ratio": iv_ratio,
        "strike_count": strike_count,
        "paired_strike_count": paired_strike_count,
        "paired_strike_ratio": paired_strike_ratio,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_ratio": duplicate_ratio,
        "selected_expiry": selected_expiry,
        "expiry_count": expiry_count,
        "expiry_missing_rows": expiry_missing_rows,
        "provider_health": provider_health,
    }
