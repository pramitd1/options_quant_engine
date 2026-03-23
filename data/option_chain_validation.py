"""
Module: option_chain_validation.py

Purpose:
    Implement option chain validation data-ingestion utilities for the repository.

Role in the System:
    Part of the data layer that downloads, normalizes, validates, and stores market snapshots.

Key Outputs:
    Normalized dataframes, validation payloads, and persisted market snapshots.

Downstream Usage:
    Consumed by analytics, the signal engine, replay tooling, and research datasets.
"""

from __future__ import annotations

import pandas as pd

from config.policy_resolver import get_parameter_value
from data.expiry_resolver import ordered_expiries


COLUMN_ALIASES = {
    "strikePrice": ["STRIKE_PR"],
    "OPTION_TYP": ["OPTION_TYPE", "optionType", "type"],
    "lastPrice": ["LAST_PRICE"],
    "bidPrice": ["BID_PRICE", "best_bid_price", "bestBidPrice", "bid_price", "bid"],
    "askPrice": ["ASK_PRICE", "best_ask_price", "bestAskPrice", "ask_price", "ask"],
    "openInterest": ["OPEN_INT"],
    "impliedVolatility": ["IV"],
}


def _resolve_column_name(option_chain, canonical_name: str) -> str | None:
    """
    Purpose:
        Resolve column name needed by downstream logic.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        canonical_name (str): Human-readable name for canonical.
    
    Returns:
        str | None: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    candidates = [canonical_name] + COLUMN_ALIASES.get(canonical_name, [])
    for column in candidates:
        if column in option_chain.columns:
            return column
    return None


def _series_or_empty(option_chain, canonical_name: str) -> pd.Series:
    """
    Purpose:
        Process series or empty for downstream use.
    
    Context:
        Internal helper within the data layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
        canonical_name (str): Human-readable name for canonical.
    
    Returns:
        pd.Series: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    column = _resolve_column_name(option_chain, canonical_name)
    if column is None:
        return pd.Series(dtype="object")
    return option_chain[column]


def _detect_iv_anomalies(iv_series: pd.Series, thresholds: dict | None = None) -> list[str]:
    """
    Detect anomalous IV values that might indicate data corruption or feed issues.
    
    Returns list of warnings if anomalies detected.
    """
    warnings = []
    if iv_series is None or iv_series.empty:
        return warnings
    
    thresholds = thresholds or {}
    max_iv_pct = float(thresholds.get("max_iv_percent", 500.0))  # 5.0x is normal, >500% is extreme
    min_valid_iv = float(thresholds.get("min_valid_iv", 0.01))
    
    # Filter positive IVs
    valid_iv = iv_series[(iv_series > min_valid_iv) & (iv_series < 10.0)]
    if valid_iv.empty:
        return warnings
    
    median_iv = valid_iv.median()
    mean_iv = valid_iv.mean()
    
    if median_iv < min_valid_iv:
        return warnings
    
    # Check for extreme spikes
    spike_threshold = median_iv * (max_iv_pct / 100.0)
    spike_count = int((iv_series > spike_threshold).sum())
    if spike_count > 0:
        max_iv = iv_series.max()
        spike_ratio = max_iv / median_iv if median_iv > 0 else 0
        warnings.append(f"iv_extreme_spike_detected:{spike_count}rows_max_ratio_{spike_ratio:.1f}x")
    
    # Check for abnormal variance
    std_iv = valid_iv.std()
    if std_iv > median_iv * 2.0:  # Std dev more than 2x median
        warnings.append(f"iv_high_variance:{std_iv:.4f}_vs_median_{median_iv:.4f}")
    
    return warnings


def _detect_spot_jump(current_spot: float, day_high: float, day_low: float, prev_close: float, thresholds: dict | None = None) -> list[str]:
    """
    Detect if current spot price jumped abnormally from previous close or day range.
    
    Returns list of warnings if anomalies detected.
    """
    warnings = []
    if current_spot is None or prev_close is None or prev_close == 0:
        return warnings
    
    thresholds = thresholds or {}
    max_gap_pct = float(thresholds.get("max_normal_gap_percent", 3.0))  # 3% is normal for options expiry
    max_intraday_pct = float(thresholds.get("max_intraday_move_percent", 8.0))  # 8% is normal
    
    # Check gap from previous close
    gap_pct = abs((current_spot - prev_close) / prev_close) * 100.0
    if gap_pct > max_gap_pct:
        warnings.append(f"spot_gap_from_prev_close:{gap_pct:.2f}pct")
    
    # Check intraday range consistency
    if day_high is not None and day_low is not None and day_high > day_low:
        intraday_range_pct = ((day_high - day_low) / current_spot) * 100.0 if current_spot > 0 else 0
        if intraday_range_pct > max_intraday_pct:
            warnings.append(f"abnormal_intraday_range:{intraday_range_pct:.2f}pct")
        
        # Check if spot is wildly outside day range
        if current_spot > day_high * 1.02:  # More than 2% above high
            warnings.append(f"spot_above_day_high_by_{((current_spot/day_high - 1)*100):.2f}pct")
        elif current_spot < day_low * 0.98:  # More than 2% below low
            warnings.append(f"spot_below_day_low_by_{((1 - current_spot/day_low)*100):.2f}pct")
    
    return warnings


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        text = str(value).strip().lower()
    except Exception:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def validate_option_chain(option_chain, spot=None):
    """
    Purpose:
        Process validate option chain for downstream use.
    
    Context:
        Public function within the data layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        option_chain (Any): Input associated with option chain.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
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
    bid_price_series = pd.to_numeric(_series_or_empty(option_chain, "bidPrice"), errors="coerce")
    ask_price_series = pd.to_numeric(_series_or_empty(option_chain, "askPrice"), errors="coerce")
    iv_series = pd.to_numeric(_series_or_empty(option_chain, "impliedVolatility"), errors="coerce")

    ce_rows = int(option_type_series.eq("CE").sum()) if not option_type_series.empty else 0
    pe_rows = int(option_type_series.eq("PE").sum()) if not option_type_series.empty else 0
    priced_rows = int(last_price_series.gt(0).sum()) if not last_price_series.empty else 0
    has_bid_column = not bid_price_series.empty
    has_ask_column = not ask_price_series.empty
    has_quote_columns = has_bid_column or has_ask_column
    has_two_sided_quote_columns = has_bid_column and has_ask_column

    bid_present_mask = bid_price_series.gt(0) if has_bid_column else pd.Series(False, index=option_chain.index)
    ask_present_mask = ask_price_series.gt(0) if has_ask_column else pd.Series(False, index=option_chain.index)

    # Count "quoted" rows only when both sides of book are populated.
    if has_two_sided_quote_columns:
        quoted_mask = bid_present_mask & ask_present_mask
    else:
        quoted_mask = bid_present_mask | ask_present_mask

    quoted_rows = int(quoted_mask.sum()) if has_quote_columns else 0
    bid_present_rows = int(bid_present_mask.sum()) if has_bid_column else 0
    ask_present_rows = int(ask_present_mask.sum()) if has_ask_column else 0
    one_sided_quote_rows = int((bid_present_mask ^ ask_present_mask).sum()) if has_two_sided_quote_columns else 0
    iv_rows = int(iv_series.gt(0).sum()) if not iv_series.empty else 0
    strike_count = int(strike_series.dropna().nunique()) if not strike_series.empty else 0

    # Check for IV anomalies
    iv_anomalies = _detect_iv_anomalies(iv_series)
    warnings.extend(iv_anomalies)

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
    if has_two_sided_quote_columns and one_sided_quote_rows > 0:
        warnings.append(f"one_sided_quote_rows:{one_sided_quote_rows}")

    source = None
    if "source" in option_chain.columns:
        try:
            non_null_sources = option_chain["source"].dropna().astype(str).str.upper().str.strip().unique().tolist()
            source = non_null_sources[0] if non_null_sources else None
        except Exception:
            source = None

    priced_ratio = round(priced_rows / max(row_count, 1), 4)
    quoted_ratio = round(quoted_rows / max(row_count, 1), 4) if has_quote_columns else 0.0
    effective_priced_rows = max(priced_rows, quoted_rows)
    effective_priced_ratio = round(effective_priced_rows / max(row_count, 1), 4)
    iv_ratio = round(iv_rows / max(row_count, 1), 4)

    trade_price_health = "GOOD" if priced_ratio >= 0.55 else "CAUTION" if priced_ratio >= 0.35 else "WEAK"
    quote_health = None
    if has_quote_columns:
        quote_health = "GOOD" if quoted_ratio >= 0.55 else "CAUTION" if quoted_ratio >= 0.35 else "WEAK"

    def _health_from_ratio(value, good_floor, caution_floor):
        if value >= good_floor:
            return "GOOD"
        if value >= caution_floor:
            return "CAUTION"
        return "WEAK"

    provider_health = {
        "source": source,
        "row_health": "GOOD" if row_count >= 120 else "THIN" if row_count >= 60 else "WEAK",
        # pricing_health is based on effective coverage (trade print OR quote),
        # so low trade-print density alone does not over-penalize sessions with
        # valid quoted liquidity.
        "pricing_health": "GOOD" if effective_priced_ratio >= 0.55 else "CAUTION" if effective_priced_ratio >= 0.35 else "WEAK",
        "trade_price_health": trade_price_health,
        "quote_health": quote_health,
        "pricing_basis": "TRADE_OR_QUOTE" if has_quote_columns else "TRADE_ONLY",
        "quote_coverage_mode": (
            "TWO_SIDED"
            if has_two_sided_quote_columns and bid_present_rows > 0 and ask_present_rows > 0
            else "STRUCT_ONLY"
            if has_two_sided_quote_columns
            else "ONE_SIDED"
        ),
        "pairing_health": "GOOD" if paired_strike_ratio >= 0.75 else "CAUTION" if paired_strike_ratio >= 0.5 else "WEAK",
        "iv_health": "GOOD" if iv_ratio >= 0.4 else "CAUTION" if iv_ratio >= 0.15 else "WEAK",
        "duplicate_health": "GOOD" if duplicate_ratio == 0 else "CAUTION" if duplicate_ratio <= 0.05 else "WEAK",
        "summary_status": "GOOD",
        "row_health_escalation_applied": False,
    }

    # Core marketability checks approximate execution-critical quality by focusing
    # on strikes nearest to spot (or strike median when spot is unavailable).
    core_valid = strike_series.notna() & option_type_series.isin(["CE", "PE"])
    if core_valid.any():
        if spot is None:
            try:
                spot = float(strike_series.dropna().median())
            except Exception:
                spot = None

        if spot is None:
            core_distance = (strike_series - float(strike_series.dropna().median())).abs()
        else:
            core_distance = (strike_series - float(spot)).abs()

        core_window_points = float(
            get_parameter_value(
                "option_chain_validation.provider_health.core_window_points",
                400.0,
            )
        )
        core_min_rows = int(
            float(
                get_parameter_value(
                    "option_chain_validation.provider_health.core_min_rows",
                    40.0,
                )
            )
        )

        core_mask = core_valid & core_distance.le(core_window_points)
        if int(core_mask.sum()) < max(core_min_rows, 10):
            nearest_rank = core_distance.where(core_valid).rank(method="first", ascending=True)
            core_mask = core_valid & nearest_rank.le(max(core_min_rows, 10))
    else:
        core_mask = pd.Series(False, index=option_chain.index)

    core_rows = int(core_mask.sum())
    core_priced_rows = int((last_price_series.gt(0) & core_mask).sum()) if core_rows > 0 else 0
    core_quoted_rows = int((quoted_mask & core_mask).sum()) if (core_rows > 0 and has_quote_columns) else 0
    core_effective_priced_rows = max(core_priced_rows, core_quoted_rows)
    core_effective_priced_ratio = round(core_effective_priced_rows / max(core_rows, 1), 4) if core_rows > 0 else 0.0
    core_iv_rows = int((iv_series.gt(0) & core_mask).sum()) if core_rows > 0 else 0
    core_iv_ratio = round(core_iv_rows / max(core_rows, 1), 4) if core_rows > 0 else 0.0

    core_paired_ratio = 0.0
    core_one_sided_ratio = 0.0
    if core_rows > 0:
        core_strike_df = pd.DataFrame(
            {
                "strike": strike_series.where(core_mask),
                "option_type": option_type_series.where(core_mask),
            }
        ).dropna(subset=["strike"])
        if not core_strike_df.empty:
            core_strike_count = int(core_strike_df["strike"].nunique())
            core_type_counts = core_strike_df.groupby("strike")["option_type"].nunique()
            core_paired_count = int(core_type_counts.ge(2).sum())
            core_paired_ratio = round(core_paired_count / max(core_strike_count, 1), 4)

        if has_two_sided_quote_columns:
            core_one_sided = int(((bid_present_mask ^ ask_present_mask) & core_mask).sum())
            core_one_sided_ratio = round(core_one_sided / max(core_rows, 1), 4)

    core_marketability_health = _health_from_ratio(core_effective_priced_ratio, 0.65, 0.45)
    core_pairing_health = _health_from_ratio(core_paired_ratio, 0.85, 0.65)
    core_iv_health = _health_from_ratio(core_iv_ratio, 0.50, 0.25)
    if has_two_sided_quote_columns:
        core_quote_integrity_health = (
            "GOOD"
            if core_one_sided_ratio <= 0.20
            else "CAUTION"
            if core_one_sided_ratio <= 0.45
            else "WEAK"
        )
    else:
        core_quote_integrity_health = "N/A"

    provider_health.update(
        {
            "core_rows": core_rows,
            "core_effective_priced_ratio": core_effective_priced_ratio,
            "core_paired_ratio": core_paired_ratio,
            "core_iv_ratio": core_iv_ratio,
            "core_one_sided_quote_ratio": core_one_sided_ratio if has_two_sided_quote_columns else None,
            "core_marketability_health": core_marketability_health,
            "core_pairing_health": core_pairing_health,
            "core_iv_health": core_iv_health,
            "core_quote_integrity_health": core_quote_integrity_health,
        }
    )

    critical_failures = []
    if core_marketability_health == "WEAK":
        critical_failures.append("core_marketability_weak")
    if core_pairing_health == "WEAK":
        critical_failures.append("core_pairing_weak")
    if core_iv_health == "WEAK":
        critical_failures.append("core_iv_weak")
    quote_integrity_standalone_block = _as_bool(
        get_parameter_value(
            "option_chain_validation.provider_health.core_quote_integrity_standalone_block",
            False,
        ),
        default=False,
    )
    if has_two_sided_quote_columns and core_quote_integrity_health == "WEAK":
        # One-sided quote books in fast markets can coexist with actionable trade prints.
        # Treat quote-integrity weakness as blocking only when marketability is already weak,
        # unless strict standalone blocking is explicitly enabled.
        if quote_integrity_standalone_block or core_marketability_health == "WEAK":
            critical_failures.append("core_quote_integrity_weak")

    provider_health["trade_blocking_status"] = "BLOCK" if critical_failures else "PASS"
    provider_health["trade_blocking_reasons"] = critical_failures

    # Summary status should be driven by core execution-critical checks.
    # Non-critical subchecks (e.g. trade-only/quote-only weakness in tail strikes)
    # are still emitted for diagnostics but do not force a WEAK summary.
    _summary_core_keys = [
        "row_health",
        "pricing_health",
        "pairing_health",
        "iv_health",
        "duplicate_health",
        "core_marketability_health",
        "core_pairing_health",
        "core_iv_health",
    ]
    if has_two_sided_quote_columns:
        _summary_core_keys.append("core_quote_integrity_health")
    _summary_core_values = [provider_health.get(k) for k in _summary_core_keys]

    _non_critical_weak_keys = []
    for _k in ("trade_price_health", "quote_health"):
        if provider_health.get(_k) == "WEAK":
            _non_critical_weak_keys.append(_k)
    provider_health["non_critical_weak_components"] = _non_critical_weak_keys

    if "WEAK" in _summary_core_values:
        provider_health["summary_status"] = "WEAK"
    elif "CAUTION" in _summary_core_values:
        provider_health["summary_status"] = "CAUTION"

    # Optional strictness toggle for thin chains: when enabled, a THIN row
    # profile is surfaced as summary CAUTION even if all other checks are GOOD.
    thin_escalation = _as_bool(
        get_parameter_value(
            "option_chain_validation.provider_health.thin_row_escalates_to_caution",
            False,
        ),
        default=False,
    )
    if (
        thin_escalation
        and provider_health.get("summary_status") == "GOOD"
        and provider_health.get("row_health") == "THIN"
    ):
        provider_health["summary_status"] = "CAUTION"
        provider_health["row_health_escalation_applied"] = True

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "row_count": row_count,
        "ce_rows": ce_rows,
        "pe_rows": pe_rows,
        "priced_rows": priced_rows,
        "priced_ratio": priced_ratio,
        "quoted_rows": quoted_rows,
        "quoted_ratio": quoted_ratio if has_quote_columns else None,
        "bid_present_rows": bid_present_rows if has_quote_columns else 0,
        "ask_present_rows": ask_present_rows if has_quote_columns else 0,
        "one_sided_quote_rows": one_sided_quote_rows if has_two_sided_quote_columns else 0,
        "effective_priced_rows": effective_priced_rows,
        "effective_priced_ratio": effective_priced_ratio,
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
