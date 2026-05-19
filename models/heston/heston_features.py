"""Feature extraction for the research-only Heston diagnostics layer."""

from __future__ import annotations

import json
import math
from datetime import time as dt_time
from typing import Any

import pandas as pd

from config.settings import (
    HESTON_PRICE_GAP_REJECT_PCT,
    HESTON_PRICE_GAP_WEAK_PCT,
    HESTON_SELECTED_IV_HIGH_PCT,
    HESTON_SELECTED_IV_LOW_PCT,
    HESTON_SHORT_TTE_REJECT_DAYS,
    HESTON_SHORT_TTE_WEAK_DAYS,
)

from .heston_calibration import HestonCalibrationResult, calibrate_heston_to_chain
from .heston_pricer import (
    HestonParams,
    black_scholes_price,
    heston_forward_variance,
    heston_greek_snapshot,
    heston_implied_vol_proxy,
    normalize_option_type,
)


HESTON_FEATURE_COLUMNS = [
    "heston_research_enabled",
    "heston_calibration_status",
    "heston_calibration_reason",
    "heston_calibration_sample_size",
    "heston_kappa",
    "heston_theta",
    "heston_vol_of_vol",
    "heston_rho",
    "heston_v0",
    "heston_calibration_error",
    "heston_surface_quality",
    "heston_quality_flags",
    "heston_bound_hit_count",
    "heston_tte_days",
    "heston_tte_bucket",
    "heston_expiry_context",
    "heston_short_tte_guard",
    "heston_selected_iv_quality",
    "heston_skew_state",
    "heston_forward_variance_proxy",
    "heston_model_price",
    "heston_model_delta",
    "heston_model_gamma",
    "heston_model_iv_proxy",
    "bs_model_price_for_heston",
    "bs_vs_heston_price_gap",
    "heston_price_gap_rel_pct",
    "bs_vs_heston_greek_gap",
    "greek_model_divergence_score",
    "heston_diagnostics_json",
]

NSE_EXPIRY_CLOSE_TIME = dt_time(hour=15, minute=30)


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _round(value, digits=6):
    parsed = _safe_float(value, None)
    if parsed is None:
        return None
    return round(parsed, digits)


def _coerce_ts(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def _time_to_expiry_years(expiry_value, valuation_time=None) -> float | None:
    expiry_ts = _coerce_ts(expiry_value)
    if expiry_ts is None:
        return None
    if (
        expiry_ts.hour == 0
        and expiry_ts.minute == 0
        and expiry_ts.second == 0
        and expiry_ts.microsecond == 0
    ):
        expiry_ts = expiry_ts.replace(
            hour=NSE_EXPIRY_CLOSE_TIME.hour,
            minute=NSE_EXPIRY_CLOSE_TIME.minute,
        )
    valuation_ts = _coerce_ts(valuation_time) or pd.Timestamp.now(tz="Asia/Kolkata")
    seconds = (expiry_ts - valuation_ts).total_seconds()
    if seconds <= 60:
        return None
    return max(seconds / (365.0 * 24.0 * 3600.0), 1e-6)


def _time_to_expiry_days(expiry_value, valuation_time=None) -> float | None:
    years = _time_to_expiry_years(expiry_value, valuation_time=valuation_time)
    return years * 365.0 if years is not None else None


def _tte_bucket(tte_days: float | None) -> str:
    if tte_days is None:
        return "UNKNOWN_TTE"
    if tte_days <= float(HESTON_SHORT_TTE_REJECT_DAYS):
        return "ULTRA_SHORT_TTE"
    if tte_days <= float(HESTON_SHORT_TTE_WEAK_DAYS):
        return "EXPIRY_DAY"
    if tte_days <= 7.0:
        return "FRONT_WEEK"
    if tte_days <= 35.0:
        return "NEXT_MONTH"
    return "LONGER_TTE"


def _expiry_context(tte_days: float | None) -> str:
    if tte_days is None:
        return "UNKNOWN"
    return "EXPIRY_DAY" if tte_days <= float(HESTON_SHORT_TTE_WEAK_DAYS) else "NON_EXPIRY"


def _empty_features(*, enabled: bool, status: str, reason: str) -> dict[str, Any]:
    out = {column: None for column in HESTON_FEATURE_COLUMNS}
    out.update(
        {
            "heston_research_enabled": bool(enabled),
            "heston_calibration_status": status,
            "heston_calibration_reason": reason,
            "heston_surface_quality": status,
            "heston_diagnostics_json": json.dumps({"status": status, "reason": reason}, sort_keys=True),
        }
    )
    return out


def default_heston_research_features(*, enabled: bool = False, reason: str | None = None) -> dict[str, Any]:
    """Return explicit default fields for payloads that exit before selection."""

    status = "PENDING_SELECTION" if enabled else "DISABLED"
    return _empty_features(
        enabled=enabled,
        status=status,
        reason=reason or ("awaiting_contract_selection" if enabled else "heston_research_disabled"),
    )


def _skew_state(params: HestonParams | None) -> str:
    if params is None:
        return "UNAVAILABLE"
    skew_pressure = params.rho * params.vol_of_vol
    if skew_pressure <= -0.20:
        return "NEGATIVE_SKEW"
    if skew_pressure >= 0.10:
        return "POSITIVE_SKEW"
    return "FLAT_SKEW"


def _divergence_score(
    *,
    bs_price,
    heston_price_value,
    bs_delta,
    heston_delta_value,
    calibration_error,
    surface_quality,
) -> int | None:
    components = []
    bs_price_value = _safe_float(bs_price, None)
    heston_price_value = _safe_float(heston_price_value, None)
    if bs_price_value is not None and heston_price_value is not None and bs_price_value > 0:
        components.append(min(abs(heston_price_value - bs_price_value) / bs_price_value, 1.0) * 55.0)
    bs_delta_value = _safe_float(bs_delta, None)
    heston_delta_value = _safe_float(heston_delta_value, None)
    if bs_delta_value is not None and heston_delta_value is not None:
        components.append(min(abs(heston_delta_value - bs_delta_value), 1.0) * 35.0)
    error_value = _safe_float(calibration_error, None)
    if error_value is not None:
        components.append(min(error_value, 1.0) * 20.0)
    quality_penalty = {"GOOD": 0, "CAUTION": 8, "WEAK": 16, "REJECTED": 35, "FAILED": 35}.get(
        str(surface_quality or "").upper(),
        12,
    )
    components.append(float(quality_penalty))
    if not components:
        return None
    return int(max(0, min(100, round(sum(components)))))


_QUALITY_SEVERITY = {
    "GOOD": 0,
    "CAUTION": 1,
    "WEAK": 2,
    "REJECTED": 3,
    "FAILED": 3,
}


def _worse_quality(current: str, floor: str) -> str:
    current_token = str(current or "FAILED").upper()
    floor_token = str(floor or "FAILED").upper()
    return current_token if _QUALITY_SEVERITY.get(current_token, 3) >= _QUALITY_SEVERITY.get(floor_token, 3) else floor_token


def _result_status(result: HestonCalibrationResult) -> str:
    if result.success:
        return "CALIBRATED"
    if result.surface_quality in {"INSUFFICIENT_DATA", "FAILED", "REJECTED"}:
        return result.surface_quality
    return "FAILED"


def _append_reason(reason: str, guard_reason: str) -> str:
    reason = str(reason or "").strip()
    return f"{reason}|{guard_reason}" if reason else guard_reason


def _apply_selected_price_gap_guard(
    *,
    status: str,
    surface_quality: str,
    reason: str,
    price_gap_rel_pct: float | None,
    selected_iv_quality: str,
) -> tuple[str, str, str, list[str]]:
    """Downgrade a selected-contract diagnostic when BS/Heston prices diverge wildly."""

    flags: list[str] = []
    if price_gap_rel_pct is None:
        return status, surface_quality, reason, flags
    if selected_iv_quality != "OK":
        flags.append("PRICE_GAP_SUPPRESSED_IV_QUALITY")
        return status, surface_quality, reason, flags
    if price_gap_rel_pct >= float(HESTON_PRICE_GAP_REJECT_PCT):
        flags.append("PRICE_GAP_REJECT")
        return (
            "REJECTED",
            "REJECTED",
            _append_reason(reason, f"selected_price_gap_reject:{round(price_gap_rel_pct, 4)}pct"),
            flags,
        )
    if price_gap_rel_pct >= float(HESTON_PRICE_GAP_WEAK_PCT):
        flags.append("PRICE_GAP_WEAK")
        return (
            status,
            _worse_quality(surface_quality, "WEAK"),
            _append_reason(reason, f"selected_price_gap_weak:{round(price_gap_rel_pct, 4)}pct"),
            flags,
        )
    return status, surface_quality, reason, flags


def _short_tte_guard(
    *,
    status: str,
    surface_quality: str,
    reason: str,
    tte_days: float | None,
) -> tuple[str, str, str, str, list[str]]:
    if tte_days is None:
        return status, surface_quality, reason, "UNKNOWN_TTE", []
    if tte_days <= float(HESTON_SHORT_TTE_REJECT_DAYS):
        return (
            "REJECTED",
            "REJECTED",
            _append_reason(reason, f"short_tte_reject:{round(tte_days, 4)}d"),
            "SHORT_TTE_REJECT",
            ["SHORT_TTE_REJECT"],
        )
    if tte_days <= float(HESTON_SHORT_TTE_WEAK_DAYS):
        return (
            status,
            _worse_quality(surface_quality, "WEAK"),
            _append_reason(reason, f"short_tte_weak:{round(tte_days, 4)}d"),
            "SHORT_TTE_WEAK",
            ["SHORT_TTE_WEAK"],
        )
    return status, surface_quality, reason, "NONE", []


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().upper()
    return token in {"1", "TRUE", "YES", "Y"}


def _selected_iv_quality(selected_iv_decimal: float | None, *, selected_iv_is_proxy=False) -> tuple[str, list[str]]:
    if selected_iv_decimal is None or selected_iv_decimal <= 0:
        return "MISSING", ["SELECTED_IV_MISSING"]
    if _boolish(selected_iv_is_proxy):
        return "PROXY", ["SELECTED_IV_PROXY"]
    iv_pct = selected_iv_decimal * 100.0
    if iv_pct < float(HESTON_SELECTED_IV_LOW_PCT):
        return "EXTREME_LOW", ["SELECTED_IV_EXTREME_LOW"]
    if iv_pct > float(HESTON_SELECTED_IV_HIGH_PCT):
        return "EXTREME_HIGH", ["SELECTED_IV_EXTREME_HIGH"]
    return "OK", []


def build_heston_research_features(
    option_chain: pd.DataFrame,
    *,
    spot,
    selected_strike=None,
    selected_option_type=None,
    selected_expiry=None,
    selected_iv=None,
    selected_iv_is_proxy=False,
    selected_iv_proxy_source=None,
    bs_delta=None,
    bs_gamma=None,
    valuation_time=None,
    enabled: bool = False,
    min_rows: int = 8,
    max_rows: int = 80,
    reject_error: float = 0.35,
    timeout_seconds: float = 2.5,
) -> dict[str, Any]:
    """Return Heston diagnostics without changing trade decisions."""

    if not enabled:
        return _empty_features(enabled=False, status="DISABLED", reason="heston_research_disabled")

    spot_value = _safe_float(spot, None)
    if option_chain is None or option_chain.empty or spot_value is None or spot_value <= 0:
        return _empty_features(enabled=True, status="FAILED", reason="missing_option_chain_or_spot")

    expiry_tte = _time_to_expiry_years(selected_expiry, valuation_time=valuation_time)
    tte_days = _time_to_expiry_days(selected_expiry, valuation_time=valuation_time)
    selected_iv_decimal = _safe_float(selected_iv, None)
    if selected_iv_decimal is not None and selected_iv_decimal > 1.0:
        selected_iv_decimal /= 100.0
    selected_iv_quality, selected_iv_flags = _selected_iv_quality(
        selected_iv_decimal,
        selected_iv_is_proxy=selected_iv_is_proxy,
    )

    result = calibrate_heston_to_chain(
        option_chain,
        spot=spot_value,
        valuation_time=valuation_time,
        min_rows=min_rows,
        max_rows=max_rows,
        reject_error=reject_error,
        timeout_seconds=timeout_seconds,
    )
    status = _result_status(result)
    if result.params is None:
        out = _empty_features(enabled=True, status=status, reason=result.reason)
        _, _, _, short_tte_guard, tte_flags = _short_tte_guard(
            status=status,
            surface_quality=status,
            reason=result.reason,
            tte_days=tte_days,
        )
        quality_flags = [*tte_flags, *selected_iv_flags]
        out["heston_calibration_sample_size"] = result.sample_size
        out["heston_calibration_error"] = result.calibration_error
        out["heston_tte_days"] = _round(tte_days, 4)
        out["heston_tte_bucket"] = _tte_bucket(tte_days)
        out["heston_expiry_context"] = _expiry_context(tte_days)
        out["heston_short_tte_guard"] = short_tte_guard
        out["heston_selected_iv_quality"] = selected_iv_quality
        out["heston_quality_flags"] = ",".join(quality_flags)
        diagnostics = result.to_dict()
        diagnostics.update(
            {
                "quality_flags": quality_flags,
                "selected_tte_days": _round(tte_days, 4),
                "tte_bucket": _tte_bucket(tte_days),
                "expiry_context": _expiry_context(tte_days),
                "short_tte_guard": short_tte_guard,
                "selected_iv_quality": selected_iv_quality,
                "selected_iv_proxy_source": str(selected_iv_proxy_source or "") or None,
            }
        )
        out["heston_diagnostics_json"] = json.dumps(diagnostics, sort_keys=True, default=str)
        return out

    params = result.params
    strike = _safe_float(selected_strike, None)
    option_type = normalize_option_type(selected_option_type)

    heston_snapshot = {"price": None, "delta": None, "gamma": None}
    heston_iv_proxy = None
    bs_model_price = None
    if strike is not None and option_type is not None and expiry_tte is not None:
        heston_snapshot = heston_greek_snapshot(
            spot=spot_value,
            strike=strike,
            time_to_expiry_years=expiry_tte,
            option_type=option_type,
            params=params,
        )
        heston_iv_proxy = heston_implied_vol_proxy(
            spot=spot_value,
            strike=strike,
            time_to_expiry_years=expiry_tte,
            params=params,
        )
        if selected_iv_decimal is not None and selected_iv_decimal > 0:
            bs_model_price = black_scholes_price(
                spot=spot_value,
                strike=strike,
                time_to_expiry_years=expiry_tte,
                volatility=selected_iv_decimal,
                option_type=option_type,
            )

    price_gap = None
    price_gap_rel_pct = None
    if bs_model_price is not None and heston_snapshot.get("price") is not None:
        price_gap = float(heston_snapshot["price"]) - float(bs_model_price)
        if bs_model_price > 0:
            price_gap_rel_pct = abs(price_gap) / float(bs_model_price) * 100.0

    bs_delta_value = _safe_float(bs_delta, None)
    delta_gap = None
    if bs_delta_value is not None and heston_snapshot.get("delta") is not None:
        delta_gap = float(heston_snapshot["delta"]) - bs_delta_value
    bs_gamma_value = _safe_float(bs_gamma, None)
    gamma_gap = None
    if bs_gamma_value is not None and heston_snapshot.get("gamma") is not None:
        gamma_gap = float(heston_snapshot["gamma"]) - bs_gamma_value
    greek_gap = None
    if delta_gap is not None or gamma_gap is not None:
        greek_gap = abs(delta_gap or 0.0) + min(abs(gamma_gap or 0.0) * 1000.0, 1.0)

    forward_variance = heston_forward_variance(params, 30.0 / 365.0)
    status = _result_status(result)
    surface_quality = result.surface_quality
    reason = result.reason
    status, surface_quality, reason, short_tte_guard, tte_flags = _short_tte_guard(
        status=status,
        surface_quality=surface_quality,
        reason=reason,
        tte_days=tte_days,
    )
    status, surface_quality, reason, gap_flags = _apply_selected_price_gap_guard(
        status=status,
        surface_quality=surface_quality,
        reason=reason,
        price_gap_rel_pct=price_gap_rel_pct,
        selected_iv_quality=selected_iv_quality,
    )
    bound_flags = [f"BOUND:{field}" for field in result.bound_hit_fields]
    quality_flags = [*bound_flags, *tte_flags, *selected_iv_flags, *gap_flags]

    diagnostics = result.to_dict()
    diagnostics.update(
        {
            "effective_status": status,
            "effective_surface_quality": surface_quality,
            "quality_flags": quality_flags,
            "selected_strike": strike,
            "selected_option_type": option_type,
            "selected_expiry": str(selected_expiry) if selected_expiry not in (None, "") else None,
            "selected_tte_years": _round(expiry_tte, 8),
            "selected_tte_days": _round(tte_days, 4),
            "tte_bucket": _tte_bucket(tte_days),
            "expiry_context": _expiry_context(tte_days),
            "short_tte_guard": short_tte_guard,
            "short_tte_weak_days": HESTON_SHORT_TTE_WEAK_DAYS,
            "short_tte_reject_days": HESTON_SHORT_TTE_REJECT_DAYS,
            "selected_iv_quality": selected_iv_quality,
            "selected_iv_proxy_source": str(selected_iv_proxy_source or "") or None,
            "price_gap_rel_pct": _round(price_gap_rel_pct, 4),
            "price_gap_weak_threshold_pct": HESTON_PRICE_GAP_WEAK_PCT,
            "price_gap_reject_threshold_pct": HESTON_PRICE_GAP_REJECT_PCT,
            "bs_gamma_gap": _round(gamma_gap, 8),
        }
    )

    return {
        "heston_research_enabled": True,
        "heston_calibration_status": status,
        "heston_calibration_reason": reason,
        "heston_calibration_sample_size": int(result.sample_size),
        "heston_kappa": _round(params.kappa, 6),
        "heston_theta": _round(params.theta, 8),
        "heston_vol_of_vol": _round(params.vol_of_vol, 6),
        "heston_rho": _round(params.rho, 6),
        "heston_v0": _round(params.v0, 8),
        "heston_calibration_error": _round(result.calibration_error, 6),
        "heston_surface_quality": surface_quality,
        "heston_quality_flags": ",".join(quality_flags),
        "heston_bound_hit_count": len(result.bound_hit_fields),
        "heston_tte_days": _round(tte_days, 4),
        "heston_tte_bucket": _tte_bucket(tte_days),
        "heston_expiry_context": _expiry_context(tte_days),
        "heston_short_tte_guard": short_tte_guard,
        "heston_selected_iv_quality": selected_iv_quality,
        "heston_skew_state": _skew_state(params),
        "heston_forward_variance_proxy": _round(forward_variance, 8),
        "heston_model_price": _round(heston_snapshot.get("price"), 4),
        "heston_model_delta": _round(heston_snapshot.get("delta"), 8),
        "heston_model_gamma": _round(heston_snapshot.get("gamma"), 10),
        "heston_model_iv_proxy": _round(heston_iv_proxy, 8),
        "bs_model_price_for_heston": _round(bs_model_price, 4),
        "bs_vs_heston_price_gap": _round(price_gap, 4),
        "heston_price_gap_rel_pct": _round(price_gap_rel_pct, 4),
        "bs_vs_heston_greek_gap": _round(greek_gap, 8),
        "greek_model_divergence_score": _divergence_score(
            bs_price=bs_model_price,
            heston_price_value=heston_snapshot.get("price"),
            bs_delta=bs_delta_value,
            heston_delta_value=heston_snapshot.get("delta"),
            calibration_error=result.calibration_error,
            surface_quality=surface_quality,
        ),
        "heston_diagnostics_json": json.dumps(diagnostics, sort_keys=True, default=str),
    }
