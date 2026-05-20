"""Robust Heston calibration against live or replay option chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from datetime import time as dt_time
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - scipy is a declared dependency
    minimize = None

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE
from config.settings import HESTON_BOUND_GUARD_REJECT_COUNT, HESTON_BOUND_GUARD_TOLERANCE_PCT
from utils.regime_normalization import normalize_iv_decimal

from .heston_pricer import DEFAULT_HESTON_PARAMS, HestonParams, heston_price, normalize_option_type


PARAMETER_BOUNDS = {
    "kappa": (0.10, 8.00),
    "theta": (0.0001, 1.00),
    "vol_of_vol": (0.01, 3.00),
    "rho": (-0.95, 0.25),
    "v0": (0.0001, 1.00),
}

NSE_EXPIRY_CLOSE_TIME = dt_time(hour=15, minute=30)


@dataclass(frozen=True)
class CalibrationPoint:
    strike: float
    option_type: str
    time_to_expiry_years: float
    market_price: float
    implied_vol: float | None
    weight: float = 1.0


@dataclass(frozen=True)
class HestonCalibrationResult:
    success: bool
    params: HestonParams | None
    calibration_error: float | None
    surface_quality: str
    reason: str
    sample_size: int
    elapsed_ms: float
    rejected: bool = False
    bound_hit_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "params": asdict(self.params) if self.params is not None else None,
            "calibration_error": self.calibration_error,
            "surface_quality": self.surface_quality,
            "reason": self.reason,
            "sample_size": self.sample_size,
            "elapsed_ms": self.elapsed_ms,
            "rejected": self.rejected,
            "bound_hit_fields": list(self.bound_hit_fields),
        }


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _coerce_timestamp(value) -> pd.Timestamp | None:
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
    expiry_ts = _coerce_timestamp(expiry_value)
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
    valuation_ts = _coerce_timestamp(valuation_time) or pd.Timestamp.now(tz="Asia/Kolkata")
    seconds = (expiry_ts - valuation_ts).total_seconds()
    if seconds <= 60:
        return None
    return max(seconds / (365.0 * 24.0 * 3600.0), 1e-6)


def _estimate_atm_vol(points: list[CalibrationPoint], spot: float) -> float:
    iv_points = [
        (abs(point.strike - spot), point.implied_vol)
        for point in points
        if point.implied_vol is not None and point.implied_vol > 0
    ]
    if iv_points:
        return float(sorted(iv_points, key=lambda item: item[0])[0][1])
    return math.sqrt(DEFAULT_HESTON_PARAMS.v0)


def _initial_guess(points: list[CalibrationPoint], spot: float) -> np.ndarray:
    atm_vol = min(max(_estimate_atm_vol(points, spot), 0.05), 1.25)
    variance = atm_vol * atm_vol
    put_ivs = [p.implied_vol for p in points if p.option_type == "PE" and p.strike < spot and p.implied_vol]
    call_ivs = [p.implied_vol for p in points if p.option_type == "CE" and p.strike > spot and p.implied_vol]
    skew = (float(np.nanmean(put_ivs)) - float(np.nanmean(call_ivs))) if put_ivs and call_ivs else 0.0
    rho = -0.35 - min(max(skew, -0.10), 0.25)
    return np.array([1.50, variance, 0.60, min(max(rho, -0.85), 0.05), variance], dtype=float)


def _params_from_array(values: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(values[0]),
        theta=float(values[1]),
        vol_of_vol=float(values[2]),
        rho=float(values[3]),
        v0=float(values[4]),
    ).clipped()


def _objective(values: np.ndarray, *, points: list[CalibrationPoint], spot: float) -> float:
    params = _params_from_array(values)
    errors = []
    weights = []
    for point in points:
        model_price = heston_price(
            spot=spot,
            strike=point.strike,
            time_to_expiry_years=point.time_to_expiry_years,
            option_type=point.option_type,
            params=params,
            risk_free_rate=RISK_FREE_RATE,
            dividend_yield=DIVIDEND_YIELD,
        )
        if model_price is None:
            continue
        scale = max(point.market_price, 1.0)
        rel_error = (float(model_price) - point.market_price) / scale
        errors.append(rel_error * rel_error)
        weights.append(point.weight)
    if not errors:
        return 1e6
    weighted = np.average(np.array(errors, dtype=float), weights=np.array(weights, dtype=float))
    feller_shortfall = max(0.0, (params.vol_of_vol ** 2) - (2.0 * params.kappa * params.theta))
    return float(weighted + min(feller_shortfall, 2.0) * 0.002)


def _calibration_error(params: HestonParams, *, points: list[CalibrationPoint], spot: float) -> float | None:
    abs_errors = []
    weights = []
    for point in points:
        model_price = heston_price(
            spot=spot,
            strike=point.strike,
            time_to_expiry_years=point.time_to_expiry_years,
            option_type=point.option_type,
            params=params,
        )
        if model_price is None:
            continue
        scale = max(point.market_price, 1.0)
        abs_errors.append(abs(float(model_price) - point.market_price) / scale)
        weights.append(point.weight)
    if not abs_errors:
        return None
    return round(float(np.average(np.array(abs_errors), weights=np.array(weights))), 6)


def _surface_quality(error: float | None, *, reject_error: float) -> str:
    if error is None:
        return "FAILED"
    if error <= min(0.08, reject_error * 0.35):
        return "GOOD"
    if error <= min(0.18, reject_error * 0.70):
        return "CAUTION"
    if error <= reject_error:
        return "WEAK"
    return "REJECTED"


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


def _bound_hit_fields(
    params: HestonParams,
    *,
    tolerance_pct: float = HESTON_BOUND_GUARD_TOLERANCE_PCT,
) -> tuple[str, ...]:
    """Return Heston parameters that are effectively pinned to optimizer bounds."""

    p = params.clipped()
    values = {
        "kappa": p.kappa,
        "theta": p.theta,
        "vol_of_vol": p.vol_of_vol,
        "rho": p.rho,
        "v0": p.v0,
    }
    hits: list[str] = []
    tolerance = max(float(tolerance_pct or 0.0), 0.0)
    for field, value in values.items():
        lower, upper = PARAMETER_BOUNDS[field]
        band = max((upper - lower) * tolerance, 1e-10)
        if value <= lower + band:
            hits.append(f"{field}_lower")
        elif value >= upper - band:
            hits.append(f"{field}_upper")
    return tuple(hits)


def _apply_bound_guard(
    *,
    params: HestonParams,
    quality: str,
    reason: str,
    reject_count: int = HESTON_BOUND_GUARD_REJECT_COUNT,
) -> tuple[str, str, bool, tuple[str, ...]]:
    """Downgrade Heston fits that solve by pinning parameters to bounds."""

    bound_hits = _bound_hit_fields(params)
    if not bound_hits:
        return quality, reason, False, bound_hits

    hit_text = ",".join(bound_hits)
    if len(bound_hits) >= int(reject_count):
        return "REJECTED", f"{reason}|parameter_bound_guard_reject:{hit_text}", True, bound_hits
    return _worse_quality(quality, "WEAK"), f"{reason}|parameter_bound_guard_weak:{hit_text}", False, bound_hits


def prepare_calibration_points(
    option_chain: pd.DataFrame,
    *,
    spot: float,
    valuation_time=None,
    max_rows: int = 80,
) -> list[CalibrationPoint]:
    """Normalize an option chain into calibration points."""

    if option_chain is None or option_chain.empty or spot <= 0:
        return []
    frame = option_chain.copy()
    strike_col = _first_column(frame, ("strikePrice", "STRIKE_PR", "strike", "strike_price"))
    type_col = _first_column(frame, ("OPTION_TYP", "option_type", "optionType", "instrument_type"))
    price_col = _first_column(frame, ("lastPrice", "LAST_PRICE", "ltp", "close"))
    expiry_col = _first_column(frame, ("EXPIRY_DT", "expiry", "expiryDate", "expiry_date"))
    iv_col = _first_column(frame, ("impliedVolatility", "IV", "iv"))
    volume_col = _first_column(frame, ("totalTradedVolume", "VOLUME", "volume"))
    oi_col = _first_column(frame, ("openInterest", "OPEN_INT", "open_interest", "oi"))
    bid_col = _first_column(frame, ("bidPrice", "BID_PRICE", "bid"))
    ask_col = _first_column(frame, ("askPrice", "ASK_PRICE", "ask"))
    required = (strike_col, type_col, price_col, expiry_col)
    if any(column is None for column in required):
        return []

    frame["_strike"] = pd.to_numeric(frame[strike_col], errors="coerce")
    frame["_price"] = pd.to_numeric(frame[price_col], errors="coerce")
    frame["_option_type"] = frame[type_col].map(normalize_option_type)
    frame["_iv"] = frame[iv_col].map(lambda value: normalize_iv_decimal(value, default=np.nan)) if iv_col else np.nan
    frame["_volume"] = pd.to_numeric(frame[volume_col], errors="coerce").fillna(0.0) if volume_col else 0.0
    frame["_oi"] = pd.to_numeric(frame[oi_col], errors="coerce").fillna(0.0) if oi_col else 0.0
    if bid_col and ask_col:
        frame["_bid"] = pd.to_numeric(frame[bid_col], errors="coerce").fillna(0.0)
        frame["_ask"] = pd.to_numeric(frame[ask_col], errors="coerce").fillna(0.0)
        two_sided_quote = (frame["_bid"] > 0) & (frame["_ask"] > 0) & (frame["_ask"] >= frame["_bid"])
        frame["_spread_ratio"] = np.where(
            two_sided_quote,
            (frame["_ask"] - frame["_bid"]).abs() / frame["_price"].replace(0, np.nan),
            np.nan,
        )
    else:
        frame["_spread_ratio"] = np.nan
    frame["_dist"] = (frame["_strike"] - float(spot)).abs()
    frame = frame.dropna(subset=["_strike", "_price", "_option_type"])
    frame = frame[(frame["_strike"] > 0) & (frame["_price"] > 0)]
    frame = frame[frame["_spread_ratio"].isna() | (frame["_spread_ratio"] <= 0.75)]
    if frame.empty:
        return []

    frame["_liquidity_rank"] = frame["_volume"].fillna(0.0) + (frame["_oi"].fillna(0.0) * 0.05)
    frame = frame.sort_values(["_dist", "_liquidity_rank"], ascending=[True, False], kind="mergesort")
    if max_rows > 0:
        frame = frame.head(int(max_rows))

    points: list[CalibrationPoint] = []
    max_liquidity = max(float(frame["_liquidity_rank"].max()), 1.0)
    for _, row in frame.iterrows():
        tte = _time_to_expiry_years(row.get(expiry_col), valuation_time=valuation_time)
        if tte is None:
            continue
        option_type = normalize_option_type(row.get("_option_type"))
        if option_type is None:
            continue
        iv = _safe_float(row.get("_iv"), None)
        liquidity_weight = 1.0 + min(max(float(row.get("_liquidity_rank", 0.0)) / max_liquidity, 0.0), 2.0)
        points.append(
            CalibrationPoint(
                strike=float(row["_strike"]),
                option_type=option_type,
                time_to_expiry_years=float(tte),
                market_price=float(row["_price"]),
                implied_vol=iv if iv is not None and iv > 0 else None,
                weight=float(liquidity_weight),
            )
        )
    return points


def calibrate_heston_to_chain(
    option_chain: pd.DataFrame,
    *,
    spot: float,
    valuation_time=None,
    min_rows: int = 8,
    max_rows: int = 80,
    reject_error: float = 0.35,
    timeout_seconds: float = 2.5,
) -> HestonCalibrationResult:
    """Calibrate Heston parameters and return a guarded result."""

    started = time.perf_counter()
    spot_value = _safe_float(spot, None)
    if spot_value is None or spot_value <= 0:
        return HestonCalibrationResult(False, None, None, "FAILED", "invalid_spot", 0, 0.0, True)

    points = prepare_calibration_points(
        option_chain,
        spot=spot_value,
        valuation_time=valuation_time,
        max_rows=max_rows,
    )
    if len(points) < int(min_rows):
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return HestonCalibrationResult(
            False,
            None,
            None,
            "INSUFFICIENT_DATA",
            "insufficient_calibration_points",
            len(points),
            elapsed_ms,
            True,
        )

    if minimize is None:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return HestonCalibrationResult(False, None, None, "FAILED", "scipy_unavailable", len(points), elapsed_ms, True)

    x0 = _initial_guess(points, spot_value)
    bounds = [
        PARAMETER_BOUNDS["kappa"],
        PARAMETER_BOUNDS["theta"],
        PARAMETER_BOUNDS["vol_of_vol"],
        PARAMETER_BOUNDS["rho"],
        PARAMETER_BOUNDS["v0"],
    ]
    timeout = max(0.05, float(timeout_seconds or 2.5))
    deadline = started + timeout
    maxiter = max(12, min(80, int(timeout * 24)))
    try:
        def objective(values: np.ndarray) -> float:
            if time.perf_counter() > deadline:
                raise TimeoutError("heston_calibration_timeout")
            return _objective(values, points=points, spot=spot_value)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return HestonCalibrationResult(
            False,
            None,
            None,
            "FAILED",
            f"optimizer_exception:{type(exc).__name__}",
            len(points),
            elapsed_ms,
            True,
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if not bool(getattr(result, "success", False)):
        candidate_params = _params_from_array(np.asarray(getattr(result, "x", x0), dtype=float))
        error = _calibration_error(candidate_params, points=points, spot=spot_value)
        quality = _surface_quality(error, reject_error=reject_error)
        quality, reason, bound_rejected, bound_hits = _apply_bound_guard(
            params=candidate_params,
            quality=quality if quality != "GOOD" else "CAUTION",
            reason=f"optimizer_not_converged:{getattr(result, 'message', 'unknown')}",
        )
        return HestonCalibrationResult(
            False,
            candidate_params,
            error,
            quality,
            reason,
            len(points),
            elapsed_ms,
            bool(bound_rejected or quality == "REJECTED"),
            bound_hits,
        )

    params = _params_from_array(np.asarray(result.x, dtype=float))
    error = _calibration_error(params, points=points, spot=spot_value)
    quality = _surface_quality(error, reject_error=reject_error)
    quality, reason, bound_rejected, bound_hits = _apply_bound_guard(
        params=params,
        quality=quality,
        reason="ok" if quality != "REJECTED" else "calibration_error_above_reject_threshold",
    )
    rejected = bound_rejected or quality == "REJECTED"
    return HestonCalibrationResult(
        not rejected,
        params,
        error,
        quality,
        reason,
        len(points),
        elapsed_ms,
        rejected,
        bound_hits,
    )
