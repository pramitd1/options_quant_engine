"""
Black-Scholes Greeks engine for per-contract and aggregate Greek calculations.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _parse_expiry_years(expiry_value, default_days: float = 7.0):
    if isinstance(expiry_value, str):
        raw = expiry_value.strip().upper()
        match = re.fullmatch(r"T\+(\d+)", raw)
        if match:
            return max(float(match.group(1)) / 365.0, 1.0 / (365.0 * 24.0))
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)

    if pd.isna(parsed):
        parsed = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
        if pd.isna(parsed):
            return default_days / 365.0

    now_ts = pd.Timestamp.utcnow()

    time_years = (parsed - now_ts).total_seconds() / (365.0 * 24 * 3600.0)
    return max(float(time_years), 1.0 / (365.0 * 24.0))


def compute_option_greeks(
    *,
    spot,
    strike,
    time_to_expiry_years,
    volatility_pct,
    option_type,
    risk_free_rate=RISK_FREE_RATE,
    dividend_yield=DIVIDEND_YIELD,
):
    spot = _safe_float(spot, None)
    strike = _safe_float(strike, None)
    t = _safe_float(time_to_expiry_years, None)
    sigma_pct = _safe_float(volatility_pct, None)

    if spot in (None, 0) or strike in (None, 0) or t in (None, 0) or sigma_pct in (None, 0):
        return None

    sigma = max(float(sigma_pct) / 100.0, 1e-6)
    t = max(float(t), 1e-6)
    r = float(risk_free_rate)
    q = float(dividend_yield)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    pdf = _norm_pdf(d1)
    discount_q = math.exp(-q * t)
    discount_r = math.exp(-r * t)

    option_type = str(option_type).upper().strip()

    charm_common = (
        discount_q
        * pdf
        * ((2.0 * (r - q) * t) - (d2 * sigma * sqrt_t))
        / (2.0 * t * sigma * sqrt_t)
    )

    if option_type == "CE":
        delta = discount_q * _norm_cdf(d1)
        theta = (
            -(spot * discount_q * pdf * sigma) / (2.0 * sqrt_t)
            - r * strike * discount_r * _norm_cdf(d2)
            + q * spot * discount_q * _norm_cdf(d1)
        ) / 365.0
        rho = (strike * t * discount_r * _norm_cdf(d2)) / 100.0
        charm = (q * discount_q * _norm_cdf(d1) - charm_common) / 365.0
    else:
        delta = discount_q * (_norm_cdf(d1) - 1.0)
        theta = (
            -(spot * discount_q * pdf * sigma) / (2.0 * sqrt_t)
            + r * strike * discount_r * _norm_cdf(-d2)
            - q * spot * discount_q * _norm_cdf(-d1)
        ) / 365.0
        rho = (-strike * t * discount_r * _norm_cdf(-d2)) / 100.0
        charm = (-q * discount_q * _norm_cdf(-d1) - charm_common) / 365.0

    gamma = (discount_q * pdf) / (spot * sigma * sqrt_t)
    vega = (spot * discount_q * pdf * sqrt_t) / 100.0
    vanna = (-(discount_q * pdf * d2) / sigma) / 100.0

    return {
        "DELTA": delta,
        "GAMMA": gamma,
        "THETA": theta,
        "VEGA": vega,
        "RHO": rho,
        "VANNA": vanna,
        "CHARM": charm,
        "TTE": t,
    }


def enrich_chain_with_greeks(
    option_chain: pd.DataFrame,
    *,
    spot,
    risk_free_rate=RISK_FREE_RATE,
    dividend_yield=DIVIDEND_YIELD,
):
    if option_chain is None or option_chain.empty:
        return option_chain

    df = option_chain.copy()

    for col in ["strikePrice", "IV", "impliedVolatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    deltas = []
    gammas = []
    thetas = []
    vegas = []
    rhos = []
    vannas = []
    charms = []
    ttes = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        iv = row_dict.get("IV", row_dict.get("impliedVolatility"))
        greeks = compute_option_greeks(
            spot=spot,
            strike=row_dict.get("strikePrice"),
            time_to_expiry_years=_parse_expiry_years(row_dict.get("EXPIRY_DT")),
            volatility_pct=iv,
            option_type=row_dict.get("OPTION_TYP"),
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        if greeks is None:
            deltas.append(row_dict.get("DELTA"))
            gammas.append(row_dict.get("GAMMA"))
            thetas.append(row_dict.get("THETA"))
            vegas.append(row_dict.get("VEGA"))
            rhos.append(row_dict.get("RHO"))
            vannas.append(row_dict.get("VANNA"))
            charms.append(row_dict.get("CHARM"))
            ttes.append(None)
            continue

        deltas.append(greeks["DELTA"])
        gammas.append(greeks["GAMMA"])
        thetas.append(greeks["THETA"])
        vegas.append(greeks["VEGA"])
        rhos.append(greeks["RHO"])
        vannas.append(greeks["VANNA"])
        charms.append(greeks["CHARM"])
        ttes.append(greeks["TTE"])

    df["DELTA"] = deltas
    df["GAMMA"] = gammas
    df["THETA"] = thetas
    df["VEGA"] = vegas
    df["RHO"] = rhos
    df["VANNA"] = vannas
    df["CHARM"] = charms
    df["TTE"] = ttes

    return df


def _exposure_regime(value: float | None, gross: float | None, *, positive_label: str, negative_label: str, neutral_label: str):
    if value is None or gross is None or gross <= 0:
        return "UNKNOWN"
    if abs(value) <= gross * 0.05:
        return neutral_label
    if value > 0:
        return positive_label
    return negative_label


def summarize_greek_exposures(option_chain: pd.DataFrame):
    if option_chain is None or option_chain.empty:
        return {
            "delta_exposure": None,
            "gamma_exposure_greeks": None,
            "theta_exposure": None,
            "vega_exposure": None,
            "rho_exposure": None,
            "vanna_exposure": None,
            "charm_exposure": None,
            "vanna_regime": None,
            "charm_regime": None,
        }

    df = option_chain.copy()
    oi = pd.to_numeric(df.get("OPEN_INT", df.get("openInterest")), errors="coerce").fillna(0.0)

    def _series(name):
        return pd.to_numeric(df.get(name), errors="coerce").fillna(0.0)

    delta_exposure = float((_series("DELTA") * oi).sum())
    gamma_exposure = float((_series("GAMMA") * oi).sum())
    theta_exposure = float((_series("THETA") * oi).sum())
    vega_exposure = float((_series("VEGA") * oi).sum())
    rho_exposure = float((_series("RHO") * oi).sum())
    vanna_exposure = float((_series("VANNA") * oi).sum())
    charm_exposure = float((_series("CHARM") * oi).sum())

    gross_vanna = float((_series("VANNA").abs() * oi).sum())
    gross_charm = float((_series("CHARM").abs() * oi).sum())

    return {
        "delta_exposure": round(delta_exposure, 2),
        "gamma_exposure_greeks": round(gamma_exposure, 6),
        "theta_exposure": round(theta_exposure, 2),
        "vega_exposure": round(vega_exposure, 2),
        "rho_exposure": round(rho_exposure, 2),
        "vanna_exposure": round(vanna_exposure, 4),
        "charm_exposure": round(charm_exposure, 4),
        "vanna_regime": _exposure_regime(
            vanna_exposure,
            gross_vanna,
            positive_label="POSITIVE_VANNA",
            negative_label="NEGATIVE_VANNA",
            neutral_label="NEUTRAL_VANNA",
        ),
        "charm_regime": _exposure_regime(
            charm_exposure,
            gross_charm,
            positive_label="POSITIVE_CHARM",
            negative_label="NEGATIVE_CHARM",
            neutral_label="NEUTRAL_CHARM",
        ),
    }
