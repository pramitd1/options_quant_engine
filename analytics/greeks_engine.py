"""
Module: greeks_engine.py

Purpose:
    Compute greeks analytics from option-chain inputs.

Role in the System:
    Part of the analytics layer that transforms raw option-chain and market snapshots into interpretable features.

Key Outputs:
    Structured features, regime labels, and market-state diagnostics derived from market data.

Downstream Usage:
    Consumed by market-state assembly, probability estimation, risk overlays, and research diagnostics.
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
import logging

import pandas as pd

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE
from utils.numerics import safe_float as _safe_float  # noqa: F401
from utils.math_helpers import norm_pdf as _norm_pdf, norm_cdf as _norm_cdf  # noqa: F401


_LOG = logging.getLogger(__name__)
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_GREEKS_MIN_TTE_SECONDS = float(os.getenv("OQE_GREEKS_MIN_TTE_SECONDS", "60"))
_GREEKS_MIN_TTE_YEARS_FLOOR = float(os.getenv("OQE_GREEKS_MIN_TTE_YEARS_FLOOR", "1e-6"))


# ---------------------------------------------------------------------------
# Newton-Raphson implied-vol estimation (fallback when provider IV is 0/None)
# ---------------------------------------------------------------------------

def _bs_price_for_iv(spot, strike, t, sigma, option_type, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """Black-Scholes European price for IV estimation."""
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discount_q = math.exp(-q * t)
    if option_type == "CE":
        return spot * discount_q * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * discount_q * _norm_cdf(-d1)


def _bs_vega_for_iv(spot, strike, t, sigma, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return spot * math.exp(-q * t) * math.sqrt(t) * _norm_pdf(d1)


def estimate_iv_from_price(market_price, spot, strike, t, option_type,
                           r=RISK_FREE_RATE, q=DIVIDEND_YIELD, tol=1e-6, max_iter=50):
    """Newton-Raphson IV solver.  Returns IV as percentage points (e.g. 18.0)
    or 0.0 when estimation fails."""
    if market_price is None or spot is None or strike is None or t is None:
        return 0.0
    market_price = float(market_price)
    spot = float(spot)
    strike = float(strike)
    t = float(t)
    if market_price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return 0.0
    intrinsic = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    if market_price <= intrinsic:
        return 0.0
    sigma = 0.20
    for _ in range(max_iter):
        price = _bs_price_for_iv(spot, strike, t, sigma, option_type, r, q)
        vega = _bs_vega_for_iv(spot, strike, t, sigma, r, q)
        if vega <= 1e-12:
            break
        sigma -= (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.001
        if sigma > 5.0:
            return 0.0
        if abs(price - market_price) < tol:
            break
    if sigma > 5.0:
        return 0.0
    return round(sigma * 100.0, 2)


def _coerce_valuation_timestamp(value):
    """
    Purpose:
        Normalize the valuation timestamp used when converting expiry dates into time-to-expiry.
    
    Context:
        Internal helper in the analytics layer. Greek calculations require a consistent valuation timestamp so the same option chain produces stable time-to-expiry values across providers and replay workflows.
    
    Inputs:
        value (Any): Timestamp-like input supplied by live runtime, replay mode, or ad hoc research code.
    
    Returns:
        pd.Timestamp: UTC timestamp used as the valuation anchor for time-to-expiry calculations.
    
    Notes:
        Invalid or missing timestamps fall back to the current UTC time so the helper stays usable even when providers omit explicit valuation metadata.
    """
    if value is None:
        return pd.Timestamp.utcnow()

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(parsed):
        return pd.Timestamp.utcnow()
    return parsed


def _parse_expiry_years(expiry_value, valuation_time=None):
    """
    Purpose:
        Convert expiry metadata into fractional years so closed-form Greeks can be evaluated.
    
    Context:
        Internal helper in the Greeks engine. The pricing formulas expect continuous time in years, while exchange data may provide calendar dates or short `T+N` style offsets.
    
    Inputs:
        expiry_value (Any): Raw expiry representation from the option chain, typically a date-like value or a `T+N` shorthand.
        valuation_time (Any): Timestamp used as the start of the expiry interval.
    
    Returns:
        float | None: Time to expiry in years, floored to a small positive value, or `None` when the expiry cannot be parsed.
    
    Notes:
        The floor prevents division-by-zero explosions in near-expiry Greeks while still treating expiring contracts as extremely short-dated.
    """
    if isinstance(expiry_value, str):
        raw = expiry_value.strip().upper()
        match = re.fullmatch(r"T\+(\d+)", raw)
        if match:
            years = float(match.group(1)) / 365.0
            if years * _SECONDS_PER_YEAR < _GREEKS_MIN_TTE_SECONDS:
                return None
            return max(years, _GREEKS_MIN_TTE_YEARS_FLOOR)
        parsed = _parse_expiry_timestamp_cached(expiry_value)
    else:
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)

    if pd.isna(parsed):
        parsed = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
        if pd.isna(parsed):
            return None

    now_ts = _coerce_valuation_timestamp(valuation_time)

    time_years = (parsed - now_ts).total_seconds() / (365.0 * 24 * 3600.0)
    if time_years <= 0:
        return None
    if float(time_years) * _SECONDS_PER_YEAR < _GREEKS_MIN_TTE_SECONDS:
        return None
    return max(float(time_years), _GREEKS_MIN_TTE_YEARS_FLOOR)


@lru_cache(maxsize=2048)
def _parse_expiry_timestamp_cached(expiry_value: str):
    """Parse and cache a string expiry value to a UTC pd.Timestamp.

    The ``@lru_cache`` decorator requires a hashable argument.  The caller
    ``_parse_expiry_years`` guards this function with ``isinstance(expiry_value,
    str)`` so only strings reach it, but we add an explicit assertion here to
    make that contract visible and to fail loudly if the guard is ever relaxed.
    """
    assert isinstance(expiry_value, str), (
        f"_parse_expiry_timestamp_cached expects a str, got {type(expiry_value).__name__!r}"
    )
    parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
        if pd.isna(parsed):
            return None
    return parsed


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
    """
    Purpose:
        Compute Black-Scholes style option sensitivities for a single contract snapshot.
    
    Context:
        This function sits in the analytics layer beneath market-state assembly. It is used when providers do not supply Greeks directly or when the engine wants a consistent cross-provider approximation.
    
    Inputs:
        spot (Any): Underlying spot price.
        strike (Any): Option strike price.
        time_to_expiry_years (Any): Time to expiry expressed in fractional years.
        volatility_pct (Any): Annualized implied volatility expressed in percentage points, for example `18.5` for 18.5%.
        option_type (Any): Option side, expected to be `CE` or `PE`.
        risk_free_rate (Any): Annualized risk-free rate used by the approximation.
        dividend_yield (Any): Annualized dividend yield applied to the underlying.
    
    Returns:
        dict | None: Dictionary containing delta, gamma, theta, vega, rho, vanna, charm, and the normalized time-to-expiry, or `None` when inputs are not usable.
    
    Notes:
        The formulas are Black-Scholes approximations. They assume European-style dynamics and are used here as a consistent sensitivity proxy rather than a perfect microstructure model.
    """
    spot = _safe_float(spot, None)
    strike = _safe_float(strike, None)
    t = _safe_float(time_to_expiry_years, None)
    sigma_pct = _safe_float(volatility_pct, None)

    if spot in (None, 0) or strike in (None, 0) or t in (None, 0) or sigma_pct in (None, 0):
        return None

    if float(t) * _SECONDS_PER_YEAR < _GREEKS_MIN_TTE_SECONDS:
        _LOG.debug("Skipping Greeks due to ultra-short TTE: %.2f seconds", float(t) * _SECONDS_PER_YEAR)
        return None

    sigma = max(float(sigma_pct) / 100.0, 1e-6)
    t = max(float(t), _GREEKS_MIN_TTE_YEARS_FLOOR)
    r = float(risk_free_rate)
    q = float(dividend_yield)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    pdf = _norm_pdf(d1)
    discount_q = math.exp(-q * t)
    discount_r = math.exp(-r * t)

    option_type = str(option_type).upper().strip()
    if option_type not in {"CE", "PE"}:
        return None

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
    valuation_time=None,
    risk_free_rate=RISK_FREE_RATE,
    dividend_yield=DIVIDEND_YIELD,
):
    """
    Purpose:
        Populate an option chain with internally computed Greeks when provider data is missing or inconsistent.
    
    Context:
        This function bridges raw option-chain ingestion and downstream analytics. It ensures later layers can rely on a uniform Greek schema regardless of which broker or replay source supplied the chain.
    
    Inputs:
        option_chain (pd.DataFrame): Option-chain snapshot to enrich.
        spot (Any): Underlying spot price used in the Greeks calculation.
        valuation_time (Any): Timestamp used as the valuation anchor for time-to-expiry.
        risk_free_rate (Any): Annualized risk-free rate supplied to the pricing approximation.
        dividend_yield (Any): Annualized dividend yield supplied to the pricing approximation.
    
    Returns:
        pd.DataFrame: Copy of the option chain with normalized Greek columns added or refreshed.
    
    Notes:
        Existing provider Greeks are preserved when the approximation cannot be computed, which keeps the enrichment path robust for partially populated chains.
    """
    if option_chain is None or option_chain.empty:
        return option_chain

    df = option_chain.copy()

    numeric_candidate_cols = [
        col for col in ["strikePrice", "STRIKE_PR", "IV", "impliedVolatility", "lastPrice", "LAST_PRICE"]
        if col in df.columns
    ]
    for col in numeric_candidate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    strike_col = "strikePrice" if "strikePrice" in df.columns else ("STRIKE_PR" if "STRIKE_PR" in df.columns else None)
    expiry_col = "EXPIRY_DT" if "EXPIRY_DT" in df.columns else ("expiry" if "expiry" in df.columns else None)
    option_type_col = "OPTION_TYP" if "OPTION_TYP" in df.columns else ("optionType" if "optionType" in df.columns else None)
    market_price_col = "lastPrice" if "lastPrice" in df.columns else ("LAST_PRICE" if "LAST_PRICE" in df.columns else None)
    iv_col = "IV" if "IV" in df.columns else ("impliedVolatility" if "impliedVolatility" in df.columns else None)

    # Pre-coerce valuation timestamp once (avoids re-parsing per row)
    coerced_valuation_time = _coerce_valuation_timestamp(valuation_time)

    # Cache expiry→TTE lookups (most chains share a single expiry)
    _tte_cache: dict = {}

    def _cached_parse_expiry_years(expiry_value):
        key = expiry_value
        if key not in _tte_cache:
            _tte_cache[key] = _parse_expiry_years(expiry_value, valuation_time=coerced_valuation_time)
        return _tte_cache[key]

    row_count = len(df)
    strike_values = df[strike_col].tolist() if strike_col else [None] * row_count
    expiry_values = df[expiry_col].tolist() if expiry_col else [None] * row_count
    option_type_values = (
        df[option_type_col].astype(str).tolist()
        if option_type_col else [""] * row_count
    )
    market_price_values = df[market_price_col].tolist() if market_price_col else [None] * row_count
    iv_values = df[iv_col].tolist() if iv_col else [None] * row_count
    tte_values = [_cached_parse_expiry_years(expiry_value) for expiry_value in expiry_values]

    existing_greek_values = {
        name: (df[name].tolist() if name in df.columns else [None] * row_count)
        for name in ["DELTA", "GAMMA", "THETA", "VEGA", "RHO", "VANNA", "CHARM"]
    }

    deltas = []
    gammas = []
    thetas = []
    vegas = []
    rhos = []
    vannas = []
    charms = []
    ttes = []
    ivs = []

    for idx, (iv, expiry_tte, strike_value, option_type_value, market_price) in enumerate(
        zip(iv_values, tte_values, strike_values, option_type_values, market_price_values)
    ):
        # Fallback: estimate IV from market price when provider gives 0/None
        if (iv is None or (isinstance(iv, (int, float)) and iv <= 0)) and spot is not None:
            estimated = estimate_iv_from_price(
                market_price,
                spot,
                strike_value,
                expiry_tte,
                option_type_value,
                r=risk_free_rate,
                q=dividend_yield,
            )
            if estimated > 0:
                iv = estimated

        greeks = compute_option_greeks(
            spot=spot,
            strike=strike_value,
            time_to_expiry_years=expiry_tte,
            volatility_pct=iv,
            option_type=option_type_value,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        if greeks is None:
            deltas.append(existing_greek_values["DELTA"][idx])
            gammas.append(existing_greek_values["GAMMA"][idx])
            thetas.append(existing_greek_values["THETA"][idx])
            vegas.append(existing_greek_values["VEGA"][idx])
            rhos.append(existing_greek_values["RHO"][idx])
            vannas.append(existing_greek_values["VANNA"][idx])
            charms.append(existing_greek_values["CHARM"][idx])
            ttes.append(None)
            ivs.append(iv)
            continue

        deltas.append(greeks["DELTA"])
        gammas.append(greeks["GAMMA"])
        thetas.append(greeks["THETA"])
        vegas.append(greeks["VEGA"])
        rhos.append(greeks["RHO"])
        vannas.append(greeks["VANNA"])
        charms.append(greeks["CHARM"])
        ttes.append(greeks["TTE"])
        ivs.append(iv)
    df["DELTA"] = deltas
    df["GAMMA"] = gammas
    df["THETA"] = thetas
    df["VEGA"] = vegas
    df["RHO"] = rhos
    df["VANNA"] = vannas
    df["CHARM"] = charms
    df["TTE"] = ttes

    # Write back estimated IV so downstream (atm_vol, vol_surface) can use it
    df["IV"] = ivs
    df["impliedVolatility"] = ivs

    return df


def _exposure_regime(value: float | None, gross: float | None, *, positive_label: str, negative_label: str, neutral_label: str):
    """
    Purpose:
        Classify a signed exposure into positive, negative, neutral, or unknown regime labels.
    
    Context:
        Internal helper used after exposure aggregation. The signal engine does not need the raw number alone; it also needs a compact regime label that can be carried into diagnostics and rule-based logic.
    
    Inputs:
        value (float | None): Signed aggregate exposure for one Greek.
        gross (float | None): Gross absolute exposure used to decide whether the signed value is materially different from zero.
        positive_label (str): Label returned when exposure is materially positive.
        negative_label (str): Label returned when exposure is materially negative.
        neutral_label (str): Label returned when exposure is near flat relative to gross exposure.
    
    Returns:
        str: Exposure regime label.
    
    Notes:
        The neutral band is proportional to gross exposure so thin chains are not over-classified into strong positive or negative regimes.
    """
    if value is None or gross is None or gross <= 0:
        return "UNKNOWN"
    if abs(value) <= gross * 0.05:
        return neutral_label
    if value > 0:
        return positive_label
    return negative_label


def summarize_greek_exposures(option_chain: pd.DataFrame):
    """
    Purpose:
        Aggregate chain-level Greek exposures into compact diagnostics used by market-state assembly.
    
    Context:
        This function converts contract-level Greeks into book-level exposure proxies. The signal engine later uses these diagnostics to reason about dealer positioning, convexity, and time-decay regimes.
    
    Inputs:
        option_chain (pd.DataFrame): Option-chain snapshot whose Greeks should be aggregated.
    
    Returns:
        dict: Aggregated exposure values and high-level vanna/charm regime labels.
    
    Notes:
        Exposures are weighted by open interest, so the output is a positioning proxy rather than a theoretical portfolio Greek for a tradable strategy.
    """
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

    df = option_chain
    oi_raw = df.get("OPEN_INT", df.get("openInterest"))
    if isinstance(oi_raw, pd.Series):
        oi = pd.to_numeric(oi_raw, errors="coerce").fillna(0.0)
    else:
        oi = pd.Series(0.0, index=df.index, dtype=float)

    greek_names = ["DELTA", "GAMMA", "THETA", "VEGA", "RHO", "VANNA", "CHARM"]
    greek_columns = {}
    missing_greek_columns = []
    for name in greek_names:
        raw = df.get(name)
        if isinstance(raw, pd.Series):
            greek_columns[name] = pd.to_numeric(raw, errors="coerce").fillna(0.0)
        else:
            greek_columns[name] = pd.Series(0.0, index=df.index, dtype=float)
            missing_greek_columns.append(name)

    weighted_exposures = {
        name: float((series * oi).sum())
        for name, series in greek_columns.items()
    }

    delta_exposure = weighted_exposures["DELTA"]
    gamma_exposure = weighted_exposures["GAMMA"]
    theta_exposure = weighted_exposures["THETA"]
    vega_exposure = weighted_exposures["VEGA"]
    rho_exposure = weighted_exposures["RHO"]
    vanna_exposure = weighted_exposures["VANNA"]
    charm_exposure = weighted_exposures["CHARM"]

    gross_vanna = float((greek_columns["VANNA"].abs() * oi).sum())
    gross_charm = float((greek_columns["CHARM"].abs() * oi).sum())

    return {
        "delta_exposure": round(delta_exposure, 2),
        "gamma_exposure_greeks": round(gamma_exposure, 6),
        "theta_exposure": round(theta_exposure, 2),
        "vega_exposure": round(vega_exposure, 2),
        "rho_exposure": round(rho_exposure, 2),
        "vanna_exposure": round(vanna_exposure, 4),
        "charm_exposure": round(charm_exposure, 4),
        "missing_greek_columns": missing_greek_columns,
        "greeks_data_warning": (
            "missing_greek_columns"
            if missing_greek_columns
            else None
        ),
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
