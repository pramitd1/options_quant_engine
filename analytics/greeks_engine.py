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
import re
from functools import lru_cache

import pandas as pd

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE
from utils.numerics import safe_float as _safe_float  # noqa: F401
from utils.math_helpers import norm_pdf as _norm_pdf, norm_cdf as _norm_cdf  # noqa: F401


# ---------------------------------------------------------------------------
# Newton-Raphson implied-vol estimation (fallback when provider IV is 0/None)
# ---------------------------------------------------------------------------

def _bs_price_for_iv(spot, strike, t, sigma, option_type, r=RISK_FREE_RATE):
    """Black-Scholes European price for IV estimation."""
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _bs_vega_for_iv(spot, strike, t, sigma, r=RISK_FREE_RATE):
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return spot * math.sqrt(t) * _norm_pdf(d1)


def estimate_iv_from_price(market_price, spot, strike, t, option_type,
                           r=RISK_FREE_RATE, tol=1e-6, max_iter=50):
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
        price = _bs_price_for_iv(spot, strike, t, sigma, option_type, r)
        vega = _bs_vega_for_iv(spot, strike, t, sigma, r)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        if sigma <= 0:
            sigma = 0.001
        if sigma > 5.0:
            return 500.0
        if abs(price - market_price) < tol:
            break
    if sigma > 5.0:
        return 500.0
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
            return max(float(match.group(1)) / 365.0, 1.0 / (365.0 * 24.0))
        parsed = _parse_expiry_timestamp_cached(expiry_value)
    else:
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)

    if pd.isna(parsed):
        parsed = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
        if pd.isna(parsed):
            return None

    now_ts = _coerce_valuation_timestamp(valuation_time)

    time_years = (parsed - now_ts).total_seconds() / (365.0 * 24 * 3600.0)
    return max(float(time_years), 1.0 / (365.0 * 24.0))


@lru_cache(maxsize=2048)
def _parse_expiry_timestamp_cached(expiry_value: str):
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

    for col in ["strikePrice", "IV", "impliedVolatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pre-coerce valuation timestamp once (avoids re-parsing per row)
    coerced_valuation_time = _coerce_valuation_timestamp(valuation_time)

    # Cache expiry→TTE lookups (most chains share a single expiry)
    _tte_cache: dict = {}

    def _cached_parse_expiry_years(expiry_value):
        key = expiry_value
        if key not in _tte_cache:
            _tte_cache[key] = _parse_expiry_years(expiry_value, valuation_time=coerced_valuation_time)
        return _tte_cache[key]

    deltas = []
    gammas = []
    thetas = []
    vegas = []
    rhos = []
    vannas = []
    charms = []
    ttes = []
    ivs = []

    for row in df.itertuples(index=False):
        iv = getattr(row, "IV", None)
        if iv is None:
            iv = getattr(row, "impliedVolatility", None)

        expiry_value = getattr(row, "EXPIRY_DT", None)
        strike_value = getattr(row, "strikePrice", None)
        option_type_value = getattr(row, "OPTION_TYP", "")

        # Fallback: estimate IV from market price when provider gives 0/None
        tte_for_iv = _cached_parse_expiry_years(expiry_value)
        if (iv is None or (isinstance(iv, (int, float)) and iv <= 0)) and spot is not None:
            market_price = getattr(row, "lastPrice", None)
            if market_price is None:
                market_price = getattr(row, "LAST_PRICE", None)
            estimated = estimate_iv_from_price(
                market_price,
                spot,
                strike_value,
                tte_for_iv,
                option_type_value,
            )
            if estimated > 0:
                iv = estimated

        greeks = compute_option_greeks(
            spot=spot,
            strike=strike_value,
            time_to_expiry_years=tte_for_iv,
            volatility_pct=iv,
            option_type=option_type_value,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        if greeks is None:
            deltas.append(getattr(row, "DELTA", None))
            gammas.append(getattr(row, "GAMMA", None))
            thetas.append(getattr(row, "THETA", None))
            vegas.append(getattr(row, "VEGA", None))
            rhos.append(getattr(row, "RHO", None))
            vannas.append(getattr(row, "VANNA", None))
            charms.append(getattr(row, "CHARM", None))
            ttes.append(None)
            ivs.append(getattr(row, "IV", getattr(row, "impliedVolatility", None)))
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

    df = option_chain.copy()
    oi = pd.to_numeric(df.get("OPEN_INT", df.get("openInterest")), errors="coerce").fillna(0.0)

    def _series(name):
        """
        Purpose:
            Return a numeric series for the requested Greek or sensitivity column.

        Context:
            Internal helper inside the exposure-aggregation path. It keeps the
            Greeks rollup compact while ensuring every requested column is
            coerced into the same numeric representation before weighted sums
            are computed.

        Inputs:
            name (Any): Column name for the requested Greek or derived sensitivity.

        Returns:
            pd.Series: Numeric series aligned with the option-chain rows and
            defaulted to zero where the provider omitted values.

        Notes:
            Missing or malformed provider values are treated as zero so the
            aggregate exposure calculation stays robust across data sources.
        """
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
