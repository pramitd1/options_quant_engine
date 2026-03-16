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

import pandas as pd

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE


def _safe_float(value, default=None):
    """
    Purpose:
        Safely coerce an input to `float` while preserving a fallback.

    Context:
        Function inside the `greeks engine` module. The module sits in the analytics layer that turns option-chain and market-structure data into tradable features.

    Inputs:
        value (Any): Raw value supplied by the caller.
        default (Any): Fallback value used when the preferred path is unavailable.

    Returns:
        float: Parsed floating-point value or the fallback.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _norm_pdf(x: float) -> float:
    """
    Purpose:
        Evaluate the standard normal probability density used by the Black-Scholes Greeks formulas.
    
    Context:
        Internal helper in the Greeks engine. The closed-form Greeks calculation uses the standard normal density when converting option inputs into delta, gamma, vega, vanna, and charm.
    
    Inputs:
        x (float): Standard-normal z-score at which the density should be evaluated.
    
    Returns:
        float: Standard normal PDF evaluated at `x`.
    
    Notes:
        This helper keeps the analytical formula explicit and avoids re-deriving the density inside each Greek expression.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """
    Purpose:
        Evaluate the standard normal cumulative distribution function.

    Context:
        Function inside the `greeks engine` module. The module sits in the analytics layer that turns option-chain and market-structure data into tradable features.

    Inputs:
        x (float): Raw scalar input supplied by the caller.

    Returns:
        float: Cumulative probability for the supplied z-score.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(expiry_value, errors="coerce", utc=True)

    if pd.isna(parsed):
        parsed = pd.to_datetime(expiry_value, errors="coerce", dayfirst=True, utc=True)
        if pd.isna(parsed):
            return None

    now_ts = _coerce_valuation_timestamp(valuation_time)

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
            time_to_expiry_years=_parse_expiry_years(
                row_dict.get("EXPIRY_DT"),
                valuation_time=valuation_time,
            ),
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
