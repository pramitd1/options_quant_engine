"""Heston stochastic-volatility pricing approximations for research.

This module intentionally does not replace the production Black-Scholes Greek
engine. It provides a compact, robust Heston-style approximation suitable for
diagnostics, calibration experiments, and signal-evaluation features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from config.settings import DIVIDEND_YIELD, RISK_FREE_RATE
from utils.math_helpers import norm_cdf, norm_pdf


OptionType = Literal["CE", "PE"]


@dataclass(frozen=True)
class HestonParams:
    """Parameter bundle for the Heston variance process.

    Variance dynamics:

    ``dv_t = kappa * (theta - v_t) dt + vol_of_vol * sqrt(v_t) dW_v``

    with spot/variance Brownian correlation ``rho`` and initial variance ``v0``.
    All variance fields are annualized variance units, not volatility points.
    """

    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    v0: float

    def clipped(self) -> "HestonParams":
        """Return parameters clipped to conservative research bounds."""

        return HestonParams(
            kappa=float(min(max(self.kappa, 0.10), 8.0)),
            theta=float(min(max(self.theta, 0.0001), 1.0)),
            vol_of_vol=float(min(max(self.vol_of_vol, 0.01), 3.0)),
            rho=float(min(max(self.rho, -0.95), 0.25)),
            v0=float(min(max(self.v0, 0.0001), 1.0)),
        )


DEFAULT_HESTON_PARAMS = HestonParams(
    kappa=1.50,
    theta=0.04,
    vol_of_vol=0.60,
    rho=-0.45,
    v0=0.04,
)


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def normalize_option_type(value) -> str | None:
    token = str(value or "").upper().strip()
    if token in {"CE", "CALL", "C"}:
        return "CE"
    if token in {"PE", "PUT", "P"}:
        return "PE"
    return None


def black_scholes_price(
    *,
    spot,
    strike,
    time_to_expiry_years,
    volatility,
    option_type,
    risk_free_rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float | None:
    """Return a Black-Scholes price using decimal annualized volatility."""

    spot = _safe_float(spot, None)
    strike = _safe_float(strike, None)
    t = _safe_float(time_to_expiry_years, None)
    sigma = _safe_float(volatility, None)
    opt_type = normalize_option_type(option_type)
    if spot is None or strike is None or t is None or sigma is None or opt_type is None:
        return None
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0) if opt_type == "CE" else max(strike - spot, 0.0)
        return float(intrinsic)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * t) / (
        sigma * sqrt_t
    )
    d2 = d1 - sigma * sqrt_t
    discount_q = math.exp(-dividend_yield * t)
    discount_r = math.exp(-risk_free_rate * t)
    if opt_type == "CE":
        return float(spot * discount_q * norm_cdf(d1) - strike * discount_r * norm_cdf(d2))
    return float(strike * discount_r * norm_cdf(-d2) - spot * discount_q * norm_cdf(-d1))


def black_scholes_delta(
    *,
    spot,
    strike,
    time_to_expiry_years,
    volatility,
    option_type,
    risk_free_rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float | None:
    """Return Black-Scholes delta for comparison diagnostics."""

    spot = _safe_float(spot, None)
    strike = _safe_float(strike, None)
    t = _safe_float(time_to_expiry_years, None)
    sigma = _safe_float(volatility, None)
    opt_type = normalize_option_type(option_type)
    if spot is None or strike is None or t is None or sigma is None or opt_type is None:
        return None
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        if opt_type == "CE":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * t) / (
        sigma * math.sqrt(t)
    )
    if opt_type == "CE":
        return math.exp(-dividend_yield * t) * norm_cdf(d1)
    return math.exp(-dividend_yield * t) * (norm_cdf(d1) - 1.0)


def heston_forward_variance(params: HestonParams, horizon_years: float) -> float:
    """Return the expected variance at a forward horizon under Heston."""

    p = params.clipped()
    horizon = max(float(horizon_years or 0.0), 0.0)
    return float(p.theta + (p.v0 - p.theta) * math.exp(-p.kappa * horizon))


def heston_average_variance(params: HestonParams, time_to_expiry_years: float) -> float:
    """Return the average variance over the option life under mean reversion."""

    p = params.clipped()
    t = max(float(time_to_expiry_years or 0.0), 1e-6)
    mean_reversion_weight = (1.0 - math.exp(-p.kappa * t)) / max(p.kappa * t, 1e-8)
    avg_var = p.theta + (p.v0 - p.theta) * mean_reversion_weight
    return float(max(avg_var, 1e-8))


def heston_implied_vol_proxy(
    *,
    spot,
    strike,
    time_to_expiry_years,
    params: HestonParams,
) -> float | None:
    """Approximate the Heston local smile as a volatility proxy.

    The approximation preserves the Heston mean-reverting variance term while
    using ``rho`` and ``vol_of_vol`` to bend the smile. Negative ``rho`` raises
    downside/put-wing volatility and lowers upside/call-wing volatility.
    """

    spot = _safe_float(spot, None)
    strike = _safe_float(strike, None)
    t = _safe_float(time_to_expiry_years, None)
    if spot is None or strike is None or t is None or spot <= 0 or strike <= 0 or t <= 0:
        return None

    p = params.clipped()
    avg_var = heston_average_variance(p, t)
    log_moneyness = math.log(strike / spot)
    skew_term = p.rho * p.vol_of_vol * math.sqrt(max(t, 1e-6)) * log_moneyness
    curvature_term = 0.18 * (p.vol_of_vol ** 2) * t * (log_moneyness ** 2)
    variance_proxy = avg_var * math.exp(skew_term + curvature_term)
    volatility = math.sqrt(max(variance_proxy, 1e-8))
    return float(min(max(volatility, 0.01), 3.0))


def heston_price(
    *,
    spot,
    strike,
    time_to_expiry_years,
    option_type,
    params: HestonParams,
    risk_free_rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float | None:
    """Return a Heston-style option price using the robust smile proxy."""

    volatility = heston_implied_vol_proxy(
        spot=spot,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        params=params,
    )
    if volatility is None:
        return None
    return black_scholes_price(
        spot=spot,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        volatility=volatility,
        option_type=option_type,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )


def heston_delta(
    *,
    spot,
    strike,
    time_to_expiry_years,
    option_type,
    params: HestonParams,
) -> float | None:
    """Finite-difference delta for Heston comparison diagnostics."""

    spot_value = _safe_float(spot, None)
    if spot_value is None or spot_value <= 0:
        return None
    bump = max(spot_value * 0.001, 0.50)
    up = heston_price(
        spot=spot_value + bump,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        option_type=option_type,
        params=params,
    )
    down = heston_price(
        spot=max(spot_value - bump, 1e-6),
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        option_type=option_type,
        params=params,
    )
    if up is None or down is None:
        return None
    return float((up - down) / (2.0 * bump))


def heston_gamma(
    *,
    spot,
    strike,
    time_to_expiry_years,
    option_type,
    params: HestonParams,
) -> float | None:
    """Finite-difference gamma for Heston comparison diagnostics."""

    spot_value = _safe_float(spot, None)
    if spot_value is None or spot_value <= 0:
        return None
    bump = max(spot_value * 0.001, 0.50)
    center = heston_price(
        spot=spot_value,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        option_type=option_type,
        params=params,
    )
    up = heston_price(
        spot=spot_value + bump,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        option_type=option_type,
        params=params,
    )
    down = heston_price(
        spot=max(spot_value - bump, 1e-6),
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        option_type=option_type,
        params=params,
    )
    if center is None or up is None or down is None:
        return None
    return float((up - (2.0 * center) + down) / (bump ** 2))


def heston_greek_snapshot(
    *,
    spot,
    strike,
    time_to_expiry_years,
    option_type,
    params: HestonParams,
) -> dict[str, float | None]:
    """Return price, delta, and gamma diagnostics from the Heston layer."""

    return {
        "price": heston_price(
            spot=spot,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            option_type=option_type,
            params=params,
        ),
        "delta": heston_delta(
            spot=spot,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            option_type=option_type,
            params=params,
        ),
        "gamma": heston_gamma(
            spot=spot,
            strike=strike,
            time_to_expiry_years=time_to_expiry_years,
            option_type=option_type,
            params=params,
        ),
    }
