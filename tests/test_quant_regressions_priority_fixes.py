from __future__ import annotations

import math

import pandas as pd
import pytest

from analytics.dealer_gamma_path import simulate_gamma_path
from analytics.gamma_exposure import approximate_gamma, calculate_gamma_exposure, gamma_signal
from analytics.greeks_engine import _bs_price_for_iv, compute_option_greeks, estimate_iv_from_price
from strategy.enhanced_strike_scoring import (
    compute_dealer_pressure,
    compute_enhanced_strike_scores,
    compute_premium_efficiency,
)


def test_dealer_gamma_path_uses_unstruck_weighting_consistent_with_market_gamma_map():
    option_chain = pd.DataFrame(
        {
            "strikePrice": [100.0, 200.0],
            "openInterest": [1.0, 1.0],
            "GAMMA": [1.0, 1.0],
            "OPTION_TYP": ["CE", "CE"],
        }
    )

    prices, curve = simulate_gamma_path(option_chain, spot=150.0, step=50, range_points=100)
    idx = int(list(prices).index(150.0))

    # simulate_gamma_path infers step from strikes (100 here), so width=100 and
    # each strike contributes exp(-0.5 * 0.5^2) at the center price.
    expected = 2.0 * math.exp(-0.125)
    assert curve[idx] == pytest.approx(expected, rel=1e-6)
    assert curve[idx] < 5.0


def test_estimate_iv_from_price_recovers_dividend_adjusted_sigma():
    spot = 100.0
    strike = 100.0
    t = 0.5
    sigma = 0.20
    r = 0.05
    q = 0.03

    market_price = _bs_price_for_iv(spot, strike, t, sigma, "CE", r=r, q=q)
    estimated_iv_pct = estimate_iv_from_price(
        market_price,
        spot,
        strike,
        t,
        "CE",
        r=r,
        q=q,
    )

    assert estimated_iv_pct == pytest.approx(20.0, abs=0.15)


def test_approximate_gamma_uses_moneyness_scaled_distance():
    near_small = approximate_gamma(101.0, 100.0)
    near_large = approximate_gamma(10100.0, 10000.0)

    assert near_small == pytest.approx(near_large, rel=1e-12)


def test_gamma_signal_uses_dealer_side_proxy_not_call_put_cancellation():
    chain = pd.DataFrame(
        {
            "strikePrice": [100.0, 100.0],
            "openInterest": [1000.0, 1000.0],
            "GAMMA": [0.2, 0.2],
            "OPTION_TYP": ["CE", "PE"],
        }
    )

    assert gamma_signal(chain, spot=100.0) == "LONG_GAMMA"


def test_calculate_gamma_exposure_fallback_is_scale_invariant():
    chain_small = pd.DataFrame(
        {
            "strikePrice": [99.0, 100.0, 101.0],
            "openInterest": [100.0, 200.0, 120.0],
            "OPTION_TYP": ["CE", "PE", "CE"],
        }
    )
    chain_large = pd.DataFrame(
        {
            "strikePrice": [s * 100.0 for s in [99.0, 100.0, 101.0]],
            "openInterest": [100.0, 200.0, 120.0],
            "OPTION_TYP": ["CE", "PE", "CE"],
        }
    )

    e_small = calculate_gamma_exposure(chain_small, spot=100.0)
    e_large = calculate_gamma_exposure(chain_large, spot=10000.0)

    assert e_small == pytest.approx(e_large, rel=1e-12)


def test_premium_efficiency_penalizes_zero_premium_rows():
    rows = pd.DataFrame(
        {
            "lastPrice": [0.0, 50.0],
            "strikePrice": [100.0, 100.0],
        }
    )

    scores = compute_premium_efficiency(
        rows,
        spot=100.0,
        atm_iv=20.0,
        days_to_expiry=7.0,
        expected_move=10.0,
    )

    assert float(scores.iloc[0]) == pytest.approx(0.0)
    assert float(scores.iloc[1]) == pytest.approx(0.5)


def test_compute_enhanced_strike_scores_handles_invalid_spot_without_crash():
    rows = pd.DataFrame(
        {
            "strikePrice": [22900.0, 23000.0, 23100.0],
            "lastPrice": [140.0, 110.0, 85.0],
            "openInterest": [15000.0, 20000.0, 14000.0],
            "totalTradedVolume": [2000.0, 3000.0, 1500.0],
            "OPTION_TYP": ["CE", "CE", "CE"],
        }
    )

    out = compute_enhanced_strike_scores(
        rows,
        spot=0.0,
        direction="CALL",
        atm_iv=18.0,
        days_to_expiry=3.0,
    )

    assert len(out) == len(rows)
    assert "enhanced_strike_score" in out.columns


def test_compute_dealer_pressure_uses_flip_context_and_gamma_intensity():
    strikes = pd.Series([23000.0, 23100.0])

    baseline = compute_dealer_pressure(
        strikes,
        gamma_regime="NEUTRAL_GAMMA",
        spot_vs_flip="ABOVE_FLIP",
        dealer_hedging_bias="NEUTRAL",
        gamma_flip_distance_pct=5.0,
        dealer_gamma_exposure=10.0,
    )
    stressed = compute_dealer_pressure(
        strikes,
        gamma_regime="NEGATIVE_GAMMA",
        spot_vs_flip="AT_FLIP",
        dealer_hedging_bias="DOWNSIDE_HEDGING_ACCELERATION",
        gamma_flip_distance_pct=0.1,
        dealer_gamma_exposure=2_000_000.0,
    )

    assert float(stressed.iloc[0]) > float(baseline.iloc[0])


def test_greeks_monotonic_gamma_as_tte_shrinks():
    g_long = compute_option_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry_years=5.0 / 365.0,
        volatility_pct=20.0,
        option_type="CE",
    )
    g_short = compute_option_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry_years=1.0 / 365.0,
        volatility_pct=20.0,
        option_type="CE",
    )

    assert g_long is not None
    assert g_short is not None
    assert g_short["GAMMA"] > g_long["GAMMA"]


def test_greeks_black_scholes_reference_values():
    # ATM call benchmark: S=K=100, T=0.5y, sigma=20%, r=0, q=0
    greeks = compute_option_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry_years=0.5,
        volatility_pct=20.0,
        option_type="CE",
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )

    assert greeks is not None
    assert greeks["DELTA"] == pytest.approx(0.5282, abs=1e-3)
    assert greeks["GAMMA"] == pytest.approx(0.0281, abs=1e-3)
    assert greeks["VEGA"] == pytest.approx(0.2814, abs=2e-3)


def test_greeks_ultra_short_tte_policy_behavior():
    # Default policy skips Greeks below 60 seconds to expiry.
    greeks = compute_option_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry_years=30.0 / (365.0 * 24.0 * 3600.0),
        volatility_pct=20.0,
        option_type="CE",
    )
    assert greeks is None
