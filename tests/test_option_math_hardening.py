from __future__ import annotations

import pandas as pd

from analytics.greeks_engine import summarize_greek_exposures
from analytics.gamma_walls import classify_walls, detect_gamma_walls
from analytics.volatility_surface import compute_risk_reversal


def test_gamma_walls_use_signed_gamma_exposure_when_available():
    chain = pd.DataFrame(
        {
            "STRIKE_PR": [100, 100, 105, 105, 95, 95],
            "OPTION_TYP": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "OPEN_INT": [1000, 200, 900, 300, 300, 1200],
            "GAMMA": [0.02, 0.02, 0.03, 0.03, 0.01, 0.01],
        }
    )

    walls = classify_walls(chain)

    # Highest positive signed gamma exposure at 105, most negative at 95.
    assert walls["resistance_wall"] == 105.0
    assert walls["support_wall"] == 95.0

    top = detect_gamma_walls(chain, top_n=2)
    assert top[0] == 105.0
    assert 105.0 in top


def test_gamma_walls_fall_back_to_oi_when_gamma_unavailable():
    chain = pd.DataFrame(
        {
            "STRIKE_PR": [100, 100, 105, 105],
            "OPTION_TYP": ["CE", "PE", "CE", "PE"],
            "OPEN_INT": [900, 1200, 1400, 800],
        }
    )

    walls = classify_walls(chain)

    assert walls["resistance_wall"] == 105.0
    assert walls["support_wall"] == 100.0


def test_risk_reversal_prefers_delta_target_over_fixed_moneyness():
    # Construct a chain where a fixed 1.5% moneyness rule would likely choose
    # 10100 CE, but delta-target logic should pick 10200 CE for ~25d.
    chain = pd.DataFrame(
        {
            "STRIKE_PR": [9800, 9900, 10000, 10100, 10200, 10300, 10400],
            "OPTION_TYP": ["PE", "PE", "CE", "CE", "CE", "CE", "CE"],
            "IV": [22.0, 21.0, 20.0, 19.5, 18.0, 17.5, 17.0],
            "TTE": [30 / 365.0] * 7,
            "EXPIRY_DT": ["2026-04-30"] * 7,
        }
    )

    rr = compute_risk_reversal(chain, spot=10000.0, delta_target=0.25)

    # Delta-target logic should move call-side selection farther OTM than a
    # naive near-ATM fixed-moneyness proxy.
    assert rr["put_iv_25d"] == 22.0
    assert rr["call_iv_25d"] <= 18.0
    assert rr["rr_value"] >= 4.0
    assert rr["rr_regime"] == "PUT_SKEW"


def test_summarize_greek_exposures_handles_string_inputs_and_regimes():
    chain = pd.DataFrame(
        {
            "OPEN_INT": ["100", "200"],
            "DELTA": ["0.5", "-0.2"],
            "GAMMA": ["0.01", "0.015"],
            "THETA": ["-1.5", "-0.5"],
            "VEGA": ["2.0", "1.0"],
            "RHO": ["0.4", "-0.1"],
            "VANNA": ["0.04", "-0.01"],
            "CHARM": ["0.03", "-0.005"],
        }
    )

    summary = summarize_greek_exposures(chain)

    assert summary["delta_exposure"] == 10.0
    assert summary["gamma_exposure_greeks"] == 4.0
    assert summary["theta_exposure"] == -250.0
    assert summary["vega_exposure"] == 400.0
    assert summary["rho_exposure"] == 20.0
    assert summary["vanna_exposure"] == 2.0
    assert summary["charm_exposure"] == 2.0
    assert summary["vanna_regime"] == "POSITIVE_VANNA"
    assert summary["charm_regime"] == "POSITIVE_CHARM"
