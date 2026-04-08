from __future__ import annotations

import pandas as pd

from analytics.dealer_hedging_flow import dealer_hedging_flow
from tuning.runtime import temporary_parameter_pack


def test_dealer_flow_changes_with_gamma_when_delta_equal():
    base = pd.DataFrame(
        {
            "DELTA": [0.0, 0.0],
            "OPEN_INT": [1000.0, 1000.0],
        }
    )
    assert dealer_hedging_flow(base) == "SELL_FUTURES"

    gamma_up = pd.DataFrame(
        {
            "DELTA": [0.0, 0.0],
            "GAMMA": [0.8, 0.2],
            "OPEN_INT": [1000.0, 1000.0],
        }
    )
    assert dealer_hedging_flow(gamma_up) == "BUY_FUTURES"


def test_dealer_flow_fallback_when_gamma_missing():
    chain = pd.DataFrame(
        {
            "DELTA": [-0.3, -0.2],
            "OPEN_INT": [1000.0, 1000.0],
        }
    )
    assert dealer_hedging_flow(chain) == "SELL_FUTURES"


def test_dealer_flow_respects_runtime_policy_override():
    chain = pd.DataFrame(
        {
            "DELTA": [0.0, 0.0],
            "GAMMA": [0.4, 0.4],
            "CHARM": [0.0, 0.0],
            "OPEN_INT": [1000.0, 1000.0],
        }
    )
    with temporary_parameter_pack(
        "dealer_flow_override_test",
        overrides={
            "analytics.dealer_flow.gamma_weight": 0.0,
            "analytics.dealer_flow.charm_weight": 0.0,
        },
    ):
        out_flat = dealer_hedging_flow(chain)

    with temporary_parameter_pack(
        "dealer_flow_override_test",
        overrides={
            "analytics.dealer_flow.gamma_weight": 1.0,
            "analytics.dealer_flow.charm_weight": 0.0,
        },
    ):
        out_weighted = dealer_hedging_flow(chain)

    assert out_flat == "SELL_FUTURES"
    assert out_weighted == "BUY_FUTURES"
