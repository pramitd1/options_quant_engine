from __future__ import annotations

import pandas as pd

from scripts.calibrate_ivhv_dealer_walkforward import _dealer_signal_components, _govern_recommendation


def test_dealer_signal_components_prefer_actual_exposures_over_proxies() -> None:
    frame = pd.DataFrame(
        {
            "dealer_hedging_flow": [0.4, -0.2],
            "market_gamma_exposure": [12.0, -8.0],
            "market_charm_exposure": [-1.5, 0.7],
            "hedging_flow_ratio": [0.9, 0.9],
            "upside_hedging_pressure": [1.0, 1.0],
            "downside_hedging_pressure": [0.0, 0.0],
            "gamma_vol_acceleration_score": [90.0, 90.0],
        }
    )

    flow, gamma_component, charm_component = _dealer_signal_components(frame)

    assert flow.tolist() == [0.4, -0.2]
    assert gamma_component.tolist() == [12.0, -8.0]
    assert charm_component.tolist() == [-1.5, 0.7]


def test_govern_recommendation_blocks_flat_lift_even_when_best_candidate_differs() -> None:
    split_df = pd.DataFrame(
        {
            "split_id": ["s1", "s2", "s1", "s2"],
            "gamma_weight": [0.5, 0.5, 0.1, 0.1],
            "charm_weight": [0.25, 0.25, 0.0, 0.0],
            "objective": [0.0, 0.0, 0.0, 0.0],
            "hit_rate": [0.5, 0.5, 0.5, 0.5],
            "coverage": [0.0, 0.0, 0.0, 0.0],
        }
    )
    summary_df = pd.DataFrame(
        {
            "gamma_weight": [0.5, 0.1],
            "charm_weight": [0.25, 0.0],
            "splits": [2, 2],
            "avg_objective": [0.0, 0.0],
            "avg_hit_rate": [0.5, 0.5],
            "avg_coverage": [0.0, 0.0],
            "min_objective": [0.0, 0.0],
            "max_objective": [0.0, 0.0],
        }
    )

    decision = _govern_recommendation(
        best_params={"gamma_weight": 0.5, "charm_weight": 0.25},
        baseline_params={"gamma_weight": 0.1, "charm_weight": 0.0},
        param_cols=["gamma_weight", "charm_weight"],
        split_df=split_df,
        summary_df=summary_df,
        minimum_trading_days=20,
        observed_trading_days=8,
        minimum_completed_signals=500,
        observed_completed_signals=80,
        minimum_splits=4,
        minimum_objective_lift=0.005,
        override_prefix="analytics.dealer_flow",
    )

    assert decision.status == "BLOCKED"
    assert "insufficient_trading_days" in decision.reasons
    assert "objective_lift_below_floor" in decision.reasons
    assert "objective_lift_confidence_not_positive" in decision.reasons
    assert decision.recommended_overrides == {}