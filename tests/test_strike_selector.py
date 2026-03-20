from __future__ import annotations

import pandas as pd

from config.policy_resolver import temporary_parameter_pack
from strategy.strike_selector import rank_strike_candidates, select_best_strike


def _option_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strikePrice": 21950,
                "OPTION_TYP": "CE",
                "lastPrice": 132.0,
                "totalTradedVolume": 4200,
                "openInterest": 95000,
                "impliedVolatility": 17.0,
            },
            {
                "strikePrice": 22000,
                "OPTION_TYP": "CE",
                "lastPrice": 118.0,
                "totalTradedVolume": 6200,
                "openInterest": 120000,
                "impliedVolatility": 16.0,
            },
            {
                "strikePrice": 22050,
                "OPTION_TYP": "CE",
                "lastPrice": 89.0,
                "totalTradedVolume": 5800,
                "openInterest": 112000,
                "impliedVolatility": 18.0,
            },
        ]
    )


def test_rank_strike_candidates_returns_deterministic_sorted_scores():
    ranked = rank_strike_candidates(
        option_chain=_option_chain(),
        direction="CALL",
        spot=22010.0,
        lot_size=50,
        max_capital=10000,
    )
    repeated = rank_strike_candidates(
        option_chain=_option_chain(),
        direction="CALL",
        spot=22010.0,
        lot_size=50,
        max_capital=10000,
    )

    assert [row["strike"] for row in ranked] == [row["strike"] for row in repeated]
    assert set(row["strike"] for row in ranked[:3]) == {21950, 22000, 22050}
    assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"]


def test_select_best_strike_keeps_hook_adjustments():
    strike, ranked = select_best_strike(
        option_chain=_option_chain(),
        direction="CALL",
        spot=22010.0,
        candidate_score_hook=lambda row, payload: {
            "score_adjustment": 3 if payload["strike"] == 22050 else 0,
            "hook_tag": "applied",
        },
    )

    assert strike == 22050
    assert ranked[0]["score_breakdown"]["option_efficiency_score_adjustment"] == 3
    assert ranked[0]["hook_tag"] == "applied"


def test_moneyness_score_interpolates_smoothly_in_continuous_mode():
    from config.strike_selection_policy import get_strike_selection_score_config

    cfg = get_strike_selection_score_config()
    atm_dist = float(cfg["atm_distance_pct"])
    far_dist = float(cfg["far_distance_pct"])
    mid_dist_pct = (atm_dist + far_dist) / 2.0
    spot = 22000.0
    atm_strike = spot * (1 + atm_dist / 100.0)
    mid_strike = spot * (1 + mid_dist_pct / 100.0)
    far_strike = spot * (1 + far_dist / 100.0)

    def _score(strike: float):
        ranked = rank_strike_candidates(
            option_chain=pd.DataFrame(
                [
                    {
                        "strikePrice": strike,
                        "OPTION_TYP": "CE",
                        "lastPrice": 100.0,
                        "totalTradedVolume": 5000,
                        "openInterest": 100000,
                        "impliedVolatility": 18.0,
                    }
                ]
            ),
            direction="CALL",
            spot=spot,
        )
        return ranked[0]["score_breakdown"]["moneyness_score"] if ranked else None

    with temporary_parameter_pack(
        "cont_strike_test",
        overrides={"strike_selection.scoring.strike_scoring_mode": "continuous"},
    ):
        score_atm = _score(atm_strike)
        score_mid = _score(mid_strike)
        score_far = _score(far_strike)

    assert score_atm is not None and score_mid is not None and score_far is not None
    assert score_far <= score_mid <= score_atm, (
        f"Expected smooth decay: far={score_far} <= mid={score_mid} <= atm={score_atm}"
    )


def test_trade_strength_continuous_mode_includes_breakdown_scores():
    from strategy.trade_strength import compute_trade_strength

    total_score, breakdown = compute_trade_strength(
        direction="CALL",
        flow_signal_value="BULLISH_FLOW",
        smart_money_signal_value="BULLISH_FLOW",
        gamma_event="NONE",
        dealer_pos="NET_LONG",
        vol_regime="LOW_VOL",
        void_signal=False,
        vacuum_state={},
        spot_vs_flip="ABOVE_FLIP",
        hedging_bias="UPSIDE_ACCELERATION",
        gamma_regime="NEGATIVE_GAMMA",
        intraday_gamma_state={},
        support_wall=22800.0,
        resistance_wall=23500.0,
        spot=23050.0,
        flip_distance_pct=0.3,
        scoring_mode="continuous",
    )

    assert "spot_vs_flip_score" in breakdown
    assert "wall_proximity_score" in breakdown
    assert isinstance(total_score, (int, float))
