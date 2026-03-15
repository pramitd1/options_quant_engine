from __future__ import annotations

import pandas as pd

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
