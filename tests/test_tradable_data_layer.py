from __future__ import annotations

import pandas as pd

from data.tradable_data_layer import evaluate_tradable_data_layer
from config.policy_resolver import temporary_parameter_pack


def test_tradable_layer_flags_crossed_locked_quotes() -> None:
    df = pd.DataFrame(
        [
            {"strikePrice": 22000, "OPTION_TYP": "CE", "lastPrice": 100, "bidPrice": 101, "askPrice": 100},
            {"strikePrice": 22000, "OPTION_TYP": "PE", "lastPrice": 110, "bidPrice": 109, "askPrice": 111},
            {"strikePrice": 22100, "OPTION_TYP": "CE", "lastPrice": 95, "bidPrice": 95, "askPrice": 95},
            {"strikePrice": 22100, "OPTION_TYP": "PE", "lastPrice": 120, "bidPrice": 119, "askPrice": 121},
            {"strikePrice": 22200, "OPTION_TYP": "CE", "lastPrice": 80, "bidPrice": 79, "askPrice": 81},
            {"strikePrice": 22200, "OPTION_TYP": "PE", "lastPrice": 130, "bidPrice": 129, "askPrice": 131},
            {"strikePrice": 22300, "OPTION_TYP": "CE", "lastPrice": 70, "bidPrice": 69, "askPrice": 71},
            {"strikePrice": 22300, "OPTION_TYP": "PE", "lastPrice": 140, "bidPrice": 139, "askPrice": 141},
            {"strikePrice": 22400, "OPTION_TYP": "CE", "lastPrice": 60, "bidPrice": 59, "askPrice": 61},
            {"strikePrice": 22400, "OPTION_TYP": "PE", "lastPrice": 150, "bidPrice": 149, "askPrice": 151},
            {"strikePrice": 22500, "OPTION_TYP": "CE", "lastPrice": 50, "bidPrice": 49, "askPrice": 51},
            {"strikePrice": 22500, "OPTION_TYP": "PE", "lastPrice": 160, "bidPrice": 159, "askPrice": 161},
            {"strikePrice": 22600, "OPTION_TYP": "CE", "lastPrice": 40, "bidPrice": 39, "askPrice": 41},
            {"strikePrice": 22600, "OPTION_TYP": "PE", "lastPrice": 170, "bidPrice": 169, "askPrice": 171},
            {"strikePrice": 22700, "OPTION_TYP": "CE", "lastPrice": 30, "bidPrice": 29, "askPrice": 31},
            {"strikePrice": 22700, "OPTION_TYP": "PE", "lastPrice": 180, "bidPrice": 179, "askPrice": 181},
            {"strikePrice": 22800, "OPTION_TYP": "CE", "lastPrice": 20, "bidPrice": 19, "askPrice": 21},
            {"strikePrice": 22800, "OPTION_TYP": "PE", "lastPrice": 190, "bidPrice": 189, "askPrice": 191},
            {"strikePrice": 22900, "OPTION_TYP": "CE", "lastPrice": 10, "bidPrice": 9, "askPrice": 11},
            {"strikePrice": 22900, "OPTION_TYP": "PE", "lastPrice": 200, "bidPrice": 199, "askPrice": 201},
            {"strikePrice": 23000, "OPTION_TYP": "CE", "lastPrice": 8, "bidPrice": 7, "askPrice": 9},
            {"strikePrice": 23000, "OPTION_TYP": "PE", "lastPrice": 210, "bidPrice": 209, "askPrice": 211},
            {"strikePrice": 23100, "OPTION_TYP": "CE", "lastPrice": 6, "bidPrice": 5, "askPrice": 7},
            {"strikePrice": 23100, "OPTION_TYP": "PE", "lastPrice": 220, "bidPrice": 219, "askPrice": 221},
            {"strikePrice": 23200, "OPTION_TYP": "CE", "lastPrice": 4, "bidPrice": 3, "askPrice": 5},
            {"strikePrice": 23200, "OPTION_TYP": "PE", "lastPrice": 230, "bidPrice": 229, "askPrice": 231},
            {"strikePrice": 23300, "OPTION_TYP": "CE", "lastPrice": 3, "bidPrice": 2, "askPrice": 4},
            {"strikePrice": 23300, "OPTION_TYP": "PE", "lastPrice": 240, "bidPrice": 239, "askPrice": 241},
            {"strikePrice": 23400, "OPTION_TYP": "CE", "lastPrice": 2, "bidPrice": 1, "askPrice": 3},
            {"strikePrice": 23400, "OPTION_TYP": "PE", "lastPrice": 250, "bidPrice": 249, "askPrice": 251},
            {"strikePrice": 23500, "OPTION_TYP": "CE", "lastPrice": 1, "bidPrice": 0.5, "askPrice": 1.5},
            {"strikePrice": 23500, "OPTION_TYP": "PE", "lastPrice": 260, "bidPrice": 259, "askPrice": 261},
            {"strikePrice": 23600, "OPTION_TYP": "CE", "lastPrice": 0.8, "bidPrice": 0.4, "askPrice": 1.2},
            {"strikePrice": 23600, "OPTION_TYP": "PE", "lastPrice": 270, "bidPrice": 269, "askPrice": 271},
            {"strikePrice": 23700, "OPTION_TYP": "CE", "lastPrice": 0.5, "bidPrice": 0.2, "askPrice": 0.8},
            {"strikePrice": 23700, "OPTION_TYP": "PE", "lastPrice": 280, "bidPrice": 279, "askPrice": 281},
            {"strikePrice": 23800, "OPTION_TYP": "CE", "lastPrice": 0.3, "bidPrice": 0.1, "askPrice": 0.6},
            {"strikePrice": 23800, "OPTION_TYP": "PE", "lastPrice": 290, "bidPrice": 289, "askPrice": 291},
            {"strikePrice": 23900, "OPTION_TYP": "CE", "lastPrice": 0.2, "bidPrice": 0.05, "askPrice": 0.4},
            {"strikePrice": 23900, "OPTION_TYP": "PE", "lastPrice": 300, "bidPrice": 299, "askPrice": 301},
        ]
    )

    with temporary_parameter_pack(
        "tradable_outlier_rejection_test",
        overrides={"tradable_data_layer.execution.max_outlier_ratio": 0.05},
    ):
        out = evaluate_tradable_data_layer(df)
    assert out["crossed_locked"]["crossed_or_locked_rows"] >= 2
    assert out["crossed_locked"]["crossed_or_locked_ratio"] > 0.0


def test_tradable_layer_rejects_extreme_outliers_for_execution_suggestion() -> None:
    rows = []
    for strike in range(22000, 23000, 25):
        rows.append({"strikePrice": strike, "OPTION_TYP": "CE", "lastPrice": 100, "bidPrice": 99, "askPrice": 101})
        rows.append({"strikePrice": strike, "OPTION_TYP": "PE", "lastPrice": 110, "bidPrice": 109, "askPrice": 111})

    # Inject multiple outliers so outlier_ratio crosses default threshold.
    rows.extend(
        [
            {"strikePrice": 23050, "OPTION_TYP": "CE", "lastPrice": 9000, "bidPrice": 8990, "askPrice": 9010},
            {"strikePrice": 23100, "OPTION_TYP": "PE", "lastPrice": 11000, "bidPrice": 10990, "askPrice": 11010},
            {"strikePrice": 23150, "OPTION_TYP": "CE", "lastPrice": 13000, "bidPrice": 12990, "askPrice": 13010},
            {"strikePrice": 23200, "OPTION_TYP": "PE", "lastPrice": 15000, "bidPrice": 14990, "askPrice": 15010},
            {"strikePrice": 23250, "OPTION_TYP": "CE", "lastPrice": 17000, "bidPrice": 16990, "askPrice": 17010},
            {"strikePrice": 23300, "OPTION_TYP": "PE", "lastPrice": 19000, "bidPrice": 18990, "askPrice": 19010},
            {"strikePrice": 23350, "OPTION_TYP": "CE", "lastPrice": 21000, "bidPrice": 20990, "askPrice": 21010},
            {"strikePrice": 23400, "OPTION_TYP": "PE", "lastPrice": 23000, "bidPrice": 22990, "askPrice": 23010},
            {"strikePrice": 23450, "OPTION_TYP": "CE", "lastPrice": 25000, "bidPrice": 24990, "askPrice": 25010},
        ]
    )

    df = pd.DataFrame(rows)
    with temporary_parameter_pack(
        "tradable_outlier_rejection_test",
        overrides={"tradable_data_layer.execution.max_outlier_ratio": 0.05},
    ):
        out = evaluate_tradable_data_layer(df)

    assert out["outlier_rejection"]["outlier_rows"] > 0
    assert out["execution_suggestion_usable"] is False
    assert "outlier_ratio_high" in out["reasons"]
