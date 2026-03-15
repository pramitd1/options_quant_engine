from __future__ import annotations

import unittest

import pandas as pd

from data.option_chain_validation import validate_option_chain


class OptionChainValidationTests(unittest.TestCase):
    def test_accepts_alias_columns_and_reports_pairing(self):
        option_chain = pd.DataFrame(
            [
                {"STRIKE_PR": 22000, "OPTION_TYP": "CE", "LAST_PRICE": 110, "OPEN_INT": 1200, "IV": 18.5, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22000, "OPTION_TYP": "PE", "LAST_PRICE": 98, "OPEN_INT": 1300, "IV": 19.1, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22100, "OPTION_TYP": "CE", "LAST_PRICE": 76, "OPEN_INT": 1100, "IV": 17.8, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22100, "OPTION_TYP": "PE", "LAST_PRICE": 122, "OPEN_INT": 1050, "IV": 20.0, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22200, "OPTION_TYP": "CE", "LAST_PRICE": 51, "OPEN_INT": 980, "IV": 18.2, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22200, "OPTION_TYP": "PE", "LAST_PRICE": 141, "OPEN_INT": 990, "IV": 21.2, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22300, "OPTION_TYP": "CE", "LAST_PRICE": 33, "OPEN_INT": 910, "IV": 19.4, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22300, "OPTION_TYP": "PE", "LAST_PRICE": 162, "OPEN_INT": 930, "IV": 22.5, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22400, "OPTION_TYP": "CE", "LAST_PRICE": 21, "OPEN_INT": 870, "IV": 20.1, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22400, "OPTION_TYP": "PE", "LAST_PRICE": 188, "OPEN_INT": 900, "IV": 23.6, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22500, "OPTION_TYP": "CE", "LAST_PRICE": 12, "OPEN_INT": 820, "IV": 21.7, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22500, "OPTION_TYP": "PE", "LAST_PRICE": 214, "OPEN_INT": 860, "IV": 24.4, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22600, "OPTION_TYP": "CE", "LAST_PRICE": 7, "OPEN_INT": 770, "IV": 22.0, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22600, "OPTION_TYP": "PE", "LAST_PRICE": 240, "OPEN_INT": 800, "IV": 25.0, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22700, "OPTION_TYP": "CE", "LAST_PRICE": 4, "OPEN_INT": 710, "IV": 23.2, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22700, "OPTION_TYP": "PE", "LAST_PRICE": 266, "OPEN_INT": 760, "IV": 26.1, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22800, "OPTION_TYP": "CE", "LAST_PRICE": 2, "OPEN_INT": 690, "IV": 24.0, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22800, "OPTION_TYP": "PE", "LAST_PRICE": 292, "OPEN_INT": 720, "IV": 27.3, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22900, "OPTION_TYP": "CE", "LAST_PRICE": 1, "OPEN_INT": 640, "IV": 24.8, "EXPIRY_DT": "2026-03-26"},
                {"STRIKE_PR": 22900, "OPTION_TYP": "PE", "LAST_PRICE": 318, "OPEN_INT": 680, "IV": 28.0, "EXPIRY_DT": "2026-03-26"},
            ]
        )

        validation = validate_option_chain(option_chain)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["strike_count"], 10)
        self.assertEqual(validation["paired_strike_count"], 10)
        self.assertEqual(validation["paired_strike_ratio"], 1.0)
        self.assertEqual(validation["selected_expiry"], "2026-03-26")

    def test_flags_sparse_unpaired_chain(self):
        option_chain = pd.DataFrame(
            [
                {"strikePrice": 22000, "OPTION_TYP": "CE", "lastPrice": 110},
                {"strikePrice": 22100, "OPTION_TYP": "CE", "lastPrice": 76},
                {"strikePrice": 22200, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 22300, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 22400, "OPTION_TYP": "XX", "lastPrice": 10},
                {"strikePrice": 22500, "OPTION_TYP": "CE", "lastPrice": 12},
                {"strikePrice": 22600, "OPTION_TYP": "PE", "lastPrice": 240},
                {"strikePrice": 22700, "OPTION_TYP": "CE", "lastPrice": 4},
                {"strikePrice": 22800, "OPTION_TYP": "PE", "lastPrice": 292},
                {"strikePrice": 22900, "OPTION_TYP": "CE", "lastPrice": 1},
                {"strikePrice": 23000, "OPTION_TYP": "PE", "lastPrice": 318},
                {"strikePrice": 23100, "OPTION_TYP": "CE", "lastPrice": 0},
                {"strikePrice": 23200, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 23300, "OPTION_TYP": "CE", "lastPrice": 0},
                {"strikePrice": 23400, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 23500, "OPTION_TYP": "CE", "lastPrice": 0},
                {"strikePrice": 23600, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 23700, "OPTION_TYP": "CE", "lastPrice": 0},
                {"strikePrice": 23800, "OPTION_TYP": "PE", "lastPrice": 0},
                {"strikePrice": 23900, "OPTION_TYP": "CE", "lastPrice": 0},
            ]
        )

        validation = validate_option_chain(option_chain)

        self.assertTrue(validation["is_valid"])
        self.assertLess(validation["paired_strike_ratio"], 0.5)
        self.assertIn("low_paired_strike_ratio:0/20", validation["warnings"])
        self.assertIn("unknown_option_type_rows:1", validation["warnings"])


if __name__ == "__main__":
    unittest.main()
