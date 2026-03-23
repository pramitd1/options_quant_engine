from __future__ import annotations

import unittest

import pandas as pd

from data.option_chain_validation import validate_option_chain
from config.policy_resolver import temporary_parameter_pack


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

    def test_bid_ask_columns_populate_quoted_ratio_and_effective_priced_ratio(self):
        """Chain with full bid/ask coverage → pricing_basis=TRADE_OR_QUOTE,
        quoted_ratio and effective_priced_ratio both equal 1.0."""
        rows = []
        for i, s in enumerate(range(22000, 22650, 50)):
            rows.append({
                "strikePrice": s, "OPTION_TYP": "CE",
                "lastPrice": 100, "bidPrice": 99, "askPrice": 101,
                "impliedVolatility": 18, "EXPIRY_DT": "2026-03-26",
            })
            rows.append({
                "strikePrice": s, "OPTION_TYP": "PE",
                "lastPrice": 110, "bidPrice": 109, "askPrice": 111,
                "impliedVolatility": 19, "EXPIRY_DT": "2026-03-26",
            })
        df = pd.DataFrame(rows)

        result = validate_option_chain(df)
        ph = result["provider_health"]

        self.assertEqual(ph["pricing_basis"], "TRADE_OR_QUOTE")
        self.assertEqual(ph["quote_coverage_mode"], "TWO_SIDED")
        self.assertIsNotNone(result["quoted_ratio"])
        self.assertEqual(result["priced_ratio"], 1.0)
        self.assertEqual(result["quoted_ratio"], 1.0)
        self.assertEqual(result["effective_priced_ratio"], 1.0)
        self.assertEqual(result["one_sided_quote_rows"], 0)
        self.assertEqual(ph["quote_health"], "GOOD")
        self.assertEqual(ph["pricing_health"], "GOOD")

    def test_ltp_only_chain_has_trade_only_pricing_basis(self):
        """Chain without bid/ask columns → pricing_basis=TRADE_ONLY,
        quoted_ratio is None, effective_priced_ratio equals priced_ratio."""
        rows = []
        for s in range(22000, 22650, 50):
            rows.append({
                "strikePrice": s, "OPTION_TYP": "CE",
                "lastPrice": 100, "impliedVolatility": 18, "EXPIRY_DT": "2026-03-26",
            })
            rows.append({
                "strikePrice": s, "OPTION_TYP": "PE",
                "lastPrice": 110, "impliedVolatility": 19, "EXPIRY_DT": "2026-03-26",
            })
        df = pd.DataFrame(rows)

        result = validate_option_chain(df)
        ph = result["provider_health"]

        self.assertEqual(ph["pricing_basis"], "TRADE_ONLY")
        self.assertIsNone(result["quoted_ratio"])
        self.assertEqual(result["effective_priced_ratio"], result["priced_ratio"])
        self.assertIsNone(ph["quote_health"])

    def test_thin_row_escalates_to_caution_toggle(self):
        """A chain in the THIN row band (60 ≤ rows < 120) with otherwise-GOOD
        health stays GOOD by default; enabling the toggle escalates it to
        CAUTION and sets row_health_escalation_applied=True."""
        # 34 strikes × 2 = 68 rows: squarely THIN
        rows = []
        for s in range(22000, 22850, 25):
            rows.append({
                "strikePrice": s, "OPTION_TYP": "CE",
                "lastPrice": 100, "impliedVolatility": 18, "EXPIRY_DT": "2026-03-26",
            })
            rows.append({
                "strikePrice": s, "OPTION_TYP": "PE",
                "lastPrice": 110, "impliedVolatility": 19, "EXPIRY_DT": "2026-03-26",
            })
        df = pd.DataFrame(rows)
        self.assertGreaterEqual(len(df), 60)
        self.assertLess(len(df), 120)

        default_result = validate_option_chain(df)
        self.assertEqual(default_result["provider_health"]["row_health"], "THIN")
        self.assertEqual(default_result["provider_health"]["summary_status"], "GOOD")
        self.assertFalse(default_result["provider_health"]["row_health_escalation_applied"])

        with temporary_parameter_pack(
            "thin_escalation_test",
            overrides={"option_chain_validation.provider_health.thin_row_escalates_to_caution": 1},
        ):
            toggle_result = validate_option_chain(df)

        self.assertEqual(toggle_result["provider_health"]["summary_status"], "CAUTION")
        self.assertTrue(toggle_result["provider_health"]["row_health_escalation_applied"])

    def test_non_critical_trade_price_weakness_does_not_force_summary_weak(self):
        """When quote coverage is strong but trade prints are sparse, summary should
        stay core-driven and avoid WEAK solely from trade_price_health."""
        rows = []
        for s in range(22000, 25000, 25):
            rows.append({
                "strikePrice": s,
                "OPTION_TYP": "CE",
                "lastPrice": 0,
                "bidPrice": 99,
                "askPrice": 101,
                "impliedVolatility": 18,
                "EXPIRY_DT": "2026-03-26",
            })
            rows.append({
                "strikePrice": s,
                "OPTION_TYP": "PE",
                "lastPrice": 0,
                "bidPrice": 109,
                "askPrice": 111,
                "impliedVolatility": 19,
                "EXPIRY_DT": "2026-03-26",
            })
        df = pd.DataFrame(rows)

        result = validate_option_chain(df)
        ph = result["provider_health"]

        self.assertEqual(ph["trade_price_health"], "WEAK")
        self.assertEqual(ph["quote_health"], "GOOD")
        self.assertEqual(ph["pricing_health"], "GOOD")
        self.assertEqual(ph["summary_status"], "GOOD")
        self.assertIn("trade_price_health", ph["non_critical_weak_components"])
        self.assertEqual(ph["trade_blocking_status"], "PASS")
        self.assertEqual(ph["trade_blocking_reasons"], [])

    def test_critical_core_pairing_weak_sets_trade_blocking_status_block(self):
        rows = []
        # Dense but one-sided option types around core strikes -> critical pairing weakness.
        for s in range(22400, 22800, 25):
            rows.append({
                "strikePrice": s,
                "OPTION_TYP": "CE",
                "lastPrice": 100,
                "bidPrice": 99,
                "askPrice": 101,
                "impliedVolatility": 18,
                "EXPIRY_DT": "2026-03-26",
            })
        # Add some distant PE rows that should not rescue core pairing.
        for s in range(24000, 24400, 50):
            rows.append({
                "strikePrice": s,
                "OPTION_TYP": "PE",
                "lastPrice": 10,
                "bidPrice": 9,
                "askPrice": 11,
                "impliedVolatility": 22,
                "EXPIRY_DT": "2026-03-26",
            })
        df = pd.DataFrame(rows)

        result = validate_option_chain(df, spot=22600)
        ph = result["provider_health"]

        self.assertEqual(ph["core_pairing_health"], "WEAK")
        self.assertEqual(ph["trade_blocking_status"], "BLOCK")
        self.assertIn("core_pairing_weak", ph["trade_blocking_reasons"])

    def test_core_quote_integrity_weak_is_non_blocking_when_marketability_is_good(self):
        rows = []
        # Fully priced chain, but one-sided quotes in core window.
        for s in range(22400, 22800, 25):
            rows.append(
                {
                    "strikePrice": s,
                    "OPTION_TYP": "CE",
                    "lastPrice": 100,
                    "bidPrice": 99,
                    "askPrice": 0,
                    "impliedVolatility": 18,
                    "EXPIRY_DT": "2026-03-26",
                }
            )
            rows.append(
                {
                    "strikePrice": s,
                    "OPTION_TYP": "PE",
                    "lastPrice": 110,
                    "bidPrice": 109,
                    "askPrice": 0,
                    "impliedVolatility": 19,
                    "EXPIRY_DT": "2026-03-26",
                }
            )
        df = pd.DataFrame(rows)

        result = validate_option_chain(df, spot=22600)
        ph = result["provider_health"]

        self.assertEqual(ph["core_quote_integrity_health"], "WEAK")
        self.assertEqual(ph["core_marketability_health"], "GOOD")
        self.assertEqual(ph["trade_blocking_status"], "PASS")
        self.assertNotIn("core_quote_integrity_weak", ph["trade_blocking_reasons"])

    def test_core_quote_integrity_weak_can_block_in_strict_mode(self):
        rows = []
        for s in range(22400, 22800, 25):
            rows.append(
                {
                    "strikePrice": s,
                    "OPTION_TYP": "CE",
                    "lastPrice": 100,
                    "bidPrice": 99,
                    "askPrice": 0,
                    "impliedVolatility": 18,
                    "EXPIRY_DT": "2026-03-26",
                }
            )
            rows.append(
                {
                    "strikePrice": s,
                    "OPTION_TYP": "PE",
                    "lastPrice": 110,
                    "bidPrice": 109,
                    "askPrice": 0,
                    "impliedVolatility": 19,
                    "EXPIRY_DT": "2026-03-26",
                }
            )
        df = pd.DataFrame(rows)

        with temporary_parameter_pack(
            "quote_integrity_strict_block_test",
            overrides={
                "option_chain_validation.provider_health.core_quote_integrity_standalone_block": 1,
            },
        ):
            result = validate_option_chain(df, spot=22600)

        ph = result["provider_health"]
        self.assertEqual(ph["core_quote_integrity_health"], "WEAK")
        self.assertEqual(ph["trade_blocking_status"], "BLOCK")
        self.assertIn("core_quote_integrity_weak", ph["trade_blocking_reasons"])

if __name__ == "__main__":
    unittest.main()
