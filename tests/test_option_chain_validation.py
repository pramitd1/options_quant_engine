from __future__ import annotations

import unittest

import pandas as pd

from data.option_chain_validation import validate_option_chain
from config.policy_resolver import temporary_parameter_pack


class OptionChainValidationTests(unittest.TestCase):
    def test_validation_emits_dual_usability_and_reliability_weights(self):
        rows = []
        for s in range(22000, 23000, 25):
            rows.append(
                {
                    "strikePrice": s,
                    "OPTION_TYP": "CE",
                    "lastPrice": 100,
                    "bidPrice": 99,
                    "askPrice": 101,
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
                    "askPrice": 111,
                    "impliedVolatility": 19,
                    "EXPIRY_DT": "2026-03-26",
                }
            )
        df = pd.DataFrame(rows)

        result = validate_option_chain(df)

        self.assertTrue(result["analytics_usable"])
        self.assertTrue(result["execution_suggestion_usable"])
        self.assertIn("tradable_data", result)
        self.assertIn("feature_reliability_weights", result)
        weights = result["feature_reliability_weights"]
        self.assertIn("flow", weights)
        self.assertIn("vol_surface", weights)
        self.assertIn("greeks", weights)
        self.assertIn("liquidity", weights)
        self.assertIn("macro", weights)

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

    # ------------------------------------------------------------------ #
    # Industry-grade IV quality gate tests                                 #
    # ------------------------------------------------------------------ #

    def _make_full_chain(self, spot=22500, *, iv_ce=18.5, iv_pe=19.0, n_strikes=20):
        """Build a clean paired chain centred around `spot`."""
        step = 50
        rows = []
        for i in range(-n_strikes // 2, n_strikes // 2 + 1):
            s = spot + i * step
            rows.append({
                "strikePrice": s, "OPTION_TYP": "CE",
                "lastPrice": 100, "bidPrice": 99, "askPrice": 101,
                "impliedVolatility": iv_ce, "EXPIRY_DT": "2026-06-26",
            })
            rows.append({
                "strikePrice": s, "OPTION_TYP": "PE",
                "lastPrice": 110, "bidPrice": 109, "askPrice": 111,
                "impliedVolatility": iv_pe, "EXPIRY_DT": "2026-06-26",
            })
        return pd.DataFrame(rows)

    def test_atm_iv_health_good_when_both_legs_present_and_in_range(self):
        """Full chain with IV=18.5/19.0 (percent form) → ATM both legs present,
        normalises to ~0.185/0.190, within 4%–150% → GOOD."""
        df = self._make_full_chain(spot=22500)
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertEqual(ph["atm_iv_health"], "GOOD")
        self.assertTrue(ph["atm_iv_ce_found"])
        self.assertTrue(ph["atm_iv_pe_found"])
        self.assertTrue(ph["atm_iv_in_range"])
        self.assertIsNotNone(ph["atm_iv_midpoint"])
        # midpoint should be close to (0.185 + 0.190)/2
        self.assertAlmostEqual(ph["atm_iv_midpoint"], 0.1875, places=3)

    def test_atm_iv_health_weak_when_no_iv_at_atm(self):
        """Chain where IV is 0 for all ATM strikes → WEAK, and triggers trade block."""
        df = self._make_full_chain(spot=22500)
        # Zero out IV for the 3 nearest strikes
        atm_mask = df["strikePrice"].between(22350, 22650)
        df.loc[atm_mask, "impliedVolatility"] = 0
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertEqual(ph["atm_iv_health"], "WEAK")
        self.assertFalse(ph["atm_iv_ce_found"])
        self.assertFalse(ph["atm_iv_pe_found"])
        self.assertEqual(ph["trade_blocking_status"], "BLOCK")
        self.assertIn("atm_iv_weak", ph["trade_blocking_reasons"])

    def test_atm_iv_health_caution_when_only_one_leg_present(self):
        """Chain where only CE side has ATM IV → CAUTION (not WEAK, not GOOD)."""
        df = self._make_full_chain(spot=22500)
        pe_mask = (df["strikePrice"].between(22350, 22650)) & (df["OPTION_TYP"] == "PE")
        df.loc[pe_mask, "impliedVolatility"] = 0
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertEqual(ph["atm_iv_health"], "CAUTION")
        self.assertTrue(ph["atm_iv_ce_found"])
        self.assertFalse(ph["atm_iv_pe_found"])
        # CAUTION does not trigger trade block on its own
        self.assertNotIn("atm_iv_weak", ph["trade_blocking_reasons"])

    def test_atm_iv_health_not_evaluated_without_spot(self):
        """Without spot, ATM gate is skipped entirely (cannot determine ATM)."""
        df = self._make_full_chain(spot=22500)
        # Zero all IVs — would be WEAK if spot were passed
        df["impliedVolatility"] = 0
        result = validate_option_chain(df)  # no spot argument
        ph = result["provider_health"]

        # health tag is WEAK (chain has no IV) but it must NOT be in trade_blocking_reasons
        self.assertNotIn("atm_iv_weak", ph["trade_blocking_reasons"])

    def test_atm_iv_vs_vix_consistent_field_populated_when_vix_supplied(self):
        """India VIX supplied with matching scale → consistency flag is True."""
        df = self._make_full_chain(spot=22500)
        # ATM IV ≈ 0.1875; VIX = 15.0 → decimal 0.15 → ratio ≈ 1.25 → in [0.3, 3.0]
        result = validate_option_chain(df, spot=22500, india_vix_level=15.0)
        ph = result["provider_health"]

        self.assertTrue(ph["atm_iv_vs_vix_consistent"])
        self.assertEqual(ph["atm_iv_health"], "GOOD")

    def test_atm_iv_vs_vix_inconsistent_downgrades_to_caution(self):
        """ATM IV far outside VIX band → VIX-inconsistent → CAUTION."""
        df = self._make_full_chain(spot=22500)
        # ATM IV ≈ 0.1875; VIX would need to be 0.063–0.625 for ratio to be in [0.3, 3.0]
        # Supply a VIX of 1.0 (100% → decimal) → ratio = 0.1875 → outside lower bound 0.3
        result = validate_option_chain(df, spot=22500, india_vix_level=100.0)
        ph = result["provider_health"]

        self.assertFalse(ph["atm_iv_vs_vix_consistent"])
        self.assertEqual(ph["atm_iv_health"], "CAUTION")

    def test_iv_parity_health_good_on_clean_chain(self):
        """Chain with uniform CE/PE IVs → near-zero divergence → GOOD."""
        df = self._make_full_chain(spot=22500, iv_ce=18.5, iv_pe=19.0)
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertEqual(ph["iv_parity_health"], "GOOD")
        self.assertIsNotNone(ph["iv_parity_breach_ratio"])
        self.assertGreater(ph["iv_parity_checked_pairs"], 0)

    def test_iv_parity_health_weak_on_crossed_chain(self):
        """Chain where PE IV is wildly different from CE IV on most strikes → WEAK,
        and a warning is emitted."""
        df = self._make_full_chain(spot=22500, iv_ce=18.5, iv_pe=18.5)
        # Inflate PE IV by 5× to simulate a crossed/stale feed
        df.loc[df["OPTION_TYP"] == "PE", "impliedVolatility"] = 95.0
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertEqual(ph["iv_parity_health"], "WEAK")
        iv_parity_warnings = [w for w in result["warnings"] if "iv_parity_breach_detected" in w]
        self.assertEqual(len(iv_parity_warnings), 1)

    def test_iv_staleness_health_good_on_varied_ivs(self):
        """Each strike has a distinct IV → staleness ratio near 0 → GOOD."""
        rows = []
        spot = 22500
        for i, s in enumerate(range(22000, 23050, 50)):
            rows.append({
                "strikePrice": s, "OPTION_TYP": "CE", "lastPrice": 100,
                "impliedVolatility": 15.0 + i * 0.1, "EXPIRY_DT": "2026-06-26",
            })
            rows.append({
                "strikePrice": s, "OPTION_TYP": "PE", "lastPrice": 110,
                "impliedVolatility": 16.0 + i * 0.1, "EXPIRY_DT": "2026-06-26",
            })
        df = pd.DataFrame(rows)
        result = validate_option_chain(df, spot=spot)
        ph = result["provider_health"]

        self.assertEqual(ph["iv_staleness_health"], "GOOD")

    def test_iv_staleness_health_weak_on_static_feed(self):
        """All rows share the same IV → stale ratio = 1.0 → WEAK, warning emitted."""
        df = self._make_full_chain(spot=22500, iv_ce=20.0, iv_pe=20.0, n_strikes=30)
        # Force a single repeated IV value → static feed
        df["impliedVolatility"] = 20.0
        result = validate_option_chain(df, spot=22500)
        ph = result["provider_health"]

        self.assertIn(ph["iv_staleness_health"], ("CAUTION", "WEAK"))
        staleness_warnings = [w for w in result["warnings"] if "iv_staleness_detected" in w]
        self.assertEqual(len(staleness_warnings), 1)

    def test_market_data_readiness_score_degrades_when_atm_iv_breaks(self):
        """Clean chains should score materially higher than chains with broken ATM IV."""
        clean = self._make_full_chain(spot=22500)
        clean["impliedVolatility"] = [15.0 + 0.05 * i for i in range(len(clean))]
        clean_result = validate_option_chain(clean, spot=22500)

        broken = clean.copy()
        broken.loc[broken["strikePrice"].between(22350, 22650), "impliedVolatility"] = 0
        broken_result = validate_option_chain(broken, spot=22500)

        self.assertGreater(clean_result["market_data_readiness_score"], broken_result["market_data_readiness_score"])
        self.assertIn(clean_result["market_data_readiness_tier"], {"HIGH", "MODERATE"})
        self.assertIn(broken_result["market_data_readiness_tier"], {"LOW", "FRAGILE"})
        self.assertEqual(clean_result["provider_health"]["market_data_readiness_score"], clean_result["market_data_readiness_score"])

    def test_atm_iv_summary_status_reflects_weak_atm_iv(self):
        """When ATM IV is absent (spot known), summary_status must be WEAK."""
        df = self._make_full_chain(spot=22500)
        df["impliedVolatility"] = 0  # strip all IV
        result = validate_option_chain(df, spot=22500)

        self.assertEqual(result["provider_health"]["summary_status"], "WEAK")
        self.assertEqual(result["provider_health"]["atm_iv_health"], "WEAK")


if __name__ == "__main__":
    unittest.main()
