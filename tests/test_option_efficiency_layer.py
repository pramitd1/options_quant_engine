from __future__ import annotations

import unittest

from risk.option_efficiency_layer import build_option_efficiency_state


class OptionEfficiencyLayerTests(unittest.TestCase):
    def test_direct_expected_move_uses_atm_iv_and_expiry(self):
        state = build_option_efficiency_state(
            spot=22000,
            atm_iv=18.0,
            expiry_value="2026-03-21",
            valuation_time="2026-03-14T10:00:00+05:30",
            direction="CALL",
            strike=22000,
            option_type="CE",
            entry_price=110,
            target=145,
            stop_loss=92,
        )

        self.assertEqual(state["expected_move_quality"], "DIRECT")
        self.assertIsNotNone(state["expected_move_points"])
        self.assertGreater(state["expected_move_points"], 0)
        self.assertGreater(state["target_reachability_score"], 50)

    def test_fallback_iv_path_stays_interpretable(self):
        state = build_option_efficiency_state(
            spot=22000,
            fallback_iv=0.19,
            expiry_value="2026-03-21",
            valuation_time="2026-03-14T10:00:00+05:30",
            direction="CALL",
            strike=22000,
            option_type="CE",
            entry_price=110,
            target=145,
        )

        self.assertEqual(state["expected_move_quality"], "FALLBACK")
        self.assertIn("fallback_iv_used", state["option_efficiency_diagnostics"]["warnings"])

    def test_missing_iv_and_expiry_degrades_to_neutral(self):
        state = build_option_efficiency_state(
            spot=22000,
            direction="CALL",
            strike=22000,
            option_type="CE",
            entry_price=110,
            target=145,
        )

        self.assertTrue(state["neutral_fallback"])
        self.assertEqual(state["expected_move_quality"], "UNAVAILABLE")
        self.assertEqual(state["option_efficiency_score"], 50)

    def test_far_otm_strike_scores_as_less_efficient(self):
        state = build_option_efficiency_state(
            spot=22000,
            atm_iv=15.0,
            expiry_value="2026-03-18",
            valuation_time="2026-03-14T10:00:00+05:30",
            direction="CALL",
            strike=22500,
            option_type="CE",
            entry_price=32,
            target=52,
        )

        self.assertEqual(state["strike_moneyness_bucket"], "OTM")
        self.assertLess(state["strike_efficiency_score"], 60)

    def test_overnight_poor_efficiency_can_block_hold(self):
        state = build_option_efficiency_state(
            spot=22000,
            atm_iv=13.0,
            expiry_value="2026-03-17",
            valuation_time="2026-03-14T15:10:00+05:30",
            direction="CALL",
            strike=22300,
            option_type="CE",
            entry_price=145,
            target=210,
            stop_loss=105,
            holding_profile="OVERNIGHT",
            global_risk_state={
                "global_risk_state": "GLOBAL_NEUTRAL",
                "holding_context": {
                    "holding_profile": "OVERNIGHT",
                    "overnight_relevant": True,
                },
            },
        )

        self.assertFalse(state["overnight_hold_allowed"])
        self.assertGreaterEqual(state["overnight_option_efficiency_penalty"], 5)


if __name__ == "__main__":
    unittest.main()
