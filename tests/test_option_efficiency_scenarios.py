from __future__ import annotations

import json
import unittest
from pathlib import Path

from backtest.option_efficiency_scenario_runner import run_option_efficiency_scenario
from config.settings import BASE_DIR


SCENARIO_FILE = Path(BASE_DIR) / "config/option_efficiency_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


class OptionEfficiencyScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = {item["name"]: item for item in _load_scenarios()["scenarios"]}

    def test_cheap_atm_reachable_target(self):
        state = run_option_efficiency_scenario(
            self.scenarios["cheap_atm_reachable_target"]["name"]
        )["option_efficiency_state"]

        self.assertEqual(state["expected_move_quality"], "DIRECT")
        self.assertGreaterEqual(state["option_efficiency_score"], 60)
        self.assertEqual(state["strike_moneyness_bucket"], "ATM")

    def test_expensive_option_with_unrealistic_target(self):
        state = run_option_efficiency_scenario(
            self.scenarios["expensive_option_unrealistic_target"]["name"]
        )["option_efficiency_state"]

        self.assertLessEqual(state["target_reachability_score"], 65)
        self.assertLessEqual(state["option_efficiency_score"], 55)

    def test_far_otm_weak_expected_move(self):
        state = run_option_efficiency_scenario(
            self.scenarios["far_otm_weak_expected_move"]["name"]
        )["option_efficiency_state"]

        self.assertEqual(state["strike_moneyness_bucket"], "OTM")
        self.assertIn("far_otm", state["payoff_efficiency_hint"])

    def test_supportive_convexity_improves_efficiency(self):
        state = run_option_efficiency_scenario(
            self.scenarios["strong_move_efficient_premium_supportive_convexity"]["name"]
        )["option_efficiency_state"]

        self.assertGreaterEqual(state["option_efficiency_score"], 60)
        self.assertGreaterEqual(state["premium_efficiency_score"], 58)

    def test_overnight_poor_efficiency_scenario(self):
        state = run_option_efficiency_scenario(
            self.scenarios["overnight_poor_efficiency"]["name"]
        )["option_efficiency_state"]

        self.assertFalse(state["overnight_hold_allowed"])
        self.assertGreater(state["overnight_option_efficiency_penalty"], 0)

    def test_neutral_missing_data_scenario(self):
        state = run_option_efficiency_scenario(
            self.scenarios["neutral_missing_data"]["name"]
        )["option_efficiency_state"]

        self.assertTrue(state["neutral_fallback"])
        self.assertEqual(state["expected_move_quality"], "UNAVAILABLE")
        self.assertEqual(state["option_efficiency_score"], 50)


if __name__ == "__main__":
    unittest.main()
