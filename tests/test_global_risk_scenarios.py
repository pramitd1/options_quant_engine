from __future__ import annotations

import json
import unittest
from pathlib import Path

from backtest.global_risk_scenario_runner import run_scenario
from config.settings import BASE_DIR


SCENARIO_FILE = Path(BASE_DIR) / "config/global_risk_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


class GlobalRiskScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = {item["name"]: item for item in _load_scenarios()}

    def test_neutral_environment_scenario(self):
        result = run_scenario(self.scenarios["neutral_environment"])
        state = result["global_risk_state"]
        features = result["features"]

        self.assertEqual(state["global_risk_state"], "GLOBAL_NEUTRAL")
        self.assertTrue(state["overnight_hold_allowed"])
        self.assertEqual(features["oil_shock_score"], 0.0)
        self.assertEqual(features["volatility_shock_score"], 0.0)
        self.assertEqual(features["volatility_explosion_probability"], 0.0)

    def test_oil_shock_scenario(self):
        result = run_scenario(self.scenarios["oil_shock"])
        state = result["global_risk_state"]
        features = result["features"]

        self.assertEqual(state["global_risk_state"], "GLOBAL_NEUTRAL")
        self.assertTrue(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "oil_shock_elevated")
        self.assertEqual(state["overnight_risk_penalty"], 6)
        self.assertEqual(features["oil_shock_score"], 1.0)
        self.assertAlmostEqual(features["commodity_risk_score"], 0.53, places=2)

    def test_volatility_spike_scenario(self):
        result = run_scenario(self.scenarios["volatility_spike"])
        state = result["global_risk_state"]
        features = result["features"]

        self.assertEqual(state["global_risk_state"], "GLOBAL_NEUTRAL")
        self.assertTrue(state["overnight_hold_allowed"])
        self.assertEqual(features["volatility_shock_score"], 1.0)
        self.assertEqual(features["volatility_explosion_probability"], 0.0)

    def test_macro_event_lockdown_scenario(self):
        result = run_scenario(self.scenarios["macro_event_lockdown"])
        state = result["global_risk_state"]

        self.assertEqual(state["global_risk_state"], "EVENT_LOCKDOWN")
        self.assertFalse(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "event_lockdown_block")
        self.assertEqual(state["overnight_risk_penalty"], 10)

    def test_volatility_compression_breakout_scenario(self):
        result = run_scenario(self.scenarios["volatility_compression_breakout"])
        state = result["global_risk_state"]
        features = result["features"]

        self.assertEqual(state["global_risk_state"], "VOL_SHOCK")
        self.assertFalse(state["overnight_hold_allowed"])
        self.assertEqual(state["overnight_hold_reason"], "vol_shock_block")
        self.assertEqual(state["overnight_risk_penalty"], 10)
        self.assertEqual(features["volatility_shock_score"], 0.7)
        self.assertAlmostEqual(features["volatility_explosion_probability"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
