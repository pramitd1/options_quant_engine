from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

from risk import build_dealer_hedging_pressure_state


class DealerHedgingPressureScenarioTests(unittest.TestCase):
    def test_fixture_scenarios_transition_as_expected(self):
        scenario_path = ROOT_DIR / "config" / "dealer_hedging_pressure_scenarios.json"
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))

        for scenario in payload.get("scenarios", []):
            with self.subTest(scenario=scenario.get("name")):
                state = build_dealer_hedging_pressure_state(**scenario.get("inputs", {}))
                expected = scenario.get("expected", {})

                for key, value in expected.items():
                    self.assertEqual(state.get(key), value)
