from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

from risk import build_gamma_vol_acceleration_state


class GammaVolAccelerationScenarioTests(unittest.TestCase):
    def test_fixture_scenarios_transition_as_expected(self):
        scenario_path = ROOT_DIR / "config" / "gamma_vol_acceleration_scenarios.json"
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))

        for scenario in payload.get("scenarios", []):
            with self.subTest(scenario=scenario.get("name")):
                state = build_gamma_vol_acceleration_state(**scenario.get("inputs", {}))
                expected = scenario.get("expected", {})

                for key, value in expected.items():
                    self.assertEqual(state.get(key), value)
