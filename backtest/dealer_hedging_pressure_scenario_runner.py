"""
Scenario runner for deterministic dealer hedging pressure validation.
"""

from __future__ import annotations

from pathlib import Path

from backtest.scenario_utils import find_named_scenario, load_scenarios
from risk.dealer_hedging_pressure_layer import build_dealer_hedging_pressure_state


def load_dealer_pressure_scenarios(path: str | Path = "config/dealer_hedging_pressure_scenarios.json") -> list[dict]:
    return load_scenarios(path)


def run_dealer_pressure_scenario(name: str, path: str | Path = "config/dealer_hedging_pressure_scenarios.json") -> dict:
    scenario = find_named_scenario(name, path=path)
    state = build_dealer_hedging_pressure_state(**scenario.get("inputs", {}))
    return {
        "scenario": scenario.get("name"),
        "inputs": scenario.get("inputs", {}),
        "expected": scenario.get("expected", {}),
        "dealer_pressure_state": state,
    }
