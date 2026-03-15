"""
Scenario runner for deterministic option efficiency validation.
"""

from __future__ import annotations

from pathlib import Path

from backtest.scenario_utils import find_named_scenario, load_scenarios
from risk.option_efficiency_layer import build_option_efficiency_state


def load_option_efficiency_scenarios(
    path: str | Path = "config/option_efficiency_scenarios.json",
) -> list[dict]:
    return load_scenarios(path)


def run_option_efficiency_scenario(
    name: str,
    path: str | Path = "config/option_efficiency_scenarios.json",
) -> dict:
    scenario = find_named_scenario(name, path=path)
    state = build_option_efficiency_state(**scenario.get("inputs", {}))
    return {
        "scenario": scenario.get("name"),
        "inputs": scenario.get("inputs", {}),
        "expected": scenario.get("expected", {}),
        "option_efficiency_state": state,
    }
