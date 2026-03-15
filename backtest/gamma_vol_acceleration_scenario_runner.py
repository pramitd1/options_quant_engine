"""
Scenario runner for deterministic gamma-vol acceleration validation.
"""

from __future__ import annotations

from pathlib import Path

from backtest.scenario_utils import find_named_scenario, load_scenarios
from risk.gamma_vol_acceleration_layer import build_gamma_vol_acceleration_state


def load_gamma_vol_scenarios(path: str | Path = "config/gamma_vol_acceleration_scenarios.json") -> list[dict]:
    return load_scenarios(path)


def run_gamma_vol_scenario(name: str, path: str | Path = "config/gamma_vol_acceleration_scenarios.json") -> dict:
    scenario = find_named_scenario(name, path=path)
    state = build_gamma_vol_acceleration_state(**scenario.get("inputs", {}))
    return {
        "scenario": scenario.get("name"),
        "inputs": scenario.get("inputs", {}),
        "expected": scenario.get("expected", {}),
        "gamma_vol_state": state,
    }
