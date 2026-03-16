"""
Module: scenario_utils.py

Purpose:
    Implement scenario utils logic used by historical replay and backtest evaluation.

Role in the System:
    Part of the backtest layer that replays historical data and measures strategy behavior out of sample.

Key Outputs:
    Backtest results, replay diagnostics, and evaluation summaries.

Downstream Usage:
    Consumed by research analysis, tuning validation, and promotion decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scenario_payload(path: str | Path) -> dict[str, Any]:
    """
    Purpose:
        Process load scenario payload for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    scenario_path = Path(path)
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def load_scenarios(path: str | Path, *, list_key: str = "scenarios") -> list[dict[str, Any]]:
    """
    Purpose:
        Process load scenarios for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        path (str | Path): Input associated with path.
        list_key (str): Input associated with list key.
    
    Returns:
        list[dict[str, Any]]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    payload = load_scenario_payload(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(item) for item in payload.get(list_key, [])]


def find_named_scenario(
    name: str,
    *,
    path: str | Path,
    list_key: str = "scenarios",
) -> dict[str, Any]:
    """
    Purpose:
        Process find named scenario for downstream use.
    
    Context:
        Public function within the backtest layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        name (str): Input associated with name.
        path (str | Path): Input associated with path.
        list_key (str): Input associated with list key.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    for scenario in load_scenarios(path, list_key=list_key):
        if scenario.get("name") == name:
            return scenario
    raise ValueError(f"Scenario not found: {name}")
