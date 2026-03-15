"""
Shared helpers for deterministic scenario runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scenario_payload(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path)
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def load_scenarios(path: str | Path, *, list_key: str = "scenarios") -> list[dict[str, Any]]:
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
    for scenario in load_scenarios(path, list_key=list_key):
        if scenario.get("name") == name:
            return scenario
    raise ValueError(f"Scenario not found: {name}")
