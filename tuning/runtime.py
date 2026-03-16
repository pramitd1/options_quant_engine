"""
Module: runtime.py

Purpose:
    Implement runtime utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

from typing import Any

from config.policy_resolver import (
    DEFAULT_PARAMETER_PACK,
    ParameterRuntimeContext,
    get_active_parameter_pack,
    get_parameter_value as resolve_parameter_override,
    resolve_dataclass_config,
    resolve_mapping,
    set_active_parameter_pack,
    temporary_parameter_pack,
)

def get_parameter_value(key: str, default: Any | None = None) -> Any:
    """
    Purpose:
        Return parameter value for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        key (str): Input associated with key.
        default (Any | None): Input associated with default.
    
    Returns:
        Any: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    from tuning.registry import get_parameter_registry

    registry = get_parameter_registry()
    definition = registry.get(key)
    resolved_default = definition.default_value if default is None else default
    return resolve_parameter_override(key, resolved_default)


def serialize_current_registry() -> dict[str, Any]:
    """
    Purpose:
        Process serialize current registry for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        dict[str, Any]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    from tuning.registry import get_parameter_registry

    active_pack = get_active_parameter_pack()
    overrides = dict(active_pack.get("overrides", {}))
    current_values = {}
    for key, definition in get_parameter_registry().items():
        current_values[key] = overrides.get(key, definition.default_value)
    return get_parameter_registry().serialize(current_values=current_values)
