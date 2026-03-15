"""
Runtime access to active parameter packs and overrides.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
import os
from typing import Any

from tuning.packs import resolve_parameter_pack
from tuning.registry import get_parameter_registry


DEFAULT_PARAMETER_PACK = "baseline_v1"


@dataclass(frozen=True)
class ParameterRuntimeContext:
    name: str
    overrides: dict[str, Any]


_DEFAULT_PACK_NAME = os.getenv("OQE_PARAMETER_PACK", DEFAULT_PARAMETER_PACK)
_ACTIVE_CONTEXT: ContextVar[ParameterRuntimeContext | None] = ContextVar(
    "oqe_active_parameter_context",
    default=None,
)


def _build_runtime_context(
    name: str = DEFAULT_PARAMETER_PACK,
    *,
    overrides: dict[str, Any] | None = None,
) -> ParameterRuntimeContext:
    pack_name = str(name or DEFAULT_PARAMETER_PACK).strip() or DEFAULT_PARAMETER_PACK
    try:
        base_overrides = dict(resolve_parameter_pack(pack_name).overrides)
    except Exception:
        if pack_name != DEFAULT_PARAMETER_PACK:
            pack_name = DEFAULT_PARAMETER_PACK
            try:
                base_overrides = dict(resolve_parameter_pack(pack_name).overrides)
            except Exception:
                base_overrides = {}
        else:
            base_overrides = {}

    if overrides:
        base_overrides.update(dict(overrides))
    return ParameterRuntimeContext(name=pack_name, overrides=base_overrides)


def _resolve_active_overrides() -> tuple[str, dict[str, Any]]:
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        context = _build_runtime_context(_DEFAULT_PACK_NAME)
        _ACTIVE_CONTEXT.set(context)
    return context.name, dict(context.overrides)


def get_active_parameter_pack() -> dict[str, Any]:
    pack_name, overrides = _resolve_active_overrides()
    return {
        "name": pack_name,
        "overrides": overrides,
    }


def set_active_parameter_pack(name: str = DEFAULT_PARAMETER_PACK, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    _ACTIVE_CONTEXT.set(_build_runtime_context(name, overrides=overrides))
    return get_active_parameter_pack()


@contextmanager
def temporary_parameter_pack(name: str = DEFAULT_PARAMETER_PACK, *, overrides: dict[str, Any] | None = None):
    token = _ACTIVE_CONTEXT.set(_build_runtime_context(name, overrides=overrides))
    try:
        yield get_active_parameter_pack()
    finally:
        _ACTIVE_CONTEXT.reset(token)


def get_parameter_value(key: str, default: Any | None = None) -> Any:
    registry = get_parameter_registry()
    definition = registry.get(key)
    _, overrides = _resolve_active_overrides()
    return overrides.get(key, definition.default_value if default is None else default)


def resolve_mapping(prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(defaults)
    _, overrides = _resolve_active_overrides()
    prefix = f"{prefix}."
    for key, value in overrides.items():
        if key.startswith(prefix):
            resolved[key[len(prefix):]] = value
    return resolved


def resolve_dataclass_config(prefix: str, config_obj: Any):
    if not is_dataclass(config_obj):
        raise TypeError("resolve_dataclass_config expects a dataclass instance")
    payload = resolve_mapping(prefix, asdict(config_obj))
    return type(config_obj)(**payload)


def serialize_current_registry() -> dict[str, Any]:
    _, overrides = _resolve_active_overrides()
    current_values = {}
    for key, definition in get_parameter_registry().items():
        current_values[key] = overrides.get(key, definition.default_value)
    return get_parameter_registry().serialize(current_values=current_values)
