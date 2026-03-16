"""
Module: policy_resolver.py

Purpose:
    Resolve runtime policy overrides for configuration modules without depending
    on the full tuning registry.

Role in the System:
    Part of the configuration layer that applies the active parameter pack to
    policy defaults before analytics, signal generation, and overlays read
    their runtime thresholds.

Key Outputs:
    Active parameter-pack metadata plus resolved mappings and dataclass-backed
    policy objects.

Downstream Usage:
    Consumed by config and macro policy getters, while `tuning.runtime`
    re-exports the same context for governance and promotion workflows.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
import os
from typing import Any

from tuning.packs import resolve_parameter_pack


DEFAULT_PARAMETER_PACK = "baseline_v1"


@dataclass(frozen=True)
class ParameterRuntimeContext:
    """
    Purpose:
        Capture the active parameter pack and its flattened override mapping.

    Context:
        Shared runtime state used by config getters, the live engine, replay
        tooling, and tuning workflows so every layer resolves the same active
        policy overrides.

    Attributes:
        name (str): Stable parameter-pack name selected for the current
            execution context.
        overrides (dict[str, Any]): Fully merged override mapping after pack
            inheritance has been resolved.

    Notes:
        The record is immutable so nested runtime evaluations, such as shadow
        mode, can safely swap contexts without leaking overrides across calls.
    """

    name: str
    overrides: dict[str, Any]


_DEFAULT_PACK_NAME = os.getenv("OQE_PARAMETER_PACK", DEFAULT_PARAMETER_PACK)
_ACTIVE_CONTEXT: ContextVar[ParameterRuntimeContext | None] = ContextVar(
    "oqe_active_parameter_context",
    default=None,
)
_MISSING = object()


def _load_pack_overrides(name: str) -> tuple[str, dict[str, Any]]:
    """
    Purpose:
        Load the fully inherited override mapping for a named parameter pack.

    Context:
        Internal helper used when bootstrapping the active configuration
        context. It keeps pack inheritance logic centralized so all callers see
        the same fallback behavior.

    Inputs:
        name (str): Preferred parameter-pack name supplied by the caller.

    Returns:
        tuple[str, dict[str, Any]]: The resolved pack name and its flattened
        override mapping.

    Notes:
        Unknown packs gracefully fall back to the default production-safe pack
        so configuration getters can still resolve a usable policy bundle.
    """

    pack_name = str(name or DEFAULT_PARAMETER_PACK).strip() or DEFAULT_PARAMETER_PACK
    try:
        return pack_name, dict(resolve_parameter_pack(pack_name).overrides)
    except Exception:
        if pack_name == DEFAULT_PARAMETER_PACK:
            return DEFAULT_PARAMETER_PACK, {}
        try:
            return DEFAULT_PARAMETER_PACK, dict(resolve_parameter_pack(DEFAULT_PARAMETER_PACK).overrides)
        except Exception:
            return DEFAULT_PARAMETER_PACK, {}


def _build_runtime_context(
    name: str = DEFAULT_PARAMETER_PACK,
    *,
    overrides: dict[str, Any] | None = None,
) -> ParameterRuntimeContext:
    """
    Purpose:
        Build the immutable runtime context used during policy resolution.

    Context:
        Internal helper for the configuration layer. It normalizes pack
        selection and applies ad-hoc overrides used by tests, shadow mode, and
        tuning experiments.

    Inputs:
        name (str): Preferred parameter-pack name for the new context.
        overrides (dict[str, Any] | None): Additional override mapping layered
            on top of the resolved pack.

    Returns:
        ParameterRuntimeContext: Immutable context object ready to install in
        the active context variable.

    Notes:
        Ad-hoc overrides intentionally win over pack-defined values so research
        and governance workflows can evaluate temporary deltas without writing a
        new pack file first.
    """

    pack_name, base_overrides = _load_pack_overrides(name)
    if overrides:
        base_overrides.update(dict(overrides))
    return ParameterRuntimeContext(name=pack_name, overrides=base_overrides)


def _resolve_active_context() -> ParameterRuntimeContext:
    """
    Purpose:
        Return the currently active parameter-resolution context.

    Context:
        Internal helper used by all public getters in this module so they share
        one lazy initialization path.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        ParameterRuntimeContext: Active context for the current execution scope.

    Notes:
        The context is created lazily to keep import-time configuration access
        cheap and side-effect free.
    """

    context = _ACTIVE_CONTEXT.get()
    if context is None:
        context = _build_runtime_context(_DEFAULT_PACK_NAME)
        _ACTIVE_CONTEXT.set(context)
    return context


def _resolve_active_overrides() -> tuple[str, dict[str, Any]]:
    """
    Purpose:
        Return the active pack name and a copy of its override mapping.

    Context:
        Internal helper shared by mapping, dataclass, and scalar policy
        resolvers.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        tuple[str, dict[str, Any]]: Active pack name plus copied override
        mapping.

    Notes:
        Returning a copy prevents downstream callers from mutating shared
        context state by accident.
    """

    context = _resolve_active_context()
    return context.name, dict(context.overrides)


def get_active_parameter_pack() -> dict[str, Any]:
    """
    Purpose:
        Expose the active parameter-pack selection for the current execution.

    Context:
        Public helper used by runtime orchestration, shadow mode, and tuning
        workflows to report which pack is currently authoritative.

    Inputs:
        None: This helper does not require caller-supplied inputs.

    Returns:
        dict[str, Any]: Serializable payload containing the active pack name and
        flattened override mapping.

    Notes:
        The return shape is intentionally simple so it can be embedded directly
        in logs, result payloads, and governance artifacts.
    """

    pack_name, overrides = _resolve_active_overrides()
    return {
        "name": pack_name,
        "overrides": overrides,
    }


def set_active_parameter_pack(
    name: str = DEFAULT_PARAMETER_PACK,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Purpose:
        Install a new active parameter pack for the current execution context.

    Context:
        Public helper used by tests, tuning experiments, and command-line
        workflows before they evaluate one or more snapshots.

    Inputs:
        name (str): Parameter-pack name to activate.
        overrides (dict[str, Any] | None): Optional ad-hoc overrides layered on
            top of the selected pack.

    Returns:
        dict[str, Any]: Serializable description of the newly active pack.

    Notes:
        The update affects only the current context variable scope, which keeps
        nested evaluations isolated.
    """

    _ACTIVE_CONTEXT.set(_build_runtime_context(name, overrides=overrides))
    return get_active_parameter_pack()


@contextmanager
def temporary_parameter_pack(
    name: str = DEFAULT_PARAMETER_PACK,
    *,
    overrides: dict[str, Any] | None = None,
):
    """
    Purpose:
        Temporarily activate a parameter pack for a bounded block of work.

    Context:
        Public context manager used by shadow mode, replay analysis, and tuning
        validation when the same snapshot must be evaluated under multiple packs
        without leaking state across runs.

    Inputs:
        name (str): Parameter-pack name to activate for the duration of the
            context.
        overrides (dict[str, Any] | None): Optional extra overrides layered on
            top of the selected pack.

    Returns:
        Any: Context manager yielding the active pack description.

    Notes:
        Context-variable reset guarantees that live runtime configuration is
        restored even if the nested evaluation raises an exception.
    """

    token = _ACTIVE_CONTEXT.set(_build_runtime_context(name, overrides=overrides))
    try:
        yield get_active_parameter_pack()
    finally:
        _ACTIVE_CONTEXT.reset(token)


def get_parameter_value(key: str, default: Any = _MISSING) -> Any:
    """
    Purpose:
        Resolve one scalar policy value from the active parameter pack.

    Context:
        Public helper for config modules that need one overrideable scalar
        rather than a whole mapping or dataclass.

    Inputs:
        key (str): Fully qualified parameter key, for example
            `trade_strength.direction_thresholds.min_score`.
        default (Any): Fallback value used when the active pack does not define
            the requested override.

    Returns:
        Any: Override value when present, otherwise the supplied default.

    Notes:
        This resolver intentionally does not depend on the tuning registry. The
        caller should therefore provide an explicit default when using it from
        production config code.
    """

    _, overrides = _resolve_active_overrides()
    if key in overrides:
        return overrides[key]
    if default is _MISSING:
        raise KeyError(f"No runtime override found for parameter '{key}' and no default was supplied")
    return default


def resolve_mapping(prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """
    Purpose:
        Resolve a namespaced mapping against the active parameter-pack overrides.

    Context:
        Public helper used by configuration modules whose defaults are stored as
        plain dictionaries.

    Inputs:
        prefix (str): Parameter namespace prefix, for example
            `trade_strength.runtime_thresholds`.
        defaults (dict[str, Any]): Default mapping defined in code.

    Returns:
        dict[str, Any]: Resolved mapping containing defaults plus any active
        overrides under the supplied prefix.

    Notes:
        Override keys are flattened in packs, so this function strips the prefix
        before merging values into the returned policy mapping.
    """

    resolved = dict(defaults)
    _, overrides = _resolve_active_overrides()
    dotted_prefix = f"{prefix}."
    for key, value in overrides.items():
        if key.startswith(dotted_prefix):
            resolved[key[len(dotted_prefix):]] = value
    return resolved


def resolve_dataclass_config(prefix: str, config_obj: Any):
    """
    Purpose:
        Resolve a dataclass-backed configuration object against active overrides.

    Context:
        Public helper used by configuration modules that model policies as
        immutable dataclasses for stronger field-level semantics.

    Inputs:
        prefix (str): Parameter namespace prefix used when matching flattened
            overrides.
        config_obj (Any): Dataclass instance containing code-defined defaults.

    Returns:
        Any: New dataclass instance with resolved field values.

    Notes:
        Field replacement happens through `asdict` so nested dataclass defaults
        remain serializable and easy to compare in tuning artifacts.
    """

    if not is_dataclass(config_obj):
        raise TypeError("resolve_dataclass_config expects a dataclass instance")
    payload = resolve_mapping(prefix, asdict(config_obj))
    return type(config_obj)(**payload)
