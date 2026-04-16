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
import time
from typing import Any


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
    layers: tuple[str, ...] = ()


_DEFAULT_PACK_NAME = os.getenv("OQE_PARAMETER_PACK", DEFAULT_PARAMETER_PACK)
_ACTIVE_CONTEXT: ContextVar[ParameterRuntimeContext | None] = ContextVar(
    "oqe_active_parameter_context",
    default=None,
)
_MISSING = object()

# Per-context cache for resolved dataclass configs.  The cache is keyed by
# (context_name, prefix, config_class).  It auto‑invalidates whenever the
# active context object changes (e.g. via ``set_active_parameter_pack`` or
# ``temporary_parameter_pack``).
_config_cache: dict[tuple, Any] = {}
_config_cache_context_id: int | None = None

_REGIME_MAP_FIELDS = (
    "gamma_regime",
    "vol_regime",
    "global_risk_state",
    "macro_regime",
    "event_risk_bucket",
    "overnight_bucket",
)


def _parse_pack_layers(selection: Any) -> tuple[str, ...]:
    """Parse a pack selection into normalized ordered layer names."""
    if selection is None:
        return ()
    if isinstance(selection, (list, tuple, set)):
        raw_items = [str(item or "").strip() for item in selection]
    else:
        text = str(selection or "").strip()
        if not text:
            return ()
        # Support both "a+b" and "a,b" composition syntaxes.
        text = text.replace(",", "+")
        raw_items = [part.strip() for part in text.split("+")]
    return tuple(item for item in raw_items if item)


def _format_pack_layers(layers: tuple[str, ...]) -> str:
    return "+".join(layers)


def _load_pack_overrides(name: Any) -> tuple[str, dict[str, Any], tuple[str, ...]]:
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

    requested_layers = _parse_pack_layers(name)
    if not requested_layers:
        requested_layers = (DEFAULT_PARAMETER_PACK,)

    from tuning.packs import resolve_parameter_pack

    import logging
    logger = logging.getLogger(__name__)

    merged_overrides: dict[str, Any] = {}
    resolved_layers: list[str] = []
    for layer in requested_layers:
        try:
            pack = resolve_parameter_pack(layer)
            merged_overrides.update(dict(pack.overrides))
            resolved_layers.append(layer)
        except Exception as exc:
            logger.error("Failed to load parameter pack layer '%s': %s", layer, exc)

    if not resolved_layers:
        if DEFAULT_PARAMETER_PACK in requested_layers:
            return DEFAULT_PARAMETER_PACK, {}, (DEFAULT_PARAMETER_PACK,)
        logger.warning("Falling back to default pack '%s'", DEFAULT_PARAMETER_PACK)
        try:
            fallback = resolve_parameter_pack(DEFAULT_PARAMETER_PACK)
            return DEFAULT_PARAMETER_PACK, dict(fallback.overrides), (DEFAULT_PARAMETER_PACK,)
        except Exception as exc2:
            logger.critical("Even default pack failed to load: %s", exc2)
            return DEFAULT_PARAMETER_PACK, {}, (DEFAULT_PARAMETER_PACK,)

    layers_tuple = tuple(resolved_layers)
    return _format_pack_layers(layers_tuple), merged_overrides, layers_tuple


def _build_runtime_context(
    name: Any = DEFAULT_PARAMETER_PACK,
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

    pack_name, base_overrides, layers = _load_pack_overrides(name)
    if overrides:
        base_overrides.update(dict(overrides))
    return ParameterRuntimeContext(name=pack_name, overrides=base_overrides, layers=layers)


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

    context = _resolve_active_context()
    pack_name = context.name
    overrides = dict(context.overrides)
    return {
        "name": pack_name,
        "overrides": overrides,
        "layers": list(context.layers),
    }


def set_active_parameter_pack(
    name: Any = DEFAULT_PARAMETER_PACK,
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
    name: Any = DEFAULT_PARAMETER_PACK,
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
    context = _resolve_active_context()
    overrides = context.overrides  # read-only access; no copy needed
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
    global _config_cache, _config_cache_context_id

    if not is_dataclass(config_obj):
        raise TypeError("resolve_dataclass_config expects a dataclass instance")

    context = _resolve_active_context()
    ctx_id = id(context)

    # Invalidate cache when the active context changes.
    if _config_cache_context_id != ctx_id:
        _config_cache = {}
        _config_cache_context_id = ctx_id

    cache_key = (prefix, type(config_obj))
    cached = _config_cache.get(cache_key)
    if cached is not None:
        return cached

    payload = resolve_mapping(prefix, asdict(config_obj))
    result = type(config_obj)(**payload)
    _config_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Regime-conditional parameter auto-switching
# ---------------------------------------------------------------------------


def _load_regime_auto_pack_config(config_path: str | None = None) -> dict[str, Any] | None:
    """Load regime auto-pack configuration from JSON.

    Returns None on file-not-found or parse errors.
    """
    import json as _json
    import logging as _logging
    from pathlib import Path as _Path

    log = _logging.getLogger(__name__)
    map_path = _Path(config_path) if config_path else (_Path(__file__).resolve().parent / "regime_auto_pack_map.json")

    if not map_path.exists():
        return None

    try:
        payload = _json.loads(map_path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        log.warning("regime_auto_pack: failed to load map - %s", exc)
    return None


def _normalize_label(value: Any) -> str:
    return str(value or "").upper().strip()


def get_regime_switch_policy(config_path: str | None = None) -> dict[str, Any]:
    """Return runtime switch guards for regime-based pack switching.

    Policy fields are read from ``switch_policy`` in
    ``regime_auto_pack_map.json`` and merged with safe defaults.
    """
    defaults = {
        "required_consecutive": 2,
        "cooldown_seconds": 300,
        "min_dwell_seconds": 300,
        "min_regime_confidence": 0.0,
        "shadow_enabled": False,
        "shadow_min_regime_confidence": 0.0,
        "decision_disagreement_alert": 0.20,
        "trade_status_disagreement_alert": 0.25,
        "signal_presence_disagreement_alert": 0.15,
        "overnight_disagreement_alert": 0.20,
        "session_alert_min_snapshots": 2,
        "log_decisions": True,
        "decision_log_path": "logs/regime_switch_decisions.jsonl",
    }
    config = _load_regime_auto_pack_config(config_path=config_path) or {}
    raw = config.get("switch_policy", {}) if isinstance(config, dict) else {}

    try:
        defaults["required_consecutive"] = max(int(raw.get("required_consecutive", defaults["required_consecutive"])), 1)
    except Exception:
        pass
    try:
        defaults["cooldown_seconds"] = max(int(raw.get("cooldown_seconds", defaults["cooldown_seconds"])), 0)
    except Exception:
        pass
    try:
        defaults["min_dwell_seconds"] = max(int(raw.get("min_dwell_seconds", defaults["min_dwell_seconds"])), 0)
    except Exception:
        pass
    try:
        defaults["min_regime_confidence"] = float(raw.get("min_regime_confidence", defaults["min_regime_confidence"]))
    except Exception:
        pass
    defaults["shadow_enabled"] = bool(config.get("shadow_enabled", raw.get("shadow_enabled", defaults["shadow_enabled"])))
    try:
        defaults["shadow_min_regime_confidence"] = float(
            raw.get(
                "shadow_min_regime_confidence",
                config.get("shadow_min_regime_confidence", defaults["shadow_min_regime_confidence"]),
            )
        )
    except Exception:
        pass
    for key in (
        "decision_disagreement_alert",
        "trade_status_disagreement_alert",
        "signal_presence_disagreement_alert",
        "overnight_disagreement_alert",
    ):
        try:
            defaults[key] = float(raw.get(key, config.get(key, defaults[key])))
        except Exception:
            pass
    try:
        defaults["session_alert_min_snapshots"] = max(
            int(raw.get("session_alert_min_snapshots", config.get("session_alert_min_snapshots", defaults["session_alert_min_snapshots"]))),
            1,
        )
    except Exception:
        pass
    defaults["log_decisions"] = bool(raw.get("log_decisions", defaults["log_decisions"]))
    defaults["decision_log_path"] = str(raw.get("decision_log_path", defaults["decision_log_path"]))
    return defaults


def suggest_regime_pack(
    gamma_regime: str | None,
    vol_regime: str | None,
    *,
    global_risk_state: str | None = None,
    macro_regime: str | None = None,
    event_risk_bucket: str | None = None,
    overnight_bucket: str | None = None,
    config_path: str | None = None,
    evaluation_mode: str = "live",
) -> str | None:
    """Suggest the parameter pack best suited to the current live regime.

    Reads ``config/regime_auto_pack_map.json``.  Returns ``None`` when
    auto-switching is disabled, the map is missing, or no entry matches the
    supplied regimes.  The caller decides whether to apply the suggestion via
    ``set_active_parameter_pack``.

    Parameters
    ----------
    gamma_regime:
        Gamma regime label, e.g. ``"NEGATIVE_GAMMA"`` or ``"POSITIVE_GAMMA"``.
    vol_regime:
        Volatility regime label, e.g. ``"HIGH_VOL"`` or ``"NORMAL_VOL"``.

    Returns
    -------
    str | None: Suggested pack name, or ``None`` when no switch is needed.
    """
    config = _load_regime_auto_pack_config(config_path=config_path)
    if not isinstance(config, dict):
        return None

    evaluation_mode = str(evaluation_mode or "live").strip().lower()
    live_enabled = bool(config.get("enabled", False))
    shadow_enabled = bool(config.get("shadow_enabled", False))

    if not live_enabled:
        if evaluation_mode != "shadow" or not shadow_enabled:
            return None

    fallback = config.get("fallback_pack", DEFAULT_PARAMETER_PACK)
    labels = {
        "gamma_regime": _normalize_label(gamma_regime),
        "vol_regime": _normalize_label(vol_regime),
        "global_risk_state": _normalize_label(global_risk_state),
        "macro_regime": _normalize_label(macro_regime),
        "event_risk_bucket": _normalize_label(event_risk_bucket),
        "overnight_bucket": _normalize_label(overnight_bucket),
    }

    best_pack: str | None = None
    best_rank = (-1, -1)

    for entry in config.get("map", []):
        if not isinstance(entry, dict):
            continue
        if "pack_layers" in entry:
            parsed_layers = _parse_pack_layers(entry.get("pack_layers"))
            pack = _format_pack_layers(parsed_layers) if parsed_layers else str(entry.get("pack", fallback))
        else:
            pack = str(entry.get("pack", fallback))
        priority = 0
        try:
            priority = int(entry.get("priority", 0))
        except Exception:
            priority = 0

        specificity = 0
        matched = True
        for field in _REGIME_MAP_FIELDS:
            expected = _normalize_label(entry.get(field, "*"))
            if expected in {"", "*"}:
                continue
            if labels[field] != expected:
                matched = False
                break
            specificity += 1

        if not matched:
            continue

        rank = (specificity, priority)
        if rank > best_rank:
            best_rank = rank
            best_pack = pack

    return best_pack


def evaluate_regime_pack_switch(
    *,
    suggested_pack: str | None,
    current_pack: str | None,
    regime_signature: str,
    switch_state: dict[str, Any] | None,
    required_consecutive: int,
    cooldown_seconds: int,
    min_dwell_seconds: int,
    regime_confidence: float | None = None,
    min_regime_confidence: float = 0.0,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Gate regime-driven pack switching to avoid noisy flip-flops.

    Returns a dict with:
    - apply: bool
    - reason: str
    - state: updated switch-state dict
    """
    now = float(now_ts) if now_ts is not None else time.time()
    state = dict(switch_state or {})
    state.setdefault("last_regime_signature", "")
    state.setdefault("consecutive_regime_hits", 0)
    state.setdefault("last_switch_ts", None)
    state.setdefault("active_since_ts", now)

    normalized_suggested = str(suggested_pack or "").strip()
    normalized_current = str(current_pack or "").strip()

    if regime_signature == state["last_regime_signature"]:
        state["consecutive_regime_hits"] = int(state.get("consecutive_regime_hits", 0)) + 1
    else:
        state["last_regime_signature"] = regime_signature
        state["consecutive_regime_hits"] = 1

    if not normalized_suggested:
        return {"apply": False, "reason": "no_suggestion", "state": state}

    if normalized_suggested == normalized_current:
        return {"apply": False, "reason": "already_active", "state": state}

    if regime_confidence is not None and float(regime_confidence) < float(min_regime_confidence):
        return {"apply": False, "reason": "confidence_below_floor", "state": state}

    if int(state["consecutive_regime_hits"]) < max(int(required_consecutive), 1):
        return {"apply": False, "reason": "insufficient_consecutive", "state": state}

    last_switch_ts = state.get("last_switch_ts")
    if last_switch_ts is not None and (now - float(last_switch_ts)) < max(int(cooldown_seconds), 0):
        return {"apply": False, "reason": "cooldown_active", "state": state}

    active_since_ts = state.get("active_since_ts")
    if (
        last_switch_ts is not None
        and active_since_ts is not None
        and (now - float(active_since_ts)) < max(int(min_dwell_seconds), 0)
    ):
        return {"apply": False, "reason": "min_dwell_active", "state": state}

    state["last_switch_ts"] = now
    state["active_since_ts"] = now
    state["consecutive_regime_hits"] = 0
    return {"apply": True, "reason": "switch_approved", "state": state}

