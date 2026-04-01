"""
Predictor factory — resolves the active predictor from configuration.

The ``PREDICTION_METHOD`` setting (or ``OQE_PREDICTION_METHOD`` env var)
selects which predictor the engine uses.  A singleton is cached for the
lifetime of the process; call ``reset_predictor()`` to force re-creation
(useful in tests or when hot-swapping at runtime).
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

from engine.predictors.protocol import MovePredictor

logger = logging.getLogger(__name__)

_ACTIVE_PREDICTOR: MovePredictor | None = None

# Registry of built-in predictor names → lazy constructors.
_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> dict[str, type]:
    """Populate the registry on first access (avoids circular imports)."""
    if _REGISTRY:
        return _REGISTRY

    from engine.predictors.builtin_predictors import (
        DefaultBlendedPredictor,
        PureMLPredictor,
        PureRulePredictor,
    )
    from engine.predictors.research_predictor import ResearchDualModelPredictor
    from engine.predictors.decision_policy_predictor import ResearchDecisionPolicyPredictor
    from engine.predictors.ev_sizing_predictor import EVSizingPredictor
    from engine.predictors.rank_gate_predictor import ResearchRankGatePredictor
    from engine.predictors.uncertainty_adjusted_predictor import ResearchUncertaintyAdjustedPredictor

    _REGISTRY["blended"] = DefaultBlendedPredictor
    _REGISTRY["pure_ml"] = PureMLPredictor
    _REGISTRY["pure_rule"] = PureRulePredictor
    _REGISTRY["research_dual_model"] = ResearchDualModelPredictor
    # Keep both keys: `decision_policy` is the production-facing name,
    # `research_decision_policy` remains for backward compatibility.
    _REGISTRY["decision_policy"] = ResearchDecisionPolicyPredictor
    _REGISTRY["research_decision_policy"] = ResearchDecisionPolicyPredictor
    _REGISTRY["ev_sizing"] = EVSizingPredictor
    _REGISTRY["research_rank_gate"] = ResearchRankGatePredictor
    _REGISTRY["research_uncertainty_adjusted"] = ResearchUncertaintyAdjustedPredictor
    return _REGISTRY


def register_predictor(name: str, cls: type) -> None:
    """Register a custom predictor class under *name*."""
    _ensure_registry()
    _REGISTRY[name] = cls


def get_predictor() -> MovePredictor:
    """
    Return the active predictor (singleton).

    Resolution order:
      1. Cached singleton (_ACTIVE_PREDICTOR)
      2. config.settings.PREDICTION_METHOD  (or env OQE_PREDICTION_METHOD)
      3. Falls back to "blended" (current production behaviour)
    """
    global _ACTIVE_PREDICTOR
    if _ACTIVE_PREDICTOR is not None:
        return _ACTIVE_PREDICTOR

    registry = _ensure_registry()

    method = "blended"
    try:
        from config import settings as _settings
        method = getattr(_settings, "PREDICTION_METHOD", "blended") or "blended"
    except Exception:
        pass

    cls = registry.get(method)
    if cls is None:
        logger.warning(
            "Unknown PREDICTION_METHOD=%r — falling back to 'blended'. "
            "Available: %s",
            method,
            ", ".join(sorted(registry)),
        )
        cls = registry["blended"]

    _ACTIVE_PREDICTOR = cls()
    logger.info("Predictor initialised: %s (method=%s)", _ACTIVE_PREDICTOR.name, method)
    return _ACTIVE_PREDICTOR


def reset_predictor() -> None:
    """Clear the singleton so the next ``get_predictor()`` re-reads config."""
    global _ACTIVE_PREDICTOR
    _ACTIVE_PREDICTOR = None


# Thread-local override for temporary predictor swaps (e.g. backtests).
_METHOD_OVERRIDE: str | None = None


@contextlib.contextmanager
def prediction_method_override(method: str) -> Iterator[MovePredictor]:
    """Context manager that temporarily swaps the active predictor.

    On entry the current singleton is saved, a new predictor of the requested
    *method* is created, and all engine calls within the block use it.  On
    exit (including exceptions) the previous predictor is restored.

    Usage::

        from engine.predictors.factory import prediction_method_override

        with prediction_method_override("pure_ml") as pred:
            result = run_preloaded_engine_snapshot(...)
    """
    global _ACTIVE_PREDICTOR, _METHOD_OVERRIDE

    prev_predictor = _ACTIVE_PREDICTOR
    prev_override = _METHOD_OVERRIDE

    registry = _ensure_registry()
    cls = registry.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown prediction method {method!r}. "
            f"Available: {', '.join(sorted(registry))}"
        )

    _ACTIVE_PREDICTOR = cls()
    _METHOD_OVERRIDE = method
    logger.info("Predictor override → %s", _ACTIVE_PREDICTOR.name)
    try:
        yield _ACTIVE_PREDICTOR
    finally:
        _ACTIVE_PREDICTOR = prev_predictor
        _METHOD_OVERRIDE = prev_override
        logger.info("Predictor restored → %s", prev_predictor.name if prev_predictor else "None (will re-init)")
