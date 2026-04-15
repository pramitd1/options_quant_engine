"""
GBT Model Loader — Ranking Model
==================================
Loads the GBT_shallow_v1 model from the registry and provides
predict_rank_score() for research inference.

This model has higher AUC and better quintile spread, making it
the preferred RANKING model in the dual-model architecture.

RESEARCH ONLY — does not affect production decisions.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import joblib
import numpy as np

from research.ml_models.ml_config import (
    GBT_META_PATH,
    GBT_MODEL_NAME,
    GBT_MODEL_PATH,
    ML_RESEARCH_ENABLED,
)

logger = logging.getLogger(__name__)

_cached_model = None
_cached_meta = None
_UNAVAILABLE = object()
_WARN_ONCE_KEYS: set[str] = set()


def _is_known_model_compatibility_issue(exc: Exception) -> bool:
    text = str(exc)
    compatibility_markers = (
        "not a known BitGenerator module",
        "Trying to unpickle estimator",
        "numpy.random",
        "unsupported pickle protocol",
        "No module named",
        "has no attribute",
    )
    return any(marker in text for marker in compatibility_markers)


def _warn_once(key: str, message: str, *args) -> None:
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    logger.warning(message, *args)


def _load_model():
    """Load GBT model and metadata from registry. Cached after first call."""
    global _cached_model, _cached_meta

    if _cached_model is _UNAVAILABLE:
        return None, None

    if _cached_model is not None:
        return _cached_model, _cached_meta

    if not GBT_MODEL_PATH.exists():
        logger.warning("GBT model not found at %s", GBT_MODEL_PATH)
        _cached_model = _UNAVAILABLE
        return None, None

    try:
        import warnings as _w
        with _w.catch_warnings(record=True) as _caught:
            _w.simplefilter("always")
            _cached_model = joblib.load(GBT_MODEL_PATH)
        for _cw in _caught:
            logger.warning(
                "sklearn version mismatch loading GBT model "
                "— rebuild with `python scripts/build_model_registry.py`: %s",
                _cw.message,
            )
        mismatch_warnings = [
            str(_cw.message)
            for _cw in _caught
            if "Trying to unpickle estimator" in str(_cw.message)
        ]
        if mismatch_warnings:
            _cached_model = _UNAVAILABLE
            _cached_meta = None
            _warn_once(
                "gbt_model_version_incompatible",
                "GBT model artifact is version-incompatible; disabling GBT inference until rebuilt.",
            )
            return None, None
        if GBT_META_PATH.exists():
            with open(GBT_META_PATH) as f:
                _cached_meta = json.load(f)
        logger.info("Loaded GBT ranking model: %s", GBT_MODEL_NAME)
        return _cached_model, _cached_meta
    except Exception as exc:
        if _is_known_model_compatibility_issue(exc):
            _warn_once(
                "gbt_model_known_compatibility_issue",
                "GBT model artifact is incompatible with the current runtime; disabling GBT inference until rebuilt: %s",
                exc,
            )
        else:
            logger.exception("Failed to load GBT model from %s", GBT_MODEL_PATH)
        _cached_model = _UNAVAILABLE
        _cached_meta = None
        return None, None


def get_feature_mask() -> Optional[list[bool]]:
    """Return the feature mask from model metadata, or None."""
    _, meta = _load_model()
    if meta is None:
        return None
    return meta.get("feature_mask")


def get_feature_names() -> Optional[list[str]]:
    """Return the active feature names from model metadata."""
    _, meta = _load_model()
    if meta is None:
        return None
    return meta.get("feature_names")


def predict_rank_score(feature_vector: np.ndarray) -> Optional[float]:
    """
    Run the GBT ranking model on a 33-element feature vector.

    Returns the predicted probability (class-1) as ml_rank_score,
    or None if the model is unavailable or inference fails.
    """
    if not ML_RESEARCH_ENABLED:
        return None

    model, meta = _load_model()
    if model is None:
        return None

    arr = np.asarray(feature_vector, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    try:
        # TrainedMovePredictor handles feature masking internally
        from models.trained_predictor import TrainedMovePredictor
        if isinstance(model, TrainedMovePredictor):
            result = model.predict_probability(arr[0])
            return round(float(result), 4) if result is not None else None

        # Fallback for raw sklearn models
        mask = meta.get("feature_mask") if meta else None
        n_expected = len(meta.get("feature_names", [])) if meta else arr.shape[1]
        if mask is not None and arr.shape[1] == 33 and n_expected < 33:
            arr = arr[:, mask]
        proba = model.predict_proba(arr)[:, 1]
        return round(float(proba[0]), 4)
    except Exception as exc:
        _warn_once("gbt_inference_failed", "GBT inference failed (suppressing further): %s", exc)
        return None


def predict_rank_scores_batch(feature_matrix: np.ndarray) -> Optional[np.ndarray]:
    """
    Batch inference for multiple rows. Returns array of rank scores or None.
    """
    if not ML_RESEARCH_ENABLED:
        return None

    model, meta = _load_model()
    if model is None:
        return None

    arr = np.asarray(feature_matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    try:
        from models.trained_predictor import TrainedMovePredictor
        if isinstance(model, TrainedMovePredictor):
            results = [model.predict_probability(row) for row in arr]
            return np.array([r if r is not None else np.nan for r in results])

        mask = meta.get("feature_mask") if meta else None
        n_expected = len(meta.get("feature_names", [])) if meta else arr.shape[1]
        if mask is not None and arr.shape[1] == 33 and n_expected < 33:
            arr = arr[:, mask]
        proba = model.predict_proba(arr)[:, 1]
        return np.round(proba, 4)
    except Exception as exc:
        _warn_once("gbt_batch_inference_failed", "GBT batch inference failed (suppressing further): %s", exc)
        return None
