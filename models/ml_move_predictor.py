"""
Module: ml_move_predictor.py

Purpose:
    Implement ML move predictor modeling logic used by predictive or heuristic components.

Role in the System:
    Part of the modeling layer that builds statistical features and predictive estimates.

Key Outputs:
    Model-ready feature sets, fitted estimators, or predictive outputs.

Downstream Usage:
    Consumed by analytics, the probability stack, and research workflows.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

def _clip(x: float, lo: float, hi: float) -> float:
    """
    Purpose:
        Clamp a numeric value to the configured bounds.

    Context:
        Function inside the `ml move predictor` module. The module sits in the modeling layer that turns features into probabilities, scores, and predictive outputs.

    Inputs:
        x (float): Raw scalar input supplied by the caller.
        lo (float): Inclusive lower bound for the returned value.
        hi (float): Inclusive upper bound for the returned value.

    Returns:
        float | int: Bounded value returned by the helper.

    Notes:
        Internal helper that keeps the surrounding implementation focused on higher-level trading logic.
    """
    return max(lo, min(hi, x))


class MovePredictor:
    """
    Purpose:
        Represent MovePredictor within the repository.
    
    Context:
        Used within the `ml move predictor` module. The class participates in the module's role within the trading system.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        The class groups behavior and state that need to stay explicit for maintainability and auditability.
    """
    def __init__(self, base_model=None):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `MovePredictor` within the modeling layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            base_model (Any): Input associated with base model.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        self.base_model = base_model

    def _sigmoid(self, x: float) -> float:
        """
        Purpose:
            Process sigmoid for downstream use.
        
        Context:
            Method on `MovePredictor` within the modeling layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            x (float): Input associated with x.
        
        Returns:
            float: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        x = _clip(float(x), -8.0, 8.0)
        return 1.0 / (1.0 + math.exp(-x))

    def _normalize_row(self, row: Iterable[float]) -> list[float]:
        """
        Purpose:
            Normalize row into the repository-standard form.
        
        Context:
            Method on `MovePredictor` within the modeling layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            row (Iterable[float]): Input associated with row.
        
        Returns:
            list[float]: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        vals = [float(x) for x in row]

        # Keep backward compatibility with the existing 7-feature expectation.
        while len(vals) < 7:
            vals.append(0.0)

        # Soft clipping / scaling so raw feature magnitude does not dominate.
        gamma_regime = _clip(vals[0], -1.0, 1.0)
        flow_signal = _clip(vals[1], -1.0, 1.0)
        vol_signal = _clip(vals[2], -1.0, 1.0)
        hedging_signal = _clip(vals[3], -1.0, 1.0)
        spot_flip_signal = _clip(vals[4], -1.0, 1.0)
        vacuum_signal = _clip(vals[5], -1.0, 1.0)
        iv_regime = _clip(vals[6], 0.0, 1.0)

        return [
            gamma_regime,
            flow_signal,
            vol_signal,
            hedging_signal,
            spot_flip_signal,
            vacuum_signal,
            iv_regime,
        ]

    def _heuristic_probability(self, X):
        """
        Purpose:
            Process heuristic probability for downstream use.
        
        Context:
            Method on `MovePredictor` within the modeling layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            X (Any): Input associated with X.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probs = []
        for row in arr:
            r = self._normalize_row(row)

            score = (
                0.55 * r[0]
                + 0.65 * r[1]
                + 0.35 * r[2]
                + 0.70 * r[3]
                + 0.45 * r[4]
                + 0.55 * r[5]
                + 0.25 * r[6]
            )

            # Lower-confidence fallback than before; keeps output in a sensible band.
            prob = 0.18 + 0.64 * self._sigmoid(score)
            probs.append(round(float(_clip(prob, 0.05, 0.95)), 2))

        return probs[0] if len(probs) == 1 else probs

    def predict_probability(self, X):
        """
        Purpose:
            Process predict probability for downstream use.
        
        Context:
            Method on `MovePredictor` within the modeling layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            X (Any): Input associated with X.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The helper keeps the surrounding module readable without changing runtime behavior.
        """
        if self.base_model is not None:
            try:
                result = self.base_model.predict_probability(X)
                return float(result[0]) if hasattr(result, "__len__") else float(result)
            except Exception:
                pass

        return self._heuristic_probability(X)
