from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from models.move_predictor import MovePredictor as BaseMovePredictor
except Exception:
    BaseMovePredictor = None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MovePredictor:
    def __init__(self):
        self.base_model = BaseMovePredictor() if BaseMovePredictor is not None else None

    def _sigmoid(self, x: float) -> float:
        x = _clip(float(x), -8.0, 8.0)
        return 1.0 / (1.0 + math.exp(-x))

    def _normalize_row(self, row: Iterable[float]) -> list[float]:
        vals = [float(x) for x in row]

        # Keep backward compatibility with the existing 7-feature expectation.
        while len(vals) < 7:
            vals.append(0.0)

        # Soft clipping / scaling so raw feature magnitude does not dominate.
        gamma_regime = _clip(vals[0], -1.0, 1.0)
        vacuum_signal = _clip(vals[1], -1.0, 1.0)
        hedging_signal = _clip(vals[2], -1.0, 1.0)
        flow_signal = _clip(vals[3], -1.0, 1.0)
        iv_regime = _clip(vals[4], -2.0, 2.0) / 2.0
        range_signal = _clip(vals[5], -2.0, 2.0) / 2.0
        extra_signal = _clip(vals[6], -2.0, 2.0) / 2.0

        return [
            gamma_regime,
            vacuum_signal,
            hedging_signal,
            flow_signal,
            iv_regime,
            range_signal,
            extra_signal,
        ]

    def _heuristic_probability(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probs = []
        for row in arr:
            r = self._normalize_row(row)

            score = (
                0.55 * r[0]
                + 0.80 * r[1]
                + 0.70 * r[2]
                + 0.55 * r[3]
                + 0.35 * r[4]
                + 0.40 * r[5]
                + 0.20 * r[6]
            )

            # Lower-confidence fallback than before; keeps output in a sensible band.
            prob = 0.18 + 0.64 * self._sigmoid(score)
            probs.append(round(float(_clip(prob, 0.05, 0.95)), 2))

        return probs[0] if len(probs) == 1 else probs

    def predict_probability(self, X):
        if self.base_model is not None:
            try:
                result = self.base_model.predict_probability(X)
                return float(result[0]) if hasattr(result, "__len__") else float(result)
            except Exception:
                pass

        return self._heuristic_probability(X)