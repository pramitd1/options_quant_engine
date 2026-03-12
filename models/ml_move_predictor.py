import numpy as np

try:
    from models.move_predictor import MovePredictor as BaseMovePredictor
except Exception:
    BaseMovePredictor = None


class MovePredictor:
    def __init__(self):
        self.base_model = BaseMovePredictor() if BaseMovePredictor is not None else None

    def _heuristic_probability(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probs = []
        for row in arr:
            score = (
                0.7 * row[0] +
                1.0 * row[1] +
                0.8 * row[2] +
                0.7 * row[3] +
                0.5 * row[4] +
                0.6 * row[5] +
                0.2 * row[6]
            )
            prob = 1 / (1 + np.exp(-score))
            probs.append(round(float(prob), 2))
        return probs[0] if len(probs) == 1 else probs

    def predict_probability(self, X):
        if self.base_model is not None:
            try:
                result = self.base_model.predict_probability(X)
                return float(result[0]) if hasattr(result, "__len__") else float(result)
            except Exception:
                pass
        return self._heuristic_probability(X)