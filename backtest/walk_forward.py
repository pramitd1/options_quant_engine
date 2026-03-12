import numpy as np

from config.settings import WF_TRAIN_RATIO, WF_MIN_TRAIN_SAMPLES
from models.move_predictor import MovePredictor


def walk_forward_retrain(feature_rows, labels):
    """
    Simple walk-forward retraining scaffold.
    feature_rows: list/array of feature vectors
    labels: list/array of 0/1 labels
    """
    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(labels, dtype=int)

    if len(X) < WF_MIN_TRAIN_SAMPLES:
        return None

    split = max(int(len(X) * WF_TRAIN_RATIO), WF_MIN_TRAIN_SAMPLES)
    if split >= len(X):
        return None

    model = MovePredictor()
    model.train(X[:split], y[:split])

    probs = model.predict_probability(X[split:])
    return {
        "train_size": int(split),
        "test_size": int(len(X) - split),
        "predictions": probs
    }