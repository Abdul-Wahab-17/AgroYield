"""Picklable model wrappers shared by training and the API.

A saved model is a fitted preprocessing step plus an estimator trained on log(yield); `predict`
returns yields in the series' own units. `Blend` averages several models in log space.
"""
import numpy as np


class YieldModel:
    def __init__(self, prep, estimator, name, params=None):
        self.prep, self.estimator, self.name, self.params = prep, estimator, name, params or {}

    def predict_log(self, X):
        return np.asarray(self.estimator.predict(self.prep.transform(X)), dtype="float64")

    def predict(self, X):
        return np.exp(self.predict_log(X))


class Blend:
    """Weighted average of YieldModels in log space (weights learned on validation years)."""

    def __init__(self, models, weights, name="Blend"):
        self.models, self.weights, self.name = list(models), np.asarray(weights, dtype="float64"), name
        self.params = {"members": [m.name for m in self.models], "weights": self.weights.round(4).tolist()}

    def predict_log(self, X):
        return sum(w * m.predict_log(X) for w, m in zip(self.weights, self.models))

    def predict(self, X):
        return np.exp(self.predict_log(X))
