from __future__ import annotations

"""Likelihood calibration model (scaffold).

Wraps scikit-learn LogisticRegression (or any binary probabilistic classifier)
behind a stable API so that the core detector doesn't depend on scikit-learn
internals.
"""

from typing import Any
import logging

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # Soft dependency
    LogisticRegression = None  # type: ignore

logger = logging.getLogger(__name__)


class LikelihoodModel:
    def __init__(self):
        if LogisticRegression is None:
            logger.warning("scikit-learn not installed; LikelihoodModel is disabled")
            self.model = None
        else:
            self.model = LogisticRegression(max_iter=1000)

    # ------------------------------------------------------------------
    def fit(self, X, y):  # X shape (n_samples, n_features)
        if self.model is None:
            raise RuntimeError("LikelihoodModel requires scikit-learn")
        self.model.fit(X, y)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("LikelihoodModel requires scikit-learn")
        return self.model.predict_proba(X)[:, 1]  # probability of class 1

    def save(self, path: str):
        import joblib, os
        if self.model is None:
            raise RuntimeError("LikelihoodModel requires scikit-learn")
        joblib.dump(self.model, path)
        logger.info(f"Saved likelihood model → {path}")

    @classmethod
    def load(cls, path: str) -> "LikelihoodModel":
        import joblib, os
        obj = cls()
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn not installed, cannot load model")
        obj.model = joblib.load(path)
        return obj 