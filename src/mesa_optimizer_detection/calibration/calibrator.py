from __future__ import annotations

"""Calibration pipeline for mesa-optimizer detection risk scores.

This module implements a lightweight *post-hoc* calibration step that maps the
raw, aggregated risk scores (and the per-method risk scores) produced by the
`MesaOptimizerDetector` into a calibrated probability that the analysed model
contains a mesa-optimizer.

The design purposefully keeps external dependencies optional: if
`scikit-learn` is available we back the calibration model with
`sklearn.linear_model.LogisticRegression`.  Otherwise, the module still
imports successfully but raises a clear error at *fit()* / *predict()* time
so that downstream code can decide to skip calibration gracefully.
"""

from typing import List, Sequence
import logging
import numpy as np

from ..core.results import DetectionResults
from .likelihood_model import LikelihoodModel

logger = logging.getLogger(__name__)

__all__ = ["DetectionCalibrator"]


class DetectionCalibrator:  # pylint: disable=too-few-public-methods
    """Wrap `LikelihoodModel` and provide convenience helpers.

    Parameters
    ----------
    method_order : list[str] | None
        Fixed ordering of detection-method names to form the feature vector.
        If *None*, the order is inferred from the first *fit()* call and then
        frozen.
    """

    def __init__(self, method_order: Sequence[str] | None = None):
        self.method_order: list[str] | None = list(method_order) if method_order else None
        self._model: LikelihoodModel = LikelihoodModel()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _extract_features(self, result: DetectionResults) -> np.ndarray:
        """Convert a *DetectionResults* object into a numeric feature vector.

        Current implementation:
        1. One feature per detection method → the *risk_score* from that method
           (0 if the method is missing/failed).
        2. A final feature with the **overall** aggregated risk score.
        """
        if self.method_order is None:
            # Lazily initialise feature order
            self.method_order = sorted(result.method_results.keys())
            logger.debug("Calibrator inferred method order: %s", self.method_order)

        features: list[float] = []
        for method in self.method_order:
            score = 0.0
            method_result = result.method_results.get(method)
            if method_result is not None and hasattr(method_result, "risk_score"):
                try:
                    score = float(method_result.risk_score)
                except Exception:  # pylint: disable=broad-except
                    pass
            features.append(score)

        # Append overall risk score as last feature
        features.append(float(result.risk_score))
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, results: List[DetectionResults], labels: List[int]):
        """Fit calibrator on a dataset of results with ground-truth labels.

        Parameters
        ----------
        results : list[DetectionResults]
            Detection outputs to use as features.
        labels : list[int]
            Ground-truth binary labels where ``1`` indicates the presence of a
            mesa-optimizer and ``0`` indicates its absence.
        """
        if len(results) != len(labels):
            raise ValueError("results and labels must have the same length")

        X = np.stack([self._extract_features(r) for r in results])
        y = np.asarray(labels, dtype=np.int32)

        self._model.fit(X, y)
        self._fitted = True
        logger.info("DetectionCalibrator fitted on %d samples", len(results))

    def predict_proba(self, result: DetectionResults) -> float:
        """Predict calibrated mesa-optimizer probability for *result*."""
        if not self._fitted:
            raise RuntimeError("DetectionCalibrator has not been fitted yet")
        X = self._extract_features(result)[None, :]
        prob = float(self._model.predict_proba(X)[0])
        logger.debug("Calibrated probability: %.4f", prob)
        return prob

    # ------------------------------------------------------------------
    # serialization helpers – delegate to *LikelihoodModel*
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Persist underlying likelihood model to *path* (via joblib)."""
        self._model.save(path)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str, method_order: Sequence[str] | None = None) -> "DetectionCalibrator":
        """Load previously saved calibrator from *path*."""
        obj = cls(method_order)
        obj._model = LikelihoodModel.load(path)
        obj._fitted = True
        logger.info("Calibrator loaded from %s", path)
        return obj 