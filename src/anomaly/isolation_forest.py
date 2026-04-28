"""
Isolation Forest anomaly detector.
Best for: tabular / IoT data with sparse, point anomalies.
Fast, unsupervised, no training labels required.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest as _IsolationForest
from sklearn.preprocessing import MinMaxScaler

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Wraps scikit-learn IsolationForest with a normalised [0,1] anomaly score
    and the shared BaseDetector interface.

    Args:
        contamination: Expected proportion of anomalies in training data.
                       Used to calibrate the decision threshold automatically.
        n_estimators:  Number of isolation trees.
        threshold:     Score cutoff above which a point is flagged as anomalous.
                       Defaults to 0.5 (mid-range after normalisation).
        random_state:  Reproducibility seed.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        super().__init__(threshold=threshold)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model = _IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = MinMaxScaler()

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on (presumed) normal training data.

        Args:
            X: Shape (n_samples,) or (n_samples, n_features).
        """
        X = self._reshape(X)
        self._model.fit(X)

        # Calibrate scaler on raw decision scores so predict() is consistent
        raw_scores = -self._model.score_samples(X)      # negative → higher = worse
        self._scaler.fit(raw_scores.reshape(-1, 1))

        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalised anomaly scores in [0, 1].
        Higher score → more anomalous.
        """
        self._check_fitted()
        X = self._reshape(X)
        raw = -self._model.score_samples(X)
        normalised = self._scaler.transform(raw.reshape(-1, 1)).flatten()
        return np.clip(normalised, 0, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
