"""
Base interface for all anomaly detectors.
Every detector implements fit(), predict(), and score().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AnomalyResult:
    """Result object returned by all detectors."""
    timestamps: list
    values: np.ndarray
    scores: np.ndarray          # raw anomaly scores (higher = more anomalous)
    is_anomaly: np.ndarray      # boolean mask
    threshold: float
    model_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def anomaly_indices(self) -> np.ndarray:
        return np.where(self.is_anomaly)[0]

    @property
    def anomaly_count(self) -> int:
        return int(self.is_anomaly.sum())

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "anomaly_count": self.anomaly_count,
            "anomalies": [
                {
                    "timestamp": self.timestamps[i],
                    "value": float(self.values[i]),
                    "score": float(self.scores[i]),
                    "is_anomaly": bool(self.is_anomaly[i]),
                }
                for i in range(len(self.timestamps))
            ],
        }


class BaseDetector(ABC):
    """Abstract base class for all anomaly detection models."""

    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """
        Fit the model on normal (training) data.

        Args:
            X: Array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for each sample.
        Higher scores = more anomalous.

        Args:
            X: Array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            scores: Array of shape (n_samples,) with values in [0, 1]
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return boolean anomaly flags using self.threshold.

        Args:
            X: Input array

        Returns:
            Boolean array, True where anomalous
        """
        scores = self.score(X)
        return scores > self.threshold

    def detect(
        self,
        X: np.ndarray,
        timestamps: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> AnomalyResult:
        """
        End-to-end detection: score + flag + wrap in AnomalyResult.

        Args:
            X: Input time series array
            timestamps: Optional list of timestamps aligned with X
            metadata: Optional context dict passed through to AnomalyResult

        Returns:
            AnomalyResult
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling detect(). Run fit() first.")

        if timestamps is None:
            timestamps = list(range(len(X)))

        scores = self.score(X)
        flags = scores > self.threshold

        return AnomalyResult(
            timestamps=timestamps,
            values=np.asarray(X),
            scores=scores,
            is_anomaly=flags,
            threshold=self.threshold,
            model_name=self.__class__.__name__,
            metadata=metadata or {},
        )

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")
