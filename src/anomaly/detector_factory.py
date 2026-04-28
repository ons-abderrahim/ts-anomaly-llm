"""
Factory for creating anomaly detectors from config or string name.
"""

from __future__ import annotations

from typing import Literal, Union

from .base import BaseDetector
from .isolation_forest import IsolationForestDetector
from .lstm_autoencoder import LSTMAEDetector
from .transformer_ae import TransformerAEDetector

ModelName = Literal["isolation_forest", "lstm_ae", "transformer_ae"]

_REGISTRY: dict[str, type[BaseDetector]] = {
    "isolation_forest": IsolationForestDetector,
    "lstm_ae": LSTMAEDetector,
    "transformer_ae": TransformerAEDetector,
}


def get_detector(
    model: Union[ModelName, str],
    **kwargs,
) -> BaseDetector:
    """
    Instantiate an anomaly detector by name.

    Args:
        model: One of "isolation_forest", "lstm_ae", "transformer_ae".
        **kwargs: Passed directly to the detector constructor.

    Returns:
        Unfitted detector instance.

    Raises:
        ValueError: If model name is not recognised.

    Example:
        >>> detector = get_detector("lstm_ae", window_size=60, epochs=50)
        >>> detector.fit(train_data)
        >>> result = detector.detect(test_data, timestamps=ts_list)
    """
    if model not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model](**kwargs)


def list_detectors() -> list[str]:
    """Return available detector names."""
    return list(_REGISTRY.keys())
