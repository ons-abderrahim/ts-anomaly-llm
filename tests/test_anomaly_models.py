"""
Tests for all three anomaly detection models.
Run with: pytest tests/test_anomaly_models.py -v
"""

import numpy as np
import pytest

from src.anomaly import (
    IsolationForestDetector,
    LSTMAEDetector,
    TransformerAEDetector,
    get_detector,
    list_detectors,
)
from src.anomaly.base import AnomalyResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_series() -> np.ndarray:
    """200 points of clean sinusoidal + Gaussian noise (no anomalies)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 200)
    return np.sin(t) + rng.normal(0, 0.05, 200)


@pytest.fixture
def series_with_spike(normal_series) -> tuple[np.ndarray, int]:
    """Same series with a hard spike injected at index 150."""
    s = normal_series.copy()
    s[150] = 20.0
    return s, 150


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestDetectorFactory:
    def test_list_detectors(self):
        names = list_detectors()
        assert "isolation_forest" in names
        assert "lstm_ae" in names
        assert "transformer_ae" in names

    def test_get_known_detector(self):
        det = get_detector("isolation_forest")
        assert isinstance(det, IsolationForestDetector)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_detector("magic_detector")

    def test_kwargs_forwarded(self):
        det = get_detector("isolation_forest", threshold=0.7, n_estimators=50)
        assert det.threshold == 0.7
        assert det.n_estimators == 50


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

class TestIsolationForest:
    def test_fit_returns_self(self, normal_series):
        det = IsolationForestDetector()
        result = det.fit(normal_series)
        assert result is det
        assert det._is_fitted

    def test_score_shape(self, normal_series):
        det = IsolationForestDetector().fit(normal_series)
        scores = det.score(normal_series)
        assert scores.shape == (len(normal_series),)

    def test_score_in_range(self, normal_series):
        det = IsolationForestDetector().fit(normal_series)
        scores = det.score(normal_series)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_detects_spike(self, normal_series, series_with_spike):
        series, spike_idx = series_with_spike
        det = IsolationForestDetector(threshold=0.5).fit(normal_series)
        result = det.detect(series)
        assert isinstance(result, AnomalyResult)
        # Spike index should be flagged
        assert result.is_anomaly[spike_idx], "Spike was not detected"

    def test_predict_before_fit_raises(self):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            det.predict(np.array([1.0, 2.0]))

    def test_multivariate_input(self, normal_series):
        X = np.column_stack([normal_series, normal_series * 0.5])
        det = IsolationForestDetector().fit(X)
        scores = det.score(X)
        assert scores.shape == (len(normal_series),)


# ---------------------------------------------------------------------------
# LSTM AE (fast settings for tests)
# ---------------------------------------------------------------------------

class TestLSTMAE:
    @pytest.fixture
    def fast_detector(self):
        return LSTMAEDetector(window_size=10, hidden_dim=16, num_layers=1, epochs=3, batch_size=16)

    def test_fit_and_score(self, fast_detector, normal_series):
        fast_detector.fit(normal_series)
        scores = fast_detector.score(normal_series)
        assert scores.shape == (len(normal_series),)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_detects_spike(self, fast_detector, normal_series, series_with_spike):
        series, spike_idx = series_with_spike
        fast_detector.fit(normal_series)
        result = fast_detector.detect(series)
        assert result.scores[spike_idx] > result.scores[:100].mean(), (
            "Spike score should exceed baseline mean score"
        )

    def test_anomaly_result_fields(self, fast_detector, normal_series):
        fast_detector.fit(normal_series)
        timestamps = [str(i) for i in range(len(normal_series))]
        result = fast_detector.detect(normal_series, timestamps=timestamps)
        assert result.model_name == "LSTMAEDetector"
        assert len(result.timestamps) == len(normal_series)
        assert result.threshold == fast_detector.threshold


# ---------------------------------------------------------------------------
# Transformer AE (fast settings for tests)
# ---------------------------------------------------------------------------

class TestTransformerAE:
    @pytest.fixture
    def fast_detector(self):
        return TransformerAEDetector(
            window_size=10, d_model=16, nhead=2,
            num_encoder_layers=1, dim_feedforward=32,
            epochs=3, batch_size=16,
        )

    def test_fit_and_score(self, fast_detector, normal_series):
        fast_detector.fit(normal_series)
        scores = fast_detector.score(normal_series)
        assert scores.shape == (len(normal_series),)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_detects_spike(self, fast_detector, normal_series, series_with_spike):
        series, spike_idx = series_with_spike
        fast_detector.fit(normal_series)
        scores = fast_detector.score(series)
        assert scores[spike_idx] > scores[:100].mean()


# ---------------------------------------------------------------------------
# AnomalyResult
# ---------------------------------------------------------------------------

class TestAnomalyResult:
    def test_to_dict(self):
        result = AnomalyResult(
            timestamps=["t0", "t1", "t2"],
            values=np.array([1.0, 5.0, 1.0]),
            scores=np.array([0.1, 0.9, 0.1]),
            is_anomaly=np.array([False, True, False]),
            threshold=0.5,
            model_name="TestModel",
        )
        d = result.to_dict()
        assert d["anomaly_count"] == 1
        assert d["model"] == "TestModel"
        assert d["anomalies"][1]["is_anomaly"] is True

    def test_anomaly_indices(self):
        result = AnomalyResult(
            timestamps=list(range(5)),
            values=np.zeros(5),
            scores=np.array([0.1, 0.8, 0.2, 0.9, 0.1]),
            is_anomaly=np.array([False, True, False, True, False]),
            threshold=0.5,
            model_name="Test",
        )
        np.testing.assert_array_equal(result.anomaly_indices, [1, 3])
