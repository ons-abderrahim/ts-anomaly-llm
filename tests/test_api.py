"""
FastAPI route integration tests using TestClient.
Run with: pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_list_models(self):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        names = [m["name"] for m in data["available"]]
        assert "isolation_forest" in names
        assert "lstm_ae" in names
        assert "transformer_ae" in names

    def test_model_has_required_fields(self):
        resp = client.get("/api/v1/models")
        for model in resp.json()["available"]:
            assert "name" in model
            assert "description" in model
            assert "best_for" in model
            assert "avg_inference_ms" in model


# ---------------------------------------------------------------------------
# POST /api/v1/detect
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    def _series(self, n=50, spike_at=None):
        vals = np.random.normal(1.0, 0.1, n).tolist()
        if spike_at is not None:
            vals[spike_at] = 20.0
        return vals

    def test_detect_returns_200(self):
        resp = client.post("/api/v1/detect", json={
            "series": self._series(50, spike_at=40),
            "model": "isolation_forest",
        })
        assert resp.status_code == 200

    def test_detect_response_structure(self):
        resp = client.post("/api/v1/detect", json={
            "series": self._series(50),
            "model": "isolation_forest",
        })
        data = resp.json()
        assert "anomalies" in data
        assert "anomaly_count" in data
        assert "model_used" in data
        assert "threshold" in data

    def test_detect_anomaly_point_fields(self):
        resp = client.post("/api/v1/detect", json={
            "series": self._series(50),
            "model": "isolation_forest",
        })
        for point in resp.json()["anomalies"]:
            assert "timestamp" in point
            assert "value" in point
            assert "score" in point
            assert "is_anomaly" in point

    def test_detect_counts_match(self):
        series = self._series(50)
        resp = client.post("/api/v1/detect", json={"series": series, "model": "isolation_forest"})
        data = resp.json()
        flagged = sum(1 for p in data["anomalies"] if p["is_anomaly"])
        assert flagged == data["anomaly_count"]

    def test_detect_with_timestamps(self):
        series = self._series(20)
        timestamps = [f"2024-01-01T00:{i:02d}:00Z" for i in range(20)]
        resp = client.post("/api/v1/detect", json={
            "series": series,
            "timestamps": timestamps,
            "model": "isolation_forest",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["anomalies"][0]["timestamp"] == timestamps[0]

    def test_detect_unknown_model_returns_422(self):
        resp = client.post("/api/v1/detect", json={
            "series": self._series(20),
            "model": "does_not_exist",
        })
        assert resp.status_code == 422

    def test_detect_too_short_series_returns_422(self):
        resp = client.post("/api/v1/detect", json={
            "series": [1.0, 2.0],
            "model": "isolation_forest",
        })
        assert resp.status_code == 422

    def test_detect_custom_threshold(self):
        series = self._series(50, spike_at=25)
        resp_low = client.post("/api/v1/detect", json={
            "series": series, "model": "isolation_forest", "threshold": 0.1
        })
        resp_high = client.post("/api/v1/detect", json={
            "series": series, "model": "isolation_forest", "threshold": 0.95
        })
        # Lower threshold should flag more anomalies
        assert resp_low.json()["anomaly_count"] >= resp_high.json()["anomaly_count"]


# ---------------------------------------------------------------------------
# POST /api/v1/explain  (mocked LLM)
# ---------------------------------------------------------------------------

MOCK_EXPLAIN_RESPONSE = {
    "explanation": "Mocked explanation text.",
    "suggested_actions": ["Action 1", "Action 2"],
    "confidence": 0.94,
    "domain": "iot",
    "model_used": "gpt-4o-mini",
}

EXPLAIN_PAYLOAD = {
    "anomaly": {"timestamp": "2024-01-01T00:04:00Z", "value": 9.8, "score": 0.94},
    "context_window": [1.1, 1.2, 1.0, 1.3, 9.8, 1.1, 1.2],
    "metadata": {"sensor_id": "temp-42", "unit": "celsius", "location": "Pump Room", "domain": "iot"},
}


class TestExplainEndpoint:
    def test_explain_returns_200(self):
        with patch("src.api.routes.explain._explainer.explain") as mock_explain:
            from src.llm.explainer import ExplanationResult
            mock_explain.return_value = ExplanationResult(**MOCK_EXPLAIN_RESPONSE)
            resp = client.post("/api/v1/explain", json=EXPLAIN_PAYLOAD)
        assert resp.status_code == 200

    def test_explain_response_structure(self):
        with patch("src.api.routes.explain._explainer.explain") as mock_explain:
            from src.llm.explainer import ExplanationResult
            mock_explain.return_value = ExplanationResult(**MOCK_EXPLAIN_RESPONSE)
            resp = client.post("/api/v1/explain", json=EXPLAIN_PAYLOAD)
        data = resp.json()
        assert "explanation" in data
        assert "suggested_actions" in data
        assert "confidence" in data
        assert "domain" in data
        assert "model_used" in data

    def test_explain_with_missing_metadata(self):
        payload = {**EXPLAIN_PAYLOAD, "metadata": None}
        with patch("src.api.routes.explain._explainer.explain") as mock_explain:
            from src.llm.explainer import ExplanationResult
            mock_explain.return_value = ExplanationResult(**MOCK_EXPLAIN_RESPONSE)
            resp = client.post("/api/v1/explain", json=payload)
        assert resp.status_code == 200
