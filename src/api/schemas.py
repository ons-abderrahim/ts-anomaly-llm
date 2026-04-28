"""
Pydantic request/response schemas for the FastAPI app.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    series: list[float] = Field(..., description="Time series values", min_length=10)
    timestamps: Optional[list[str]] = Field(None, description="ISO timestamps aligned with series")
    model: str = Field("isolation_forest", description="Detector: isolation_forest | lstm_ae | transformer_ae")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Anomaly score cutoff")
    metadata: Optional[dict[str, Any]] = Field(None, description="Pass-through metadata (sensor_id, unit, etc.)")

    model_config = {"json_schema_extra": {
        "example": {
            "series": [1.1, 1.2, 1.0, 1.3, 9.8, 1.1, 1.2],
            "timestamps": [f"2024-01-01T00:0{i}:00Z" for i in range(7)],
            "model": "isolation_forest",
            "threshold": 0.5,
            "metadata": {"sensor_id": "temp-42", "unit": "celsius", "location": "Pump Room 3"},
        }
    }}


class AnomalyPoint(BaseModel):
    timestamp: str
    value: float
    score: float
    is_anomaly: bool


class DetectResponse(BaseModel):
    anomalies: list[AnomalyPoint]
    anomaly_count: int
    model_used: str
    threshold: float


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

class AnomalyInput(BaseModel):
    timestamp: str
    value: float
    score: float


class ExplainRequest(BaseModel):
    anomaly: AnomalyInput
    context_window: list[float] = Field(..., description="Values around the anomaly (±N points)")
    metadata: Optional[dict[str, Any]] = Field(None, description="sensor_id, unit, location, domain")

    model_config = {"json_schema_extra": {
        "example": {
            "anomaly": {"timestamp": "2024-01-01T00:04:00Z", "value": 9.8, "score": 0.94},
            "context_window": [1.1, 1.2, 1.0, 1.3, 9.8, 1.1, 1.2],
            "metadata": {
                "sensor_id": "temp-42",
                "unit": "celsius",
                "location": "Pump Room 3",
                "domain": "iot",
            },
        }
    }}


class ExplainResponse(BaseModel):
    explanation: str
    suggested_actions: list[str]
    confidence: Optional[float]
    domain: str
    model_used: str


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    description: str
    best_for: str
    avg_inference_ms: float


class ModelsResponse(BaseModel):
    available: list[ModelInfo]
