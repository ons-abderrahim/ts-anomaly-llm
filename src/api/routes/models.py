"""
GET /api/v1/models
List available anomaly detection models.
"""

from fastapi import APIRouter

from src.api.schemas import ModelInfo, ModelsResponse

router = APIRouter()

_MODEL_CATALOG = [
    ModelInfo(
        name="isolation_forest",
        description="Scikit-learn Isolation Forest — unsupervised, fast, no GPU required.",
        best_for="Tabular / IoT data with sparse, independent point anomalies.",
        avg_inference_ms=3.0,
    ),
    ModelInfo(
        name="lstm_ae",
        description="LSTM Autoencoder — sequence-aware, detects temporal pattern breaks.",
        best_for="Smooth univariate or low-dimensional multivariate time series.",
        avg_inference_ms=20.0,
    ),
    ModelInfo(
        name="transformer_ae",
        description="Transformer Autoencoder — self-attention, handles long-range dependencies.",
        best_for="Complex multivariate time series with non-local correlations.",
        avg_inference_ms=50.0,
    ),
]


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Return all available anomaly detection models."""
    return ModelsResponse(available=_MODEL_CATALOG)
