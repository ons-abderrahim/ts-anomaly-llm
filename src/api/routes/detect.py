"""
POST /api/v1/detect
Run anomaly detection on a time series payload.
"""

from fastapi import APIRouter, HTTPException

from src.anomaly import get_detector
from src.api.schemas import AnomalyPoint, DetectRequest, DetectResponse
import numpy as np

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect_anomalies(request: DetectRequest) -> DetectResponse:
    """
    Detect anomalies in a time series.

    - **series**: list of numeric values
    - **model**: which detector to use (`isolation_forest`, `lstm_ae`, `transformer_ae`)
    - **threshold**: score cutoff for flagging anomalies
    """
    try:
        detector = get_detector(request.model, threshold=request.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    series = np.array(request.series, dtype=float)
    timestamps = request.timestamps or [str(i) for i in range(len(series))]

    # For a REST endpoint we fit on the series itself (unsupervised / streaming mode).
    # In production, load a pre-trained model from the model registry instead.
    try:
        detector.fit(series)
        result = detector.detect(series, timestamps=timestamps, metadata=request.metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")

    points = [
        AnomalyPoint(
            timestamp=str(result.timestamps[i]),
            value=float(result.values[i]),
            score=float(result.scores[i]),
            is_anomaly=bool(result.is_anomaly[i]),
        )
        for i in range(len(result.timestamps))
    ]

    return DetectResponse(
        anomalies=points,
        anomaly_count=result.anomaly_count,
        model_used=result.model_name,
        threshold=result.threshold,
    )
