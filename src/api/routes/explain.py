"""
POST /api/v1/explain
Generate a plain-English LLM explanation for a detected anomaly.
"""

import os

from fastapi import APIRouter, HTTPException

from src.llm.context_builder import AnomalyContext, ContextBuilder
from src.llm.explainer import AnomalyExplainer
from src.api.schemas import ExplainRequest, ExplainResponse

router = APIRouter()

# Single shared explainer instance (model loaded once at import time)
_explainer = AnomalyExplainer(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
_ctx_builder = ContextBuilder()


@router.post("/explain", response_model=ExplainResponse)
async def explain_anomaly(request: ExplainRequest) -> ExplainResponse:
    """
    Generate a plain-English root-cause explanation for a detected anomaly.

    Requires OPENAI_API_KEY to be set in environment.
    """
    meta = request.metadata or {}

    # Build a synthetic AnomalyContext from the request payload
    import numpy as np
    context = AnomalyContext(
        timestamp=request.anomaly.timestamp,
        anomalous_value=request.anomaly.value,
        anomaly_score=request.anomaly.score,
        baseline_mean=float(np.mean(request.context_window)),
        baseline_std=float(np.std(request.context_window)) or 1e-6,
        deviation_sigma=(
            (request.anomaly.value - float(np.mean(request.context_window)))
            / max(float(np.std(request.context_window)), 1e-6)
        ),
        trend="spike" if request.anomaly.value > float(np.mean(request.context_window)) else "dip",
        context_window_values=request.context_window,
        sensor_id=meta.get("sensor_id"),
        unit=meta.get("unit"),
        location=meta.get("location"),
        domain=meta.get("domain", "general"),
    )

    try:
        result = _explainer.explain(context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return ExplainResponse(
        explanation=result.explanation,
        suggested_actions=result.suggested_actions,
        confidence=result.confidence,
        domain=result.domain,
        model_used=result.model_used,
    )
