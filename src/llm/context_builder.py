"""
Converts raw AnomalyResult + metadata into a structured context dict
that the LLM explainer can render into a natural-language prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.anomaly.base import AnomalyResult


@dataclass
class AnomalyContext:
    """Rich context for a single anomalous event."""
    timestamp: str
    anomalous_value: float
    anomaly_score: float
    baseline_mean: float
    baseline_std: float
    deviation_sigma: float          # how many std devs from baseline
    trend: str                      # "rising" | "falling" | "stable" | "spike" | "dip"
    context_window_values: list[float]
    sensor_id: Optional[str] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    domain: Optional[str] = None    # "iot" | "financial" | "operational"
    extra_metadata: dict = None

    def to_prompt_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "value": round(self.anomalous_value, 4),
            "score": round(self.anomaly_score, 3),
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
            "deviation_sigma": round(self.deviation_sigma, 2),
            "trend": self.trend,
            "context_window": [round(v, 4) for v in self.context_window_values],
            "sensor_id": self.sensor_id or "unknown",
            "unit": self.unit or "units",
            "location": self.location or "unspecified",
            "domain": self.domain or "general",
        }


class ContextBuilder:
    """
    Builds AnomalyContext objects from AnomalyResult + metadata.

    Args:
        baseline_window: Number of preceding points used to compute baseline stats.
    """

    def __init__(self, baseline_window: int = 50):
        self.baseline_window = baseline_window

    def build(
        self,
        result: AnomalyResult,
        anomaly_index: int,
        metadata: Optional[dict] = None,
    ) -> AnomalyContext:
        """
        Build context for a single anomalous point.

        Args:
            result:          Full AnomalyResult from the detector.
            anomaly_index:   Index into result.values of the anomalous point.
            metadata:        Optional dict with keys: sensor_id, unit, location, domain.

        Returns:
            AnomalyContext
        """
        values = result.values.flatten()
        meta = metadata or {}

        # Baseline: up to baseline_window points before the anomaly
        start = max(0, anomaly_index - self.baseline_window)
        baseline = values[start:anomaly_index]
        baseline_mean = float(baseline.mean()) if len(baseline) > 0 else 0.0
        baseline_std = float(baseline.std()) if len(baseline) > 1 else 1e-6

        anomalous_value = float(values[anomaly_index])
        deviation_sigma = (anomalous_value - baseline_mean) / max(baseline_std, 1e-6)

        # Context window: ±10 points around anomaly
        ctx_start = max(0, anomaly_index - 10)
        ctx_end = min(len(values), anomaly_index + 11)
        context_window = values[ctx_start:ctx_end].tolist()

        trend = self._classify_trend(baseline, anomalous_value)

        return AnomalyContext(
            timestamp=str(result.timestamps[anomaly_index]),
            anomalous_value=anomalous_value,
            anomaly_score=float(result.scores[anomaly_index]),
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            deviation_sigma=deviation_sigma,
            trend=trend,
            context_window_values=context_window,
            sensor_id=meta.get("sensor_id"),
            unit=meta.get("unit"),
            location=meta.get("location"),
            domain=meta.get("domain"),
            extra_metadata={k: v for k, v in meta.items()
                            if k not in ("sensor_id", "unit", "location", "domain")},
        )

    def build_all(
        self,
        result: AnomalyResult,
        metadata: Optional[dict] = None,
    ) -> list[AnomalyContext]:
        """Build contexts for every flagged anomaly in a result."""
        return [
            self.build(result, int(i), metadata)
            for i in result.anomaly_indices
        ]

    # ------------------------------------------------------------------

    @staticmethod
    def _classify_trend(baseline: np.ndarray, anomalous_value: float) -> str:
        """Classify the nature of the anomaly relative to the baseline trend."""
        if len(baseline) < 3:
            return "unknown"

        recent = baseline[-min(10, len(baseline)):]
        trend_slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        mean = baseline.mean()

        if abs(anomalous_value - mean) > 3 * baseline.std():
            return "spike" if anomalous_value > mean else "dip"
        elif trend_slope > 0.01 * abs(mean):
            return "rising"
        elif trend_slope < -0.01 * abs(mean):
            return "falling"
        else:
            return "stable"
