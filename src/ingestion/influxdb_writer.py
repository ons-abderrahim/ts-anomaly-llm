"""
InfluxDB writer for persisting anomaly results and explanations.
Uses the InfluxDB Python client v2 / v3 compatible write API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from src.anomaly.base import AnomalyResult
from src.llm.explainer import ExplanationResult

logger = logging.getLogger(__name__)


class InfluxDBWriter:
    """
    Writes anomaly detection results and LLM explanations to InfluxDB.

    Args:
        url:    InfluxDB URL (e.g. "http://localhost:8086").
        token:  InfluxDB API token.
        org:    InfluxDB organisation name.
        bucket: Target bucket.
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",
        org: str = "anomaly-org",
        bucket: str = "anomalies",
    ):
        self.url = url
        self.org = org
        self.bucket = bucket

        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def write_anomaly_result(
        self,
        result: AnomalyResult,
        sensor_id: Optional[str] = None,
    ):
        """
        Write all anomalous points from a detection result to InfluxDB.

        Args:
            result:    Populated AnomalyResult.
            sensor_id: Optional tag for the source sensor.
        """
        points = []
        for idx in result.anomaly_indices:
            p = (
                Point("anomaly_detection")
                .tag("model", result.model_name)
                .tag("sensor_id", sensor_id or "unknown")
                .field("value", float(result.values[idx]))
                .field("score", float(result.scores[idx]))
                .field("is_anomaly", True)
                .time(result.timestamps[idx], WritePrecision.NANOSECONDS)
            )
            points.append(p)

        if points:
            try:
                self._write_api.write(bucket=self.bucket, record=points)
                logger.info("Wrote %d anomaly points to InfluxDB", len(points))
            except Exception as exc:
                logger.error("InfluxDB write failed: %s", exc)

    def write_explanation(
        self,
        timestamp: str,
        explanation: ExplanationResult,
        sensor_id: Optional[str] = None,
    ):
        """
        Write an LLM explanation record to InfluxDB.

        Args:
            timestamp:   ISO timestamp of the anomaly.
            explanation: ExplanationResult from AnomalyExplainer.
            sensor_id:   Optional sensor tag.
        """
        p = (
            Point("anomaly_explanation")
            .tag("domain", explanation.domain)
            .tag("sensor_id", sensor_id or "unknown")
            .tag("llm_model", explanation.model_used)
            .field("explanation", explanation.explanation[:1000])  # InfluxDB field size limit
            .field("confidence", explanation.confidence or 0.0)
            .time(timestamp, WritePrecision.NANOSECONDS)
        )

        try:
            self._write_api.write(bucket=self.bucket, record=p)
            logger.debug("Wrote explanation to InfluxDB")
        except Exception as exc:
            logger.error("InfluxDB explanation write failed: %s", exc)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
