"""
Kafka consumer for streaming time series ingestion.
Reads messages, buffers windows, and triggers detection + explanation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Callable, Optional

from kafka import KafkaConsumer
import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesKafkaConsumer:
    """
    Consumes time series messages from a Kafka topic and maintains
    a sliding window buffer for real-time anomaly detection.

    Message format expected (JSON):
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "value": 1.23,
            "sensor_id": "temp-42",
            "unit": "celsius"
        }

    Args:
        topic:           Kafka topic name.
        bootstrap_servers: Kafka broker address(es).
        group_id:        Consumer group ID.
        window_size:     Number of data points to buffer before triggering detection.
        on_window:       Callback invoked with (window_values, window_meta) when buffer is full.
        auto_offset_reset: "latest" for live streaming, "earliest" for replay.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "anomaly-detector",
        window_size: int = 100,
        on_window: Optional[Callable[[np.ndarray, list[dict]], None]] = None,
        auto_offset_reset: str = "latest",
    ):
        self.topic = topic
        self.window_size = window_size
        self.on_window = on_window

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )

        self._buffer: deque[dict] = deque(maxlen=window_size)
        self._running = False

    def start(self):
        """Start consuming messages in a blocking loop."""
        self._running = True
        logger.info("Kafka consumer started on topic '%s'", self.topic)

        try:
            for message in self._consumer:
                if not self._running:
                    break

                record = message.value
                self._buffer.append(record)
                logger.debug("Ingested: %s", record)

                if len(self._buffer) >= self.window_size:
                    self._flush_window()

        except Exception as exc:
            logger.error("Consumer error: %s", exc)
        finally:
            self._consumer.close()
            logger.info("Kafka consumer stopped")

    def stop(self):
        self._running = False

    def _flush_window(self):
        """Extract buffer into arrays and invoke on_window callback."""
        records = list(self._buffer)
        values = np.array([r["value"] for r in records], dtype=float)
        meta_list = [{k: v for k, v in r.items() if k != "value"} for r in records]

        if self.on_window:
            try:
                self.on_window(values, meta_list)
            except Exception as exc:
                logger.error("on_window callback failed: %s", exc)
