"""
Redis Streams consumer for lightweight streaming ingestion.
Alternative to Kafka for smaller deployments.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Callable, Optional

import redis
import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesRedisConsumer:
    """
    Consumes time series messages from a Redis Stream and maintains
    a sliding window buffer for real-time anomaly detection.

    Message format expected (Redis hash fields):
        timestamp  → "2024-01-01T00:00:00Z"
        value      → "1.23"
        sensor_id  → "temp-42"
        unit       → "celsius"

    Args:
        stream_key:      Redis stream key to consume.
        host:            Redis host.
        port:            Redis port.
        db:              Redis database index.
        consumer_group:  Redis consumer group name.
        consumer_name:   This consumer's name within the group.
        window_size:     Number of data points to buffer before triggering detection.
        on_window:       Callback invoked with (window_values, window_meta) when buffer is full.
        block_ms:        How long to block on XREADGROUP (ms) waiting for new messages.
        batch_size:      Max messages to read per XREADGROUP call.
    """

    def __init__(
        self,
        stream_key: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        consumer_group: str = "anomaly-detector",
        consumer_name: str = "worker-1",
        window_size: int = 100,
        on_window: Optional[Callable[[np.ndarray, list[dict]], None]] = None,
        block_ms: int = 2000,
        batch_size: int = 50,
    ):
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.window_size = window_size
        self.on_window = on_window
        self.block_ms = block_ms
        self.batch_size = batch_size

        self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._buffer: deque[dict] = deque(maxlen=window_size)
        self._running = False

        self._ensure_group()

    # ------------------------------------------------------------------

    def start(self):
        """Start consuming messages in a blocking loop."""
        self._running = True
        logger.info("Redis consumer started on stream '%s'", self.stream_key)

        while self._running:
            try:
                messages = self._redis.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_key: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )

                if not messages:
                    continue

                for _stream, entries in messages:
                    for entry_id, fields in entries:
                        self._process_message(entry_id, fields)

            except redis.exceptions.ConnectionError as exc:
                logger.error("Redis connection error: %s. Retrying in 5s…", exc)
                time.sleep(5)
            except Exception as exc:
                logger.error("Unexpected consumer error: %s", exc)

        logger.info("Redis consumer stopped")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------

    def _process_message(self, entry_id: str, fields: dict):
        """Parse a Redis stream entry and add to buffer."""
        try:
            record = {
                "timestamp": fields.get("timestamp", entry_id),
                "value": float(fields["value"]),
                "sensor_id": fields.get("sensor_id", "unknown"),
                "unit": fields.get("unit", "units"),
            }
            # Include any extra fields as metadata
            for k, v in fields.items():
                if k not in record:
                    record[k] = v

            self._buffer.append(record)
            logger.debug("Buffered message %s: value=%s", entry_id, record["value"])

            # Acknowledge message to Redis
            self._redis.xack(self.stream_key, self.consumer_group, entry_id)

            if len(self._buffer) >= self.window_size:
                self._flush_window()

        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed message %s: %s", entry_id, exc)

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

    def _ensure_group(self):
        """Create the consumer group if it doesn't already exist."""
        try:
            self._redis.xgroup_create(
                self.stream_key, self.consumer_group, id="0", mkstream=True
            )
            logger.info(
                "Created consumer group '%s' on stream '%s'",
                self.consumer_group,
                self.stream_key,
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logger.debug("Consumer group '%s' already exists", self.consumer_group)
            else:
                raise
