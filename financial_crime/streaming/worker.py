import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from financial_crime.config import settings
from financial_crime.data.schema import FEATURES
from financial_crime.errors import ToolExecutionError
from financial_crime.streaming.events import StreamOutput, TransactionEvent

logger = logging.getLogger("financial_crime.streaming")
TOPICS = {
    "raw": "transactions.raw",
    "validated": "transactions.validated",
    "scored": "transactions.scored",
    "investigations": "investigations.created",
    "deadletter": "deadletter.transactions",
}


def process_message(
    raw: str | bytes,
    artifact: dict[str, Any] | None = None,
    review_threshold: float = 0.8,
) -> list[StreamOutput]:
    try:
        event = TransactionEvent.model_validate_json(raw)
        transaction = event.payload
    except Exception as error:
        logger.warning("invalid transaction event: %s", type(error).__name__)
        return [
            StreamOutput(
                topic=TOPICS["deadletter"],
                key="invalid",
                payload={"error": "ValidationError", "detail": "event_schema_invalid"},
            )
        ]

    data = transaction.model_dump(mode="json")
    data["is_suspicious"] = None
    outputs = [
        StreamOutput(
            topic=TOPICS["validated"],
            key=transaction.transaction_id,
            payload={
                "event_id": event.event_id,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": event.correlation_id,
                "transaction": data,
            },
        )
    ]
    score = None
    if artifact is not None:
        try:
            values = pd.DataFrame(
                [[float(getattr(transaction, feature)) for feature in FEATURES]], columns=FEATURES
            )
            score = float(artifact["model"].predict_proba(values)[0, 1])
        except Exception as error:
            logger.error("model scoring failed: %s", type(error).__name__)
            outputs.append(
                StreamOutput(
                    topic=TOPICS["deadletter"],
                    key=transaction.transaction_id,
                    payload={
                        "error": "ModelUnavailableError",
                        "transaction_id": transaction.transaction_id,
                    },
                )
            )
    outputs.append(
        StreamOutput(
            topic=TOPICS["scored"],
            key=transaction.transaction_id,
            payload={
                "transaction_id": transaction.transaction_id,
                "model_status": "available" if score is not None else "unavailable",
                "risk_score": score,
                "model_version": artifact.get("version")
                if score is not None and artifact
                else None,
                "human_review_required": True,
            },
        )
    )
    if score is not None and score >= review_threshold:
        outputs.append(
            StreamOutput(
                topic=TOPICS["investigations"],
                key=transaction.transaction_id,
                payload={
                    "investigation_id": f"CASE-{transaction.transaction_id}",
                    "transaction_id": transaction.transaction_id,
                    "status": "requires-review",
                    "risk_score": score,
                    "model_version": artifact.get("version") if artifact else None,
                    "threshold": review_threshold,
                    "threshold_version": "0.1.0",
                },
            )
        )
    return outputs


class KafkaPublisher:
    def __init__(self, bootstrap_servers: str):
        try:
            from confluent_kafka import Producer
        except ImportError as error:
            raise RuntimeError("install the streaming extra to enable Kafka") from error
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})

    def publish(self, output: StreamOutput, timeout_seconds: float = 10) -> None:
        delivery_errors: list[str] = []

        def on_delivery(error, _message) -> None:
            if error is not None:
                delivery_errors.append(str(error))

        self._producer.produce(
            output.topic,
            key=output.key.encode(),
            value=json.dumps(output.payload, separators=(",", ":")).encode(),
            callback=on_delivery,
        )
        remaining = self._producer.flush(timeout_seconds)
        if remaining or delivery_errors:
            raise ToolExecutionError(
                delivery_errors[0] if delivery_errors else "Kafka delivery timed out"
            )


def run_worker(
    bootstrap_servers: str,
    model_path: str | None = None,
    review_threshold: float = settings.risk_review_threshold,
) -> None:
    try:
        from confluent_kafka import Consumer, TopicPartition
    except ImportError as error:
        raise RuntimeError("install the streaming extra to enable Kafka") from error
    model_artifact = (
        joblib.load(Path(model_path)) if model_path and Path(model_path).exists() else None
    )
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": "financial-crime-scorer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    publisher = KafkaPublisher(bootstrap_servers)
    consumer.subscribe([TOPICS["raw"]])
    latencies: deque[float] = deque(maxlen=1000)
    window_started = time.perf_counter()
    processed = 0
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                logger.error("Kafka consumer error: %s", message.error().code())
                continue
            started = time.perf_counter()
            outputs = process_message(message.value(), model_artifact, review_threshold)
            for output in outputs:
                publisher.publish(output)
            committed = consumer.commit(message=message, asynchronous=False)
            if committed and any(partition.error for partition in committed):
                raise ToolExecutionError("Kafka offset commit failed")
            latencies.append(time.perf_counter() - started)
            processed += 1
            if processed % 100 == 0:
                topic_partition = TopicPartition(message.topic(), message.partition())
                _, high = consumer.get_watermark_offsets(topic_partition, timeout=5)
                position = consumer.position([topic_partition])[0].offset
                stats = summarize_latency(
                    list(latencies),
                    time.perf_counter() - window_started,
                    consumer_lag=max(0, high - max(0, position)),
                )
                logger.info("stream_benchmark %s", json.dumps(stats))
                window_started = time.perf_counter()
    finally:
        consumer.close()


def summarize_latency(
    latencies_seconds: list[float],
    elapsed_seconds: float,
    consumer_lag: int | None = None,
) -> dict[str, float | int | None]:
    if not latencies_seconds or elapsed_seconds <= 0:
        raise ValueError("latencies and positive elapsed time are required")
    values = np.asarray(latencies_seconds, dtype=float) * 1000
    return {
        "events_per_second": len(values) / elapsed_seconds,
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "events_processed": len(values),
        "consumer_lag": consumer_lag,
    }
