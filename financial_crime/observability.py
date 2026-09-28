import json
import logging
from datetime import UTC, datetime

from fastapi import FastAPI

logger = logging.getLogger("financial_crime.observability")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("correlation_id", "method", "path", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, separators=(",", ":"))


def configure_structured_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_fci_json", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler._fci_json = True
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def configure_tracing(app: FastAPI, endpoint: str | None) -> bool:
    if not endpoint:
        logger.info("OpenTelemetry export disabled; set OTEL_OTLP_ENDPOINT to enable it")
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry endpoint configured but optional packages are not installed")
        return False
    provider = TracerProvider(
        resource=Resource.create({"service.name": "financial-crime-investigation-api"})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    return True
