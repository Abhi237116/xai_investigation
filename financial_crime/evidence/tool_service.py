import json
import logging
import time
import uuid
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select

from financial_crime.config import settings
from financial_crime.data.database import (
    EvidenceRow,
    InvestigationRow,
    SessionFactory,
    ToolCallRow,
    TransactionRow,
)
from financial_crime.errors import ToolExecutionError
from financial_crime.evidence.graph import build_graph, detect_shared_entities, neighborhood
from financial_crime.ml.counterfactual import generate_counterfactual
from financial_crime.ml.explain import ExplanationUnavailableError, explain_case

logger = logging.getLogger("financial_crime.tools")
TOOL_METRICS = {"calls": 0, "failures": 0, "duration_seconds": []}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.isoformat()))


def _record_call(
    name: str, arguments: dict[str, Any], status: str, duration: float, result: Any
) -> None:
    input_hash = sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()
    output_hash = (
        sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
        if result is not None
        else None
    )
    with SessionFactory() as session:
        session.add(
            ToolCallRow(
                call_id=str(uuid.uuid4()),
                tool_name=name,
                input_sha256=input_hash,
                output_sha256=output_hash,
                status=status,
                duration_ms=duration * 1000,
            )
        )
        session.commit()


def _audited(name: str, arguments: dict[str, Any], operation: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    status = "success"
    result = None
    try:
        result = operation()
        if time.perf_counter() - started > 10:
            status = "timeout"
            raise ToolExecutionError("tool execution exceeded the configured 10 second limit")
        return _jsonable(result)
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - started
        TOOL_METRICS["calls"] += 1
        TOOL_METRICS["failures"] += int(status != "success")
        TOOL_METRICS["duration_seconds"].append(elapsed)
        TOOL_METRICS["duration_seconds"] = TOOL_METRICS["duration_seconds"][-1000:]
        try:
            _record_call(name, arguments, status, elapsed, result)
        except Exception as audit_error:
            logger.exception("tool audit persistence failed: %s", type(audit_error).__name__)


def _transaction(row: TransactionRow) -> dict[str, Any]:
    return {
        "transaction_id": row.transaction_id,
        "account_id": row.account_id,
        "beneficiary_id": row.beneficiary_id,
        "device_id": row.device_id,
        "merchant_id": row.merchant_id,
        "timestamp": row.timestamp,
        "amount": row.amount,
        "account_age_days": row.account_age_days,
        "beneficiary_age_days": row.beneficiary_age_days,
        "device_accounts_30d": row.device_accounts_30d,
        "transactions_1h": row.transactions_1h,
        "distance_from_usual_km": row.distance_from_usual_km,
        "source": row.source,
    }


def _limited_history(field: str, entity_id: str, limit: int) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    column = getattr(TransactionRow, field)
    with SessionFactory() as session:
        rows = session.scalars(
            select(TransactionRow)
            .where(column == entity_id)
            .order_by(TransactionRow.timestamp.desc())
            .limit(limit)
        )
        return [_transaction(row) for row in rows]


def _graph():
    with SessionFactory() as session:
        frame = pd.DataFrame([_transaction(row) for row in session.scalars(select(TransactionRow))])
    return build_graph(frame)


def _model_artifact() -> dict[str, Any] | None:
    path = settings.model_artifact_path
    return joblib.load(path) if path and Path(path).is_file() else None


def get_transaction(transaction_id: str) -> dict[str, Any]:
    def operation():
        with SessionFactory() as session:
            row = session.get(TransactionRow, transaction_id)
            return (
                {"error": "EvidenceUnavailableError", "detail": "transaction_not_found"}
                if row is None
                else _transaction(row)
            )

    return _audited("get_transaction", {"transaction_id": transaction_id}, operation)


def get_account_history(account_id: str, limit: int = 50) -> dict[str, Any]:
    return _audited(
        "get_account_history",
        {"account_id": account_id, "limit": limit},
        lambda: {
            "account_id": account_id,
            "transactions": _limited_history("account_id", account_id, limit),
        },
    )


def get_device_history(device_id: str, limit: int = 50) -> dict[str, Any]:
    return _audited(
        "get_device_history",
        {"device_id": device_id, "limit": limit},
        lambda: {
            "device_id": device_id,
            "transactions": _limited_history("device_id", device_id, limit),
        },
    )


def get_beneficiary_history(beneficiary_id: str, limit: int = 50) -> dict[str, Any]:
    return _audited(
        "get_beneficiary_history",
        {"beneficiary_id": beneficiary_id, "limit": limit},
        lambda: {
            "beneficiary_id": beneficiary_id,
            "transactions": _limited_history("beneficiary_id", beneficiary_id, limit),
        },
    )


def find_related_accounts(account_id: str, depth: int = 2) -> dict[str, Any]:
    def operation():
        graph = _graph()
        node = f"account:{account_id}"
        return {
            "account_id": account_id,
            "related_accounts": neighborhood(graph, node, depth)["connected_accounts"],
            "shared_entities": detect_shared_entities(graph, account_id),
        }

    return _audited("find_related_accounts", {"account_id": account_id, "depth": depth}, operation)


def get_network_connections(entity_type: str, entity_id: str, depth: int = 2) -> dict[str, Any]:
    def operation():
        if entity_type not in {"account", "transaction", "device", "beneficiary", "merchant"}:
            raise ValueError("unsupported entity_type")
        graph = _graph()
        return neighborhood(graph, f"{entity_type}:{entity_id}", depth)

    return _audited(
        "get_network_connections",
        {"entity_type": entity_type, "entity_id": entity_id, "depth": depth},
        operation,
    )


def get_model_prediction(transaction_id: str) -> dict[str, Any]:
    def operation():
        artifact = _model_artifact()
        if artifact is None:
            return {"status": "unavailable", "error": "ModelUnavailableError", "risk_score": None}
        with SessionFactory() as session:
            row = session.get(TransactionRow, transaction_id)
            if row is None:
                return {"error": "EvidenceUnavailableError", "detail": "transaction_not_found"}
            values = {feature: float(getattr(row, feature)) for feature in artifact["features"]}
            score = float(artifact["model"].predict_proba(pd.DataFrame([values]))[0, 1])
        return {
            "status": "available",
            "transaction_id": transaction_id,
            "risk_score": score,
            "model_version": artifact.get("version"),
            "model_name": artifact.get("name"),
        }

    return _audited("get_model_prediction", {"transaction_id": transaction_id}, operation)


def get_shap_explanation(transaction_id: str) -> dict[str, Any]:
    def operation():
        artifact = _model_artifact()
        if artifact is None:
            return {"status": "explanation_unavailable", "error": "ModelUnavailableError"}
        with SessionFactory() as session:
            row = session.get(TransactionRow, transaction_id)
            if row is None:
                return {"status": "explanation_unavailable", "error": "EvidenceUnavailableError"}
            values = {feature: float(getattr(row, feature)) for feature in artifact["features"]}
        try:
            return explain_case(artifact, values)
        except ExplanationUnavailableError:
            return {"status": "explanation_unavailable", "error": "ExplanationUnavailableError"}

    return _audited("get_shap_explanation", {"transaction_id": transaction_id}, operation)


def search_evidence(case_id: str, query: str, limit: int = 25) -> dict[str, Any]:
    def operation():
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("query is required and limit must be between 1 and 100")
        with SessionFactory() as session:
            case = session.get(InvestigationRow, case_id)
            if case is None:
                return {"case_id": case_id, "items": [], "error": "EvidenceUnavailableError"}
            rows = session.scalars(
                select(EvidenceRow).where(EvidenceRow.investigation_id == case_id)
            )
            items = [
                {
                    "evidence_id": item.evidence_id,
                    "category": item.category,
                    "source": item.source,
                    "claim": item.claim,
                    "observed_at": item.observed_at,
                }
                for item in rows
                if query.casefold() in item.claim.casefold()
            ][:limit]
            return {"case_id": case_id, "items": items}

    return _audited(
        "search_evidence", {"case_id": case_id, "query": query, "limit": limit}, operation
    )


def generate_counterfactual_for_transaction(transaction_id: str) -> dict[str, Any]:
    def operation():
        artifact = _model_artifact()
        if artifact is None:
            return {"status": "unavailable", "error": "ModelUnavailableError"}
        with SessionFactory() as session:
            row = session.get(TransactionRow, transaction_id)
            if row is None:
                return {"status": "unavailable", "error": "EvidenceUnavailableError"}
            values = {feature: float(getattr(row, feature)) for feature in artifact["features"]}
        return {
            "status": "available",
            **generate_counterfactual(artifact, values, settings.risk_review_threshold),
        }

    return _audited("generate_counterfactual", {"transaction_id": transaction_id}, operation)


_TOOL_DEFINITIONS = [
    (
        "get_transaction",
        "Retrieve one transaction and provenance fields.",
        {"transaction_id": "string"},
    ),
    (
        "get_account_history",
        "Retrieve bounded account transaction history.",
        {"account_id": "string", "limit": "integer"},
    ),
    (
        "get_device_history",
        "Retrieve bounded device transaction history.",
        {"device_id": "string", "limit": "integer"},
    ),
    (
        "get_beneficiary_history",
        "Retrieve bounded beneficiary transaction history.",
        {"beneficiary_id": "string", "limit": "integer"},
    ),
    (
        "find_related_accounts",
        "Find accounts linked by shared devices, beneficiaries or merchants.",
        {"account_id": "string", "depth": "integer"},
    ),
    (
        "get_network_connections",
        "Retrieve a bounded relationship graph neighborhood.",
        {"entity_type": "string", "entity_id": "string", "depth": "integer"},
    ),
    (
        "get_model_prediction",
        "Retrieve an official model prediction or explicit unavailable status.",
        {"transaction_id": "string"},
    ),
    (
        "get_shap_explanation",
        "Retrieve SHAP output directly from the model explainer.",
        {"transaction_id": "string"},
    ),
    (
        "search_evidence",
        "Search evidence attached to a case.",
        {"case_id": "string", "query": "string", "limit": "integer"},
    ),
    (
        "generate_counterfactual",
        "Simulate bounded feature substitutions; no real-world prevention claim.",
        {"transaction_id": "string"},
    ),
]
TOOL_MANIFEST = [
    {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {key: {"type": value} for key, value in inputs.items()},
            "required": [key for key in inputs if key not in {"limit", "depth"}],
            "additionalProperties": False,
        },
        "output_schema": {"type": "object", "additionalProperties": True},
        "permission_level": "read-only",
        "timeout_seconds": 10,
        "error_behavior": "structured error or explicit unavailable status",
    }
    for name, description, inputs in _TOOL_DEFINITIONS
]
