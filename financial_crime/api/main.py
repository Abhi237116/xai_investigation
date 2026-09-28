import json
import logging
import time
import uuid
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from financial_crime.agents.investigator import investigate_transaction
from financial_crime.agents.llm import provider_health
from financial_crime.config import settings
from financial_crime.data.database import (
    AgentRunRow,
    EvidenceRow,
    InvestigationRow,
    SessionFactory,
    TransactionRow,
)
from financial_crime.data.ingest import load_dataset
from financial_crime.evidence.graph import build_graph, detect_shared_entities, neighborhood
from financial_crime.observability import configure_structured_logging, configure_tracing

logger = logging.getLogger("financial_crime.api")
REQUEST_METRICS = {"requests": 0, "failures": 0, "latency_seconds": []}


def initialize_demo_database() -> None:
    with SessionFactory() as session:
        if session.scalar(select(TransactionRow.transaction_id).limit(1)):
            return
        frame, _ = load_dataset(rows=500, seed=42)
        suspicious_count = 0
        for record in frame.to_dict(orient="records"):
            record["timestamp"] = datetime.fromisoformat(record["timestamp"])
            row = TransactionRow(**record)
            session.add(row)
            if row.is_suspicious == 1 and suspicious_count < 30:
                case_id = f"CASE-{row.transaction_id}"
                session.add(
                    InvestigationRow(
                        investigation_id=case_id,
                        transaction_id=row.transaction_id,
                        status="requires-review",
                    )
                )
                session.add(
                    EvidenceRow(
                        evidence_id=f"EV-{row.transaction_id}-001",
                        investigation_id=case_id,
                        category="OBSERVED",
                        source="synthetic_transaction_record",
                        claim=(
                            f"Synthetic transaction {row.transaction_id} recorded amount "
                            f"{row.amount:.2f} at {row.timestamp.isoformat()}."
                        ),
                        observed_at=row.timestamp,
                    )
                )
                suspicious_count += 1
        session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_structured_logging()
    initialize_demo_database()
    configure_tracing(app, settings.otel_otlp_endpoint)
    yield


app = FastAPI(title="Financial Crime Investigation API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.api_cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Correlation-ID"],
)


@app.middleware("http")
async def correlation_id(request: Request, call_next) -> Response:
    value = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))[:100]
    started = time.perf_counter()
    response = await call_next(request)
    REQUEST_METRICS["requests"] += 1
    REQUEST_METRICS["failures"] += int(response.status_code >= 500)
    REQUEST_METRICS["latency_seconds"].append(time.perf_counter() - started)
    REQUEST_METRICS["latency_seconds"] = REQUEST_METRICS["latency_seconds"][-1000:]
    response.headers["X-Correlation-ID"] = value
    logger.info(
        "api_request",
        extra={
            "correlation_id": value,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        },
    )
    return response


def get_session() -> Generator[Session, None, None]:
    with SessionFactory() as session:
        yield session


def transaction_dict(row: TransactionRow) -> dict[str, object]:
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "investigation-api"}


@app.get("/health/llm")
def llm_health() -> dict:
    return provider_health()


@app.get("/health/ready")
def ready() -> dict[str, str]:
    return {"status": "ready", "service": "investigation-api"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    latencies = sorted(REQUEST_METRICS["latency_seconds"])
    p50 = latencies[int((len(latencies) - 1) * 0.5)] if latencies else 0
    p95 = latencies[int((len(latencies) - 1) * 0.95)] if latencies else 0
    from financial_crime.evidence.tool_service import TOOL_METRICS

    tool_latencies = sorted(TOOL_METRICS["duration_seconds"])
    tool_p95 = tool_latencies[int((len(tool_latencies) - 1) * 0.95)] if tool_latencies else 0
    body = (
        "# HELP fci_api_requests_total API requests\n# TYPE fci_api_requests_total counter\n"
        f"fci_api_requests_total {REQUEST_METRICS['requests']}\n"
        "# TYPE fci_api_failures_total counter\n"
        f"fci_api_failures_total {REQUEST_METRICS['failures']}\n"
        "# TYPE fci_api_latency_p50_seconds gauge\n"
        f"fci_api_latency_p50_seconds {p50}\n"
        "# TYPE fci_api_latency_p95_seconds gauge\n"
        f"fci_api_latency_p95_seconds {p95}\n"
        "# TYPE fci_tool_calls_total counter\n"
        f"fci_tool_calls_total {TOOL_METRICS['calls']}\n"
        "# TYPE fci_tool_failures_total counter\n"
        f"fci_tool_failures_total {TOOL_METRICS['failures']}\n"
        "# TYPE fci_tool_latency_p95_seconds gauge\n"
        f"fci_tool_latency_p95_seconds {tool_p95}\n"
    )
    return Response(body, media_type="text/plain; version=0.0.4")


@app.get("/summary")
def summary(session: Session = Depends(get_session)) -> dict:
    transactions = session.scalars(select(TransactionRow)).all()
    cases = session.scalars(select(InvestigationRow)).all()
    return {
        "transaction_count": len(transactions),
        "investigation_count": len(cases),
        "review_required": sum(case.status == "requires-review" for case in cases),
        "synthetic_only": all(row.source == "synthetic" for row in transactions),
    }


@app.get("/benchmarks")
def benchmark_report() -> dict:
    path = Path(__file__).parents[2] / "docs" / "benchmark" / "benchmark_report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="benchmark_report_unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/transactions")
def list_transactions(
    limit: int = 100, session: Session = Depends(get_session)
) -> list[dict[str, object]]:
    rows = session.scalars(
        select(TransactionRow).order_by(TransactionRow.timestamp.desc()).limit(min(limit, 500))
    )
    return [transaction_dict(row) for row in rows]


@app.get("/transactions/{transaction_id}")
def get_transaction(
    transaction_id: str, session: Session = Depends(get_session)
) -> dict[str, object]:
    row = session.get(TransactionRow, transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="transaction_not_found")
    return transaction_dict(row)


@app.get("/accounts/{account_id}/history")
def account_history(account_id: str, limit: int = 100, session: Session = Depends(get_session)):
    rows = session.scalars(
        select(TransactionRow)
        .where(TransactionRow.account_id == account_id)
        .order_by(TransactionRow.timestamp.desc())
        .limit(min(limit, 500))
    )
    return [transaction_dict(row) for row in rows]


@app.get("/devices/{device_id}/history")
def device_history(device_id: str, limit: int = 100, session: Session = Depends(get_session)):
    rows = session.scalars(
        select(TransactionRow)
        .where(TransactionRow.device_id == device_id)
        .order_by(TransactionRow.timestamp.desc())
        .limit(min(limit, 500))
    )
    return [transaction_dict(row) for row in rows]


@app.get("/beneficiaries/{beneficiary_id}")
def beneficiary_history(
    beneficiary_id: str, limit: int = 100, session: Session = Depends(get_session)
):
    rows = session.scalars(
        select(TransactionRow)
        .where(TransactionRow.beneficiary_id == beneficiary_id)
        .order_by(TransactionRow.timestamp.desc())
        .limit(min(limit, 500))
    )
    items = [transaction_dict(row) for row in rows]
    if not items:
        raise HTTPException(status_code=404, detail="beneficiary_not_found")
    return {"beneficiary_id": beneficiary_id, "transactions": items}


@app.get("/investigations")
def list_investigations(
    limit: int = 100, session: Session = Depends(get_session)
) -> list[dict[str, str]]:
    cases = session.scalars(
        select(InvestigationRow)
        .order_by(InvestigationRow.created_at.desc())
        .limit(min(max(limit, 1), 500))
    )
    return [
        {
            "investigation_id": case.investigation_id,
            "transaction_id": case.transaction_id,
            "status": case.status,
        }
        for case in cases
    ]


@app.get("/investigations/{investigation_id}")
def get_investigation(investigation_id: str, session: Session = Depends(get_session)):
    case = session.get(InvestigationRow, investigation_id)
    if case is None:
        raise HTTPException(status_code=404, detail="investigation_not_found")
    transaction = session.get(TransactionRow, case.transaction_id)
    evidence = session.scalars(
        select(EvidenceRow).where(EvidenceRow.investigation_id == case.investigation_id)
    )
    latest_run = session.scalar(
        select(AgentRunRow)
        .where(AgentRunRow.investigation_id == investigation_id)
        .order_by(AgentRunRow.created_at.desc())
        .limit(1)
    )
    return {
        "investigation_id": case.investigation_id,
        "transaction": transaction_dict(transaction) if transaction else None,
        "status": case.status,
        "human_review_required": True,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "category": item.category,
                "source": item.source,
                "claim": item.claim,
                "observed_at": item.observed_at,
            }
            for item in evidence
        ],
        "latest_report": json.loads(latest_run.report_json) if latest_run else None,
    }


@app.post("/investigations/{investigation_id}/run")
def run_case_investigation(investigation_id: str, session: Session = Depends(get_session)):
    case = session.get(InvestigationRow, investigation_id)
    if case is None:
        raise HTTPException(status_code=404, detail="investigation_not_found")
    report = investigate_transaction(case.transaction_id, investigation_id)
    run_id = str(uuid.uuid4())
    session.add(
        AgentRunRow(
            run_id=run_id,
            investigation_id=investigation_id,
            status=report["verification_result"],
            report_json=json.dumps(report, separators=(",", ":")),
        )
    )
    session.commit()
    return {"run_id": run_id, **report}


def transaction_graph(session: Session):
    records = [transaction_dict(row) for row in session.scalars(select(TransactionRow))]
    return build_graph(pd.DataFrame(records))


@app.get("/accounts/{account_id}/connections")
def account_connections(
    account_id: str,
    depth: int = 2,
    session: Session = Depends(get_session),
):
    graph = transaction_graph(session)
    start = f"account:{account_id}"
    if start not in graph:
        raise HTTPException(status_code=404, detail="account_not_found")
    return {
        "account_id": account_id,
        "neighborhood": neighborhood(graph, start, depth),
        "shared_entities": detect_shared_entities(graph, account_id),
    }


@app.get("/transactions/{transaction_id}/network")
def transaction_network(
    transaction_id: str, depth: int = 2, session: Session = Depends(get_session)
):
    transaction = session.get(TransactionRow, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="transaction_not_found")
    graph = transaction_graph(session)
    return {
        "transaction_id": transaction_id,
        "neighborhood": neighborhood(graph, f"transaction:{transaction_id}", depth, limit=150),
        "shared_entities": detect_shared_entities(graph, transaction.account_id),
    }
