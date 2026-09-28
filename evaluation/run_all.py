import argparse
import html
import json
import statistics
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import select

if __package__:
    from .agent_metrics import evaluate_report
else:
    from agent_metrics import evaluate_report
from financial_crime.agents.investigator import investigate_transaction
from financial_crime.config import settings
from financial_crime.data.database import (
    InvestigationRow,
    TransactionRow,
    make_session_factory,
)
from financial_crime.data.ingest import load_dataset
from financial_crime.ml.explain import (
    evaluate_explanation_faithfulness,
    evaluate_explanation_stability,
)
from financial_crime.ml.train import train_benchmarks
from financial_crime.streaming.events import TransactionEvent
from financial_crime.streaming.worker import process_message


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": float(statistics.median(values)) if values else 0.0,
        "p95": float(pd.Series(values).quantile(0.95)) if values else 0.0,
    }


def run(output_dir: Path, cases_file: Path, rows: int = 500, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, dataset = load_dataset(rows=rows, seed=seed)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    train_result = train_benchmarks(frame, model_dir)
    artifact_path = model_dir / "xgboost.joblib"
    artifact = joblib.load(artifact_path)
    cases = json.loads(cases_file.read_text(encoding="utf-8"))["cases"]
    test_ids = set(artifact["test_transaction_ids"])
    requested_ids = {case["transaction_id"] for case in cases}
    if not requested_ids <= test_ids:
        missing = sorted(requested_ids - test_ids)
        raise ValueError(f"evaluation cases must be held-out test rows: {missing}")

    from financial_crime.api import main as api_main
    from financial_crime.data import database as database_module
    from financial_crime.evidence import tool_service

    old_database_url = settings.database_url
    old_artifact_path = settings.model_artifact_path
    old_factory = database_module.SessionFactory
    old_api_factory = api_main.SessionFactory
    old_tool_factory = tool_service.SessionFactory
    settings.database_url = f"sqlite:///{(output_dir / 'evaluation.db').resolve().as_posix()}"
    evaluation_session_factory = make_session_factory()
    database_module.SessionFactory = evaluation_session_factory
    api_main.SessionFactory = evaluation_session_factory
    tool_service.SessionFactory = evaluation_session_factory
    try:
        api_main.initialize_demo_database()
        with evaluation_session_factory() as session:
            present = set(
                session.scalars(
                    select(InvestigationRow.transaction_id).where(
                        InvestigationRow.transaction_id.in_(requested_ids)
                    )
                )
            )
            known_transactions = set(
                session.scalars(
                    select(TransactionRow.transaction_id).where(
                        TransactionRow.transaction_id.in_(requested_ids)
                    )
                )
            )
            if known_transactions != requested_ids:
                raise ValueError(
                    "the evidence database does not contain every evaluation transaction"
                )
            for transaction_id in requested_ids - present:
                session.add(
                    InvestigationRow(
                        investigation_id=f"CASE-{transaction_id}",
                        transaction_id=transaction_id,
                        status="requires-review",
                    )
                )
            session.commit()
            db_cases = {
                row.transaction_id: row.investigation_id
                for row in session.scalars(
                    select(InvestigationRow).where(
                        InvestigationRow.transaction_id.in_(requested_ids)
                    )
                )
            }
        if set(db_cases) != requested_ids:
            raise ValueError("the seeded evidence database does not contain every evaluation case")

        settings.model_artifact_path = str(artifact_path)
        case_results = []
        for case in cases:
            started = time.perf_counter()
            report = investigate_transaction(
                case["transaction_id"], db_cases[case["transaction_id"]]
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            case_results.append(evaluate_report(report, artifact, elapsed_ms))

        api_latencies = []
        with TestClient(api_main.app) as client:
            for _ in range(5):
                started = time.perf_counter()
                response = client.get("/summary")
                if response.status_code != 200:
                    raise RuntimeError("API benchmark request failed")
                api_latencies.append((time.perf_counter() - started) * 1000)

        database_latencies = []
        for _ in range(10):
            started = time.perf_counter()
            with evaluation_session_factory() as session:
                session.scalars(select(TransactionRow).limit(100)).all()
            database_latencies.append((time.perf_counter() - started) * 1000)

        stream_latencies = []
        stream_started = time.perf_counter()
        for _, row in frame.head(100).iterrows():
            event = TransactionEvent(payload=row.to_dict())
            started = time.perf_counter()
            outputs = process_message(event.model_dump_json(), artifact)
            if not any(output.topic == "transactions.scored" for output in outputs):
                raise RuntimeError("stream processor did not emit a score status")
            stream_latencies.append((time.perf_counter() - started) * 1000)
        stream_elapsed = time.perf_counter() - stream_started
        api_benchmark = {"status": "measured_in_process", "latency_ms": _summary(api_latencies)}
        database_benchmark = {
            "status": "measured_sqlite",
            "query": "select up to 100 transactions",
            "latency_ms": _summary(database_latencies),
        }
        streaming_benchmark = {
            "status": "processor_simulation_only",
            "events": len(stream_latencies),
            "events_per_second": len(stream_latencies) / stream_elapsed,
            "latency_ms": _summary(stream_latencies),
            "consumer_lag": None,
            "broker_status": "not_run_requires_Docker",
        }
    finally:
        settings.model_artifact_path = old_artifact_path
        settings.database_url = old_database_url
        database_module.SessionFactory = old_factory
        api_main.SessionFactory = old_api_factory
        tool_service.SessionFactory = old_tool_factory

    metrics = [
        "evidence_recall_at_10",
        "citation_correctness",
        "tool_call_accuracy",
        "unsupported_claim_rate",
        "hallucination_rate",
        "counterfactual_validity",
        "task_completion",
    ]
    report = {
        "evaluation_version": "1",
        "dataset": dataset,
        "model": {
            "name": "xgboost",
            "version": artifact.get("version"),
            "test_rows": len(artifact["test_transaction_ids"]),
        },
        "llm_provider": settings.llm_provider,
        "cases": len(case_results),
        "metrics": {
            name: _summary([float(item[name]) for item in case_results]) for name in metrics
        },
        "latency_ms": _summary([float(item["latency_ms"]) for item in case_results]),
        "llm_token_usage": {
            "input_tokens": sum(int(item["llm_input_tokens"]) for item in case_results),
            "output_tokens": sum(int(item["llm_output_tokens"]) for item in case_results),
        },
        "ml_benchmark": train_result["models"],
        "case_results": case_results,
    }
    (output_dir / "agent_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    sample = frame.iloc[0]
    values = {feature: float(sample[feature]) for feature in artifact["features"]}
    xai = {
        "faithfulness": evaluate_explanation_faithfulness(artifact, values),
        "stability": evaluate_explanation_stability(artifact, values),
    }
    system = {
        "ml": {"status": "measured", "models": train_result["models"]},
        "xai": {"status": "measured", **xai},
        "agents": {"status": "measured", "metrics": report["metrics"]},
        "api": api_benchmark,
        "database": database_benchmark,
        "streaming": streaming_benchmark,
        "kafka": {"status": "not_run", "reason": "requires Docker/Redpanda broker"},
        "llm": {
            "status": "disabled" if settings.llm_provider == "disabled" else "configured",
            "token_usage": report["llm_token_usage"],
        },
        "dataset": dataset,
        "model_version": artifact.get("version"),
        "note": (
            "Synthetic development benchmark only; not representative of production performance."
        ),
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(system, indent=2), encoding="utf-8"
    )
    rows = [
        (
            f"<tr><th>{html.escape(name)}</th>"
            f"<td>{html.escape(str(value['status']))}</td>"
            f"<td><pre>{html.escape(json.dumps(value, indent=2))}</pre></td></tr>"
        )
        for name, value in system.items()
        if isinstance(value, dict) and "status" in value
    ]
    rows_html = "".join(rows)
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'><title>FCI System Benchmark</title>",
            "<style>body{font:14px system-ui;max-width:1100px;margin:32px auto;color:#182220}",
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5d2;",
            "padding:12px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;",
            "overflow-wrap:anywhere}aside{padding:12px;background:#eef4f1}</style>",
            "<h1>System benchmark</h1>",
            "<aside>Synthetic development results; not a production benchmark.</aside>",
            f"<p>Dataset hash: {html.escape(dataset['sha256'])} | Model "
            f"{html.escape(str(artifact.get('version')))}</p>",
            "<table><thead><tr><th>Subsystem</th><th>Status</th><th>Details</th></tr></thead>",
            f"<tbody>{rows_html}</tbody></table>",
        ]
    )
    (output_dir / "benchmark_report.html").write_text(page, encoding="utf-8")
    report["system_benchmark"] = system
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the financial crime investigation agents"
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark"))
    parser.add_argument("--cases", type=Path, default=Path("evaluation/fixtures/agent_cases.json"))
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.cases, args.rows, args.seed), indent=2))


if __name__ == "__main__":
    main()
