import asyncio

import joblib
from fastapi.testclient import TestClient

from financial_crime.api.main import app
from financial_crime.data.ingest import load_dataset
from financial_crime.evidence.graph import build_graph, detect_shared_entities, neighborhood
from financial_crime.evidence.tool_service import TOOL_MANIFEST
from financial_crime.ml.counterfactual import generate_counterfactual
from financial_crime.ml.explain import (
    evaluate_explanation_faithfulness,
    evaluate_explanation_stability,
    explain_case,
    global_shap_importance,
)
from financial_crime.ml.train import train_benchmarks
from financial_crime.streaming.events import TransactionEvent
from financial_crime.streaming.worker import process_message, summarize_latency


def test_seeded_data_is_reproducible() -> None:
    first, first_manifest = load_dataset(rows=120, seed=12)
    second, second_manifest = load_dataset(rows=120, seed=12)
    assert first.equals(second)
    assert first_manifest["sha256"] == second_manifest["sha256"]


def test_seed_changes_data_manifest() -> None:
    _, first_manifest = load_dataset(rows=120, seed=12)
    _, second_manifest = load_dataset(rows=120, seed=13)
    assert first_manifest["sha256"] != second_manifest["sha256"]


def test_training_emits_required_metrics_and_three_models(tmp_path) -> None:
    frame, _ = load_dataset(rows=600, seed=22)
    result = train_benchmarks(frame, tmp_path)
    assert {"logistic_regression", "random_forest", "xgboost"} <= result["models"].keys()
    for metrics in result["models"].values():
        assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score"} <= metrics.keys()
    assert (tmp_path / "benchmark_report.json").exists()
    protocol = result["protocol"]
    assert protocol["split"] == "chronological 60/20/20; validation is reporting-only"
    assert protocol["target_excluded"] is True
    assert protocol["train_validation_test_disjoint"] is True


def test_tree_explanation_has_model_provenance(tmp_path) -> None:
    frame, _ = load_dataset(rows=600, seed=23)
    train_benchmarks(frame, tmp_path)
    artifact = joblib.load(tmp_path / "xgboost.joblib")
    values = {key: float(frame.iloc[0][key]) for key in artifact["features"]}
    explanation = explain_case(artifact, values)
    assert explanation["explainer_type"] == "TreeExplainer"
    assert explanation["model_version"] == artifact["version"]
    assert len(explanation["features"]) == len(artifact["features"])
    assert global_shap_importance(artifact, frame.head(20), tmp_path / "global.json")
    assert (tmp_path / "global.json").exists()
    faithfulness = evaluate_explanation_faithfulness(artifact, values)
    stability = evaluate_explanation_stability(artifact, values)
    assert faithfulness["metric"] == "spearman_shap_vs_ablation"
    assert stability["metric"] == "spearman_rank_correlation"
    simulation = generate_counterfactual(artifact, values)
    assert simulation["simulation_only"] is True
    assert "This is an analytical model simulation" in simulation["feasibility_constraints"][-1]


def test_read_only_evidence_api_serves_seeded_records() -> None:
    with TestClient(app) as client:
        summary = client.get("/summary").json()
        assert summary["transaction_count"] == 500
        assert summary["synthetic_only"] is True
        assert client.get("/health/ready").json()["status"] == "ready"
        assert client.get("/benchmarks").status_code == 200
        assert "fci_api_requests_total" in client.get("/metrics").text
        response = client.get("/transactions", params={"limit": 1})
        assert response.status_code == 200
        transaction = response.json()[0]
        detail = client.get(f"/transactions/{transaction['transaction_id']}")
        assert detail.status_code == 200
        assert detail.json()["source"] == "synthetic"
        missing = client.get("/transactions/not-a-real-transaction")
        assert missing.status_code == 404


def test_graph_finds_shared_device_relationships() -> None:
    import pandas as pd

    records = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "account_id": "A1",
                "device_id": "D1",
                "beneficiary_id": "B1",
                "merchant_id": "M1",
            },
            {
                "transaction_id": "T2",
                "account_id": "A2",
                "device_id": "D1",
                "beneficiary_id": "B2",
                "merchant_id": "M2",
            },
        ]
    )
    graph = build_graph(records)
    findings = detect_shared_entities(graph, "A1")
    assert any(
        item["entity_type"] == "device" and item["related_account_ids"] == ["A2"]
        for item in findings
    )
    result = neighborhood(graph, "account:A1", depth=2)
    assert "A2" in result["connected_accounts"]


def test_stream_validation_and_unavailable_model_are_explicit() -> None:
    frame, _ = load_dataset(rows=100, seed=31)
    from financial_crime.data.schema import Transaction

    payload = Transaction.model_validate(frame.iloc[0].to_dict()).model_copy(
        update={"is_suspicious": None}
    )
    event = TransactionEvent(payload=payload)
    outputs = process_message(event.model_dump_json())
    assert outputs[0].topic == "transactions.validated"
    assert outputs[0].payload["transaction"]["is_suspicious"] is None
    assert outputs[1].payload["model_status"] == "unavailable"
    assert outputs[1].payload["risk_score"] is None
    assert process_message("not-json")[0].topic == "deadletter.transactions"
    assert summarize_latency([0.01, 0.02], 0.03)["events_processed"] == 2.0


def test_investigation_tool_manifest_is_read_only_and_structured() -> None:
    names = {tool["name"] for tool in TOOL_MANIFEST}
    assert {"get_transaction", "get_shap_explanation", "search_evidence"} <= names
    assert all(tool["permission_level"] == "read-only" for tool in TOOL_MANIFEST)
    assert all("input_schema" in tool and "output_schema" in tool for tool in TOOL_MANIFEST)


def test_mcp_server_registers_typed_input_and_output_schemas() -> None:
    from financial_crime.evidence.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert {"get_transaction", "get_shap_explanation", "search_evidence"} <= by_name.keys()
    assert (
        by_name["get_transaction"].inputSchema["properties"]["transaction_id"]["type"] == "string"
    )
    assert by_name["get_transaction"].outputSchema["type"] == "object"


def test_langgraph_investigation_is_bounded_and_verified(monkeypatch) -> None:
    from financial_crime.config import settings

    monkeypatch.setattr(settings, "model_artifact_path", "artifacts/not-present-for-test.joblib")
    with TestClient(app) as client:
        case = client.get("/investigations", params={"limit": 1}).json()[0]
        response = client.post(f"/investigations/{case['investigation_id']}/run")
        assert response.status_code == 200
        report = response.json()
        latest = client.get(f"/investigations/{case['investigation_id']}").json()
        assert latest["latest_report"]["verification_result"] == "PASS"
    assert report["verification_result"] == "PASS"
    assert report["human_review_status"] == "requires-review"
    assert report["risk_score"] is None
    assert report["risk_score_category"] is None
    assert report["counterfactual"]["status"] == "unavailable"
    assert report["uncertainty_status"] == "not_estimated"
    assert {"model_prediction", "shap_explanation", "counterfactual"} <= set(
        report["missing_evidence"]
    )
    assert report["llm_status"] == "disabled"
    assert {"Transaction Agent", "Identity Agent", "Network Agent", "XAI Agent"} <= {
        action["agent"] for action in report["agent_actions"]
    }
    evidence_ids = {item["evidence_id"] for item in report["observed_evidence"]}
    assert all(item["evidence_id"] in evidence_ids for item in report["observed_evidence"])
    assert all(
        item["category"] in {"OBSERVED", "MODEL_OUTPUT", "UNVERIFIED"}
        for item in report["observed_evidence"]
    )


def test_llm_fallback_only_accepts_cited_hypotheses(monkeypatch) -> None:
    from financial_crime.agents import llm
    from financial_crime.agents.llm import NarrativeDraft, ProviderResponse

    class FailingProvider:
        name = "groq"

        def generate(self, _packet):
            raise RuntimeError("rate limit")

    class WorkingProvider:
        name = "gemini"

        def generate(self, packet):
            valid_id = packet["allowed_evidence_ids"][0]
            return ProviderResponse(
                draft=NarrativeDraft.model_validate(
                    {
                        "statements": [
                            {
                                "category": "AGENT_HYPOTHESIS",
                                "text": "Consider reviewing the observed relationship.",
                                "evidence_ids": [valid_id],
                            },
                            {
                                "category": "AGENT_HYPOTHESIS",
                                "text": "Unsupported item.",
                                "evidence_ids": ["EV-UNKNOWN"],
                            },
                        ]
                    }
                ),
                input_tokens=18,
                output_tokens=12,
            )

    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    monkeypatch.setattr(llm.settings, "llm_fallback_provider", "gemini")
    monkeypatch.setattr(llm.settings, "llm_allow_evidence_export", True)
    monkeypatch.setattr(llm.settings, "llm_max_retries", 0)
    monkeypatch.setattr(
        llm, "_provider", lambda name: FailingProvider() if name == "groq" else WorkingProvider()
    )
    report = {
        "case_id": "CASE-T1",
        "risk_score": 0.76,
        "top_shap_factors": [],
        "observed_evidence": [
            {
                "evidence_id": "EV-1",
                "category": "OBSERVED",
                "source": "db",
                "claim": "A link exists.",
            }
        ],
    }
    output = llm.generate_grounded_hypotheses(report)
    assert output["provider"] == "gemini"
    assert len(output["statements"]) == 1
    assert output["unsupported_claims_rejected"] == 1


def test_llm_evidence_export_is_opt_in(monkeypatch) -> None:
    from financial_crime.agents import llm

    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    monkeypatch.setattr(llm.settings, "llm_allow_evidence_export", False)
    monkeypatch.setattr(llm, "_provider", lambda _name: (_ for _ in ()).throw(AssertionError()))
    result = llm.generate_grounded_hypotheses({"observed_evidence": []})
    assert result["status"] == "evidence_export_disabled"


def test_agent_metrics_report_citations_tools_and_counterfactual() -> None:
    import numpy as np

    from evaluation.agent_metrics import REQUIRED_EVIDENCE_SOURCES, REQUIRED_TOOLS, evaluate_report

    evidence = [
        {"evidence_id": f"EV-{index}", "source": source, "claim": "Synthetic evidence."}
        for index, source in enumerate(sorted(REQUIRED_EVIDENCE_SOURCES))
    ]
    evidence_ids = [item["evidence_id"] for item in evidence]

    class ConstantModel:
        def predict_proba(self, _frame):
            return np.array([[0.5, 0.5]])

    report = {
        "case_id": "CASE-T1",
        "verification_result": "PASS",
        "risk_score": 0.5,
        "risk_evidence_id": evidence_ids[0],
        "observed_evidence": evidence,
        "top_shap_factors": [{"feature": "amount", "evidence_id": evidence_ids[1]}],
        "network_findings": [{"evidence_id": evidence_ids[2]}],
        "agent_hypotheses": [{"evidence_ids": [evidence_ids[3]]}],
        "agent_actions": [{"tool": tool} for tool in sorted(REQUIRED_TOOLS)],
        "unsupported_claims_rejected": 0,
        "counterfactual": {
            "status": "available",
            "evidence_id": evidence_ids[4],
            "simulation_only": True,
            "feature_values": {
                "amount": 100,
                "account_age_days": 100,
                "beneficiary_age_days": 100,
                "device_accounts_30d": 1,
                "transactions_1h": 1,
                "distance_from_usual_km": 5,
            },
            "simulated_model_score": 0.5,
            "changed_features": [],
        },
    }
    artifact = {
        "model": ConstantModel(),
        "feature_minimums": dict.fromkeys(
            [
                "amount",
                "account_age_days",
                "beneficiary_age_days",
                "device_accounts_30d",
                "transactions_1h",
                "distance_from_usual_km",
            ],
            0,
        ),
        "feature_maximums": dict.fromkeys(
            [
                "amount",
                "account_age_days",
                "beneficiary_age_days",
                "device_accounts_30d",
                "transactions_1h",
                "distance_from_usual_km",
            ],
            1000,
        ),
    }
    report["counterfactual"]["threshold"] = 0.8
    report["counterfactual"]["original_model_score"] = 0.5
    report["counterfactual"]["prediction_changed"] = False
    report["counterfactual"]["unchanged_features"] = [
        "amount",
        "account_age_days",
        "beneficiary_age_days",
        "device_accounts_30d",
        "transactions_1h",
        "distance_from_usual_km",
    ]
    result = evaluate_report(report, artifact, 10.0)
    assert result["citation_correctness"] == 1
    assert result["tool_call_accuracy"] == 1
    assert result["counterfactual_validity"] == 1
    assert result["task_completion"] == 1
