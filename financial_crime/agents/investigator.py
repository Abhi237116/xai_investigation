from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from financial_crime.agents.llm import generate_grounded_hypotheses
from financial_crime.agents.report_schema import InvestigationReport
from financial_crime.evidence import tool_service


class InvestigationState(TypedDict):
    case_id: str
    transaction_id: str
    transaction_context: NotRequired[dict[str, Any]]
    risk_score: NotRequired[float | None]
    risk_evidence_id: NotRequired[str | None]
    model_version: NotRequired[str | None]
    shap_features: NotRequired[list[dict[str, Any]]]
    evidence_ids: NotRequired[list[str]]
    observations: NotRequired[list[dict[str, Any]]]
    graph_findings: NotRequired[list[dict[str, Any]]]
    agent_hypotheses: NotRequired[list[dict[str, Any]]]
    agent_actions: NotRequired[list[dict[str, Any]]]
    verification_status: NotRequired[str]
    final_report: NotRequired[dict[str, Any]]
    missing_evidence: NotRequired[list[str]]
    counterfactual: NotRequired[dict[str, Any]]


def _stable_evidence_id(tool_name: str, result: Any) -> str:
    canonical = json.dumps(result, sort_keys=True, default=str)
    digest = sha256(f"{tool_name}:{canonical}".encode()).hexdigest()[:12].upper()
    return f"EV-{digest}"


def _observation(tool: str, result: Any, claim: str, category: str = "OBSERVED") -> dict[str, Any]:
    return {
        "evidence_id": _stable_evidence_id(tool, result),
        "category": category,
        "source": tool,
        "claim": claim,
    }


def _action(agent: str, tool: str, result: Any, status: str = "success") -> dict[str, Any]:
    if isinstance(result, dict) and "transactions" in result:
        count = len(result["transactions"])
    elif isinstance(result, dict) and "related_accounts" in result:
        count = len(result["related_accounts"])
    elif isinstance(result, dict) and "features" in result:
        count = len(result["features"])
    else:
        count = int(isinstance(result, dict) and not result.get("error"))
    return {"agent": agent, "tool": tool, "status": status, "result_count": count}


def investigator_node(state: InvestigationState) -> dict[str, Any]:
    return {
        "evidence_ids": [],
        "observations": [],
        "graph_findings": [],
        "agent_hypotheses": [],
        "agent_actions": [
            {
                "agent": "Investigator Agent",
                "action": "coordinate bounded specialist evidence retrieval",
            }
        ],
        "missing_evidence": [],
    }


def transaction_agent_node(state: InvestigationState) -> dict[str, Any]:
    transaction = tool_service.get_transaction(state["transaction_id"])
    actions = [_action("Transaction Agent", "get_transaction", transaction)]
    observations = []
    if not transaction.get("error"):
        observation = _observation(
            "get_transaction",
            transaction,
            f"Transaction {transaction['transaction_id']} has amount "
            f"{transaction['amount']:.2f} and source {transaction['source']}.",
        )
        observations.append(observation)
    else:
        observations.append(
            _observation(
                "get_transaction", transaction, "Transaction record was unavailable.", "UNVERIFIED"
            )
        )
    context = (
        {
            key: transaction[key]
            for key in (
                "transaction_id",
                "account_id",
                "beneficiary_id",
                "device_id",
                "amount",
                "source",
            )
        }
        if not transaction.get("error")
        else {}
    )
    return {
        "transaction_context": context,
        "observations": observations,
        "evidence_ids": [item["evidence_id"] for item in observations],
        "agent_actions": [*state.get("agent_actions", []), *actions],
    }


def identity_agent_node(state: InvestigationState) -> dict[str, Any]:
    transaction = state.get("transaction_context", {})
    if not transaction:
        return {
            "agent_actions": [_action("Identity Agent", "get_device_history", {}, "skipped")],
            "missing_evidence": [
                *state.get("missing_evidence", []),
                "device_history",
                "beneficiary_history",
            ],
        }
    device = tool_service.get_device_history(transaction["device_id"], limit=10)
    beneficiary = tool_service.get_beneficiary_history(transaction["beneficiary_id"], limit=10)
    observations = []
    for tool, result, entity, label in [
        ("get_device_history", device, transaction["device_id"], "device"),
        ("get_beneficiary_history", beneficiary, transaction["beneficiary_id"], "beneficiary"),
    ]:
        items = result.get("transactions", [])
        observation = _observation(
            tool,
            result,
            f"The synthetic record set contains {len(items)} transactions associated "
            f"with {label} {entity}.",
        )
        observations.append(observation)
    return {
        "observations": [*state.get("observations", []), *observations],
        "evidence_ids": [
            *state.get("evidence_ids", []),
            *[item["evidence_id"] for item in observations],
        ],
        "agent_actions": [
            *state.get("agent_actions", []),
            _action("Identity Agent", "get_device_history", device),
            _action("Identity Agent", "get_beneficiary_history", beneficiary),
        ],
    }


def network_agent_node(state: InvestigationState) -> dict[str, Any]:
    transaction = state.get("transaction_context", {})
    if not transaction:
        return {"agent_actions": [_action("Network Agent", "find_related_accounts", {}, "skipped")]}
    result = tool_service.find_related_accounts(transaction["account_id"], depth=2)
    findings = result.get("shared_entities", [])
    observations = []
    for finding in findings:
        text = (
            f"Account {transaction['account_id']} shares {finding['entity_type']} "
            f"{finding['entity_id']} with {len(finding['related_account_ids'])} other account(s)."
        )
        observation = _observation("find_related_accounts", finding, text)
        finding["evidence_id"] = observation["evidence_id"]
        finding["category"] = "OBSERVED"
        finding["source"] = "find_related_accounts"
        observations.append(observation)
    hypotheses = []
    if findings:
        hypotheses.append(
            {
                "category": "AGENT_HYPOTHESIS",
                "claim": (
                    "The observed shared-entity links may warrant additional human review; "
                    "they do not establish intent."
                ),
                "evidence_ids": [item["evidence_id"] for item in observations],
            }
        )
    return {
        "graph_findings": findings,
        "observations": [*state.get("observations", []), *observations],
        "evidence_ids": [
            *state.get("evidence_ids", []),
            *[item["evidence_id"] for item in observations],
        ],
        "agent_hypotheses": hypotheses,
        "agent_actions": [
            *state.get("agent_actions", []),
            _action("Network Agent", "find_related_accounts", result),
        ],
    }


def xai_agent_node(state: InvestigationState) -> dict[str, Any]:
    prediction = tool_service.get_model_prediction(state["transaction_id"])
    explanation = tool_service.get_shap_explanation(state["transaction_id"])
    counterfactual = tool_service.generate_counterfactual_for_transaction(state["transaction_id"])
    actions = [
        _action("XAI Agent", "get_model_prediction", prediction),
        _action("XAI Agent", "get_shap_explanation", explanation),
        _action("XAI Agent", "generate_counterfactual", counterfactual),
    ]
    missing = []
    observations = []
    risk_score = prediction.get("risk_score")
    if risk_score is None:
        missing.append("model_prediction")
        risk_evidence_id = None
    else:
        risk_observation = _observation(
            "get_model_prediction",
            prediction,
            f"The {prediction['model_name']} model version "
            f"{prediction['model_version']} returned risk score {risk_score:.6f}.",
            "MODEL_OUTPUT",
        )
        observations.append(risk_observation)
        risk_evidence_id = risk_observation["evidence_id"]
    features = explanation.get("features", [])
    if explanation.get("status") != "available":
        missing.append("shap_explanation")
        shap_evidence_id = None
    else:
        shap_observation = _observation(
            "get_shap_explanation",
            explanation,
            f"SHAP TreeExplainer returned {len(features)} feature contributions "
            f"for model version {explanation['model_version']}.",
            "MODEL_OUTPUT",
        )
        observations.append(shap_observation)
        shap_evidence_id = shap_observation["evidence_id"]
        features = [
            {
                **item,
                "evidence_id": shap_evidence_id,
                "category": "MODEL_OUTPUT",
                "source": "get_shap_explanation",
            }
            for item in features
        ]
    if counterfactual.get("status") != "available":
        missing.append("counterfactual")
    else:
        cf_observation = _observation(
            "generate_counterfactual",
            counterfactual,
            "A bounded single-feature counterfactual simulation was computed; it does not "
            "predict real-world prevention.",
            "MODEL_OUTPUT",
        )
        observations.append(cf_observation)
        counterfactual = {
            **counterfactual,
            "evidence_id": cf_observation["evidence_id"],
            "category": "MODEL_OUTPUT",
            "source": "generate_counterfactual",
        }
    return {
        "risk_score": risk_score,
        "model_version": prediction.get("model_version"),
        "risk_evidence_id": risk_evidence_id,
        "shap_evidence_id": shap_evidence_id,
        "shap_features": features,
        "counterfactual": counterfactual,
        "observations": [*state.get("observations", []), *observations],
        "evidence_ids": [
            *state.get("evidence_ids", []),
            *[item["evidence_id"] for item in observations],
        ],
        "missing_evidence": [*state.get("missing_evidence", []), *missing],
        "agent_actions": [*state.get("agent_actions", []), *actions],
    }


def verification_agent_node(state: InvestigationState) -> dict[str, Any]:
    evidence_ids = set(state.get("evidence_ids", []))
    source_priority = {
        "get_transaction": 0,
        "get_device_history": 1,
        "get_beneficiary_history": 2,
        "get_model_prediction": 3,
        "get_shap_explanation": 4,
        "generate_counterfactual": 5,
    }
    observations = sorted(
        state.get("observations", []),
        key=lambda item: (source_priority.get(item.get("source"), 6), item["evidence_id"]),
    )
    valid_categories = {"OBSERVED", "MODEL_OUTPUT", "AGENT_HYPOTHESIS", "UNVERIFIED"}
    valid = (
        bool(observations)
        and any(item.get("category") in {"OBSERVED", "MODEL_OUTPUT"} for item in observations)
        and all(
            item.get("category") in valid_categories and item.get("evidence_id") in evidence_ids
            for item in observations
        )
    )
    for hypothesis in state.get("agent_hypotheses", []):
        valid = valid and hypothesis.get("category") == "AGENT_HYPOTHESIS"
        valid = valid and bool(hypothesis.get("evidence_ids"))
        valid = valid and set(hypothesis.get("evidence_ids", [])) <= evidence_ids
    if state.get("risk_score") is not None:
        valid = valid and state.get("risk_evidence_id") in evidence_ids
    if state.get("shap_features"):
        valid = valid and all(
            feature.get("evidence_id") in evidence_ids for feature in state["shap_features"][:5]
        )
    counterfactual = state.get("counterfactual", {})
    if counterfactual.get("status") == "available":
        valid = valid and counterfactual.get("evidence_id") in evidence_ids
    status = "PASS" if valid else "FAIL"
    report = {
        "case_id": state["case_id"],
        "transaction_id": state["transaction_id"],
        "risk_score": state.get("risk_score"),
        "model_version": state.get("model_version"),
        "risk_evidence_id": state.get("risk_evidence_id"),
        "risk_score_category": "MODEL_OUTPUT" if state.get("risk_score") is not None else None,
        "top_shap_factors": state.get("shap_features", [])[:5],
        "observed_evidence": observations,
        "network_findings": state.get("graph_findings", []),
        "agent_hypotheses": state.get("agent_hypotheses", []),
        "counterfactual": state.get(
            "counterfactual", {"status": "unavailable", "error": "ModelUnavailableError"}
        ),
        "agent_actions": state.get("agent_actions", []),
        "missing_evidence": state.get("missing_evidence", []),
        "verification_result": status,
        "human_review_status": "requires-review",
        "uncertainty_status": "not_estimated",
        "decision_support_notice": (
            "This report is decision support and does not determine guilt or "
            "authorize consequential action."
        ),
    }
    return {
        "verification_status": status,
        "final_report": report,
        "agent_actions": [
            {
                "agent": "Verification Agent",
                "action": "check claim categories and evidence references",
                "status": status,
            }
        ],
    }


def build_investigator_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("investigator", investigator_node)
    graph.add_node("transaction_agent", transaction_agent_node)
    graph.add_node("identity_agent", identity_agent_node)
    graph.add_node("network_agent", network_agent_node)
    graph.add_node("xai_agent", xai_agent_node)
    graph.add_node("verification_agent", verification_agent_node)
    graph.add_edge(START, "investigator")
    graph.add_edge("investigator", "transaction_agent")
    graph.add_edge("transaction_agent", "identity_agent")
    graph.add_edge("identity_agent", "network_agent")
    graph.add_edge("network_agent", "xai_agent")
    graph.add_edge("xai_agent", "verification_agent")
    graph.add_edge("verification_agent", END)
    return graph.compile()


def investigate_transaction(transaction_id: str, case_id: str | None = None) -> dict[str, Any]:
    workflow = build_investigator_graph()
    result = workflow.invoke(
        {"case_id": case_id or f"CASE-{transaction_id}", "transaction_id": transaction_id}
    )
    report = result["final_report"]
    narrative = generate_grounded_hypotheses(report)
    report["llm_status"] = narrative["status"]
    report["llm_provider"] = narrative["provider"]
    report["unsupported_claims_rejected"] = narrative.get("unsupported_claims_rejected", 0)
    report["llm_input_tokens"] = narrative.get("input_tokens", 0)
    report["llm_output_tokens"] = narrative.get("output_tokens", 0)
    report["agent_hypotheses"].extend(narrative["statements"])
    report["agent_actions"].append(
        {
            "agent": "Investigator Agent",
            "action": "request cited follow-up hypotheses",
            "provider": narrative["provider"],
            "status": narrative["status"],
            "unsupported_claims_rejected": report["unsupported_claims_rejected"],
        }
    )
    known_ids = {item["evidence_id"] for item in report["observed_evidence"]}
    narrative_verified = all(
        statement.get("category") == "AGENT_HYPOTHESIS"
        and set(statement.get("evidence_ids", [])) <= known_ids
        for statement in report["agent_hypotheses"]
    )
    report["verification_result"] = (
        "PASS" if report["verification_result"] == "PASS" and narrative_verified else "FAIL"
    )
    return InvestigationReport.model_validate(report).model_dump(mode="json")
