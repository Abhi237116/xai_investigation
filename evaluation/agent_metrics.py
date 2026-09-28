from typing import Any

import pandas as pd

from financial_crime.data.schema import FEATURES

REQUIRED_TOOLS = {
    "get_transaction",
    "get_device_history",
    "get_beneficiary_history",
    "find_related_accounts",
    "get_model_prediction",
    "get_shap_explanation",
    "generate_counterfactual",
}
REQUIRED_EVIDENCE_SOURCES = {
    "get_transaction",
    "get_device_history",
    "get_beneficiary_history",
    "get_model_prediction",
    "get_shap_explanation",
    "generate_counterfactual",
}


def evaluate_report(
    report: dict[str, Any], artifact: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    evidence = report.get("observed_evidence", [])
    known_ids = {item.get("evidence_id") for item in evidence}
    sources_at_k = {item.get("source") for item in evidence[:10]}
    evidence_recall = len(REQUIRED_EVIDENCE_SOURCES & sources_at_k) / len(REQUIRED_EVIDENCE_SOURCES)

    references: list[str] = []
    references.extend(item.get("evidence_id") for item in report.get("top_shap_factors", []))
    references.extend(item.get("evidence_id") for item in report.get("network_findings", []))
    for hypothesis in report.get("agent_hypotheses", []):
        references.extend(hypothesis.get("evidence_ids", []))
    if report.get("risk_evidence_id"):
        references.append(report["risk_evidence_id"])
    counterfactual = report.get("counterfactual", {})
    if counterfactual.get("status") == "available":
        references.append(counterfactual.get("evidence_id"))
    valid_references = sum(reference in known_ids for reference in references)
    citation_correctness = valid_references / max(1, len(references))

    actual_calls = [
        action.get("tool") for action in report.get("agent_actions", []) if action.get("tool")
    ]
    correct_calls = sum(tool in REQUIRED_TOOLS for tool in actual_calls)
    tool_accuracy = correct_calls / max(len(REQUIRED_TOOLS), len(actual_calls))

    claim_count = (
        len(evidence)
        + len(report.get("top_shap_factors", []))
        + len(report.get("network_findings", []))
        + len(report.get("agent_hypotheses", []))
        + int(report.get("risk_score") is not None)
        + int(counterfactual.get("status") == "available")
    )
    unsupported_count = len(references) - valid_references
    unsupported_count += int(report.get("unsupported_claims_rejected", 0))
    unsupported_rate = unsupported_count / max(1, claim_count)

    cf_valid = False
    if counterfactual.get("status") == "available":
        values = counterfactual.get("feature_values", {})
        try:
            score = float(
                artifact["model"].predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1]
            )
            cf_valid = abs(score - float(counterfactual["simulated_model_score"])) < 1e-6
            cf_valid = cf_valid and counterfactual.get("simulation_only") is True
            cf_valid = cf_valid and counterfactual.get("evidence_id") in known_ids
            changed = set(counterfactual.get("changed_features", []))
            unchanged = set(counterfactual.get("unchanged_features", []))
            cf_valid = cf_valid and changed <= set(FEATURES)
            cf_valid = cf_valid and unchanged == set(FEATURES) - changed
            cf_valid = cf_valid and all(
                artifact["feature_minimums"][feature]
                <= float(values[feature])
                <= artifact["feature_maximums"][feature]
                for feature in changed
            )
            cf_valid = cf_valid and counterfactual.get("prediction_changed") == (
                (score >= counterfactual["threshold"])
                != (float(counterfactual["original_model_score"]) >= counterfactual["threshold"])
            )
        except (KeyError, TypeError, ValueError):
            cf_valid = False

    calls_complete = set(actual_calls) == REQUIRED_TOOLS and len(actual_calls) == len(
        REQUIRED_TOOLS
    )
    available_sources = {"get_transaction", "get_device_history", "get_beneficiary_history"}
    available_sources |= sources_at_k & {
        "get_model_prediction",
        "get_shap_explanation",
        "generate_counterfactual",
    }
    required_available = len(available_sources)
    evidence_complete = len(available_sources & sources_at_k) == required_available
    task_complete = (
        report.get("verification_result") == "PASS"
        and evidence_complete
        and citation_correctness == 1.0
        and calls_complete
        and cf_valid
        and report.get("risk_score") is not None
        and bool(report.get("top_shap_factors"))
    )
    return {
        "case_id": report.get("case_id"),
        "evidence_recall_at_10": evidence_recall,
        "citation_correctness": citation_correctness,
        "tool_call_accuracy": tool_accuracy,
        "unsupported_claim_rate": unsupported_rate,
        "hallucination_rate": unsupported_rate,
        "counterfactual_validity": float(cf_valid),
        "task_completion": float(task_complete),
        "verification_result": report.get("verification_result"),
        "latency_ms": latency_ms,
        "llm_input_tokens": report.get("llm_input_tokens", 0),
        "llm_output_tokens": report.get("llm_output_tokens", 0),
    }
