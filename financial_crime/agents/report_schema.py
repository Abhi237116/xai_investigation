from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    category: Literal["OBSERVED", "MODEL_OUTPUT", "AGENT_HYPOTHESIS", "UNVERIFIED"]
    source: str
    claim: str


class ShapFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float
    contribution: float
    evidence_id: str
    category: Literal["MODEL_OUTPUT"]
    source: str


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    transaction_id: str
    risk_score: float | None
    model_version: str | None
    risk_evidence_id: str | None
    risk_score_category: Literal["MODEL_OUTPUT"] | None
    top_shap_factors: list[ShapFactor]
    observed_evidence: list[EvidenceClaim] = Field(max_length=100)
    network_findings: list[dict[str, Any]] = Field(max_length=100)
    agent_hypotheses: list[dict[str, Any]] = Field(max_length=20)
    counterfactual: dict[str, Any]
    missing_evidence: list[str]
    verification_result: Literal["PASS", "FAIL"]
    human_review_status: Literal["requires-review"]
    uncertainty_status: Literal["not_estimated"]
    decision_support_notice: str
    agent_actions: list[dict[str, Any]] = Field(max_length=50)
    llm_status: str
    llm_provider: str | None
    unsupported_claims_rejected: int
    llm_input_tokens: int
    llm_output_tokens: int
