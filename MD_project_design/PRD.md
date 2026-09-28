# Financial Crime Investigation AI --- PRD

## 1. Product Overview

Build a production-style, explainable, agentic financial-crime
investigation platform for synthetic/public financial data.

The system detects suspicious transactions using classical ML, explains
predictions using SHAP, investigates related evidence using
deterministic tools and graph analysis, and uses an LLM agent to
coordinate the investigation and produce a grounded case report.

The platform is a decision-support system. It must never autonomously
declare a person or account guilty of financial crime. Final decisions
remain with a human investigator.

## 2. Problem

Traditional fraud/AML demos usually stop at:

`transaction -> ML model -> fraud probability`

This project must go further:

`event -> streaming -> ML risk -> SHAP -> evidence retrieval -> graph investigation -> agent reasoning -> verification -> human review`

The goal is to demonstrate production-oriented AI engineering rather
than a generic chatbot.

## 3. Target Users

### Primary

-   Financial crime / AML investigator
-   Fraud analyst
-   Risk analyst
-   Compliance analyst

### Secondary

-   ML engineer
-   AI engineer
-   MLOps/platform engineer
-   Engineering manager reviewing investigation systems

## 4. Goals

1.  Detect suspicious financial activity with strong ML baselines.
2.  Provide model explanations using SHAP.
3.  Retrieve factual evidence from structured data.
4.  Investigate account/device/beneficiary relationships using graph
    analysis.
5.  Use an LLM agent to coordinate evidence gathering and generate a
    grounded report.
6.  Expose controlled investigation capabilities through MCP tools.
7.  Keep humans in the decision loop.
8.  Measure ML, XAI, agent, streaming and system performance.
9.  Run the core project at zero infrastructure cost.
10. Make the LLM provider replaceable.

## 5. Non-Goals

-   Autonomous accusation or legal determination.
-   Autonomous account freezing or transaction blocking.
-   Real customer/PII data.
-   Training an LLM.
-   Building a production banking core.
-   Claiming regulatory compliance.
-   Using the LLM as the source of truth for numerical risk scores.

## 6. Core User Flow

1.  Transaction enters Kafka.
2.  Stream processor validates/enriches the event.
3.  Feature service creates model features.
4.  XGBoost/LightGBM produces a risk score.
5.  SHAP calculates feature contributions.
6.  Evidence engine retrieves relevant transaction/account/device data.
7.  Network analysis identifies related entities.
8.  LangGraph orchestrates specialist investigation agents.
9.  Verification agent checks claims against evidence.
10. LLM produces a structured investigation report.
11. Human investigator reviews the case.
12. The system stores the complete audit trail.

## 7. Core Features

### F1 --- Transaction ingestion

-   Batch dataset ingestion.
-   Kafka event ingestion.
-   Schema validation.
-   Dead-letter handling.

### F2 --- Risk scoring

-   Logistic Regression baseline.
-   Random Forest baseline.
-   XGBoost production candidate.
-   Optional LightGBM comparison.
-   Versioned model artifacts.

### F3 --- Explainable AI

-   SHAP TreeExplainer for tree models.
-   Global feature importance.
-   Local per-case explanation.
-   Explanation stability test.
-   Feature ablation/faithfulness test.
-   Counterfactual analysis where feasible.

### F4 --- Evidence engine

Retrieve: - transaction history - account history - beneficiary
history - device history - location patterns - transaction velocity -
related entities - previous alerts

Every evidence item must have a stable ID and provenance.

### F5 --- Graph investigation

Use NetworkX initially.

Represent: - account -\> account - account -\> device - account -\>
beneficiary - account -\> merchant - account -\> transaction

Support: - connected-component analysis - shared-device discovery -
suspicious-path discovery - neighborhood exploration

### F6 --- Agentic investigation

Agents: - Investigator Agent - Transaction Agent - Identity Agent -
Network Agent - XAI Agent - Verification Agent

The Investigator Agent coordinates; specialist agents retrieve evidence
through tools.

### F7 --- MCP tools

Expose controlled tools for: - get_transaction - get_account_history -
get_device_history - get_beneficiary_history - find_related_accounts -
get_network_connections - get_model_prediction - get_shap_explanation -
generate_counterfactual - create_investigation

Read-only tools should be the default.

### F8 --- LLM

Primary provider: - Groq API

Initial model: - `openai/gpt-oss-120b`

Fallback provider: - Gemini API

The provider must be configurable through environment variables.

### F9 --- Human review

The UI must show: - risk score - model version - SHAP factors -
evidence - graph relationships - counterfactual result - agent reasoning
summary - citations/provenance - confidence/uncertainty - approve for
review / reject recommendation / request more evidence

The UI must clearly state that the AI recommendation is decision
support.

### F10 --- Evaluation

Evaluate: - PR-AUC - ROC-AUC - precision - recall - F1 -
calibration/Brier score - SHAP faithfulness - SHAP stability -
counterfactual validity - evidence Recall@K - evidence precision -
citation correctness - tool-call accuracy - unsupported-claim rate -
investigation completion - p50/p95 latency - Kafka throughput - consumer
lag

## 8. Dataset Strategy

Primary: - AMLSim for synthetic AML transaction/network investigation.

Secondary: - IEEE-CIS Fraud Detection for tabular fraud modeling.

Optional: - Elliptic dataset for transaction-network experimentation.

No real customer data.

## 9. Success Criteria

The project is considered complete when:

-   A reproducible training pipeline exists.
-   At least three ML baselines are compared.
-   The final model has a documented validation protocol.
-   Every investigated case can show SHAP evidence.
-   Agent claims can be traced to evidence/tool outputs.
-   Unsupported claims are measured.
-   The full pipeline works with a local Docker setup.
-   LLM provider can be changed without changing agent logic.
-   Tests cover critical tools and evidence transformations.
-   Benchmark results are reproducible from a documented command.

## 10. Deliverables

-   Backend API
-   React investigation dashboard
-   Kafka pipeline
-   ML training pipeline
-   SHAP/XAI module
-   Graph investigation module
-   LangGraph agent workflow
-   MCP server/tools
-   Evaluation suite
-   Docker Compose environment
-   Kubernetes manifests
-   Terraform architecture skeleton
-   CI workflow
-   Documentation
