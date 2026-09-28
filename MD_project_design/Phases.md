# Financial Crime Investigation AI --- Implementation Phases

## Phase 0 --- Repository and Development Environment

### Goal

Create a clean project skeleton.

### Build

-   Git repository
-   Python environment
-   React/Vite frontend
-   FastAPI backend
-   Docker Compose
-   `.env.example`
-   linting/formatting
-   pytest
-   basic CI

### Done when

``` text
pytest
ruff check .
```

run successfully.

------------------------------------------------------------------------

## Phase 1 --- Dataset and Data Model

### Goal

Get reproducible data into the system.

### Build

-   AMLSim ingestion
-   IEEE-CIS ingestion
-   schema validation
-   PostgreSQL schema
-   dataset versioning
-   data dictionary

### Done when

A clean dataset can be loaded repeatedly with the same results.

------------------------------------------------------------------------

## Phase 2 --- Baseline ML

### Goal

Establish trustworthy baselines.

### Build

-   Logistic Regression
-   Random Forest
-   XGBoost
-   train/validation/test protocol
-   class imbalance strategy
-   PR-AUC/ROC-AUC/F1/precision/recall
-   calibration

### Done when

A benchmark table is generated automatically.

------------------------------------------------------------------------

## Phase 3 --- SHAP Explainability

### Goal

Make predictions explainable.

### Build

-   SHAP TreeExplainer
-   local explanations
-   global explanations
-   feature contribution API
-   explanation persistence

### Experiments

-   feature ablation
-   SHAP faithfulness
-   SHAP stability

### Done when

A transaction can return a risk score plus reproducible SHAP evidence.

------------------------------------------------------------------------

## Phase 4 --- Transaction Investigation API

### Goal

Expose deterministic evidence retrieval.

### Build

FastAPI endpoints:

``` text
GET /transactions/{id}
GET /accounts/{id}/history
GET /devices/{id}/history
GET /beneficiaries/{id}
GET /investigations/{id}
```

### Done when

All evidence can be retrieved without an LLM.

------------------------------------------------------------------------

## Phase 5 --- Graph Investigation

### Goal

Add relationship-based investigation.

### Build

-   NetworkX graph
-   account/device/beneficiary relationships
-   connected components
-   suspicious relationship detection
-   neighborhood traversal

### Done when

A case can display related entities and paths.

------------------------------------------------------------------------

## Phase 6 --- Kafka Streaming

### Goal

Turn batch detection into event-driven detection.

### Build

Topics:

``` text
transactions.raw
transactions.validated
transactions.scored
investigations.created
deadletter.transactions
```

### Benchmark

Measure: - events/sec - p50 latency - p95 latency - consumer lag

### Done when

Synthetic transactions can flow end-to-end.

------------------------------------------------------------------------

## Phase 7 --- MCP Investigation Tools

### Goal

Expose controlled tools to agents.

### Build

Read-only tools:

``` text
get_transaction
get_account_history
get_device_history
find_related_accounts
get_network_connections
get_model_prediction
get_shap_explanation
search_evidence
```

### Done when

Tools have schemas, tests, permissions and structured outputs.

------------------------------------------------------------------------

## Phase 8 --- LangGraph Agent System

### Goal

Introduce agentic investigation.

### Agents

1.  Investigator
2.  Transaction
3.  Identity
4.  Network
5.  XAI
6.  Verification

### Rules

-   Investigator coordinates.
-   Specialist agents gather evidence.
-   Verification checks claims.
-   No autonomous consequential actions.

### Done when

A case can be investigated through an agent graph.

------------------------------------------------------------------------

## Phase 9 --- Groq LLM Integration

### Goal

Use a free/low-cost remote LLM for reasoning and report generation.

### Build

-   LLMProvider interface
-   Groq provider
-   Gemini fallback
-   retry/backoff
-   token budget
-   structured outputs
-   provider health check

Groq currently provides OpenAI-compatible APIs and supports tool use on
supported models. citeturn0search1turn0search8

### Done when

The application can switch providers through configuration without
changing agent logic.

------------------------------------------------------------------------

## Phase 10 --- Evidence-Grounded Investigation Reports

### Goal

Generate trustworthy reports.

### Report sections

``` text
Case ID
Risk score
Model version
Top SHAP factors
Observed evidence
Network findings
Agent hypotheses
Counterfactual
Missing evidence
Verification result
Human review status
```

Every factual claim must point to evidence.

### Done when

Unsupported claims are automatically detected or rejected.

------------------------------------------------------------------------

## Phase 11 --- Agent Evaluation

### Goal

Benchmark the agent separately from the ML model.

### Metrics

-   evidence Recall@K
-   citation correctness
-   tool-call accuracy
-   task completion
-   unsupported-claim rate
-   hallucination rate
-   latency
-   LLM token usage

Create a fixed evaluation set.

### Done when

`python evaluation/run_all.py` generates a report.

------------------------------------------------------------------------

## Phase 12 --- Full System Benchmark

### Goal

Measure the complete platform.

### Benchmark

``` text
ML
XAI
Agents
Kafka
API
Database
LLM
```

Generate:

``` text
benchmark_report.json
benchmark_report.html
```

### Done when

Results are reproducible and included in README.

------------------------------------------------------------------------

## Phase 13 --- Dashboard

### Goal

Create the investigator UI.

### Pages

1.  Overview
2.  Alerts
3.  Investigation workspace
4.  Transaction details
5.  Evidence graph
6.  XAI explanation
7.  Agent activity
8.  Benchmark dashboard

### Done when

An investigator can inspect a case without reading backend logs.

------------------------------------------------------------------------

## Phase 14 --- Observability

### Goal

Make the system production-like.

### Add

-   Prometheus
-   Grafana
-   OpenTelemetry
-   structured logging
-   correlation IDs

Track: - API latency - model latency - agent latency - tool failures -
Kafka lag - LLM rate limits

------------------------------------------------------------------------

## Phase 15 --- Kubernetes

### Goal

Deploy locally using Kubernetes.

### Build

-   Deployments
-   Services
-   ConfigMaps
-   Secrets
-   probes
-   resource limits
-   HPA where practical

Use kind or Minikube.

### Done when

Core services run in Kubernetes locally.

------------------------------------------------------------------------

## Phase 16 --- Terraform / AWS Target Architecture

### Goal

Demonstrate cloud readiness without requiring paid infrastructure.

### Document/prepare

-   EKS
-   MSK
-   RDS
-   S3
-   IAM
-   CloudWatch
-   Bedrock

Do not deploy expensive services by default.

### Done when

Terraform architecture is documented and safe to keep disabled.

------------------------------------------------------------------------

## Phase 17 --- Final Hardening

### Checklist

-   tests
-   security review
-   dependency review
-   error handling
-   rate-limit handling
-   reproducibility
-   documentation
-   screenshots
-   architecture diagram
-   benchmark report
-   demo script

### Final output

A recruiter can clone the repository, follow the README, run the local
stack, generate a case, inspect SHAP evidence, observe agent tool calls,
and view the final investigation report.
