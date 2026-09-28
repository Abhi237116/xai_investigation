# Financial Crime Investigation AI --- Memory

> STATUS: ACTIVE IMPLEMENTATION RECORD
>
> This file should be updated after every completed implementation
> phase. Do not treat unchecked items as completed.

## 1. Current Phase

-   Phase: 17
-   Status: Application phases and Phase 16-17 deliverables are implemented; Phase 6 broker and Phase 15 cluster completion gates remain unverified because Docker/kubectl are unavailable
-   Last updated: 2026-09-29

## 2. Project Objective

Build an explainable, agentic financial-crime investigation platform
using: - XGBoost/LightGBM - SHAP - Kafka - NetworkX - LangGraph - MCP -
Groq/Gemini - FastAPI - PostgreSQL - Redis - Docker - Kubernetes -
optional AWS architecture

The system uses synthetic/public data and keeps humans responsible for
final decisions.

## 3. Completed

-   [x] Phase 0 (Python pytest/Ruff pass; React dependencies installed and lockfile generated)
-   [x] Phase 1 (canonical schema, seeded synthetic generator, CSV normalization, SQL schema, hash manifest and data dictionary)
-   [x] Phase 2 (chronological 60/20/20 where class-valid, stratified fallback, class weighting, leakage allowlist, serialized artifacts, PR-AUC/ROC-AUC/precision/recall/F1/Brier report)
-   [x] Phase 3 (local TreeExplainer, global mean absolute SHAP, persisted local/global outputs, ablation faithfulness and perturbation stability checks)
-   [x] Phase 4 (transaction, account, device, beneficiary and investigation read-only endpoints; seeded local evidence and stable IDs)
-   [x] Phase 5 (NetworkX account/device/beneficiary/merchant/transaction links, bounded neighborhoods and shared-entity evidence)
-   [ ] Phase 6 (event schemas, producer/consumer worker, validation and scoring topics, dead-letter routing, p50/p95/throughput/lag instrumentation are implemented; end-to-end broker gate not met because Docker is unavailable)
-   [x] Phase 7 (ten typed read-only MCP tools, audit hashes/status, schemas/permissions/timeouts/error behavior, counterfactual tool)
-   [x] Phase 8 (LangGraph investigator, transaction, identity, network, XAI and verification nodes; bounded tool data, evidence IDs and human-review report)
-   [x] Phase 9 (Groq/Gemini provider interface, bounded retries/timeouts/token cap, structured cited hypotheses, opt-in evidence export and health endpoint)
-   [x] Phase 10 (strict structured reports with citation-linked facts, model/SHAP/network/counterfactual outputs, missing evidence, uncertainty and human review)
-   [x] Phase 11 (fixed five-case held-out evaluation; evidence Recall@10 1.0, citations 1.0, tool accuracy 1.0, task completion 1.0, unsupported claims 0.0, counterfactual validity 1.0; mean latency 1.147s; LLM disabled; isolated SQLite)
-   [x] Phase 12 (reproducible JSON and HTML reports in `docs/benchmark`; API/SQLite/processor metrics measured; Kafka lag unavailable without Docker; LLM disabled)
-   [x] Phase 13 (React investigator workspace with Overview, Alerts, Investigations, Transactions, Evidence graph, XAI, Agent activity, Benchmarks, and System health; dark/light themes; production build passes)
-   [x] Phase 14 (request/tool Prometheus counters and latency gauges, correlation-ID structured JSON request logs, optional OTLP FastAPI tracing, Prometheus/Grafana Compose profile and dashboard)
-   [ ] Phase 15 (namespace, config/secret template, API/dashboard/worker deployments and services, probes, bounded resources, HPA are implemented; all 9 resources parse, but local cluster runtime gate not met because Docker/kubectl are unavailable)
-   [x] Phase 16 (EKS/MSK/RDS/S3/IAM/CloudWatch/Bedrock target mapped in deployment docs; Terraform creates nothing by default and only offers an opt-in encrypted/private bucket; cloud plan/apply not run)
-   [x] Phase 17 (13 pytest tests, Ruff, Bandit, pip dependency check, npm dependency tree and production build pass; no credentials detected; reports regenerated; API/dashboard manually exercised at desktop and mobile widths; external runtime limits documented)

## 4. Current Architecture

Update this section whenever architecture changes.

``` text
Current architecture:
FastAPI + SQLAlchemy evidence API; time-aware sklearn/XGBoost baselines + SHAP; NetworkX; audited MCP read-only tools; LangGraph deterministic agent workflow; optional Groq/Gemini and OTLP tracing; React/Vite investigator UI; SQLite default and Compose PostgreSQL/Redis/Redpanda, Prometheus and Grafana profile.
```

## 5. Technology Versions

Record actual versions used by the implementation.

``` text
Python: 3.14.3 local; CI targets 3.11
FastAPI: 0.141.1 installed
Pydantic: 2.13.5 installed
XGBoost: 3.4.1 installed
LightGBM: not configured
SHAP: 0.52.0 installed
Kafka: confluent-kafka 2.15.1 installed; Redpanda Compose configured; Docker unavailable on host, so live broker integration remains unverified
Flink: optional/not configured
PostgreSQL: 16 image configured; SQLite is local default
Redis: 7 image configured
LangGraph: 1.2.12 installed
MCP SDK: 1.30.0 installed; FastMCP server and typed tool schemas verified
Groq SDK: OpenAI Python SDK 2.54.0 installed for provider integration
React: 18.3.1 configured; packages installed, lockfile generated
Node: 24.14.0 local
Docker: unavailable on host
Kubernetes: local templates in `k8s/`; cluster runtime unavailable on host
Terraform: opt-in encrypted S3 artifact bucket; EKS/MSK/RDS/IAM/CloudWatch/Bedrock architecture documented, no resources created
Observability: Prometheus-compatible `/metrics`, Compose Prometheus/Grafana profile, optional OTLP FastAPI tracing
Dashboard: Vite 6.4.3, TypeScript 5.7.3, Recharts 2.15.4, lucide-react 0.468.0; chart chunk ~530 kB / 153 kB gzip
```

## 6. Model State

``` text
Dataset: synthetic-seed-42, 500 rows
Dataset version: SHA-256 7777fbd132dfd80df94442fb578955668a62fc67ae618fdb65c39503c004aaaa, schema 1
Features: amount, account_age_days, beneficiary_age_days, device_accounts_30d, transactions_1h, distance_from_usual_km
Training protocol: chronological 60/20/20 when every partition contains both classes; otherwise reproducible stratified fallback; fixed feature allowlist excludes label and identifiers; no validation/test tuning.
Validation protocol: validation reporting-only; final benchmark measured on held-out test rows.
Model: LR/RF/XGBoost baselines; no best model selected. All synthetic scores are near-perfect because the generator encodes labels in feature distributions; not a representative fraud benchmark.
Model version: pipeline 0.1.0
PR-AUC/ROC-AUC/F1: 1.0 for all baselines on 100 synthetic held-out rows; not a real-world performance claim
Calibration/Brier: LR 0.0003503, RF 0.0002917, XGBoost 0.0000151; no calibration method fitted
SHAP faithfulness: implemented; selected benchmark sample was uninformative
SHAP stability: Spearman ranking 1.0 under seeded 1% perturbation
```

## 7. LLM State

``` text
Primary provider: disabled by default; configure groq
Primary model: openai/gpt-oss-120b
Fallback provider: disabled by default; configure gemini
Fallback model: gemini-3.8-flash
Prompt version: 1
Tool-calling strategy: local deterministic LangGraph specialists; LLM only drafts evidence-cited hypotheses
Structured output schema: NarrativeDraft with AGENT_HYPOTHESIS and one or more evidence IDs
Current known limitations: Live Groq/Gemini calls not exercised because no API credentials supplied; evidence export is opt-in
```

## 8. Agent State

``` text
Investigator Agent:
Transaction Agent: bounded transaction retrieval; tested.
Identity Agent: bounded device/beneficiary histories; tested.
Network Agent: shared-entity evidence via NetworkX; tested.
XAI Agent: model output, SHAP and bounded simulation; explicit unavailable status without artifact; tested.
Verification Agent: claim category and citation checks; evaluation cases PASS.
```

For each agent record: - responsibilities - allowed tools - known bugs -
latest test result

## 9. MCP Tools

``` text
Implemented: 10 typed read-only tools for transaction/account/device/beneficiary histories, relationships, prediction, SHAP, evidence search and counterfactual simulation.

Tested: input/output schemas, permissions, timeout manifest, MCP registration and core workflow tests.

Blocked: none; external MCP host wiring is optional and unverified.
```

## 10. Benchmark State

``` text
ML: All baselines PR-AUC 1.0, ROC-AUC 1.0, F1 1.0; synthetic chronological test split only.

XAI:
Faithfulness: uninformative for selected sample
Stability: 1.0
Counterfactual validity: 1.0 on fixed five-case agent set

Agent (five fixed held-out transactions, seed 42): evidence Recall@10 1.0; citation correctness 1.0; tool accuracy 1.0; unsupported/hallucination rate 0.0; counterfactual validity 1.0; task completion 1.0; mean latency 1147 ms (p50 496 ms, p95 3032 ms); zero LLM tokens (disabled).

System: API p50 29.18 ms / p95 38.23 ms; SQLite 100-row query p50 2.40 ms / p95 6.00 ms; processor simulation 120.30 events/sec, p50 7.09 ms / p95 9.11 ms for 100 events. Live Kafka consumer lag unavailable.
```

## 11. Known Problems

Record only actual known problems.

``` text
- Synthetic labels are strongly encoded by generator feature ranges; perfect ML metrics must not be generalized.
- Docker is unavailable in the implementation host; live Redpanda/Kafka, Docker Compose, Kubernetes, Grafana and cloud deployment remain unverified.
- Public IEEE-CIS ingestion requires authorized user-held Kaggle access; no dataset bundled.
- Remote Groq/Gemini calls were not exercised because credentials are optional and absent.
- Evaluation cases share the held-out model partition; report metrics are small-sample demonstration metrics, not general agent quality estimates.
```

## 12. Decisions Made

Record decisions that future coding sessions must preserve.

``` text
- Public-data adapters are optional; the runnable path uses reproducible synthetic data with no bundled external dataset.
- Strict model feature allowlist; temporal split when class-valid; reporting-only validation; no best model automatically selected.
- Evaluation uses an isolated SQLite store and does not modify the app's local review queue.
- LLM evidence export is opt-in; consequential decisions remain human-controlled.
- AWS Terraform creates nothing by default; only an explicitly enabled encrypted, private artifact bucket is scaffolded.
- Date: 2026-09-29
```

## 13. Do Not Change

Record architecture/contracts that should not be changed casually.

``` text
- ML remains the source of risk scores.
- SHAP remains the source of model explanations.
- LLM cannot invent evidence.
- Human review remains required for consequential decisions.
- LLM provider remains configurable.
```

## 14. Next Task

Only one immediate task should be listed.

``` text
Next task:
Run Phase 6 end-to-end on Docker/Redpanda and Phase 15 on kind/Minikube when Docker/kubectl are available; Kaggle/provider credentials remain optional.
```

## 15. Update Rules

After completing a phase: 1. Update Current Phase. 2. Check the
completed phase. 3. Record actual dependency versions. 4. Record
model/benchmark results. 5. Record important architectural decisions. 6.
Record known bugs. 7. Record the next task. 8. Do not rewrite historical
decisions.
