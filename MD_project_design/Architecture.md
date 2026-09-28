# Financial Crime Investigation AI --- Architecture

## 1. Architecture Principles

1.  Deterministic systems are the source of truth.
2.  The LLM interprets and orchestrates; it does not invent evidence.
3.  ML generates risk scores.
4.  SHAP explains the ML model.
5.  SQL and graph tools retrieve factual evidence.
6.  Human approval is required for consequential actions.
7.  Every important decision is auditable.
8.  Components must be replaceable.
9.  Development must work without paid cloud infrastructure.
10. The system must degrade gracefully when the LLM is unavailable.

## 2. High-Level Architecture

``` text
                     DATA SOURCES
                         |
             +-----------+-----------+
             |                       |
          AMLSim                 IEEE-CIS
             |                       |
             +-----------+-----------+
                         |
                       Kafka
                         |
                       Flink
                         |
                  Feature Pipeline
                         |
                +--------+--------+
                |                 |
              ML API          Evidence DB
                |                 |
          XGBoost/LightGBM        |
                |                 |
              SHAP                |
                |                 |
                +--------+--------+
                         |
                  Evidence Layer
                         |
            +------------+------------+
            |            |            |
           SQL        NetworkX       Rules
            |            |            |
            +------------+------------+
                         |
                    LangGraph
                         |
      +------------------+------------------+
      |          |          |        |      |
 Transaction  Identity   Network    XAI  Verification
   Agent      Agent      Agent     Agent    Agent
      |          |          |        |      |
      +----------+----------+--------+------+
                         |
                       MCP
                         |
                  Groq / Gemini
                         |
                 Investigation Report
                         |
                  Human Review UI
```

## 3. Technology Stack

### Backend

-   Python 3.11+
-   FastAPI
-   Pydantic
-   SQLAlchemy

### ML

-   scikit-learn
-   XGBoost
-   LightGBM
-   SHAP
-   NetworkX

### Streaming

-   Apache Kafka
-   Apache Flink

For the first milestone, Kafka is mandatory; Flink can initially be a
lightweight/optional stream-processing service so the project remains
practical on constrained hardware.

### Storage

-   PostgreSQL
-   pgvector
-   Redis

### Agents

-   LangGraph
-   Groq API
-   Gemini API fallback
-   MCP

### Frontend

-   React
-   TypeScript
-   Vite
-   Tailwind CSS
-   Recharts or another lightweight charting library

### Infrastructure

-   Docker
-   Docker Compose
-   Kubernetes
-   kind or Minikube
-   Terraform
-   GitHub Actions

### Observability

-   Prometheus
-   Grafana
-   OpenTelemetry
-   Langfuse, only if useful and configured safely

## 4. LLM Provider Design

Use an abstraction:

``` text
LLMProvider
  |
  +-- GroqProvider
  |
  +-- GeminiProvider
```

Configuration:

``` env
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
```

Fallback:

``` env
LLM_FALLBACK_PROVIDER=gemini
```

The current Groq API is OpenAI-compatible and supports tool use for
supported models. Its documented limits are model/account dependent, so
the application must implement retry/backoff and configurable rate
limiting. citeturn0search0turn0search1turn0search4

## 5. Service Layout

``` text
financial-crime-ai/
├── apps/
│   ├── api/
│   ├── worker/
│   └── dashboard/
├── agents/
│   ├── investigator/
│   ├── transaction/
│   ├── identity/
│   ├── network/
│   ├── xai/
│   └── verifier/
├── mcp/
│   └── investigation_server/
├── ml/
│   ├── training/
│   ├── inference/
│   ├── features/
│   └── explainability/
├── streaming/
│   ├── producers/
│   ├── consumers/
│   └── schemas/
├── evidence/
│   ├── sql/
│   ├── graph/
│   └── provenance/
├── evaluation/
│   ├── model/
│   ├── xai/
│   ├── agent/
│   └── system/
├── tests/
├── configs/
├── scripts/
├── infra/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
├── docs/
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
└── Memory.md
```

## 6. Data Flow

### Training

``` text
Dataset
 -> validation
 -> preprocessing
 -> feature engineering
 -> train/validation/test split
 -> baseline models
 -> XGBoost/LightGBM
 -> evaluation
 -> SHAP evaluation
 -> MLflow/model artifact
```

### Online investigation

``` text
Kafka event
 -> validation
 -> feature calculation
 -> model inference
 -> SHAP
 -> evidence retrieval
 -> graph analysis
 -> agent orchestration
 -> verification
 -> report
 -> human review
 -> audit record
```

## 7. Database Model

Core tables:

``` text
accounts
transactions
devices
beneficiaries
merchants
alerts
investigations
evidence
model_predictions
shap_explanations
agent_runs
tool_calls
audit_events
```

Important relationships:

``` text
account 1--N transactions
account N--N devices
account N--N beneficiaries
account N--N accounts
investigation 1--N evidence
investigation 1--N agent_runs
model_prediction 1--1 shap_explanation
```

## 8. Kafka Topics

``` text
transactions.raw
transactions.validated
transactions.scored
investigations.created
investigations.events
agent.tool_calls
agent.results
deadletter.transactions
```

Messages must include: - event_id - event_type - schema_version -
timestamp - correlation_id

## 9. Agent State

LangGraph state should contain only what is required:

``` text
case_id
transaction_id
risk_score
model_version
shap_features
evidence_ids
graph_findings
agent_actions
verification_status
final_report
```

Do not place entire database tables or huge documents into agent state.

## 10. MCP Architecture

MCP is used as the controlled tool interface between agents and
application capabilities. MCP servers expose tools/resources to AI
hosts. citeturn0search5

The investigation MCP server should initially expose read-only tools.

Example:

``` text
get_transaction(transaction_id)
get_account_history(account_id, limit)
get_device_history(device_id)
find_related_accounts(account_id, depth)
get_shap_explanation(transaction_id)
get_model_prediction(transaction_id)
search_evidence(case_id, query)
```

Any future write tool must require explicit authorization.

## 11. Failure Handling

If Kafka fails: - retry producer - use bounded local buffering - send
unrecoverable messages to DLQ

If model service fails: - mark prediction unavailable - never fabricate
a risk score

If SHAP fails: - report explanation unavailable - do not ask the LLM to
invent one

If database fails: - fail closed for evidence-dependent investigation

If LLM fails: - produce a deterministic investigation summary - mark
`llm_status=unavailable`

If an agent tool fails: - record the failure - retry according to
policy - continue only when the missing evidence is non-critical

## 12. Deployment

### Free development

``` text
Docker Compose
Kafka
PostgreSQL
Redis
FastAPI
React
ML service
Agent service
MCP server
```

### Local Kubernetes

``` text
kind/minikube
  |
  +-- api
  +-- ml-service
  +-- agent-service
  +-- kafka
  +-- postgres
  +-- redis
```

### Optional AWS target

``` text
EKS
MSK
RDS
S3
CloudWatch
IAM
Bedrock
```

AWS deployment is optional and must not be required for the core project
to function.

## 13. No-GPU Constraint

No GPU is required.

Training: - XGBoost/LightGBM on CPU. - SHAP TreeExplainer on CPU.

LLM: - remote API via Groq/Gemini.

Heavy experiments: - Kaggle notebooks if necessary.

Never introduce GPU-only dependencies unless explicitly approved.
