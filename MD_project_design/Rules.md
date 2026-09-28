# Financial Crime Investigation AI --- AI and Engineering Rules

## 1. Core Safety Rules

1.  The system is decision support, not an autonomous law-enforcement or
    banking decision maker.
2.  Never declare a person, account, merchant or transaction "guilty."
3.  Use terms such as `suspicious`, `high-risk`, `requires-review`, or
    `evidence-indicates`.
4.  Never fabricate evidence, transactions, SHAP values, model outputs,
    citations, or tool results.
5.  Never use real customer PII in development.
6.  Use AMLSim, IEEE-CIS and other approved synthetic/public data.
7.  Every consequential action requires human approval.

## 2. Source-of-Truth Hierarchy

When sources disagree, use this order:

1.  Database/tool result
2.  ML model output
3.  SHAP output
4.  Deterministic graph/rule output
5.  Retrieved documents
6.  LLM interpretation

The LLM must never override deterministic evidence.

## 3. LLM Rules

The LLM may: - plan investigations - choose among approved tools -
summarize evidence - explain model outputs in plain language - identify
missing evidence - propose follow-up investigations

The LLM may not: - calculate the official fraud probability - invent
missing evidence - alter SHAP values - claim a tool succeeded if it
failed - fabricate citations - approve or reject a financial-crime
case - execute irreversible actions - access arbitrary
filesystem/database commands - bypass authorization

## 4. Evidence Rules

Every claim in an investigation report must be categorized as:

``` text
OBSERVED
INFERRED
MODEL_OUTPUT
AGENT_HYPOTHESIS
UNVERIFIED
```

Example:

``` text
OBSERVED:
Beneficiary B442 was added four minutes before transaction TX123.

MODEL_OUTPUT:
XGBoost risk score = 0.94.

XAI:
transaction_amount had SHAP contribution +0.41.

AGENT_HYPOTHESIS:
The combination may warrant investigation for unusual beneficiary activity.
```

The agent must not turn `AGENT_HYPOTHESIS` into `OBSERVED`.

## 5. SHAP Rules

1.  SHAP values must come directly from the trained model and explainer.
2.  Never ask the LLM to generate SHAP values.
3.  Store model version with every explanation.
4.  Store feature values used for the explanation.
5.  Store explainer type.
6.  Test explanation faithfulness.
7.  Test explanation stability.
8.  If SHAP fails, report `explanation_unavailable`.

## 6. Counterfactual Rules

Counterfactuals are analytical simulations, not instructions to
manipulate real financial behavior.

They must: - identify changed features - identify unchanged features -
state whether the model prediction changed - report feasibility
constraints - never claim that changing a feature would actually prevent
fraud

## 7. Agent Rules

Agents must: - have explicit roles - have minimal tool permissions -
validate tool inputs - record tool calls - record failures - avoid
duplicate calls where possible - stop when sufficient evidence is
collected

Agents must not: - access arbitrary SQL - execute arbitrary shell
commands - modify production infrastructure - call undocumented tools -
bypass human approval

## 8. MCP Rules

Default tools are read-only.

Tool schemas must define: - name - description - input schema - output
schema - permission level - timeout - error behavior

Tool output must be structured JSON.

Never return raw secrets.

## 9. Error Handling

Use explicit errors:

``` text
ValidationError
ModelUnavailableError
EvidenceUnavailableError
ToolExecutionError
LLMRateLimitError
LLMProviderError
DatabaseError
```

No bare:

``` python
except:
    pass
```

Never silently swallow an exception.

## 10. LLM Rate Limits

Implement: - exponential backoff - maximum retries - request timeout -
token budget - provider fallback - request logging without sensitive
payloads

Do not hammer the free LLM API.

Groq documents RPM/RPD/TPM/TPD limits and returns 429 responses when
rate limits are exceeded, so rate limiting must be implemented in the
application. citeturn0search0

## 11. Cost Rules

Default infrastructure must be free/local: - Docker - PostgreSQL -
Redis - Kafka - NetworkX - open-source ML libraries - Kaggle for
optional compute

Cloud resources must be disabled by default.

No paid API should be required for the project to run.

## 12. Model Rules

Required baseline sequence:

``` text
Logistic Regression
Random Forest
XGBoost
```

Optional: - LightGBM

Do not claim the best model before evaluation.

Use: - stratified or time-aware validation as appropriate - leakage
checks - class-imbalance handling - reproducible random seeds

Never use test data for tuning.

## 13. Evaluation Rules

Never report accuracy alone.

Required classification metrics: - PR-AUC - ROC-AUC - precision -
recall - F1 - calibration/Brier score

Required XAI metrics: - faithfulness - stability - counterfactual
validity

Required agent metrics: - evidence recall - citation correctness -
tool-call accuracy - unsupported-claim rate - task completion rate

Required system metrics: - p50 latency - p95 latency - throughput -
Kafka consumer lag

## 14. Coding Rules

Prefer: - small functions - typed Python - Pydantic schemas - dependency
injection - testable modules - explicit interfaces - configuration via
environment variables

Avoid: - giant files - global mutable state - hardcoded secrets -
duplicated logic - magic numbers - unnecessary frameworks

## 15. Dependency Rules

Before adding a dependency: 1. Check whether the standard library or
existing dependency solves it. 2. Prefer mature open-source libraries.
3. Avoid duplicate frameworks. 4. Pin compatible versions. 5. Update
requirements only when needed.

## 16. Security Rules

Never commit: - API keys - passwords - tokens - database credentials -
private datasets

Use `.env` locally and `.env.example` in Git.

Add `.gitignore` entries for:

``` text
.env
*.pem
*.key
credentials*
data/private/
```

## 17. UI Rules

The dashboard must clearly distinguish: - model prediction - SHAP
explanation - retrieved evidence - agent hypothesis - human decision

Do not use UI language that implies certainty.

## 18. Coding-Agent Rules

When implementing a phase: 1. Read PRD.md. 2. Read Architecture.md. 3.
Read Rules.md. 4. Read Phases.md. 5. Read Memory.md if it exists. 6.
Inspect existing code before changing it. 7. Implement only the current
phase. 8. Run tests. 9. Update Memory.md. 10. Do not rewrite unrelated
code. 11. Do not introduce new architecture without documenting it.
