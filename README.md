# Financial Crime Investigation AI

Evidence-grounded financial-crime investigation decision support for synthetic and authorized public datasets. This platform produces risk indicators for human review; it does not determine guilt or take consequential actions.

## Local development

Requirements: Python 3.11+ and Node.js 22+. Docker Compose is optional for local API and dashboard work.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,agents]"
Copy-Item .env.example .env
python scripts/train.py --rows 500 --seed 42
python -m uvicorn financial_crime.api.main:app --reload
```

In a second terminal:

```powershell
cd apps/dashboard
npm ci
npm run dev
```

The API is at `http://localhost:8000`; its OpenAPI page is `/docs`. The dashboard runs at `http://localhost:5173`.

Open a second terminal in `apps/dashboard` for the UI commands above. The seeded local store initializes with deterministic synthetic data. The dashboard supports the overview, review queue, transaction records, case evidence, XAI, agent trace, benchmarks, and service health. Every case remains in `requires-review`; no approve/reject control is provided.

Install the `agents` extra (included in the command above) to run the LangGraph workflow, MCP tool server, and optional Groq/Gemini clients. An MCP host can launch the read-only stdio server with `fci-mcp-server`.

## Data

The built-in deterministic synthetic generator needs no credentials or external data. Generate its canonical CSV and SHA-256 manifest with:

```powershell
python -c "from financial_crime.data.ingest import write_dataset; print(write_dataset('data/demo'))"
```

CSV import supports the canonical schema plus basic IEEE-CIS transaction aliases. IEEE-CIS access is controlled by its Kaggle competition rules and is not bundled. Never place private/customer data in this repository. See [the data dictionary](docs/data_dictionary.md).

Authorized IEEE-CIS validation is separate from the synthetic demo schema and does not train models:

```powershell
python scripts/validate_ieee_cis.py --data-dir <ieee-cis-csv-directory> --output artifacts/ieee_cis/validation_report.json
```

In Kaggle, use `/kaggle/input/competitions/ieee-fraud-detection` as the data directory and write the report to `/kaggle/working`. The validator preserves original IEEE-CIS features for later training, keeps `isFraud` only as the target, and keeps unlabeled competition test data separate. See [the IEEE-CIS Kaggle workflow](docs/ieee_cis_kaggle_workflow.md).

## IEEE-CIS model training baseline (Phase 3)

Run the baselines only in Kaggle or another environment that has access to the competition files:

```bash
python scripts/train_ieee_cis.py \
  --data-dir /kaggle/input/competitions/ieee-fraud-detection \
  --output-dir /kaggle/working/phase3_artifacts
```

The workflow:

1. Loads and validates the real IEEE-CIS transaction and identity files.
2. Uses the validated loader output so `TransactionID` and `isFraud` are excluded from model inputs.
3. Splits on `TransactionDT` chronologically when both classes are present in the train and validation windows; otherwise it falls back to a stratified split with a documented warning.
4. Fits a preprocessing pipeline only on the training split, then evaluates DummyClassifier, logistic regression, and XGBoost (if available).
5. Saves the pipeline artifact, validation predictions, metrics JSON/CSV, and training metadata under `/kaggle/working/phase3_artifacts`.

This phase does not integrate the model into production inference or modify the demo application. It only trains and benchmarks the baseline candidates.

## Configuration

Copy `.env.example` to `.env`. Remote LLM providers are disabled by default; set a provider and its own key only when needed. The application remains useful without a provider and returns deterministic evidence summaries. Never commit `.env` or credentials.

## Local streaming demo

Train the CPU baselines first, then start the local Compose stack and publish synthetic events:

```powershell
python scripts/train.py --rows 4000
docker compose up --build
```

In another terminal, publish to the broker exposed at `localhost:19092`:

```powershell
python scripts/produce_transactions.py --rows 50
```

The worker writes validated and scored events, sends malformed events to the dead-letter topic, and logs p50/p95 processing latency, throughput, and measured consumer lag in batches. If the model artifact is missing or scoring fails, it emits `model_status=unavailable` and never fabricates a score. The local Postgres Compose service uses trust authentication on the private Compose network and is for development only.

For Prometheus and Grafana, start `docker compose --profile monitoring up --build`. Prometheus scrapes the API's private `/metrics` endpoint; Grafana is at `http://localhost:3000` (`admin` / `local-demo-only` unless `GRAFANA_ADMIN_PASSWORD` is set). OpenTelemetry tracing is opt-in with the `observability` extra and `OTEL_OTLP_ENDPOINT` pointing to an OTLP/HTTP receiver.

## Evaluation and deployment

Rebuild the reproducible held-out evaluation and full-system reports:

```powershell
python evaluation/run_all.py --output docs/benchmark --rows 500 --seed 42
```

The current reports are [agent_evaluation.json](docs/benchmark/agent_evaluation.json), [benchmark_report.json](docs/benchmark/benchmark_report.json), and [benchmark_report.html](docs/benchmark/benchmark_report.html). All model metrics are from an intentionally simple synthetic generator whose labels are encoded in its feature distributions; perfect scores are not a claim about real-world fraud detection. The XAI sample faithfulness result can be uninformative. Kafka is marked unavailable when no broker is run, and LLM token usage is zero with the default disabled provider.

Kubernetes examples are in `k8s/`; AWS target architecture and an opt-in-only encrypted artifact bucket are in `infra/aws/` and `docs/deployment.md`. Docker, a local Kubernetes cluster (`kind`/Minikube), cloud credentials, and authorized public datasets are not bundled. Terraform defaults to creating no resources.

See [the demo walkthrough](docs/demo_script.md), [security review](docs/security_review.md), [architecture diagram](docs/architecture.mmd), and [data dictionary](docs/data_dictionary.md).

## Project docs

Implementation requirements and phase tracking live in `MD_project_design/`. Benchmark output will identify the dataset provenance and model version; synthetic metrics must not be presented as public benchmark results.
