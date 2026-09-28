# Local Demo Script

This walkthrough uses synthetic records only. It demonstrates decision support, not automated case disposition.

1. Install Python 3.11+, Node.js 22+, and the repository extras with `python -m pip install -e ".[dev,agents]"`.
2. Start the API with `python -m uvicorn financial_crime.api.main:app --reload`.
3. In `apps/dashboard`, run `npm ci` and `npm run dev`; open `http://localhost:5173`.
4. Select a case from Review queue and run the investigation. Inspect the retrieved OBSERVED records, MODEL_OUTPUT, SHAP factors, counterfactual simulation, network signals, agent trace, and verification status separately.
5. Confirm the report remains `requires-review`; this demo has no approval or rejection action.
6. Run `python evaluation/run_all.py --output docs/benchmark --rows 500 --seed 42`. Open `docs/benchmark/benchmark_report.html` and review the JSON alongside it.

For streaming and Grafana, Docker Compose is required. Run `python scripts/train.py --rows 500` followed by `docker compose up --build`; publish synthetic records with `python scripts/produce_transactions.py --rows 25`. Monitoring is enabled with `docker compose --profile monitoring up --build`. Docker/Kafka integration has not been exercised on hosts without Docker.

IEEE-CIS ingestion requires the user's own Kaggle authorization and dataset acceptance. Groq or Gemini are optional and each requires its own key plus explicit evidence-export opt-in. No public dataset or API credential is bundled.
