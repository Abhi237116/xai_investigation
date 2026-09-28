FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY financial_crime ./financial_crime
COPY docs/benchmark/benchmark_report.json ./docs/benchmark/benchmark_report.json
RUN pip install --no-cache-dir ".[agents]"
EXPOSE 8000
CMD ["uvicorn", "financial_crime.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
