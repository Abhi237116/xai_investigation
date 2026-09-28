# Security and Limitations Review

- All bundled example records are synthetic. API transaction responses omit the synthetic training label.
- MCP tools are typed, bounded, audited by input/output hashes, and read-only. The LLM receives only a bounded evidence packet after `LLM_ALLOW_EVIDENCE_EXPORT=true` is set.
- Reports label observations, model outputs, hypotheses, and missing evidence separately. Every hypothesis must cite known evidence. No automatic approve/reject or external enforcement action exists.
- The default SQLite and Compose PostgreSQL trust authentication are development-only. Do not expose these services to an untrusted network or use real customer data.
- API CORS is configurable. Before production, add identity-aware authentication/authorization, TLS, secret management, rate limiting, data retention controls, durable audit storage, and reviewed database migrations.
- Prometheus metrics are unauthenticated and intended only for a private local network. Do not expose metrics endpoints publicly.
- Synthetic ML results are intentionally not a proxy for real fraud performance. The generator's feature distributions encode its labels, so perfect baseline metrics indicate a toy separable benchmark, not production quality.
- Counterfactual values are bounded model simulations and are not behavioral recommendations or causal claims.
- Groq/Gemini provider calls, Docker broker operation, Kubernetes execution, Terraform cloud deployment, and authorized public-data ingestion require external credentials/infrastructure and are not claimed as locally verified.
