# Deployment Targets

## Local Kubernetes

The manifests in `k8s/` demonstrate the API/dashboard deployments, probes, bounded CPU/memory requests, a ConfigMap, a blank Secret template, and a modest API HPA. They are templates, not a verified full production stack. Build images with `docker build -t fci-api:local .` and `docker build -t fci-dashboard:local apps/dashboard`; load them into kind/Minikube, then apply the namespace, config, API, dashboard, and HPA manifests. Add a durable PostgreSQL service, Kafka/Redpanda, model artifact volume, ingress/TLS, and secret manager before using a cluster beyond a disposable demo. SQLite and missing-model behavior are development fallbacks.

## AWS target (disabled)

The target production mapping is EKS for the API/worker, MSK for event topics, RDS PostgreSQL, S3 for versioned artifacts/data manifests, least-privilege IAM roles, CloudWatch logs/alarms, and optional Bedrock as a separately governed provider. Keep private networking, encryption at rest/in transit, VPC endpoints, scoped service accounts, retention controls, and human approval gates. No cluster, database, broker, or model endpoint is provisioned by this repository.

`infra/aws` configures a default AWS provider and an optional encrypted, public-access-blocked artifact bucket controlled by `enable_demo_bucket=false`. Review cost, account, region, naming, and retention before changing it. Run Terraform plan only with a configured AWS profile; apply is not part of the demo. EKS/MSK/RDS/Bedrock remain documented target services, not active Terraform resources.
