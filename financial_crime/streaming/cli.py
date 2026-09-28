import argparse

from financial_crime.config import settings
from financial_crime.streaming.worker import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume, validate and score Kafka transactions")
    parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
    parser.add_argument("--model-path", default=settings.model_artifact_path)
    args = parser.parse_args()
    run_worker(args.bootstrap_servers, args.model_path)


if __name__ == "__main__":
    main()
