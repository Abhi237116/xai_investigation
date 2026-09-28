import argparse
import json

from financial_crime.config import settings
from financial_crime.data.ingest import load_dataset
from financial_crime.data.schema import Transaction
from financial_crime.streaming.events import StreamOutput, TransactionEvent
from financial_crime.streaming.worker import TOPICS, KafkaPublisher

parser = argparse.ArgumentParser(description="Publish synthetic transactions to Kafka")
parser.add_argument("--bootstrap-servers", default=settings.kafka_bootstrap_servers)
parser.add_argument("--rows", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
frame, manifest = load_dataset(rows=max(args.rows, 100), seed=args.seed)
publisher = KafkaPublisher(args.bootstrap_servers)
for record in frame.head(args.rows).to_dict(orient="records"):
    transaction = Transaction.model_validate(record)
    event = TransactionEvent(payload=transaction.model_copy(update={"is_suspicious": None}))
    publisher.publish(
        output=StreamOutput(
            topic=TOPICS["raw"],
            key=transaction.transaction_id,
            payload=json.loads(event.model_dump_json()),
        )
    )
print(json.dumps({"published": args.rows, "dataset": manifest}, indent=2))
