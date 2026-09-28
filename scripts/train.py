import argparse
import json

from financial_crime.data.ingest import load_dataset
from financial_crime.ml.train import train_benchmarks

parser = argparse.ArgumentParser(description="Train baseline financial risk models")
parser.add_argument("--data", help="Optional authorized CSV dataset")
parser.add_argument("--rows", type=int, default=4000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="artifacts/models")
args = parser.parse_args()
dataset, metadata = load_dataset(args.data, rows=args.rows, seed=args.seed)
report = train_benchmarks(dataset, args.output)
report["dataset"] = metadata
print(json.dumps(report, indent=2))
