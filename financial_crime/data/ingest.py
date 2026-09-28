import hashlib
import json
from pathlib import Path

import pandas as pd

from financial_crime.data.schema import Transaction
from financial_crime.data.synthetic import generate_transactions

ALIASES = {
    "TransactionID": "transaction_id",
    "TransactionAmt": "amount",
    "isFraud": "is_suspicious",
    "account": "account_id",
    "beneficiary": "beneficiary_id",
    "device": "device_id",
}
DEFAULTS = {
    "account_id": "AC-UNKNOWN",
    "beneficiary_id": "BN-UNKNOWN",
    "device_id": "DV-UNKNOWN",
    "merchant_id": "merchant-unknown",
    "account_age_days": 365,
    "beneficiary_age_days": 365,
    "device_accounts_30d": 1,
    "transactions_1h": 1,
    "distance_from_usual_km": 0.0,
}


def load_dataset(
    path: str | Path | None = None,
    rows: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if path is None:
        frame = generate_transactions(rows, seed)
        source = f"synthetic-seed-{seed}"
    else:
        file_path = Path(path)
        frame = pd.read_csv(file_path).rename(columns=ALIASES)
        source = file_path.name
        for key, value in DEFAULTS.items():
            if key not in frame:
                frame[key] = value
        if "timestamp" not in frame:
            frame["timestamp"] = pd.date_range(
                "2025-01-01", periods=len(frame), freq="min", tz="UTC"
            )
        if "transaction_id" not in frame:
            raise ValueError("dataset needs transaction_id or TransactionID")
        frame["source"] = file_path.stem
    normalized = []
    for record in frame.to_dict(orient="records"):
        if pd.isna(record.get("is_suspicious")):
            record["is_suspicious"] = None
        normalized.append(Transaction.model_validate(record).model_dump(mode="json"))
    result = pd.DataFrame(normalized)
    canonical = result.sort_values("transaction_id").to_json(orient="records", date_format="iso")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    metadata = {"source": source, "sha256": digest, "rows": str(len(result)), "schema_version": "1"}
    return result, metadata


def write_dataset(
    output_dir: str | Path,
    path: str | Path | None = None,
    rows: int = 2000,
    seed: int = 42,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame, metadata = load_dataset(path, rows, seed)
    frame.to_csv(destination / "transactions.csv", index=False)
    manifest = json.dumps(metadata, indent=2)
    (destination / "dataset_manifest.json").write_text(manifest, encoding="utf-8")
    return metadata
