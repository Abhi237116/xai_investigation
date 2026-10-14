import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TRAIN_TRANSACTION = "train_transaction.csv"
TRAIN_IDENTITY = "train_identity.csv"
TEST_TRANSACTION = "test_transaction.csv"
TEST_IDENTITY = "test_identity.csv"
TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
EXCLUDED_MODEL_INPUT_COLUMNS = [ID_COLUMN, TARGET_COLUMN]

EXPECTED_IEEE_CIS_AUDIT = {
    "train_transaction_rows": 590_540,
    "train_transaction_columns": 394,
    "train_identity_rows": 144_233,
    "train_identity_columns": 41,
    "duplicate_train_transaction_ids": 0,
    "missing_train_labels": 0,
    "class_distribution": {0: 569_877, 1: 20_663},
    "identity_join_coverage_pct": 24.42,
}


@dataclass(frozen=True)
class IeeeCisTrainingData:
    """IEEE-CIS data kept in its source feature space for later model training."""

    train_features: pd.DataFrame
    train_target: pd.Series
    train_transaction_ids: pd.Series
    train_joined: pd.DataFrame
    test_features: pd.DataFrame
    test_transaction_ids: pd.Series
    test_joined: pd.DataFrame
    report: dict[str, Any]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required IEEE-CIS file not found: {path}")
    return pd.read_csv(path)


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].duplicated().sum())


def _missing_summary(frame: pd.DataFrame, limit: int = 20) -> dict[str, Any]:
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "columns_with_missing_values": int(len(missing)),
        "total_missing_values": int(missing.sum()),
        "top_columns": {str(column): int(value) for column, value in missing.head(limit).items()},
    }


def _validate_required_columns(
    frame: pd.DataFrame, required: list[str], table_name: str, failures: list[str]
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        failures.append(f"{table_name} missing required columns: {', '.join(missing)}")


def _left_join_identity(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    failures: list[str],
    table_name: str,
) -> pd.DataFrame:
    if ID_COLUMN not in transaction.columns or ID_COLUMN not in identity.columns:
        return transaction.copy()
    duplicate_identity_ids = _duplicate_count(identity, ID_COLUMN)
    if duplicate_identity_ids:
        failures.append(f"{table_name} identity has duplicate TransactionID values")
    joined = transaction.merge(identity, on=ID_COLUMN, how="left", suffixes=("", "_identity"))
    if len(joined) != len(transaction):
        failures.append(f"{table_name} identity join changed transaction row count")
    return joined


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXCLUDED_MODEL_INPUT_COLUMNS]


def _identity_coverage(
    transaction: pd.DataFrame, identity: pd.DataFrame
) -> tuple[int, int, float]:
    if ID_COLUMN not in transaction.columns or ID_COLUMN not in identity.columns:
        return 0, int(len(transaction)), 0.0
    matches = int(transaction[ID_COLUMN].isin(identity[ID_COLUMN]).sum())
    missing = int(len(transaction) - matches)
    coverage = round(matches / max(1, len(transaction)) * 100, 2)
    return matches, missing, coverage


def _expected_audit_discrepancies(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed = {
        "train_transaction_rows": report["tables"]["train_transaction"]["rows"],
        "train_transaction_columns": report["tables"]["train_transaction"]["columns"],
        "train_identity_rows": report["tables"]["train_identity"]["rows"],
        "train_identity_columns": report["tables"]["train_identity"]["columns"],
        "duplicate_train_transaction_ids": report["join"]["duplicate_train_transaction_ids"],
        "missing_train_labels": report["target"]["missing_labels"],
        "class_distribution": {
            int(key): value for key, value in report["target"]["class_distribution"].items()
        },
        "identity_join_coverage_pct": report["join"]["identity_join_coverage_pct"],
    }
    discrepancies: dict[str, dict[str, Any]] = {}
    for key, expected in EXPECTED_IEEE_CIS_AUDIT.items():
        actual = observed[key]
        if key == "identity_join_coverage_pct":
            matches = abs(float(actual) - float(expected)) <= 0.01
        else:
            matches = actual == expected
        if not matches:
            discrepancies[key] = {"expected": expected, "actual": actual}
    return discrepancies


def load_ieee_cis_dataset(
    dataset_dir: str | Path,
    *,
    compare_expected_audit: bool = True,
) -> IeeeCisTrainingData:
    """Load and validate IEEE-CIS files without mapping them into the demo schema."""

    root = Path(dataset_dir)
    train_transaction = _read_csv(root / TRAIN_TRANSACTION)
    train_identity = _read_csv(root / TRAIN_IDENTITY)
    test_transaction = _read_csv(root / TEST_TRANSACTION)
    test_identity = _read_csv(root / TEST_IDENTITY)

    failures: list[str] = []
    _validate_required_columns(
        train_transaction, [ID_COLUMN, TARGET_COLUMN], "train_transaction", failures
    )
    _validate_required_columns(train_identity, [ID_COLUMN], "train_identity", failures)
    _validate_required_columns(test_transaction, [ID_COLUMN], "test_transaction", failures)
    _validate_required_columns(test_identity, [ID_COLUMN], "test_identity", failures)

    duplicate_train_ids = _duplicate_count(train_transaction, ID_COLUMN)
    duplicate_test_ids = _duplicate_count(test_transaction, ID_COLUMN)
    if duplicate_train_ids:
        failures.append("train_transaction has duplicate TransactionID values")
    if duplicate_test_ids:
        failures.append("test_transaction has duplicate TransactionID values")
    if TARGET_COLUMN in test_transaction.columns:
        failures.append("test_transaction must remain unlabeled and must not include isFraud")

    train_joined = _left_join_identity(train_transaction, train_identity, failures, "training")
    test_joined = _left_join_identity(test_transaction, test_identity, failures, "test")

    if TARGET_COLUMN in train_transaction.columns:
        missing_labels = int(train_transaction[TARGET_COLUMN].isna().sum())
        if missing_labels:
            failures.append("train_transaction has missing isFraud labels")
        class_distribution = {
            int(key): int(value)
            for key, value in train_transaction[TARGET_COLUMN]
            .value_counts(dropna=False)
            .sort_index()
            .items()
            if pd.notna(key)
        }
        target = train_transaction[TARGET_COLUMN].copy()
    else:
        missing_labels = len(train_transaction)
        class_distribution = {}
        target = pd.Series(dtype="int64", name=TARGET_COLUMN)

    train_identity_matches, train_identity_missing, train_identity_coverage = _identity_coverage(
        train_transaction, train_identity
    )
    test_identity_matches, test_identity_missing, test_identity_coverage = _identity_coverage(
        test_transaction, test_identity
    )

    train_feature_columns = _feature_columns(train_joined)
    test_feature_columns = _feature_columns(test_joined)
    if TARGET_COLUMN in train_feature_columns or ID_COLUMN in train_feature_columns:
        failures.append("training model input features include TransactionID or isFraud")
    if TARGET_COLUMN in test_feature_columns or ID_COLUMN in test_feature_columns:
        failures.append("test model input features include TransactionID or isFraud")

    report: dict[str, Any] = {
        "dataset": "ieee-cis-fraud-detection",
        "source_directory": str(root),
        "tables": {
            "train_transaction": {
                "rows": int(len(train_transaction)),
                "columns": int(len(train_transaction.columns)),
            },
            "train_identity": {
                "rows": int(len(train_identity)),
                "columns": int(len(train_identity.columns)),
            },
            "test_transaction": {
                "rows": int(len(test_transaction)),
                "columns": int(len(test_transaction.columns)),
            },
            "test_identity": {
                "rows": int(len(test_identity)),
                "columns": int(len(test_identity.columns)),
            },
        },
        "columns": {
            "train_transaction": list(train_transaction.columns),
            "train_identity": list(train_identity.columns),
            "train_joined": list(train_joined.columns),
            "test_transaction": list(test_transaction.columns),
            "test_identity": list(test_identity.columns),
            "test_joined": list(test_joined.columns),
        },
        "join": {
            "method": "left join identity on TransactionID",
            "train_row_count_preserved": len(train_joined) == len(train_transaction),
            "test_row_count_preserved": len(test_joined) == len(test_transaction),
            "duplicate_train_transaction_ids": duplicate_train_ids,
            "duplicate_train_identity_ids": _duplicate_count(train_identity, ID_COLUMN),
            "duplicate_test_transaction_ids": duplicate_test_ids,
            "duplicate_test_identity_ids": _duplicate_count(test_identity, ID_COLUMN),
            "train_identity_matches": train_identity_matches,
            "train_missing_identity_records": train_identity_missing,
            "identity_join_coverage_pct": train_identity_coverage,
            "test_identity_matches": test_identity_matches,
            "test_missing_identity_records": test_identity_missing,
            "test_identity_join_coverage_pct": test_identity_coverage,
        },
        "target": {
            "column": TARGET_COLUMN,
            "missing_labels": missing_labels,
            "class_distribution": class_distribution,
            "excluded_from_model_inputs": TARGET_COLUMN not in train_feature_columns,
        },
        "features": {
            "representation": "original IEEE-CIS joined features",
            "train_input_feature_count": len(train_feature_columns),
            "test_input_feature_count": len(test_feature_columns),
            "excluded_columns": EXCLUDED_MODEL_INPUT_COLUMNS,
            "transaction_id_excluded_from_model_inputs": ID_COLUMN not in train_feature_columns,
            "unlabeled_test_data_kept_separate": TARGET_COLUMN not in test_transaction.columns,
        },
        "missing_values": {
            "train_transaction": _missing_summary(train_transaction),
            "train_identity": _missing_summary(train_identity),
            "train_joined": _missing_summary(train_joined),
            "test_transaction": _missing_summary(test_transaction),
            "test_identity": _missing_summary(test_identity),
            "test_joined": _missing_summary(test_joined),
        },
        "validation_failures": failures,
    }
    if compare_expected_audit:
        report["expected_audit"] = EXPECTED_IEEE_CIS_AUDIT
        report["expected_audit_discrepancies"] = _expected_audit_discrepancies(report)

    return IeeeCisTrainingData(
        train_features=train_joined[train_feature_columns].copy(),
        train_target=target,
        train_transaction_ids=train_transaction[ID_COLUMN].copy()
        if ID_COLUMN in train_transaction
        else pd.Series(dtype="int64"),
        train_joined=train_joined,
        test_features=test_joined[test_feature_columns].copy(),
        test_transaction_ids=test_transaction[ID_COLUMN].copy()
        if ID_COLUMN in test_transaction
        else pd.Series(dtype="int64"),
        test_joined=test_joined,
        report=report,
    )


def write_ieee_cis_report(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    compare_expected_audit: bool = True,
) -> dict[str, Any]:
    dataset = load_ieee_cis_dataset(
        dataset_dir, compare_expected_audit=compare_expected_audit
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dataset.report, indent=2), encoding="utf-8")
    return dataset.report
