import json

import pandas as pd

from financial_crime.data.ieee_cis import (
    ID_COLUMN,
    TARGET_COLUMN,
    load_ieee_cis_dataset,
    write_ieee_cis_report,
)


def _write_ieee_fixture(root, *, duplicate_identity: bool = False, test_has_target: bool = False):
    train_transaction = pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4],
            TARGET_COLUMN: [0, 1, 0, 0],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [12.5, 200.0, 7.0, 55.0],
            "card1": [1001, 1002, 1003, 1004],
        }
    )
    train_identity = pd.DataFrame(
        {
            ID_COLUMN: [1, 3, 3 if duplicate_identity else 5],
            "id_01": [0.0, -5.0, -7.0],
            "DeviceType": ["desktop", "mobile", "mobile"],
        }
    )
    test_transaction = pd.DataFrame(
        {
            ID_COLUMN: [10, 11],
            "TransactionDT": [50, 60],
            "TransactionAmt": [13.0, 99.0],
        }
    )
    if test_has_target:
        test_transaction[TARGET_COLUMN] = [0, 1]
    test_identity = pd.DataFrame(
        {
            ID_COLUMN: [10],
            "id_01": [-1.0],
            "DeviceType": ["desktop"],
        }
    )
    train_transaction.to_csv(root / "train_transaction.csv", index=False)
    train_identity.to_csv(root / "train_identity.csv", index=False)
    test_transaction.to_csv(root / "test_transaction.csv", index=False)
    test_identity.to_csv(root / "test_identity.csv", index=False)


def test_ieee_cis_loader_preserves_rows_and_separates_target(tmp_path) -> None:
    _write_ieee_fixture(tmp_path)
    dataset = load_ieee_cis_dataset(tmp_path, compare_expected_audit=False)

    assert len(dataset.train_joined) == 4
    assert len(dataset.test_joined) == 2
    assert dataset.train_target.tolist() == [0, 1, 0, 0]
    assert ID_COLUMN not in dataset.train_features.columns
    assert TARGET_COLUMN not in dataset.train_features.columns
    assert TARGET_COLUMN not in dataset.test_features.columns
    assert dataset.report["join"]["train_missing_identity_records"] == 2
    assert dataset.report["join"]["identity_join_coverage_pct"] == 50.0
    assert dataset.report["target"]["class_distribution"] == {0: 3, 1: 1}
    assert dataset.report["validation_failures"] == []


def test_ieee_cis_validation_reports_join_and_test_label_failures(tmp_path) -> None:
    _write_ieee_fixture(tmp_path, duplicate_identity=True, test_has_target=True)
    dataset = load_ieee_cis_dataset(tmp_path, compare_expected_audit=False)

    failures = dataset.report["validation_failures"]
    assert "training identity has duplicate TransactionID values" in failures
    assert "training identity join changed transaction row count" in failures
    assert "test_transaction must remain unlabeled and must not include isFraud" in failures


def test_ieee_cis_report_writer_includes_expected_audit_discrepancies(tmp_path) -> None:
    _write_ieee_fixture(tmp_path)
    report_path = tmp_path / "reports" / "ieee_report.json"

    report = write_ieee_cis_report(tmp_path, report_path)

    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["dataset"] == "ieee-cis-fraud-detection"
    assert "train_transaction_rows" in saved["expected_audit_discrepancies"]
    assert report["features"]["unlabeled_test_data_kept_separate"] is True
