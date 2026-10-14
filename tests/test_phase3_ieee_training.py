import json

import joblib
import pandas as pd

from financial_crime.data.ieee_cis import ID_COLUMN, TARGET_COLUMN, load_ieee_cis_dataset
from financial_crime.ml.train import train_ieee_cis_baselines


def _write_phase3_fixture(root) -> None:
    train_transaction = pd.DataFrame(
        {
            ID_COLUMN: list(range(1, 21)),
            TARGET_COLUMN: [0, 1] * 10,
            "TransactionDT": list(range(1, 21)),
            "TransactionAmt": [10.0, 120.0, 9.5, 101.0, 15.0, 130.0, 12.0, 115.0, 11.5, 119.0,
                               8.0, 125.0, 14.0, 135.0, 17.0, 140.0, 10.5, 127.0, 13.0, 132.0],
            "ProductCD": ["W", "C", "W", "C", "W", "C", "W", "C", "W", "C",
                           "W", "C", "W", "C", "W", "C", "W", "C", "W", "C"],
            "DeviceType": ["desktop", "mobile", "desktop", "mobile", "desktop", "mobile",
                           "desktop", "mobile", "desktop", "mobile", "desktop", "mobile",
                           "desktop", "mobile", "desktop", "mobile", "desktop", "mobile",
                           "desktop", "mobile"],
        }
    )
    train_identity = pd.DataFrame(
        {
            ID_COLUMN: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            "id_01": [0.1, -1.2, 0.2, -2.0, 0.3, -1.5, 0.4, -2.1, 0.5, -1.7],
            "DeviceInfo": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    test_transaction = pd.DataFrame(
        {
            ID_COLUMN: [101, 102],
            "TransactionDT": [21, 22],
            "TransactionAmt": [40.0, 90.0],
            "ProductCD": ["W", "C"],
            "DeviceType": ["desktop", "mobile"],
        }
    )
    test_identity = pd.DataFrame(
        {
            ID_COLUMN: [101],
            "id_01": [0.7],
            "DeviceInfo": ["A"],
        }
    )
    train_transaction.to_csv(root / "train_transaction.csv", index=False)
    train_identity.to_csv(root / "train_identity.csv", index=False)
    test_transaction.to_csv(root / "test_transaction.csv", index=False)
    test_identity.to_csv(root / "test_identity.csv", index=False)


def test_ieee_phase3_feature_and_target_are_separated(tmp_path) -> None:
    _write_phase3_fixture(tmp_path)
    dataset = load_ieee_cis_dataset(tmp_path, compare_expected_audit=False)

    assert ID_COLUMN not in dataset.train_features.columns
    assert TARGET_COLUMN not in dataset.train_features.columns
    assert len(dataset.train_features.columns) > 0
    assert len(dataset.train_target) == len(dataset.train_features)
    assert dataset.train_target.nunique() == 2


def test_ieee_phase3_training_saves_pipeline_and_metrics(tmp_path) -> None:
    _write_phase3_fixture(tmp_path)
    dataset = load_ieee_cis_dataset(tmp_path, compare_expected_audit=False)
    output_dir = tmp_path / "phase3_artifacts"

    summary = train_ieee_cis_baselines(dataset, output_dir)
    assert "dummy_classifier" in summary["models"]
    assert "logistic_regression" in summary["models"]
    assert summary["dataset"]["feature_count"] > 0
    assert summary["dataset"]["train_rows"] > 0
    assert summary["dataset"]["validation_rows"] > 0
    assert (output_dir / "phase3_training_summary.json").exists()

    artifact = joblib.load(output_dir / "logistic_regression_baseline.joblib")
    assert "pipeline" in artifact
    assert artifact["feature_order"]
    assert artifact["thresholds"]["selected_method"].startswith("maximize validation F1")


def test_ieee_phase3_preprocessing_uses_training_only_categories_and_keeps_test_unlabeled(
    tmp_path,
) -> None:
    _write_phase3_fixture(tmp_path)
    dataset = load_ieee_cis_dataset(tmp_path, compare_expected_audit=False)
    output_dir = tmp_path / "phase3_artifacts"

    train_ieee_cis_baselines(dataset, output_dir)
    logistic_artifact = joblib.load(output_dir / "logistic_regression_baseline.joblib")
    pipeline = logistic_artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    assert "ProductCD" in preprocessor.feature_names_in_

    validation_predictions = pd.read_csv(
        output_dir / "logistic_regression_validation_predictions.csv"
    )
    required_columns = {"TransactionID", "actual", "score", "selected_prediction"}
    assert set(validation_predictions.columns) >= required_columns
    assert dataset.report["features"]["unlabeled_test_data_kept_separate"] is True
    reported_strategy = json.loads(
        (output_dir / "phase3_training_summary.json").read_text(encoding="utf-8")
    )["dataset"]["split_strategy"]
    assert reported_strategy
