# IEEE-CIS Kaggle Validation Workflow

This phase validates the authorized IEEE-CIS Fraud Detection files without committing raw data,
derived tables, credentials, or large artifacts.

## Kaggle notebook steps

1. Attach the IEEE-CIS Fraud Detection competition dataset to the notebook.
2. Confirm these files exist under the Kaggle input directory:

   ```text
   /kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv
   /kaggle/input/competitions/ieee-fraud-detection/train_identity.csv
   /kaggle/input/competitions/ieee-fraud-detection/test_transaction.csv
   /kaggle/input/competitions/ieee-fraud-detection/test_identity.csv
   /kaggle/input/competitions/ieee-fraud-detection/sample_submission.csv
   ```

3. Run validation from the repository checkout:

   ```bash
   python scripts/validate_ieee_cis.py \
     --data-dir /kaggle/input/competitions/ieee-fraud-detection \
     --output /kaggle/working/ieee_cis_validation_report.json
   ```

4. Review `/kaggle/working/ieee_cis_validation_report.json`.

5. Download or attach only the compact JSON report if needed. Do not commit the raw CSV files.

## What the validator checks

- Loads transaction and identity files separately.
- Left-joins identity records on `TransactionID`, preserving all transaction rows.
- Reports duplicate `TransactionID` values, missing identity records, row-count changes, missing
  labels, class distribution, missing values, columns and identity join coverage.
- Keeps `isFraud` strictly as the supervised target.
- Excludes `TransactionID` and `isFraud` from the model input feature frame.
- Keeps unlabeled competition test data separate from training and validation.

## Phase 3 baseline training workflow

After validation passes, train the real-data baselines with:

```bash
python scripts/train_ieee_cis.py \
  --data-dir /kaggle/input/competitions/ieee-fraud-detection \
  --output-dir /kaggle/working/phase3_artifacts
```

The training workflow implements the Phase 3 requirements:

- Keeps the labeled train split separate from the unlabeled competition test data.
- Excludes `TransactionID` and `isFraud` from the feature frame.
- Uses `TransactionDT` to prefer a chronological validation split when both classes are present in both windows.
- Falls back to a stratified split only when chronological validation is not suitable.
- Fits preprocessing only on the training split.
- Uses a categorical/numeric preprocessing pipeline with missing-value handling and one-hot encoding.
- Compares a dummy baseline, logistic regression, and XGBoost (when available).
- Saves model artifacts, validation predictions, confusion summaries, and metadata under
  `/kaggle/working/phase3_artifacts`.

The intended artifact directory structure is:

```text
/kaggle/working/phase3_artifacts/
  dummy_classifier_baseline.joblib
  logistic_regression_baseline.joblib
  xgboost_baseline.joblib
  dummy_classifier_validation_predictions.csv
  logistic_regression_validation_predictions.csv
  xgboost_validation_predictions.csv
  phase3_training_summary.json
```

The metrics and artifact files are intended for download only after a successful Kaggle run. Do not
claim the real-data baseline is complete until the artifacts are produced and verified in Kaggle.

## Expected full-dataset audit

The validator calculates these values from the files and reports discrepancies from the currently
known IEEE-CIS audit:

- Training transactions: 590,540 rows and 394 columns.
- Training identity: 144,233 rows and 41 columns.
- Duplicate training `TransactionID` values: 0.
- Missing `isFraud` labels: 0.
- Fraud distribution: 569,877 legitimate and 20,663 fraudulent transactions.
- Identity join coverage: 24.42%.

Discrepancies are reported under `expected_audit_discrepancies`; they are not silently assumed away.
