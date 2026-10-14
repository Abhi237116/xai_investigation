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
