# Data Dictionary

The demo dataset is synthetic and reproducible. `is_suspicious` is a synthetic label for model exercises, not evidence of actual misconduct. The CSV manifest records the canonical data SHA-256 and schema version.

| Field | Type | Meaning |
|---|---|---|
| transaction_id | string | Stable synthetic or source transaction identifier |
| account_id | string | Pseudonymous source account key |
| beneficiary_id | string | Pseudonymous destination entity key |
| device_id | string | Pseudonymous device key |
| merchant_id | string | Pseudonymous merchant key |
| timestamp | ISO datetime | Event time |
| amount | float | Transaction amount in arbitrary synthetic units |
| account_age_days | integer | Account age at event |
| beneficiary_age_days | integer | Beneficiary relationship age |
| device_accounts_30d | integer | Distinct accounts on device in prior 30 days |
| transactions_1h | integer | Account transaction count in prior hour |
| distance_from_usual_km | float | Distance from synthetic usual location |
| is_suspicious | integer or null | Synthetic/public source label; omitted for unlabeled records |
| source | string | Dataset provenance label |

AMLSim CSVs can be mapped into the canonical demo schema. Unavailable fields receive explicit neutral defaults and source columns remain available for audit; the dataset manifest never describes transformed benchmark data as untouched source data.

IEEE-CIS is kept in a separate training-data representation. The ingestion validator preserves the original transaction and identity feature columns, left-joins identity rows on `TransactionID`, keeps `isFraud` only as the target, and excludes both `TransactionID` and `isFraud` from model input features. The unlabeled competition test files remain separate from supervised training and validation. See [the Kaggle validation workflow](ieee_cis_kaggle_workflow.md).
