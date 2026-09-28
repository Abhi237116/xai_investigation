import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from financial_crime.data.schema import FEATURES

SEED = 42


def make_models(positive_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            n_jobs=4,
            random_state=SEED,
        )
    except ImportError:
        pass
    return models


def train_benchmarks(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    labeled = frame.dropna(subset=["is_suspicious"]).copy()
    if labeled["is_suspicious"].nunique() != 2 or len(labeled) < 100:
        raise ValueError("at least 100 labeled rows with both classes are required")
    forbidden = {
        "is_suspicious",
        "transaction_id",
        "timestamp",
        "account_id",
        "beneficiary_id",
        "device_id",
        "merchant_id",
    }
    if set(FEATURES) & forbidden or not set(FEATURES) <= set(labeled.columns):
        raise ValueError("feature allowlist contains a target/identifier or missing feature")
    features = labeled[FEATURES].astype(float)
    labels = labeled["is_suspicious"].astype(int)
    chronological = "timestamp" in labeled.columns
    if chronological:
        order = pd.to_datetime(labeled["timestamp"], utc=True).sort_values().index
        train_end, validation_end = int(len(order) * 0.6), int(len(order) * 0.8)
        train_idx, validation_idx, test_idx = (
            order[:train_end],
            order[train_end:validation_end],
            order[validation_end:],
        )
        chronological = all(
            labels.loc[index].nunique() == 2 for index in (train_idx, validation_idx, test_idx)
        )
    if not chronological:
        train_idx, remaining_idx = train_test_split(
            features.index, test_size=0.4, stratify=labels, random_state=SEED
        )
        validation_idx, test_idx = train_test_split(
            remaining_idx, test_size=0.5, stratify=labels.loc[remaining_idx], random_state=SEED
        )
    x_train, y_train = features.loc[train_idx], labels.loc[train_idx]
    x_validation, y_validation = features.loc[validation_idx], labels.loc[validation_idx]
    x_test, y_test = features.loc[test_idx], labels.loc[test_idx]
    ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, Any] = {
        "protocol": {
            "train_rows": len(y_train),
            "validation_rows": len(y_validation),
            "test_rows": len(y_test),
            "seed": SEED,
            "split": ("chronological 60/20/20" if chronological else "stratified 60/20/20")
            + "; validation is reporting-only",
            "feature_allowlist": FEATURES,
            "target_excluded": "is_suspicious" not in FEATURES,
            "train_validation_test_disjoint": not (
                set(train_idx) & set(validation_idx)
                or set(train_idx) & set(test_idx)
                or set(validation_idx) & set(test_idx)
            ),
        },
        "models": {},
    }
    for name, model in make_models(ratio).items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "threshold": 0.5,
            "test_rows": len(y_test),
        }
        table["models"][name] = metrics
        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "version": "0.1.0",
                "name": name,
                "feature_medians": x_train.median().to_dict(),
                "feature_minimums": x_train.min().to_dict(),
                "feature_maximums": x_train.max().to_dict(),
                "training_rows": len(x_train),
                "validation_rows": len(x_validation),
                "training_transaction_ids": labeled.loc[x_train.index, "transaction_id"].tolist(),
                "validation_transaction_ids": labeled.loc[
                    x_validation.index, "transaction_id"
                ].tolist(),
                "test_rows": len(x_test),
                "test_transaction_ids": labeled.loc[x_test.index, "transaction_id"].tolist(),
            },
            out / f"{name}.joblib",
        )
    (out / "benchmark_report.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table
