import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financial_crime.data.ieee_cis import ID_COLUMN, TARGET_COLUMN
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


def _build_ieee_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = list(features.select_dtypes(include=["object", "string", "category"]).columns)
    numeric = [column for column in features.columns if column not in categorical]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        raise ValueError("no IEEE-CIS training features are available for preprocessing")

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def _choose_ieee_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    validation_fraction: float = 0.2,
    min_validation_rows: int = 200,
    seed: int = SEED,
) -> tuple[list[int], list[int], str]:
    if ID_COLUMN in features.columns:
        features = features.drop(columns=[ID_COLUMN])

    if "TransactionDT" in features.columns:
        transaction_dt = pd.to_numeric(features["TransactionDT"], errors="coerce")
        valid_rows = transaction_dt.notna()
        if valid_rows.sum() >= 2 * min_validation_rows and target.loc[valid_rows].nunique() == 2:
            ordered_index = transaction_dt.loc[valid_rows].sort_values().index.to_list()
            val_size = max(min_validation_rows, int(len(ordered_index) * validation_fraction))
            if len(ordered_index) - val_size < min_validation_rows:
                val_size = max(min_validation_rows, len(ordered_index) // 5)
            train_idx = ordered_index[:-val_size]
            validation_idx = ordered_index[-val_size:]
            if target.loc[train_idx].nunique() == 2 and target.loc[validation_idx].nunique() == 2:
                return train_idx, validation_idx, "chronological TransactionDT"

    idx = np.arange(len(features))
    train_idx, validation_idx = train_test_split(
        idx,
        test_size=validation_fraction,
        stratify=target,
        random_state=seed,
    )
    return train_idx.tolist(), validation_idx.tolist(), "stratified fallback"


def _ieee_model_specs(positive_weight: float, *, seed: int = SEED) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dummy_classifier": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=seed,
        )
    except ImportError:
        pass
    return models


def _model_thresholds(y_true: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    default_threshold = 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {
            "default_threshold": default_threshold,
            "selected_threshold": default_threshold,
            "selected_method": "default threshold; no recall/precision curve available",
        }

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_idx = int(np.nanargmax(f1_scores))
    selected_threshold = float(thresholds[best_idx])
    return {
        "default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "selected_method": "maximize validation F1 on the precision-recall curve",
    }


def _save_validation_summary(
    output_dir: Path,
    *,
    model_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    default_threshold: float,
    selected_threshold: float,
    transaction_ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    default_predictions = (scores >= default_threshold).astype(int)
    selected_predictions = (scores >= selected_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, default_predictions, labels=[0, 1]).ravel()
    selected_confusion = confusion_matrix(y_true, selected_predictions, labels=[0, 1]).ravel()

    summary = {
        "model": model_name,
        "default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "default_precision": float(precision_score(y_true, default_predictions, zero_division=0)),
        "default_recall": float(recall_score(y_true, default_predictions, zero_division=0)),
        "default_f1": float(f1_score(y_true, default_predictions, zero_division=0)),
        "selected_precision": float(
            precision_score(y_true, selected_predictions, zero_division=0)
        ),
        "selected_recall": float(recall_score(y_true, selected_predictions, zero_division=0)),
        "selected_f1": float(f1_score(y_true, selected_predictions, zero_division=0)),
        "default_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "selected_confusion_matrix": {
            "tn": int(selected_confusion[0]),
            "fp": int(selected_confusion[1]),
            "fn": int(selected_confusion[2]),
            "tp": int(selected_confusion[3]),
        },
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }

    predictions = pd.DataFrame(
        {
            "TransactionID": transaction_ids.to_numpy(),
            "actual": y_true.to_numpy(),
            "score": scores,
            "default_prediction": default_predictions,
            "selected_prediction": selected_predictions,
            "default_threshold": default_threshold,
            "selected_threshold": selected_threshold,
        }
    )
    predictions_path = output_dir / f"{model_name}_validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_curve = pd.DataFrame(
        {
            "threshold": np.concatenate([[np.nan], thresholds]),
            "precision": precision,
            "recall": recall,
        }
    )
    pr_curve_path = output_dir / f"{model_name}_precision_recall_curve.csv"
    pr_curve.to_csv(pr_curve_path, index=False)

    confusion_path = output_dir / f"{model_name}_confusion_matrix.csv"
    pd.DataFrame(
        {
            "label": ["tn", "fp", "fn", "tp"],
            "count": [
                int(confusion_matrix(y_true, default_predictions, labels=[0, 1]).ravel()[0]),
                int(confusion_matrix(y_true, default_predictions, labels=[0, 1]).ravel()[1]),
                int(confusion_matrix(y_true, default_predictions, labels=[0, 1]).ravel()[2]),
                int(confusion_matrix(y_true, default_predictions, labels=[0, 1]).ravel()[3]),
            ],
        }
    ).to_csv(confusion_path, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.plot(recall[:-1], precision[:-1], color="tab:blue", lw=2)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model_name} precision-recall curve")
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_name}_precision_recall_curve.png")
        plt.close(fig)
    except Exception:
        pass

    return predictions, summary


def train_ieee_cis_baselines(
    dataset: Any,
    output_dir: str | Path,
    *,
    seed: int = SEED,
    validation_fraction: float = 0.2,
    min_validation_rows: int = 200,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    feature_frame = dataset.train_features.copy()
    target = dataset.train_target.astype(int).copy()
    transaction_ids = dataset.train_transaction_ids.copy()

    if ID_COLUMN in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=[ID_COLUMN])
    if target.isna().any():
        raise ValueError("IEEE-CIS training labels must not contain missing values")

    train_idx, validation_idx, split_strategy = _choose_ieee_split(
        feature_frame,
        target,
        validation_fraction=validation_fraction,
        min_validation_rows=min_validation_rows,
        seed=seed,
    )

    x_train = feature_frame.iloc[train_idx].copy()
    x_validation = feature_frame.iloc[validation_idx].copy()
    y_train = target.iloc[train_idx].copy()
    y_validation = target.iloc[validation_idx].copy()
    validation_ids = transaction_ids.iloc[validation_idx].copy()

    if y_train.nunique() != 2 or y_validation.nunique() != 2:
        raise ValueError("training and validation splits must contain both classes")

    feature_columns = list(feature_frame.columns)
    positive_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    model_specs = _ieee_model_specs(positive_weight, seed=seed)

    models_report: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "dataset": "ieee-cis-fraud-detection",
        "seed": seed,
        "split_strategy": split_strategy,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "feature_count": int(len(feature_columns)),
        "feature_order": feature_columns,
        "positive_class": 1,
        "negative_class": 0,
        "class_distribution": {
            "train": {"negative": int((y_train == 0).sum()), "positive": int((y_train == 1).sum())},
            "validation": {
                "negative": int((y_validation == 0).sum()),
                "positive": int((y_validation == 1).sum()),
            },
        },
        "transaction_dt_range": None,
    }
    if "TransactionDT" in feature_frame.columns:
        train_dt = pd.to_numeric(
            feature_frame.loc[x_train.index, "TransactionDT"], errors="coerce"
        )
        val_dt = pd.to_numeric(
            feature_frame.loc[x_validation.index, "TransactionDT"], errors="coerce"
        )
        metadata["transaction_dt_range"] = {
            "train_min": int(train_dt.min()) if train_dt.notna().any() else None,
            "train_max": int(train_dt.max()) if train_dt.notna().any() else None,
            "validation_min": int(val_dt.min()) if val_dt.notna().any() else None,
            "validation_max": int(val_dt.max()) if val_dt.notna().any() else None,
        }

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_ieee_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)

        validation_scores = pipeline.predict_proba(x_validation)[:, 1]
        thresholds = _model_thresholds(y_validation, validation_scores)
        _, summary = _save_validation_summary(
            root,
            model_name=model_name,
            y_true=y_validation,
            scores=validation_scores,
            default_threshold=thresholds["default_threshold"],
            selected_threshold=thresholds["selected_threshold"],
            transaction_ids=validation_ids,
        )

        artifact = {
            "pipeline": pipeline,
            "model_name": model_name,
            "feature_order": feature_columns,
            "split_strategy": split_strategy,
            "seed": seed,
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "target_column": TARGET_COLUMN,
            "id_column": ID_COLUMN,
            "thresholds": thresholds,
            "metrics": summary,
        }
        joblib.dump(artifact, root / f"{model_name}_baseline.joblib")

        models_report[model_name] = {
            "thresholds": thresholds,
            "metrics": {
                "average_precision": summary["average_precision"],
                "roc_auc": summary["roc_auc"],
                "precision": summary["default_precision"],
                "recall": summary["default_recall"],
                "f1": summary["default_f1"],
                "brier_score": summary["brier_score"],
                "selected_precision": summary["selected_precision"],
                "selected_recall": summary["selected_recall"],
                "selected_f1": summary["selected_f1"],
                "default_threshold": thresholds["default_threshold"],
                "selected_threshold": thresholds["selected_threshold"],
            },
            "predictions_path": str(root / f"{model_name}_validation_predictions.csv"),
            "artifact_path": str(root / f"{model_name}_baseline.joblib"),
        }

    metrics_summary = {
        "dataset": metadata,
        "models": models_report,
    }
    (root / "phase3_training_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )
    return metrics_summary
