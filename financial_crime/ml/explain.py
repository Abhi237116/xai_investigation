import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from financial_crime.data.schema import FEATURES


class ExplanationUnavailableError(RuntimeError):
    """A model explanation could not be generated from the configured artifact."""


def _one_row_shap(explainer: Any, values: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(explainer.shap_values(values))
    if raw.ndim == 3:
        raw = raw[0, :, -1]
    elif raw.ndim == 2:
        raw = raw[0]
    return raw.astype(float)


def explain_case(
    artifact: dict[str, Any],
    values: dict[str, float],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import shap

        model = artifact["model"]
        name = artifact["name"]
        if name not in {"random_forest", "xgboost"}:
            raise ExplanationUnavailableError("TreeExplainer is available for tree baselines only")
        row = pd.DataFrame(
            [{feature: float(values[feature]) for feature in FEATURES}], columns=FEATURES
        )
        explainer = shap.TreeExplainer(model)
        contributions = _one_row_shap(explainer, row)
        score = float(model.predict_proba(row)[0, 1])
        baseline = np.asarray(explainer.expected_value).reshape(-1)
        base_score = float(baseline[-1] if baseline.size > 1 else baseline[0])
        features = [
            {"feature": key, "value": float(row.iloc[0][key]), "contribution": float(contribution)}
            for key, contribution in zip(FEATURES, contributions, strict=True)
        ]
        features.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        result = {
            "status": "available",
            "model_name": name,
            "model_version": artifact.get("version", "unknown"),
            "explainer_type": "TreeExplainer",
            "risk_score": score,
            "base_value": base_score,
            "feature_values": {key: float(row.iloc[0][key]) for key in FEATURES},
            "features": features,
        }
        if output_dir:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "shap_explanation.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        return result
    except ExplanationUnavailableError:
        raise
    except Exception as error:
        raise ExplanationUnavailableError("explanation_unavailable") from error


def evaluate_explanation_stability(
    artifact: dict[str, Any], values: dict[str, float], seed: int = 19
) -> dict[str, float | str]:
    original = explain_case(artifact, values)
    rng = np.random.default_rng(seed)
    perturbed = {key: value * (1 + float(rng.normal(0, 0.01))) for key, value in values.items()}
    nearby = explain_case(artifact, perturbed)
    base = {item["feature"]: abs(item["contribution"]) for item in original["features"]}
    changed = {item["feature"]: abs(item["contribution"]) for item in nearby["features"]}
    original_values = [base[key] for key in FEATURES]
    changed_values = [changed[key] for key in FEATURES]
    if np.ptp(original_values) == 0 or np.ptp(changed_values) == 0:
        return {"metric": "spearman_rank_correlation", "stability": None, "status": "uninformative"}
    correlation = spearmanr(original_values, changed_values)
    return {"metric": "spearman_rank_correlation", "stability": float(correlation.statistic)}


def evaluate_explanation_faithfulness(
    artifact: dict[str, Any], values: dict[str, float]
) -> dict[str, float | str]:
    result = explain_case(artifact, values)
    model = artifact["model"]
    row = pd.DataFrame([{key: values[key] for key in FEATURES}], columns=FEATURES)
    baseline_score = float(model.predict_proba(row)[0, 1])
    median = pd.DataFrame([artifact.get("feature_medians", values)], columns=FEATURES)
    impacts = []
    attributions = []
    for feature in FEATURES:
        ablated = row.copy()
        ablated[feature] = median.iloc[0][feature]
        impacts.append(abs(baseline_score - float(model.predict_proba(ablated)[0, 1])))
        contribution = next(
            item["contribution"] for item in result["features"] if item["feature"] == feature
        )
        attributions.append(abs(contribution))
    if np.ptp(attributions) == 0 or np.ptp(impacts) == 0:
        return {
            "metric": "spearman_shap_vs_ablation",
            "faithfulness": None,
            "status": "uninformative",
        }
    correlation = spearmanr(attributions, impacts)
    return {"metric": "spearman_shap_vs_ablation", "faithfulness": float(correlation.statistic)}


def global_importance(model: Any, name: str) -> list[dict[str, float | str]]:
    if name == "xgboost":
        values = model.feature_importances_
    elif name == "random_forest":
        values = model.feature_importances_
    else:
        raise ExplanationUnavailableError("global tree importance unavailable")
    return sorted(
        (
            {"feature": key, "importance": float(value)}
            for key, value in zip(FEATURES, values, strict=True)
        ),
        key=lambda item: float(item["importance"]),
        reverse=True,
    )


def global_shap_importance(
    artifact: dict[str, Any], frame: pd.DataFrame, output_path: str | Path | None = None
) -> list[dict[str, float | str]]:
    try:
        import shap

        rows = frame[FEATURES].astype(float)
        values = np.asarray(shap.TreeExplainer(artifact["model"]).shap_values(rows))
        if values.ndim == 3:
            values = values[:, :, -1]
        scores = np.abs(values).mean(axis=0)
        result = sorted(
            (
                {"feature": feature, "mean_abs_shap": float(score)}
                for feature, score in zip(FEATURES, scores, strict=True)
            ),
            key=lambda item: float(item["mean_abs_shap"]),
            reverse=True,
        )
        if output_path:
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as error:
        raise ExplanationUnavailableError("explanation_unavailable") from error
