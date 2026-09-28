from typing import Any

import pandas as pd

from financial_crime.data.schema import FEATURES


def generate_counterfactual(
    artifact: dict[str, Any], values: dict[str, float], threshold: float = 0.8
) -> dict[str, Any]:
    """Simulate bounded single-feature substitutions; never prescribe real behavior."""
    model = artifact["model"]
    original = {feature: float(values[feature]) for feature in FEATURES}
    current_frame = pd.DataFrame([original], columns=FEATURES)
    baseline_score = float(model.predict_proba(current_frame)[0, 1])
    baseline_label = baseline_score >= threshold
    candidates = []
    minimums = artifact.get("feature_minimums", {})
    maximums = artifact.get("feature_maximums", {})
    medians = artifact.get("feature_medians", {})
    for feature in FEATURES:
        candidate_value = float(medians.get(feature, original[feature]))
        if (
            not float(minimums.get(feature, candidate_value))
            <= candidate_value
            <= float(maximums.get(feature, candidate_value))
        ):
            continue
        candidate = dict(original)
        candidate[feature] = candidate_value
        score = float(model.predict_proba(pd.DataFrame([candidate], columns=FEATURES))[0, 1])
        changed = (score >= threshold) != baseline_label
        if changed:
            candidates.append((abs(candidate_value - original[feature]), feature, candidate, score))
    if candidates:
        _, changed_feature, selected, simulated_score = min(candidates, key=lambda item: item[0])
    else:
        changed_feature = None
        selected = original
        simulated_score = baseline_score
    return {
        "simulation_only": True,
        "model_name": artifact.get("name"),
        "model_version": artifact.get("version"),
        "threshold": threshold,
        "original_model_score": baseline_score,
        "simulated_model_score": simulated_score,
        "prediction_changed": (simulated_score >= threshold) != baseline_label,
        "changed_features": ([changed_feature] if changed_feature else []),
        "unchanged_features": [feature for feature in FEATURES if feature != changed_feature],
        "feature_values": selected,
        "feasibility_constraints": [
            "One feature at a time is substituted with its training median.",
            "The substituted value is bounded by the model training range.",
            "This is an analytical model simulation, not a prediction of real-world behavior.",
        ],
    }
