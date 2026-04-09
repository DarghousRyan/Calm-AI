"""
Single-input inference for Calm AI risk prediction.

Usage:
    from app.ml.inference import predict_risk

    result = predict_risk({
        "mood": "down",
        "stress": 6.2,
        "craving": 7.1,
        "sleep_hours": 5.4,
        "exercise_minutes": 20,
        "social_interaction": 25,
        "trigger_boredom": 1,
        "trigger_loneliness": 0,
        "trigger_conflict": 1,
        "days_since_last_relapse": 2,
    })
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.features import ALL_FEATURES

_Label = Literal["low", "medium", "high"]


def _project_root() -> Path:
    # app/ml/inference.py -> app/ml/ -> app/ -> project root
    return Path(__file__).resolve().parents[2]


def _load_payload() -> dict[str, Any]:
    artifacts_dir = _project_root() / "app" / "ml" / "artifacts"
    artifact_path = artifacts_dir / "risk_model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. "
            "Run `python3 scripts/train_model.py` first."
        )
    return joblib.load(artifact_path)


@lru_cache(maxsize=1)
def _load_payload_cached() -> dict[str, Any]:
    # Cache artifact loading so repeated predictions don't re-hit disk.
    return _load_payload()


def predict_risk(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict low/medium/high risk from a single input dict.

    Notes:
    - `input_data` must contain at least `mood` plus the numeric features used
      during training.
    - Extra keys are ignored.
    """

    payload = _load_payload_cached()
    model: Pipeline = payload["model"]
    label_names: list[str] = payload["label_names"]
    categorical_features: list[str] = payload["categorical_features"]
    numeric_features: list[str] = payload["numeric_features"]

    expected_cols = categorical_features + numeric_features
    # Guard against drift if artifact metadata doesn't match code.
    if sorted(expected_cols) != sorted(list(ALL_FEATURES)):
        raise RuntimeError(
            "Feature mismatch between loaded artifact and current code. "
            f"artifact={expected_cols} code={list(ALL_FEATURES)}"
        )

    missing = [c for c in expected_cols if c not in input_data]
    if missing:
        raise KeyError(
            "Missing required features for prediction: "
            + ", ".join(missing)
        )

    row = {k: input_data[k] for k in expected_cols}
    X = pd.DataFrame([row])

    proba = model.predict_proba(X)[0]  # shape: (3,)
    pred_idx = int(np.argmax(proba))
    pred_label = str(label_names[pred_idx])

    # Cast to a narrow literal where possible, but keep runtime flexible.
    typed_label: _Label | str = pred_label  # type: ignore[assignment]

    return {
        "risk_class": typed_label,
        "risk_probabilities": {
            str(label_names[i]): float(proba[i]) for i in range(len(label_names))
        },
    }

