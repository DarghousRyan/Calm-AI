"""
Train a multiclass risk classifier for Calm AI.

This script:
1. Loads `data/synthetic/daily_logs.csv`
2. Builds a semi-derived multiclass target (low/medium/high) from the
   synthetic signals (risk_label + wellness indicators).
3. Trains a simple, readable sklearn pipeline:
   - One-hot encode `mood`
   - Standardize numeric features
   - Train a multinomial logistic regression classifier
4. Prints accuracy, macro-F1, and a confusion matrix summary
5. Saves the trained pipeline to `app/ml/artifacts/risk_model.joblib`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/train_model.py` from the repo root (imports `app.*`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _project_root() -> Path:
    # scripts/train_model.py -> scripts/ -> project root
    return Path(__file__).resolve().parents[1]


def _derive_risk_classes(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Convert the synthetic binary `risk_label` into a multiclass target:
    low / medium / high risk.
    """

    # Normalize/shape signals into a scalar "risk_score".
    # (This keeps the training target learnable and correlated with inputs.)
    sleep_hours = df["sleep_hours"].astype(float)
    stress = df["stress"].astype(float)
    craving = df["craving"].astype(float)
    exercise_minutes = df["exercise_minutes"].astype(float)
    social_interaction = df["social_interaction"].astype(float)
    days_since_last_relapse = df["days_since_last_relapse"].astype(float)

    # Lower sleep => higher risk. Map sleep roughly from [3.5, 10.5] -> [1, 0].
    sleep_factor = 1.0 - ((sleep_hours - 3.5) / (10.5 - 3.5)).clip(0.0, 1.0)
    social_factor = 1.0 - (social_interaction / 180.0).clip(0.0, 1.0)
    exercise_factor = 1.0 - (exercise_minutes / 140.0).clip(0.0, 1.0)

    # Recency: fewer days since relapse => higher immediate pressure.
    relapse_pressure = 1.0 / (1.0 + days_since_last_relapse / 3.0)

    triggers = (
        0.9 * df["trigger_loneliness"].astype(float)
        + 1.0 * df["trigger_conflict"].astype(float)
        + 0.6 * df["trigger_boredom"].astype(float)
    )

    # Binary label adds structure to "high" risk and helps separate classes.
    # (Synthetic data generation already correlates these features.)
    risk_label = df["risk_label"].astype(float)

    risk_score = (
        0.30 * stress
        + 0.25 * craving
        + 2.00 * sleep_factor
        + 1.20 * triggers
        + 1.00 * relapse_pressure
        + 0.40 * social_factor
        + 0.40 * exercise_factor
        + 1.20 * risk_label
    )

    # Tercile binning -> multiclass target.
    q1 = float(np.quantile(risk_score, 0.33))
    q2 = float(np.quantile(risk_score, 0.66))

    y = np.zeros(len(df), dtype=int)
    y[risk_score > q1] = 1
    y[risk_score > q2] = 2

    # Ensure labels cover all classes (in case of degenerate quantiles).
    # If the dataset is extremely skewed, this may collapse into fewer classes.
    # In that case, the metrics will still run, but may be less informative.
    label_names = ["low", "medium", "high"]
    return y, label_names


def main() -> None:
    project_root = _project_root()
    csv_path = project_root / "data" / "synthetic" / "daily_logs.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing synthetic dataset: {csv_path}. "
            "Run `python3 scripts/generate_synthetic_data.py --users 500 --days 60` first."
        )

    df = pd.read_csv(csv_path)

    # Feature columns used for inference and training.
    categorical_features = list(CATEGORICAL_FEATURES)
    numeric_features = list(NUMERIC_FEATURES)

    # Multiclass target: low / medium / high.
    y, label_names = _derive_risk_classes(df)

    # Basic split for evaluation.
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        # StandardScaler keeps coefficients stable for logistic regression.
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    # Use a simple, broadly-compatible configuration so it works across
    # a wide range of scikit-learn versions.
    clf = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
    )

    model: Pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    model.fit(df_train, y_train)

    y_pred = model.predict(df_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # Print a compact "summary" that is still easy to read.
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=label_names, columns=label_names))

    # Save trained model + metadata for inference.
    artifacts_dir = project_root / "app" / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifacts_dir / "risk_model.joblib"
    payload: dict[str, Any] = {
        "model": model,
        "label_names": label_names,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }
    joblib.dump(payload, artifact_path)

    print(f"Saved model to: {artifact_path}")


if __name__ == "__main__":
    main()

