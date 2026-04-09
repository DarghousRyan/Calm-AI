"""
Shared ML feature definitions.

Keeping feature names in one place prevents drift between:
- training (`scripts/train_model.py`)
- inference (`app/ml/inference.py`)
- API request validation (`app/api/routes/ml.py`)
"""

from __future__ import annotations

from typing import Final


CATEGORICAL_FEATURES: Final[list[str]] = ["mood"]

NUMERIC_FEATURES: Final[list[str]] = [
    "stress",
    "craving",
    "sleep_hours",
    "exercise_minutes",
    "social_interaction",
    "trigger_boredom",
    "trigger_loneliness",
    "trigger_conflict",
    "days_since_last_relapse",
]

ALL_FEATURES: Final[list[str]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

