"""
Prediction Pydantic schemas.

These are used for API I/O (not DB persistence).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionBase(BaseModel):
    daily_log_id: int
    predicted_calm_score: float
    explanation: str | None = None


class PredictionCreate(PredictionBase):
    pass


class PredictionRead(PredictionBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

