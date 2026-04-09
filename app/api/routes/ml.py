"""
ML-related API routes.

These routes wrap the inference code for use by the frontend.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.inference import predict_risk


router = APIRouter(prefix="/ml", tags=["ml"])


class PredictRequest(BaseModel):
    # Keep this aligned with `scripts/train_model.py` features.
    mood: str = Field(..., description="One of: great/good/okay/down/bad (or any string).")
    stress: float = Field(..., ge=0.0, le=10.0)
    craving: float = Field(..., ge=0.0, le=10.0)
    sleep_hours: float = Field(..., ge=0.0, le=24.0)
    exercise_minutes: int = Field(..., ge=0, le=1440)
    social_interaction: int = Field(..., ge=0, le=1440)
    trigger_boredom: int = Field(0, ge=0, le=1)
    trigger_loneliness: int = Field(0, ge=0, le=1)
    trigger_conflict: int = Field(0, ge=0, le=1)
    days_since_last_relapse: int = Field(..., ge=0, le=36500)

    # Optional, ignored by the model but useful for upstream storage.
    log_date: str | None = None
    user_id: int | None = None


class PredictResponse(BaseModel):
    risk_class: Literal["low", "medium", "high"] | str
    risk_probabilities: dict[str, float]


@router.post("/predict", response_model=PredictResponse)
def ml_predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return predict_risk(req.model_dump())
    except FileNotFoundError as e:
        # Model not trained yet.
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

