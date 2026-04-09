"""
Recommendation API routes.

These routes are intentionally simple:
- Take a latest log + risk level
- Return supportive, non-clinical recommendations
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.recommendation_service import DISCLAIMER_TEXT, generate_recommendations


router = APIRouter(tags=["recommendations"])


class RecommendationsRequest(BaseModel):
    latest_log: dict[str, Any] = Field(..., description="Latest daily log fields.")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Model predicted risk level.")


class RecommendationItem(BaseModel):
    title: str
    suggestion: str
    explanation: str
    disclaimer: str


class RecommendationsResponse(BaseModel):
    disclaimer: str
    recommendations: list[RecommendationItem]


@router.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(req: RecommendationsRequest) -> dict[str, Any]:
    try:
        recs = generate_recommendations(req.latest_log, risk_level=req.risk_level)
        return {"disclaimer": DISCLAIMER_TEXT, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {e}") from e

