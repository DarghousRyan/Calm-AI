"""
Check-in API routes.

Allows the frontend to save and retrieve past check-ins.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.checkin import CheckInLog

router = APIRouter(tags=["checkins"])


class CheckInCreateRequest(BaseModel):
    log_date: date
    mood: str = Field(..., min_length=1, max_length=64)
    stress: float = Field(..., ge=0.0, le=10.0)
    craving: float = Field(..., ge=0.0, le=10.0)
    sleep_hours: float = Field(..., ge=0.0, le=24.0)
    exercise_minutes: int = Field(..., ge=0, le=1440)
    social_interaction: int = Field(..., ge=0, le=1440)
    trigger_boredom: int = Field(0, ge=0, le=1)
    trigger_loneliness: int = Field(0, ge=0, le=1)
    trigger_conflict: int = Field(0, ge=0, le=1)
    days_since_last_relapse: int = Field(..., ge=0, le=36500)


class CheckInRead(CheckInCreateRequest):
    id: int
    created_at: datetime


class CheckInListResponse(BaseModel):
    checkins: list[CheckInRead]


@router.post("/checkins", response_model=CheckInRead)
def create_checkin(req: CheckInCreateRequest, db: Session = Depends(get_db)) -> CheckInRead:
    try:
        row = CheckInLog(**req.model_dump())
        db.add(row)
        db.commit()
        db.refresh(row)
        return CheckInRead(**row.__dict__)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save check-in: {e}") from e


@router.get("/checkins", response_model=CheckInListResponse)
def list_checkins(db: Session = Depends(get_db), limit: int = 100) -> CheckInListResponse:
    safe_limit = max(1, min(limit, 500))
    rows = (
        db.query(CheckInLog)
        .order_by(CheckInLog.log_date.desc(), CheckInLog.created_at.desc())
        .limit(safe_limit)
        .all()
    )
    return CheckInListResponse(checkins=[CheckInRead(**row.__dict__) for row in rows])
