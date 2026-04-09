"""
DailyLog Pydantic schemas.

These are used for API I/O (not DB persistence).
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class DailyLogBase(BaseModel):
    log_date: date
    mood: str | None = None
    notes: str | None = None


class DailyLogCreate(DailyLogBase):
    pass


class DailyLogRead(DailyLogBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

