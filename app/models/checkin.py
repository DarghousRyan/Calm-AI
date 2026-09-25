"""
Check-in log model.

Stores full user check-in payloads so the UI can render past history.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CheckInLog(Base):
    __tablename__ = "checkin_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Nullable keeps existing MVP databases startable; all new check-ins are owned.
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True)
    log_date: Mapped[date] = mapped_column(Date, index=True)
    mood: Mapped[str] = mapped_column(String(64))
    stress: Mapped[float] = mapped_column(Float)
    craving: Mapped[float] = mapped_column(Float)
    sleep_hours: Mapped[float] = mapped_column(Float)
    exercise_minutes: Mapped[int] = mapped_column(Integer)
    social_interaction: Mapped[int] = mapped_column(Integer)
    trigger_boredom: Mapped[int] = mapped_column(Integer)
    trigger_loneliness: Mapped[int] = mapped_column(Integer)
    trigger_conflict: Mapped[int] = mapped_column(Integer)
    custom_trigger: Mapped[str | None] = mapped_column(String(500), nullable=True)
    days_since_last_relapse: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
