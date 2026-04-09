"""
DailyLog model.

Represents a user's log entry for a date (simple MVP entity).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DailyLog(Base):
    __tablename__ = "daily_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    log_date: Mapped[date] = mapped_column(Date, unique=True, index=True)

    # Minimal example signals; adjust as the product evolves.
    mood: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # One log can have many predictions over time.
    predictions = relationship(
        "Prediction",
        back_populates="daily_log",
        cascade="all, delete-orphan",
    )

