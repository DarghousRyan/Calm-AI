"""
Prediction model.

Represents an AI-generated prediction linked to a DailyLog.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    daily_log_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("daily_logs.id", ondelete="CASCADE"), index=True
    )

    # Example: a numeric calm score and an optional explanation string.
    predicted_calm_score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str | None] = mapped_column(String(4000), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    daily_log = relationship("DailyLog", back_populates="predictions")

