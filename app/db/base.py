"""
SQLAlchemy declarative base.

Why this file exists:
- `Base` is imported by all models.
- This file also imports model modules so SQLAlchemy "registers" tables
  when we call `Base.metadata.create_all(...)`.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models here so their tables are attached to Base.metadata.
# Keep imports at the bottom to avoid circular import issues.
from app.models.daily_log import DailyLog  # noqa: E402,F401
from app.models.prediction import Prediction  # noqa: E402,F401

