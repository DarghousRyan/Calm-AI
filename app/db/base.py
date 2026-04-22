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

