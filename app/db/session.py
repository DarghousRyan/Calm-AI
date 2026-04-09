"""
SQLAlchemy session setup.

This module is the single place where we configure:
- SQLite engine
- Session factory (`SessionLocal`)
- FastAPI dependency (`get_db`) that yields a DB session per request
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Local SQLite database file in the project root.
SQLALCHEMY_DATABASE_URL = "sqlite:///./calm_ai.db"

# `check_same_thread=False` is required when using SQLite with FastAPI (multiple threads).
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Typical "unit of work" session factory. We keep it minimal and explicit.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy Session.

    Usage:
        def route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

