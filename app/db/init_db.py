"""
Database initialization utilities.

For this MVP we create tables on startup. In a production setup you'd typically
use migrations (Alembic) instead of `create_all`.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from app.db.base import Base


def init_db(engine: Engine) -> None:
    # Import models here so their tables are registered before create_all.
    import app.models.checkin  # noqa: F401
    import app.models.daily_log  # noqa: F401
    import app.models.prediction  # noqa: F401

    Base.metadata.create_all(bind=engine)

