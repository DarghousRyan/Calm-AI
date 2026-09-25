"""
Database initialization utilities.

For this MVP we create tables on startup. In a production setup you'd typically
use migrations (Alembic) instead of `create_all`.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.base import Base


def init_db(engine: Engine) -> None:
    # Import models here so their tables are registered before create_all.
    import app.models.checkin  # noqa: F401
    import app.models.chat  # noqa: F401
    import app.models.daily_log  # noqa: F401
    import app.models.prediction  # noqa: F401
    import app.models.user  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # The MVP database may already contain check-ins created before accounts
    # existed. Keep those legacy rows inaccessible to new accounts while
    # allowing the upgraded service to start without a manual migration.
    if "checkin_logs" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("checkin_logs")}
        if "user_id" not in columns and engine.dialect.name == "sqlite":
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE checkin_logs ADD COLUMN user_id INTEGER"))
    if "users" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("users")}
        if "supabase_user_id" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE users ADD COLUMN supabase_user_id VARCHAR(128)"))
    if "checkin_logs" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("checkin_logs")}
        if "custom_trigger" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE checkin_logs ADD COLUMN custom_trigger VARCHAR(500)"))
