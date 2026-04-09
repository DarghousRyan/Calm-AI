"""
FastAPI entrypoint for Calm AI.

Minimal structure:
- `app/db/session.py` configures engine + SessionLocal + `get_db`.
- `app/db/init_db.py` initializes tables (MVP-only).
- `app/models/*` define SQLAlchemy tables.
- `app/schemas/*` define Pydantic request/response models.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.db.init_db import init_db
from app.db.session import engine

@asynccontextmanager
async def lifespan(_: FastAPI):
    # For a minimal MVP, create tables on startup.
    init_db(engine)
    yield


app = FastAPI(title="Calm AI API", lifespan=lifespan)
app.include_router(api_router)

