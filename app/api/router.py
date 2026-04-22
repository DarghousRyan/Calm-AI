"""
Central API router.

Keeping route inclusion in one place makes `app/main.py` small and makes it
easy to see what the API exposes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.ml import router as ml_router
from app.api.routes.chat import router as chat_router
from app.api.routes.checkins import router as checkins_router
from app.api.routes.recommendations import router as recommendations_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(ml_router)
api_router.include_router(chat_router)
api_router.include_router(checkins_router)
api_router.include_router(recommendations_router)

