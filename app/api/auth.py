"""Supabase Auth integration and local user ownership lookup."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User

_bearer = HTTPBearer(auto_error=False)
load_dotenv()


def _supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "").strip().rstrip("/")


def _publishable_key() -> str:
    return os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()


def supabase_auth_request(path: str, payload: dict[str, str]) -> dict:
    """Call Supabase Auth without exposing a secret key to the client."""
    url = _supabase_url()
    key = _publishable_key()
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")
    try:
        response = requests.post(
            f"{url}/auth/v1/{path.lstrip('/')}",
            headers={"apikey": key, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Could not reach Supabase Auth") from exc
    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("msg") or body.get("message") or body.get("error_description") or "Authentication failed"
        except ValueError:
            detail = "Authentication failed"
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


def supabase_user_from_token(token: str) -> dict:
    url = _supabase_url()
    key = _publishable_key()
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")
    try:
        response = requests.get(
            f"{url}/auth/v1/user",
            headers={"apikey": key, "Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Could not reach Supabase Auth") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired login")
    return response.json()


def get_or_create_local_user(db: Session, supabase_user: dict) -> User:
    external_id = str(supabase_user.get("id", "")).strip()
    email = str(supabase_user.get("email", "")).strip().lower()
    if not external_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Supabase user is incomplete")

    user = db.scalar(select(User).where(User.supabase_user_id == external_id))
    if user is None:
        user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, password_hash="", supabase_user_id=external_id)
        db.add(user)
    else:
        user.email = email
        user.supabase_user_id = external_id
    db.commit()
    db.refresh(user)
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
    return get_or_create_local_user(db, supabase_user_from_token(credentials.credentials))
