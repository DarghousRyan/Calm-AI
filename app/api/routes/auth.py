"""Account registration and login."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, get_or_create_local_user, supabase_auth_request
from app.db.session import get_db
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])
class Credentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=128)


class AuthResponse(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    user_id: int
    email: str
    message: str | None = None


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(req: Credentials, db: Session = Depends(get_db)) -> AuthResponse:
    email = req.email.strip().lower()
    payload = supabase_auth_request("signup", {"email": email, "password": req.password})
    supabase_user = payload.get("user") or {}
    access_token = payload.get("access_token")
    if not access_token:
        return AuthResponse(
            email=email,
            user_id=0,
            message="Account created. Check your email to confirm your account, then log in.",
        )
    user = get_or_create_local_user(db, supabase_user)
    return AuthResponse(
        access_token=access_token,
        refresh_token=payload.get("refresh_token"),
        user_id=user.id,
        email=user.email,
    )


@router.post("/login", response_model=AuthResponse)
def login(req: Credentials, db: Session = Depends(get_db)) -> AuthResponse:
    email = req.email.strip().lower()
    payload = supabase_auth_request("token?grant_type=password", {"email": email, "password": req.password})
    access_token = str(payload.get("access_token", ""))
    if not access_token:
        raise HTTPException(status_code=401, detail="Supabase did not return a login token")
    user = get_or_create_local_user(db, payload.get("user") or {})
    return AuthResponse(
        access_token=access_token,
        refresh_token=payload.get("refresh_token"),
        user_id=user.id,
        email=user.email,
    )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


@router.post("/refresh", response_model=AuthResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)) -> AuthResponse:
    payload = supabase_auth_request(
        "token?grant_type=refresh_token",
        {"refresh_token": req.refresh_token},
    )
    access_token = str(payload.get("access_token", ""))
    if not access_token:
        raise HTTPException(status_code=401, detail="Supabase did not return a refreshed login token")
    user = get_or_create_local_user(db, payload.get("user") or {})
    return AuthResponse(
        access_token=access_token,
        refresh_token=payload.get("refresh_token", req.refresh_token),
        user_id=user.id,
        email=user.email,
    )


@router.get("/me", response_model=AuthResponse)
def me(user: User = Depends(get_current_user)) -> AuthResponse:
    return AuthResponse(user_id=user.id, email=user.email)
