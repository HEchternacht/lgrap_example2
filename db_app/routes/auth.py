"""
Authentication routes.

POST /auth/login   — validate user ID format, ensure user row exists, set session
POST /auth/logout  — clear the session
GET  /auth/me      — return the currently authenticated user (or 401)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from db_app.crud.user import get_or_create_user
from db_app.database import get_db
from db_app.models.user import USER_ID_PATTERN

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class LoginPayload(BaseModel):
    user: str

    @field_validator("user")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not USER_ID_PATTERN.match(v):
            raise ValueError(
                "User ID must be 4 uppercase letters followed by 4 digits "
                "(e.g. KSKS0771)."
            )
        return v


class AuthResponse(BaseModel):
    status: str
    user: str | None = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Log in with a user ID and start a session",
)
def login(
    payload: LoginPayload,
    request: Request,
    db: Session = Depends(get_db),
) -> AuthResponse:
    """
    Accepts a user ID in ``AAAA0000`` format.  Creates the user record on first
    login, then stores the ID in the signed session cookie.
    """
    user, created = get_or_create_user(db, payload.user)
    request.session["user"] = user.user_id
    return AuthResponse(
        status="created" if created else "ok",
        user=user.user_id,
    )


@router.post(
    "/logout",
    response_model=AuthResponse,
    summary="End the current session",
)
def logout(request: Request) -> AuthResponse:
    """Clear the session cookie."""
    request.session.clear()
    return AuthResponse(status="logged_out")


@router.get(
    "/me",
    response_model=AuthResponse,
    summary="Return the currently authenticated user",
    responses={401: {"description": "Not authenticated"}},
)
def me(request: Request) -> AuthResponse:
    """Return the user stored in the current session, or 401 if not logged in."""
    user_id: str | None = request.session.get("user")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    return AuthResponse(status="ok", user=user_id)
