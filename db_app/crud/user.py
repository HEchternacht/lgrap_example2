"""CRUD helpers for the users table."""
from __future__ import annotations

from sqlalchemy.orm import Session

from db_app.models.user import User


def get_user(db: Session, user_id: str) -> User | None:
    """Return the User record for *user_id*, or None if it does not exist."""
    return db.get(User, user_id)


def get_or_create_user(db: Session, user_id: str) -> tuple[User, bool]:
    """
    Look up *user_id*; insert a new row if it does not exist.

    Returns ``(user, created)`` where ``created`` is True on first login.
    """
    user = db.get(User, user_id)
    if user is not None:
        return user, False

    user = User(user_id=user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, True
