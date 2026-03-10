"""
users table.

Each row represents one user identified by a unique code in the format
``AAAA0000`` — 4 uppercase letters followed by 4 digits (e.g. ``KSKS0771``).

The user_id itself is the primary key; no surrogate integer/UUID needed.
"""
from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db_app.database import Base

# Compiled once — reused by both the ORM layer and the validation helpers.
USER_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Z]{4}\d{4}$")


def is_valid_user_id(value: str) -> bool:
    """Return True when *value* matches the AAAA0000 format."""
    return bool(USER_ID_PATTERN.match(value))


class User(Base):
    __tablename__ = "users"

    # "KSKS0771" — 8 chars, globally unique, human-readable
    user_id: Mapped[str] = mapped_column(String(9), primary_key=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
