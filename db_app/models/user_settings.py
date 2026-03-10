"""
user_settings table.

One row per user — stores language, tone, and any arbitrary extra preferences.
Upserted on each request that carries `extra_body` prompt params.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db_app.database import Base


class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Unique per user — AAAA0000 format (e.g. KSKS0771)
    user_id: Mapped[str] = mapped_column(
        String(9), nullable=False, unique=True, index=True
    )

    language: Mapped[str] = mapped_column(
        String(50), nullable=False, default="English"
    )
    tone: Mapped[str] = mapped_column(
        String(100), nullable=False, default="helpful and concise"
    )

    # Catch-all for future settings: {"get_weather.temperature": "fahrenheit", …}
    extra_preferences: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
