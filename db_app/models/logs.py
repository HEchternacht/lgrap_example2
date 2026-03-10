"""
app_logs table.

Structured application log — supplements (not replaces) stdout logging.
Useful for per-user audit trails and chat-scoped error tracking.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from db_app.database import Base

LogLevel = Literal["info", "warning", "error"]


class AppLog(Base):
    __tablename__ = "app_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Nullable — AAAA0000 format (e.g. KSKS0771) or None for system logs
    user_id: Mapped[str | None] = mapped_column(
        String(9), nullable=True, index=True
    )

    # Nullable FK to chat_history.id (not enforced at DB level for portability)
    chat_history_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )

    # "info" | "warning" | "error"
    level: Mapped[str] = mapped_column(String(10), nullable=False, default="info")

    message: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
