"""
chat_history table.

One row = one complete conversation (request + response messages
in OpenAI format).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from db_app.database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    # Primary key: UUID stored as a 36-char string (works on all backends)
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Who made the request — AAAA0000 format (e.g. KSKS0771) or "anonymous"
    user_id: Mapped[str] = mapped_column(String(9), nullable=False, index=True)

    # Short human-readable summary (first user message, truncated to 120 chars)
    title: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # Model name used for this completion
    model: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    # Full conversation in OpenAI message format:
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, …]
    messages: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
