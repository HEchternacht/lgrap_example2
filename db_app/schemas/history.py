"""Pydantic schemas for the chat history API responses."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ChatHistoryItem(BaseModel):
    """Single chat history record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: Optional[str] = None
    model: str
    messages: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ChatHistoryList(BaseModel):
    """Paginated list of chat history records."""

    items: list[ChatHistoryItem]
    total: int
    skip: int
    limit: int
