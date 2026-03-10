"""
CRUD helpers for the chat_history table.

All functions are synchronous — call via asyncio.to_thread when used from
async code so the event loop is not blocked.
"""
from __future__ import annotations

import uuid

from sqlalchemy import desc
from sqlalchemy.orm import Session

from db_app.models.chat_history import ChatHistory


def get_user_history(
    db: Session,
    user_id: str,
    skip: int = 0,
    limit: int = 20,
) -> tuple[list[ChatHistory], int]:
    """
    Return a paginated, most-recent-first list of histories for *user_id*,
    plus the total un-paginated count.
    """
    q = db.query(ChatHistory).filter(ChatHistory.user_id == user_id)
    total = q.count()
    items = q.order_by(desc(ChatHistory.created_at)).offset(skip).limit(limit).all()
    return items, total


def get_history_by_id(
    db: Session,
    history_id: str,
    user_id: str,
) -> ChatHistory | None:
    """Return a single record owned by *user_id*, or None if not found."""
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.id == history_id, ChatHistory.user_id == user_id)
        .first()
    )


def save_chat_history(
    db: Session,
    user_id: str,
    model: str,
    messages: list[dict],
    title: str | None = None,
) -> ChatHistory:
    """
    Persist a completed conversation.

    *messages* should be the full conversation in OpenAI wire format:
    input messages + final assistant reply.
    """
    record = ChatHistory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        model=model,
        messages=messages,
        title=title,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
