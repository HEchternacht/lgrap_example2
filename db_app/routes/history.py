"""
GET /v1/users/{user_id}/history      — paginated list
GET /v1/users/{user_id}/history/{id} — single record detail
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db_app.crud.history import get_history_by_id, get_user_history
from db_app.database import get_db
from db_app.schemas.history import ChatHistoryItem, ChatHistoryList

router = APIRouter(tags=["Chat History"])


@router.get(
    "/users/{user_id}/history",
    response_model=ChatHistoryList,
    summary="List chat history for a user",
)
def list_user_history(
    user_id: str,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    db: Session = Depends(get_db),
) -> ChatHistoryList:
    """Return paginated chat history for *user_id*, newest first."""
    items, total = get_user_history(db, user_id, skip=skip, limit=limit)
    return ChatHistoryList(items=items, total=total, skip=skip, limit=limit)


@router.get(
    "/users/{user_id}/history/{history_id}",
    response_model=ChatHistoryItem,
    summary="Get a single chat history record",
    responses={404: {"description": "Not found"}},
)
def get_history_detail(
    user_id: str,
    history_id: str,
    db: Session = Depends(get_db),
) -> ChatHistoryItem:
    """Return a single chat history record owned by *user_id*."""
    record = get_history_by_id(db, history_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Chat history not found.")
    return record
