"""
GET /v1/models — OpenAI-compatible model listing endpoints.
"""
from __future__ import annotations

import time

from fastapi import APIRouter

from app.schemas.openai import ModelCard, ModelList
from app.utils.config import settings

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelList,
    summary="List available models",
    tags=["Models"],
)
async def list_models() -> ModelList:
    """Return the list of models served by this API."""
    return ModelList(
        data=[
            ModelCard(
                id=settings.model_name,
                created=int(time.time()),
                owned_by="local",
            )
        ]
    )


@router.get(
    "/models/{model_id}",
    response_model=ModelCard,
    summary="Retrieve a model",
    tags=["Models"],
)
async def retrieve_model(model_id: str) -> ModelCard:
    """Return metadata for a specific model ID."""
    return ModelCard(
        id=model_id,
        created=int(time.time()),
        owned_by="local",
    )
