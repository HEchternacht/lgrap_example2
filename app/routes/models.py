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
    now = int(time.time())
    return ModelList(
        data=[
            ModelCard(id=model_id, created=now, owned_by="local")
            for model_id in settings.available_models
        ]
    )


@router.get(
    "/models/{model_id}",
    response_model=ModelCard,
    summary="Retrieve a model",
    tags=["Models"],
    responses={404: {"description": "Model not available"}},
)
async def retrieve_model(model_id: str) -> ModelCard:
    """Return metadata for a specific model ID."""
    from fastapi import HTTPException
    if model_id not in settings.available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelCard(
        id=model_id,
        created=int(time.time()),
        owned_by="local",
    )
