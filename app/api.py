"""
FastAPI application factory.

Registers routers, CORS middleware, and a lifespan handler that pre-warms
the LangGraph agent on startup.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import get_agent
from app.routes import chat, models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting lgrap — pre-warming LangGraph agent …")
    get_agent()  # build + cache the compiled graph before the first request
    logger.info("Agent ready.")
    yield
    logger.info("lgrap shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LGRap Agent API",
        description=(
            "OpenAI-compatible LangGraph agent API.\n\n"
            "Supports streaming (SSE), tool calling, run cancellation, "
            "and LangSmith tracing."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Expose OpenAPI under /openapi.json (default) — compatible with
        # tools that consume the OpenAI schema.
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow all origins in development; tighten for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")

    return app
