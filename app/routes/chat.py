"""
POST /v1/chat/completions — OpenAI-compatible chat completion endpoint.

Supports:
  • streaming (SSE, text/event-stream) and non-streaming JSON
  • tool calling via LangGraph ReAct agent
  • run cancellation via DELETE /v1/chat/completions/{run_id}
  • client-disconnect detection during streaming
  • LangSmith tracing metadata (run_name, tags)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.graph import get_agent
from app.managers.run_manager import run_manager
from app.utils.config import settings
from app.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)
from app.utils.context import parse_extra_body, prompt_params, tool_config
from app.utils.streaming import build_agent_messages, lc_message_to_openai, openai_to_lc_messages

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
    tags=["Chat"],
    responses={
        200: {"description": "Successful completion (JSON or SSE stream)"},
        500: {"description": "Internal server error"},
    },
)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completion endpoint.

    - Set `stream: true` to receive a Server-Sent Events stream.
    - The response `X-Run-Id` header contains the run ID usable for cancellation.
    - Cancel an active stream via `DELETE /v1/chat/completions/{run_id}`.
    """
    # Validate the requested model against the configured allowlist.
    if body.model not in settings.available_models:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{body.model}' is not available. "
                f"Allowed models: {settings.available_models}"
            ),
        )

    run_id = run_manager.create_run()
    # Parse extra_body and distribute into per-namespace context vars
    parsed = parse_extra_body(body.extra_body)
    params = parsed.get("prompt", {})
    prompt_params.set(params)
    tool_config.set({ns: cfg for ns, cfg in parsed.items() if ns != "prompt"})
    # Build messages with rendered system prompt + user template applied NOW,
    # in the route handler where params are definitively available.
    lc_messages = build_agent_messages(openai_to_lc_messages(body.messages), params)
    created = int(time.time())

    agent = get_agent(body.model)

    if body.stream:
        return StreamingResponse(
            _stream_response(
                run_id, body.model, lc_messages, created, request,
                agent=agent,
                user_id=body.user or "anonymous",
                raw_messages=[m.model_dump() for m in body.messages],
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
                "X-Run-Id": run_id,
            },
        )

    # ---- non-streaming ----
    try:
        config = _run_config(run_id)
        result = await agent.ainvoke({"messages": lc_messages}, config=config)
        _log_messages(run_id, result["messages"])
        final_msg = result["messages"][-1]
        response_oa = lc_message_to_openai(final_msg)
        # Persist the full conversation to DB (non-blocking)
        await asyncio.to_thread(
            _sync_save_history,
            body.user or "anonymous",
            body.model,
            [m.model_dump() for m in body.messages],
            response_oa.content or "",
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{run_id}",
            created=created,
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message=response_oa,
                    finish_reason="stop",
                )
            ],
            usage=Usage(),
        )
    except Exception as exc:
        logger.error("Agent error (run=%s): %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        run_manager.complete_run(run_id)


# ---------------------------------------------------------------------------
# DELETE /v1/chat/completions/{run_id}  — cancel an active streaming run
# ---------------------------------------------------------------------------


@router.delete(
    "/chat/completions/{run_id}",
    summary="Cancel an active completion",
    tags=["Chat"],
    responses={
        200: {"description": "Run cancelled successfully"},
        404: {"description": "Run ID not found"},
    },
)
async def cancel_completion(run_id: str):
    """Cancel a streaming completion that is still in progress."""
    if not run_manager.cancel_run(run_id):
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found or already completed.",
        )
    return {"id": run_id, "cancelled": True}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_config(run_id: str) -> RunnableConfig:
    return RunnableConfig(
        run_name=f"chat-{run_id}",
        tags=["lgrap", "chat"],
        metadata={"run_id": run_id},
    )


async def _stream_response(
    run_id: str,
    model: str,
    lc_messages: list[BaseMessage],
    created: int,
    request: Request,
    agent,
    user_id: str = "anonymous",
    raw_messages: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted chunks from the agent."""
    try:
        # --- opening chunk: announce the assistant role ---
        yield _sse(
            ChatCompletionChunk(
                id=f"chatcmpl-{run_id}",
                created=created,
                model=model,
                choices=[ChunkChoice(index=0, delta=Delta(role="assistant"), finish_reason=None)],
            ).model_dump_json()
        )

        config = _run_config(run_id)
        finish_reason = "stop"
        content_buf: list[str] = []  # accumulates the final assistant text

        async for event in agent.astream_events(
            {"messages": lc_messages},
            version="v2",
            config=config,
        ):
            # Check explicit cancellation
            if run_manager.is_cancelled(run_id):
                break

            # Check client disconnect (avoid lingering server-side work)
            if await request.is_disconnected():
                return

            event_type = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # --- tool call started ---
            if event_type == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                logger.info(
                    "[run=%s] TOOL CALL  >> %s  input=%s",
                    run_id, tool_name, json.dumps(tool_input, ensure_ascii=False),
                )
                yield _sse(json.dumps({
                    "type": "tool_call",
                    "name": tool_name,
                    "input": tool_input,
                }))
                continue

            # --- tool call finished ---
            if event_type == "on_tool_end":
                tool_name = event.get("name", "unknown")
                tool_output = event.get("data", {}).get("output", "")
                output_str = tool_output.content if hasattr(tool_output, "content") else str(tool_output)
                logger.info(
                    "[run=%s] TOOL RESULT << %s  output=%s",
                    run_id, tool_name, output_str,
                )
                yield _sse(json.dumps({
                    "type": "tool_result",
                    "name": tool_name,
                    "output": output_str,
                }))
                continue

            # --- final answer (last agent LLM call that produces plain text) ---
            if event_type == "on_chat_model_end" and node == "agent":
                output = event.get("data", {}).get("output")
                if output is not None:
                    content = output.content if isinstance(output.content, str) else ""
                    has_tool_calls = bool(getattr(output, "tool_calls", None))
                    if content and not has_tool_calls:
                        logger.info("[run=%s] ANSWER     >> %s", run_id, content)
                continue

            if event_type != "on_chat_model_stream":
                continue

            # Only surface tokens from the agent node (not tool nodes)
            if node != "agent":
                continue

            chunk_msg = event["data"]["chunk"]
            content: str = chunk_msg.content if isinstance(chunk_msg.content, str) else ""
            # Skip tool-call chunks (they have no text content for the end-user)
            has_tool_calls = bool(getattr(chunk_msg, "tool_call_chunks", None))

            if content and not has_tool_calls:
                content_buf.append(content)
                yield _sse(
                    ChatCompletionChunk(
                        id=f"chatcmpl-{run_id}",
                        created=created,
                        model=model,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=Delta(content=content),
                                finish_reason=None,
                            )
                        ],
                    ).model_dump_json()
                )

        # --- closing chunk: finish_reason ---
        yield _sse(
            ChatCompletionChunk(
                id=f"chatcmpl-{run_id}",
                created=created,
                model=model,
                choices=[ChunkChoice(index=0, delta=Delta(), finish_reason=finish_reason)],
            ).model_dump_json()
        )
        yield "data: [DONE]\n\n"

        # Save completed conversation to DB after client has received all data
        if raw_messages and content_buf:
            await asyncio.to_thread(
                _sync_save_history,
                user_id,
                model,
                raw_messages,
                "".join(content_buf),
            )

    except Exception as exc:
        logger.error("Streaming error (run=%s): %s", run_id, exc, exc_info=True)
        error_payload = json.dumps(
            {"error": {"message": str(exc), "type": "server_error", "code": 500}}
        )
        yield _sse(error_payload)
    finally:
        run_manager.complete_run(run_id)


def _sse(payload: str) -> str:
    """Wrap a JSON payload as an SSE data line."""
    return f"data: {payload}\n\n"


def _sync_save_history(
    user_id: str,
    model: str,
    raw_messages: list[dict],
    response_content: str,
) -> None:
    """
    Persist the completed conversation to the database.
    Runs in a thread (via asyncio.to_thread) so the event loop is never blocked.
    """
    try:
        from db_app.crud.history import save_chat_history
        from db_app.database import SessionLocal

        messages = list(raw_messages) + [{"role": "assistant", "content": response_content}]
        first_user = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"), None
        )
        title: str | None = None
        if first_user:
            title = (first_user[:117] + "…") if len(first_user) > 120 else first_user

        db = SessionLocal()
        try:
            save_chat_history(db, user_id, model, messages, title)
        finally:
            db.close()
    except Exception:
        logger.warning("Failed to save chat history (user=%s)", user_id, exc_info=True)


def _log_messages(run_id: str, messages: list) -> None:
    """Log tool calls, tool results, and the final answer from a completed agent run."""
    for msg in messages:
        if isinstance(msg, AIMessage):
            # Log any tool calls the model issued
            for tc in getattr(msg, "tool_calls", []) or []:
                logger.info(
                    "[run=%s] TOOL CALL  >> %s  input=%s",
                    run_id,
                    tc.get("name", "unknown"),
                    json.dumps(tc.get("args", {}), ensure_ascii=False),
                )
            # Log plain-text assistant turns (the final answer)
            if isinstance(msg.content, str) and msg.content and not msg.tool_calls:
                logger.info("[run=%s] ANSWER     >> %s", run_id, msg.content)
        elif isinstance(msg, ToolMessage):
            logger.info(
                "[run=%s] TOOL RESULT << %s  output=%s",
                run_id,
                msg.name or "tool",
                str(msg.content),
            )
