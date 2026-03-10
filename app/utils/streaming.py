"""
Conversion utilities between OpenAI message format and LangChain message objects.
"""
from __future__ import annotations

import json
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.schemas.openai import ChatMessage, FunctionCall, ToolCall


def openai_to_lc_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert a list of OpenAI ChatMessage objects to LangChain BaseMessage objects."""
    result: list[BaseMessage] = []
    for msg in messages:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content or ""))
        elif msg.role == "system":
            result.append(SystemMessage(content=msg.content or ""))
        elif msg.role == "tool":
            result.append(
                ToolMessage(
                    content=msg.content or "",
                    tool_call_id=msg.tool_call_id or "",
                )
            )
    return result


def lc_message_to_openai(message: BaseMessage) -> ChatMessage:
    """Convert a LangChain AIMessage to an OpenAI-compatible ChatMessage."""
    if isinstance(message, AIMessage):
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=json.dumps(tc.get("args", {})),
                    ),
                )
                for tc in message.tool_calls
            ]
        content = message.content if isinstance(message.content, str) else None
        return ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
    return ChatMessage(role="assistant", content=str(message.content))
