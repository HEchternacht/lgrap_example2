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


def build_agent_messages(
    messages: list[BaseMessage],
    params: dict[str, str],
) -> list[BaseMessage]:
    """
    Prepend the rendered system prompt and wrap the last HumanMessage
    with USER_PROMPT_TEMPLATE.  Called once in the route handler so
    the agent receives fully-rendered messages and needs no context vars.
    """
    from app.prompts.system import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, render  # local to avoid circular

    result = [SystemMessage(content=render(SYSTEM_PROMPT, params))] + list(messages)

    # Wrap the last human message (the current user turn)
    for i in range(len(result) - 1, -1, -1):
        if isinstance(result[i], HumanMessage):
            raw = result[i].content if isinstance(result[i].content, str) else ""
            result[i] = HumanMessage(content=render(USER_PROMPT_TEMPLATE, {**params, "user_input": raw}))
            break

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
