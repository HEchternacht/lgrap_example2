"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Mutable state threaded through the agent graph."""

    messages: Annotated[list, add_messages]
