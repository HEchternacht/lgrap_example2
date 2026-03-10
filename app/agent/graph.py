"""
LangGraph ReAct agent graph.

Architecture:
  [START] → agent → (tool_calls?) → tools → agent → … → [END]

The agent node calls the LLM (bound with tools). If the model emits
tool calls the graph routes to the tools node, executes them, and loops
back.  When the model responds with plain text the graph ends.

The compiled graph is cached as a module-level singleton via
`get_agent()` so it is built once on first use and reused afterwards.
"""
from __future__ import annotations

import functools
import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.state import AgentState
from app.tools.registry import get_tools
from app.utils.config import settings

logger = logging.getLogger(__name__)


def _build_agent():
    """Compile and return the LangGraph agent."""
    logger.info(
        "Building LangGraph agent (model=%s, base_url=%s)",
        settings.model_name,
        settings.openai_base_url,
    )

    llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.model_name,
        streaming=True,
    )

    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        # Messages already contain the rendered system prompt and wrapped user turn —
        # built once in the route handler before the agent was invoked.
        response = llm_with_tools.invoke(list(state["messages"]))
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    # Route to tools if the model issued tool calls, otherwise end
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


@functools.lru_cache(maxsize=1)
def get_agent():
    """Return the compiled LangGraph agent (built once, cached forever)."""
    return _build_agent()
