"""Dummy web search tool — returns pre-seeded results for demonstration purposes."""
from __future__ import annotations

from langchain_core.tools import tool

_KNOWLEDGE_BASE: dict[str, str] = {
    "langchain": (
        "LangChain is an open-source framework for building applications powered by "
        "large language models. It provides composable primitives for chains, agents, "
        "memory, and tool use."
    ),
    "langgraph": (
        "LangGraph is a library built on top of LangChain for constructing stateful, "
        "multi-actor agent workflows using a graph-based execution model. It supports "
        "cycles, conditional edges, and streaming."
    ),
    "fastapi": (
        "FastAPI is a modern Python web framework for building APIs. It leverages "
        "Python type hints and Pydantic for automatic request validation, serialization, "
        "and interactive OpenAPI documentation."
    ),
    "openai": (
        "OpenAI is an AI research company behind the GPT series of language models, "
        "DALL-E image generation, and the ChatGPT product. They also publish the "
        "OpenAI API — a widely adopted standard for LLM integrations."
    ),
    "python": (
        "Python is a high-level, dynamically-typed programming language renowned for "
        "its readability and rich ecosystem. It is the dominant language for data "
        "science, machine learning, and backend web development."
    ),
    "langsmith": (
        "LangSmith is an observability and evaluation platform for LLM applications "
        "built with LangChain. It enables tracing, debugging, dataset management, "
        "and automated testing of language model pipelines."
    ),
    "uvicorn": (
        "Uvicorn is a lightning-fast ASGI server implementation for Python, built on "
        "uvloop and httptools. It is the recommended server for running FastAPI applications."
    ),
}


@tool
def web_search(query: str) -> str:
    """
    Search the internet for information on a topic or question.

    Returns a concise summary of the most relevant results found.
    Note: This is a demonstration tool using a static knowledge base.

    Args:
        query: The search query or question to look up.
    """
    query_lower = query.lower()
    matches = [
        result
        for keyword, result in _KNOWLEDGE_BASE.items()
        if keyword in query_lower
    ]

    if matches:
        header = f"Search results for '{query}':\n\n"
        return header + "\n\n".join(matches)

    return (
        f"Search results for '{query}':\n\n"
        "No specific results found in the demo knowledge base. "
        "In a production environment this tool would query a real search engine."
    )
