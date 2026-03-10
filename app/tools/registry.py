"""Tool registry — single source of truth for all agent tools."""
from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.calculator import calculator
from app.tools.search import web_search
from app.tools.weather import get_weather


def get_tools() -> list[BaseTool]:
    """Return the list of tools available to the LangGraph agent."""
    return [calculator, get_weather, web_search]
