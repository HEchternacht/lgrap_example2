"""
Prompt templates for the LangGraph agent.

Both templates support {param} placeholders filled at runtime from
extra_body prompt.* values in the chat completion request.

Example extra_body:
    {"prompt.language": "Portuguese", "prompt.tone": "formal"}

Defaults (used when no extra_body overrides are provided):
    language → "English"
    tone     → "helpful and concise"
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to the following tools:

- **calculator** — evaluate mathematical expressions (e.g., "2 + 3 * 4", "(10 - 3) / 2")
- **get_weather** — retrieve simulated weather data for any city
- **web_search** — search for information on any topic

Guidelines:
- Use tools whenever they can improve accuracy or provide real data.
- For calculations, always show the expression you evaluated.
- If you cannot answer with the available tools, say so clearly.
- Always respond in **{language}**.
- Your tone must be **{tone}**.
"""

# Wraps the user's raw input before it is sent to the LLM.
# {user_input} is replaced with the actual last user message.
USER_PROMPT_TEMPLATE = """\
{user_input}

---
Important: answer only in **{language}**, using a **{tone}** tone.\
"""

# ---------------------------------------------------------------------------
# Defaults + renderer
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, str] = {
    "language": "English",
    "tone": "helpful and concise",
}


class _SafeFormatMap(dict):
    """Leave unknown {placeholders} intact instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render(template: str, params: dict[str, str]) -> str:
    """
    Render *template* by substituting {placeholders}.

    *params* is merged over _DEFAULTS so callers only need to pass
    the keys they want to override.  Unknown keys are left as-is.
    """
    merged = {**_DEFAULTS, **params}
    return template.format_map(_SafeFormatMap(merged))

