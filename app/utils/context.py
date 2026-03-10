"""
Request-scoped tool configuration via Python contextvars.

Tools read from `tool_config` to receive per-request settings injected
from the `extra_body` field of the chat completion request.

Key format in extra_body:  "<tool_name>.<config_key>": "<value>"

Example:
    extra_body: {"get_weather.temperature": "fahrenheit"}

This is stored internally as:
    {"get_weather": {"temperature": "fahrenheit"}}
"""
from __future__ import annotations

from contextvars import ContextVar

# Maps tool_name -> {config_key: value}
tool_config: ContextVar[dict[str, dict[str, str]]] = ContextVar(
    "tool_config", default={}
)


def parse_extra_body(extra_body: dict | None) -> dict[str, dict[str, str]]:
    """
    Convert a flat extra_body dict with dot-notation keys into a
    nested {tool_name: {key: value}} mapping.

    Keys without a dot are ignored.
    """
    if not extra_body:
        return {}
    result: dict[str, dict[str, str]] = {}
    for raw_key, value in extra_body.items():
        if "." in raw_key:
            tool_name, cfg_key = raw_key.split(".", 1)
            result.setdefault(tool_name, {})[cfg_key] = str(value)
    return result
