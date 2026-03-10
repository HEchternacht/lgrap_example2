"""
Request-scoped configuration via Python contextvars.

Namespaces in extra_body (dot-notation keys):
  - "prompt.*"      → prompt_params  (language, tone, …)
  - "<tool_name>.*" → tool_config    (per-tool settings, e.g. get_weather.temperature)

Example extra_body:
    {
        "prompt.language": "Portuguese",
        "prompt.tone": "formal",
        "get_weather.temperature": "fahrenheit"
    }
"""
from __future__ import annotations

from contextvars import ContextVar

# Maps tool_name -> {config_key: value}  (excludes "prompt" namespace)
tool_config: ContextVar[dict[str, dict[str, str]]] = ContextVar(
    "tool_config", default={}
)

# Flat prompt parameters: {"language": "Portuguese", "tone": "formal"}
prompt_params: ContextVar[dict[str, str]] = ContextVar(
    "prompt_params", default={}
)


def parse_extra_body(extra_body: dict | None) -> dict[str, dict[str, str]]:
    """
    Convert a flat extra_body dict with dot-notation keys into a
    nested {namespace: {key: value}} mapping.

    Keys without a dot are ignored.
    """
    if not extra_body:
        return {}
    result: dict[str, dict[str, str]] = {}
    for raw_key, value in extra_body.items():
        if "." in raw_key:
            namespace, cfg_key = raw_key.split(".", 1)
            result.setdefault(namespace, {})[cfg_key] = str(value)
    return result

