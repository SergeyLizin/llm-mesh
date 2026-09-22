"""Provider-neutral connection settings and per-route request options.

A catalog route replaces LLM_OPTIONS atomically. Its missing request options use
client defaults, so values left by an earlier route cannot affect a new client.
Standalone clients may still use individual LLM_* settings without LLM_OPTIONS.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

CONNECTION_ENV = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_PROVIDER_LABEL",
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_AUTH_URL", "LLM_AUTH_SCOPE",
    "LLM_VERIFY_SSL",
)

# Catalog field names, not a second list of environment variables.
ROUTE_OPTIONS = frozenset({
    "disable_reasoning", "reasoning_off", "reasoning_on", "reasoning_field",
    "disable_tools", "sanitize_enums", "disable_thinking_for_tools",
    "force_temperature", "force_top_p", "response_format", "open_object_schemas",
    "cache_hit_field", "cache_miss_field", "cache_nested_field",
    "extra_body", "extra_headers", "tool_choice_pref", "max_output_tokens",
    "min_output_tokens", "max_concurrent", "reasoning_effort",
})


def read_options(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Read an optional JSON object, preserving empty objects and false/zero values."""
    raw = (os.environ if environ is None else environ).get("LLM_OPTIONS", "").strip()
    if not raw:
        return None
    try:
        options = json.loads(raw)
    except ValueError as exc:
        raise ValueError("LLM_OPTIONS must be a valid JSON object") from exc
    if not isinstance(options, dict):
        raise ValueError("LLM_OPTIONS must be a JSON object")
    return options


def get_env(name: str, default: str = "", *, environ: Mapping[str, str] | None = None) -> str:
    """Resolve a neutral setting from options, then the standalone environment.

    Explicit options win. Within a catalog configuration, absent per-route
    options use the default; process-level controls such as retry budgets and
    batch mode may still come from standalone LLM_* variables.
    """
    source = os.environ if environ is None else environ
    if name in CONNECTION_ENV or not name.startswith("LLM_"):
        return source.get(name, "") or default
    options = read_options(source)
    key = name.removeprefix("LLM_").lower()
    if options is not None:
        if key in options:
            value = options[key]
            if value is None:
                return default
            if isinstance(value, (dict, list, bool)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        if key in ROUTE_OPTIONS:
            return default
    return source.get(name, "") or default


def route_options(route: Mapping[str, Any]) -> dict[str, Any]:
    """Extract request options from a catalog route without connection secrets."""
    options = {key: route[key] for key in ROUTE_OPTIONS if key in route and route[key] is not None}
    if route.get("max_tokens") is not None:
        options["max_output_tokens"] = route["max_tokens"]
    for key in ("http_timeout", "no_degrade", "length_retries", "length_retry_cap"):
        if route.get(key) is not None:
            options[key] = route[key]
    if route.get("kind") == "gigachat" and isinstance(route.get("reasoning_on"), dict):
        effort = route["reasoning_on"].get("reasoning_effort")
        if effort is not None:
            options.setdefault("reasoning_effort", effort)
    return options
