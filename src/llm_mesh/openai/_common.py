"""Shared OpenAI-compatible gateway error detectors and stateless helpers. Request construction,
structured-output tiers, and retry policies remain client-specific. Detection requires both
status and error text: a bare 404 can mean a missing model, which changing tool_choice cannot
fix.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Used only for annotations; no runtime import is required.
    from .client import OpenAIError


def _is_reasoning_mandatory_error(detail: str) -> bool:
    """Return whether the provider rejects reasoning.enabled=false because reasoning is mandatory."""
    low = detail.lower()
    return "mandatory" in low or "cannot be disabled" in low


def _is_temperature_unsupported_error(detail: str) -> bool:
    """Detect rejection of temperature=0 by models that accept only the default value 1. This
    triggers the temperature=1 fallback in _post_inner.
    """
    low = detail.lower()
    return "temperature" in low and (
        "does not support" in low or "only the default" in low
    )


def _google_enum_literal(value: Any) -> str:
    """Convert a JSON primitive to a string enum literal for Google FunctionDeclaration."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


# OpenRouter can return 404 when no active provider accepts named-function tool_choice. Alibaba
# MaaS can return 400 when only auto is supported, rejecting both required and a named function.
# These indicate unsupported parameter forms, not missing tool capability or a model response.
_TOOL_CHOICE_UNSUPPORTED_RE = re.compile(
    r"no endpoints found that support the provided 'tool_choice'"
    r"|tool_choice parameter does not support",
    re.IGNORECASE,
)


def _is_tool_choice_unsupported_error(err: "OpenAIError") -> bool:
    """Detect tool_choice form rejection: require status 400 or 404 and matching response text.
    Missing models, schema errors, and authentication failures must propagate. Match detail,
    falling back to str(err).
    """
    if getattr(err, "status_code", None) not in (404, 400):
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_TOOL_CHOICE_UNSUPPORTED_RE.search(haystack))


# Some llama.cpp Jinja templates cannot construct a tool parser for numeric fields and return
# 400 with a parser-generation error. Fall back to text without tools; changing tool_choice does
# not repair the template grammar.
_TOOL_PARSER_BROKEN_RE = re.compile(
    r"unable to generate parser for this template"
    r"|automatic parser generation failed",
    re.IGNORECASE,
)


def _is_tool_parser_unsupported_error(err: "OpenAIError") -> bool:
    """Detect a 400 template tool-parser failure. Both strict and required still send tools, so
    this requires text fallback.
    """
    if getattr(err, "status_code", None) != 400:
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_TOOL_PARSER_BROKEN_RE.search(haystack))


# llama.cpp PEG parsers may reject large structured responses with a format-mismatch 500. Text
# fallback bypasses the server-side tool parser and lets the client parse JSON content.
_PEG_FORMAT_ERROR_RE = re.compile(
    r"output that does not match the expected peg-",
    re.IGNORECASE,
)


def _is_peg_format_error(err: "OpenAIError") -> bool:
    """Detect a llama.cpp PEG output-format 500 and fall back to text. The model generated output,
    but the server tool parser could not process it; retries with another tool_choice form do
    not bypass that parser.
    """
    if getattr(err, "status_code", None) != 500:
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_PEG_FORMAT_ERROR_RE.search(haystack))


# Nonstandard vendor body keys used for reasoning control. Strict gateways may reject the entire
# request when an unknown key is present. reasoning_effort is deliberately excluded: it is a
# standard OpenAI parameter, and unsupported values are handled separately.
VENDOR_BODY_KEYS: tuple[str, ...] = (
    "chat_template_kwargs", "reasoning", "thinking", "enable_thinking",
)

# Match rejection of an unknown body key, rather than an invalid value. Gateway wording varies;
# callers restrict this detector to HTTP 400.
_UNKNOWN_BODY_FIELD_RE = re.compile(
    r"extra arguments"
    # Allow intervening words such as REQUEST in the standard wording Unrecognized request
    # argument supplied.
    r"|(unknown|unrecognized) (?:\w+ ){0,2}(field|parameter|argument)"
    r"|(does not|doesn't) (support|accept) .{0,30}(parameter|argument|field)",
    re.IGNORECASE,
)


def _is_unknown_body_field_error(detail: str) -> bool:
    """Detect an unknown body-field error so the HTTP 400 handler can retry without
    VENDOR_BODY_KEYS. Strict gateways reject the complete request rather than silently ignoring
    these fields.
    """
    return bool(_UNKNOWN_BODY_FIELD_RE.search(detail))


def chat_completions_url(base_url: str) -> str:
    """Construct <base>/chat/completions from an OpenAI-style base URL."""
    return f"{base_url.rstrip('/')}/chat/completions"
