"""Async HTTP client for OpenAI-compatible chat/completions endpoints. Configure model, endpoint,
credentials, and provider options explicitly or through environment variables. Supports plain
text, streaming, and structured tool output, with text emulation for providers without function
calling. Bearer authentication does not use token refresh.
"""

from __future__ import annotations

import asyncio
import json
import logging
from llm_mesh.config import get_env
import re
from typing import Any, AsyncIterator, Literal, cast

import httpx

from llm_mesh._common import (
    _env_flag,
    _env_float_default,
    _env_int_default,
    _env_is_disabled,
    _env_nonneg_int,
    _env_positive_int,
    _parse_json_dict_env,
    build_text_messages,
    _args_satisfy_schema,
    finish_reason_opt,
    post_with_length_retry,
    warn_if_truncated,
)
from llm_mesh.base import BaseLLMClient, Capability
from llm_mesh.openai._common import (
    _PEG_FORMAT_ERROR_RE,
    VENDOR_BODY_KEYS,
    _google_enum_literal,
    _is_peg_format_error,
    _is_reasoning_mandatory_error,
    _is_unknown_body_field_error,
    _is_temperature_unsupported_error,
    _is_tool_choice_unsupported_error,
    _is_tool_parser_unsupported_error,
    chat_completions_url,
)
from llm_mesh._retry import backoff_with_jitter, retry_after_delay
from llm_mesh._streaming import (chunk_from_sse_payload, iter_sse_payloads,
    ToolCallAccumulator, events_from_sse_payload)
from llm_mesh.stream_events import Complete, ContentDelta, Error, StreamEvent
from llm_mesh.text_parsing import (
    _CONTROL_MARKER_RE,
    extract_json_array_from_text,
    extract_json_from_text,
)
from llm_mesh.types import (
    LLMAuthError,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    LLMUsage,
    LLMValidationError,
)
logger = logging.getLogger(__name__)

# Transient HTTP statuses eligible for exponential-backoff retries.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# Connection settings use provider-neutral environment names.
_API_KEY_ENVS = ("LLM_API_KEY",)
# OpenAI-compatible base URL, such as https://host/v1.
_BASE_URL_ENVS = ("LLM_BASE_URL",)


class OpenAIError(RuntimeError):
    """OpenAI client error with optional HTTP status_code and detail. Callers can classify gateway
    failures structurally rather than parsing the formatted exception message.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or ""


class ToolArgsCorruptedError(LLMValidationError):
    """Unparseable tool arguments, distinguished from other validation failures for structured-tier
    fallback. Remains an LLMValidationError for existing handlers. Some gateways corrupt
    argument strings under auto while returning valid data for a forced function on the same
    schema.
    """


# Kimi can reject named-function tool_choice as incompatible with enabled thinking. Fall back to
# required without disabling reasoning, using the same tier transition as other tool_choice form
# rejections.
_TOOL_CHOICE_THINKING_CONFLICT_RE = re.compile(
    r"tool_choice .* incompatible with thinking enabled",
    re.IGNORECASE,
)


def _is_tool_choice_thinking_conflict_error(err: "OpenAIError") -> bool:
    """Detect HTTP 400 for named tool_choice being incompatible with enabled thinking or reasoning."""
    if getattr(err, "status_code", None) != 400:
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_TOOL_CHOICE_THINKING_CONFLICT_RE.search(haystack))


def _stringify_numeric_enums(node: Any) -> Any:
    """Copy a schema into Google FunctionDeclaration-compatible form. Serialize enum literals as
    strings without changing their declared scalar type, and reduce nullable type lists to the
    first non-null type. Optionality still comes from required. Apply only to native
    function-calling bodies when LLM_SANITIZE_ENUMS is enabled.
    """
    if isinstance(node, dict):
        out = {k: _stringify_numeric_enums(v) for k, v in node.items()}
        enum = out.get("enum")
        if isinstance(enum, list):
            out["enum"] = [
                v if isinstance(v, str) else _google_enum_literal(v) for v in enum
            ]
        declared_type = out.get("type")
        if isinstance(declared_type, list):
            non_null = [t for t in declared_type if t != "null"]
            out["type"] = non_null[0] if non_null else "string"
        return out
    if isinstance(node, list):
        return [_stringify_numeric_enums(item) for item in node]
    return node


# Detect a gateway rejecting an open dictionary schema. Match the specific schema error so
# unrelated HTTP failures still propagate through the normal error path.
_OPEN_OBJECT_REJECTED_RE = re.compile(
    r"Object fields require at least one of:\s*'properties'\s*or\s*'anyOf'",
    re.IGNORECASE,
)


def _is_open_object_schema_rejected(err: "OpenAIError") -> bool:
    """Detect HTTP 400 caused by an open dictionary in a tool schema."""
    if getattr(err, "status_code", None) != 400:
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_OPEN_OBJECT_REJECTED_RE.search(haystack))


# Gateway 5xx statuses eligible for one text tier attempt. Exclude 429: removing tools does not
# bypass a rate limit.
_GATEWAY_5XX = (500, 502, 503, 504)


def _is_gateway_5xx(err: "OpenAIError") -> bool:
    """Return whether this is a gateway 5xx, excluding client errors and rate limits."""
    return getattr(err, "status_code", None) in _GATEWAY_5XX


def _schema_has_open_object(node: Any) -> bool:
    """Find object schemas with neither properties nor a combinator/$ref. An explicit empty
    properties map is not an open object for this detector.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and not any(
            k in node for k in ("properties", "anyOf", "oneOf", "allOf", "$ref")
        ):
            return True
        return any(_schema_has_open_object(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_has_open_object(v) for v in node)
    return False


def _open_object_paths(node: Any, prefix: str = "") -> list[str]:
    """Return deterministic paths to open dictionary nodes for actionable warnings. Preserve
    traversal order so logs identify the same offending fields across runs.
    """
    found: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" and not any(
            k in node for k in ("properties", "anyOf", "oneOf", "allOf", "$ref")
        ):
            return [prefix or "<schema root>"]
        for k, v in node.items():
            found += _open_object_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += _open_object_paths(v, f"{prefix}[{i}]")
    return found


def _request_open_object_paths(request: LLMRequest) -> list[str]:
    """Find open-object paths in both the single-function schema and multi-tool schemas."""
    return (
        _open_object_paths(request.schema_, "schema")
        + _open_object_paths(request.tools, "tools")
    )


def _request_has_open_object(request: LLMRequest) -> bool:
    """Return whether any schema in the request contains an open object."""
    return (
        _schema_has_open_object(request.schema_)
        or _schema_has_open_object(request.tools)
    )


class _BufferedStreamResponse:
    """Minimal response wrapper reconstructed from SSE, exposing status_code, text, and json().
    Existing _post_inner retry/error handling can use it like a non-streaming httpx response.
    """

    __slots__ = ("status_code", "text", "_data")

    def __init__(self, status_code: int, text: str, data: dict[str, Any] | None) -> None:
        self.status_code = status_code
        self.text = text
        self._data = data

    def json(self) -> dict[str, Any]:
        if self._data is None:
            raise ValueError("no JSON body (error response)")
        return self._data


def _is_tool_choice_param_reject(err: "OpenAIError") -> bool:
    """Detect a 400 attributed to the tool_choice parameter even without a matching error message.
    Complements the shared text detector and signals a request-form fallback.
    """
    if getattr(err, "status_code", None) != 400:
        return False
    haystack = getattr(err, "detail", "") or str(err)
    return bool(_TOOL_CHOICE_PARAM_REJECT_RE.search(haystack))

_TOOL_CHOICE_PARAM_REJECT_RE = re.compile(r"[\"']param[\"']\s*:\s*[\"']tool_choice[\"']", re.IGNORECASE)

class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible text, streaming, and structured-output client. base_url is required;
    explicit arguments take precedence over environment aliases. api_key supplies bearer
    authentication, label names logs, and extra_headers extends requests. LLM_MAX_RETRIES and
    LLM_RETRY_BACKOFF_S control retries; LLM_DISABLE_TOOLS selects text emulation.
    """

    CAPABILITIES = frozenset({
        Capability.TEXT,
        Capability.STREAM,
        Capability.STREAM_EVENTS,
        Capability.STRUCTURED,
        # Native tools and tool_choice. LLM_DISABLE_TOOLS switches one instance
        # to text emulation; the class still implements the native path.
        Capability.TOOLS,
        # tool_choice "auto" may select among request.tools. A gateway that
        # ignores them falls back to one forced function.
        Capability.MULTI_TOOL,
        # tools_required stays on the native tool loop and does not text-emulate.
        Capability.TOOLS_REQUIRED,
        # response_format json_schema/json_object is a declared tier, not the default.
        Capability.JSON_SCHEMA_MODE,
        Capability.LENGTH_RETRY,
    })

    def __init__(
        self,
        model: str = "",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        label: str | None = None,
        extra_headers: dict[str, str] | None = None,
        http_timeout: float | None = None,
        verify: bool | None = None,
        length_retry_cap: int = 32768,
        reasoning_field: str | None = None,
        no_degrade: bool | None = None,
        disable_thinking_for_tools: bool | None = None,
        tool_choice_pref: str | None = None,
        validate_schema: bool = True,
        fallback_policy: Literal["recover", "preserve"] = "recover",
    ) -> None:
        if fallback_policy not in ("recover", "preserve"):
            raise ValueError("fallback_policy must be 'recover' or 'preserve'")
        self._preserve_responses = fallback_policy == "preserve"
        base = base_url or self._env_first(_BASE_URL_ENVS)
        if not base:
            raise OpenAIError(
                "OpenAI provider: missing base_url "
                f"({'/'.join(_BASE_URL_ENVS)} are not set) — "
                "specify endpoint, for example 'https://host/v1'"
            )
        self.URL = chat_completions_url(base)
        self.PROVIDER = label or get_env("LLM_PROVIDER_LABEL") or "openai"
        self._model = model
        self._extra_headers = extra_headers or {}
        self._key = api_key or self._env_first(_API_KEY_ENVS)
        if not self._key:
            raise OpenAIError(
                f"{self.PROVIDER}: missing API key "
                f"({'/'.join(_API_KEY_ENVS)} are not set)"
            )
        self._max_retries = _env_int_default("LLM_MAX_RETRIES", 3, logger=logger)
        self._retry_backoff_s = float(get_env("LLM_RETRY_BACKOFF_S", "1.0"))
        # Limit outbound concurrency with LLM_MAX_CONCURRENT. Create the per-instance semaphore
        # lazily inside the active event loop.
        self._max_concurrent = _env_positive_int("LLM_MAX_CONCURRENT")
        self._semaphore: asyncio.Semaphore | None = None
        # Optional LLM_MAX_OUTPUT_TOKENS ceiling. There is no universal OpenAI-compatible
        # model-limit map, so the default leaves clipping to the endpoint.
        self._max_output_tokens = _env_positive_int("LLM_MAX_OUTPUT_TOKENS")
        self._max_tokens_warned = False
        # Optional LLM_MIN_OUTPUT_TOKENS floor reserves enough budget for hidden reasoning plus
        # visible content. Disabled by default. A sufficient initial budget avoids repeated
        # length-limited attempts that spend every token on reasoning.
        self._min_output_tokens = _env_positive_int("LLM_MIN_OUTPUT_TOKENS")
        # On length truncation, retry with a doubled budget up to the ceiling instead of masking
        # incomplete JSON with repair. Defaults: LLM_LENGTH_RETRIES=2 and
        # LLM_LENGTH_RETRY_CAP=32768. Zero is a valid retry budget.
        self._length_retries = _env_nonneg_int("LLM_LENGTH_RETRIES", 2)
        self._length_retry_cap = length_retry_cap
        _lrc = _env_positive_int("LLM_LENGTH_RETRY_CAP")
        _lrc_explicit = _lrc is not None
        if _lrc_explicit:
            self._length_retry_cap = _lrc
        # A floor at or above the retry cap prevents escalation. Respect an explicitly
        # configured cap and warn; raise an implicit default cap to twice the floor so the
        # default does not silently disable the requested behavior.
        if self._min_output_tokens is not None and (
            self._min_output_tokens >= self._length_retry_cap
        ):
            if _lrc_explicit:
                logger.warning(
                    "LLM_MIN_OUTPUT_TOKENS=%d >= LLM_LENGTH_RETRY_CAP=%d — "
                    "length-retry disabled (no room for growth). Increase the ceiling "
                    "if truncation retries are required.",
                    self._min_output_tokens, self._length_retry_cap,
                )
            else:
                raised = self._min_output_tokens * 2
                logger.info(
                    "LLM_MIN_OUTPUT_TOKENS=%d reaches the default "
                    "LLM_LENGTH_RETRY_CAP=%d — ceiling raised to %d, otherwise "
                    "length-retry would make no attempts",
                    self._min_output_tokens, self._length_retry_cap, raised,
                )
                self._length_retry_cap = raised
        # TLS verification precedence: explicit argument, LLM_VERIFY_SSL, then True. Custom-CA
        # or self-signed deployments may override verification.
        if verify is not None:
            self._verify = verify
        else:
            # No strip: a padded value was never treated as off.
            self._verify = not _env_is_disabled("LLM_VERIFY_SSL", default="1")
        # Optionally use stream=true for non-streaming APIs, then reconstruct the response from
        # SSE. Receiving bytes during generation avoids reverse-proxy idle timeouts on long
        # requests. No strip, matching the historical parser.
        self._stream_transport = _env_flag("LLM_STREAM_TRANSPORT", strip=False)
        # LLM_DISABLE_TOOLS selects text-based structured-output emulation for endpoints without
        # native tools/tool_choice support. No strip, matching the historical parser.
        self._tools_enabled = not _env_flag("LLM_DISABLE_TOOLS", strip=False)
        # LLM_SANITIZE_ENUMS stringifies schema enum literals for Google FunctionDeclaration.
        # Disabled by default because it changes the wire schema for every call on this client.
        self._sanitize_enums = _env_flag("LLM_SANITIZE_ENUMS")
        # LLM_DISABLE_THINKING_FOR_TOOLS disables reasoning only for native function calls whose
        # forced tool_choice conflicts with thinking. Text generation and text fallback retain
        # reasoning.
        self._disable_thinking_for_tools = (
            disable_thinking_for_tools if disable_thinking_for_tools is not None else
            _env_flag("LLM_DISABLE_THINKING_FOR_TOOLS")
        )
        # Cache the learned structured tier per instance: strict, then required, then text.
        # LLM_TOOL_CHOICE_PREF may explicitly select the initial tier for endpoints that
        # silently ignore a named function.
        self._reasoning_field: str = (reasoning_field if reasoning_field is not None
            else get_env("LLM_REASONING_FIELD", "").strip())
        # Optional cache-counter response field names complement LLMUsage.from_raw defaults,
        # including provider-specific nested fields.
        self._cache_hit_field: str | None = (
            get_env("LLM_CACHE_HIT_FIELD", "").strip() or None
        )
        self._cache_miss_field: str | None = (
            get_env("LLM_CACHE_MISS_FIELD", "").strip() or None
        )
        self._cache_nested_field: str | None = (
            get_env("LLM_CACHE_NESTED_FIELD", "").strip() or None
        )
        self._validate_schema = validate_schema
        _forced_pref = (tool_choice_pref if tool_choice_pref is not None else
                        get_env("LLM_TOOL_CHOICE_PREF", "")).strip().lower()
        self._text_requested = _forced_pref == "text"
        self._tool_choice_pref: str | None = (
            _forced_pref if _forced_pref in ("auto", "required", "text") else None
        )
        # Cache multi-tool support per instance: unknown, confirmed function selection, or tools
        # ignored in favor of JSON content. The last case falls back to a forced single
        # function.
        self._multitool_supported: bool | None = None
        # Record the tier that actually served the last structured call: multi, strict,
        # required, response_format, or text. A valid response alone does not reveal whether
        # native function calling or emulation produced it.
        self._last_served_tier: str | None = None
        # Optional native structured output: json_schema carries a schema; json_object
        # guarantees JSON syntax only. Enable this tier through an explicit route declaration.
        self._response_format = get_env("LLM_RESPONSE_FORMAT", "").strip()
        # A declared or learned open-object rejection bypasses unsupported schema-bearing tiers
        # and goes directly to text. Undeclared routes learn the restriction after a matching
        # 400.
        self._open_objects_unsupported = get_env(
            "LLM_OPEN_OBJECT_SCHEMAS", ""
        ).strip().lower() == "unsupported"
        # Track whether the restriction was declared or learned so diagnostics do not point to a
        # nonexistent catalog declaration.
        self._open_objects_declared = self._open_objects_unsupported
        # Log the affected gateway/model route. Learned state lasts for this client instance; a
        # measured catalog declaration persists across instances. Per-call model
        # overrides compute a separate id so concurrent requests cannot race
        # on this construction-time value.
        self._route_id = f"{self.PROVIDER}/{self._model}"
        # LLM_NO_DEGRADE makes native capability failures visible by rejecting implicit text
        # emulation. Native tool_choice form selection may still proceed; explicitly requested
        # text mode remains valid.
        # No strip, matching the historical parser.
        self._no_degrade = (no_degrade if no_degrade is not None else
            _env_flag("LLM_NO_DEGRADE", strip=False))
        # Additional provider body settings apply to text and structured requests.
        # LLM_DISABLE_REASONING controls reasoning-off behavior, LLM_REASONING_OFF supplies the
        # route dialect, and LLM_EXTRA_BODY supplies other vendor settings.
        self._extra_body: dict[str, Any] = {}
        _parsed_extra = _parse_json_dict_env("LLM_EXTRA_BODY", logger=logger)
        if _parsed_extra is not None:
            self._extra_body.update(_parsed_extra)
        # Merge LLM_EXTRA_HEADERS into HTTP headers, for example a provider-specific project
        # identifier.
        _parsed_headers = _parse_json_dict_env("LLM_EXTRA_HEADERS", logger=logger)
        if _parsed_headers is not None:
            self._extra_headers.update({str(k): str(v) for k, v in _parsed_headers.items()})
        # LLM_REASONING_ON supplies the gateway's explicit enabling dialect. There is no
        # universal payload that reliably enables reasoning across providers.
        self._reasoning_on_body: dict[str, Any] = (
            _parse_json_dict_env("LLM_REASONING_ON", logger=logger) or {}
        )
        off_body = _parse_json_dict_env("LLM_REASONING_OFF", logger=logger)
        self._reasoning_off_body = off_body or {}
        self._reasoning_off_declared = off_body is not None
        self._dtft_no_off_warned = False
        for key, value in self._reasoning_on_body.items():
            if isinstance(value, dict) and isinstance(self._extra_body.get(key), dict):
                self._extra_body[key].update(value)
            else:
                self._extra_body[key] = value
        if _env_flag("LLM_DISABLE_REASONING") and self._reasoning_off_declared:
            for key in self._reasoning_on_body:
                self._extra_body.pop(key, None)
            self._extra_body.update(self._reasoning_off_body)
            self._reasoning_on_body = {}
        # Optional frequency_penalty and repetition_penalty environment overrides mitigate
        # repetitive generation. Apply them only when configured, preserving the existing
        # defaults.
        for _env, _key in (("LLM_FREQUENCY_PENALTY", "frequency_penalty"),
                           ("LLM_REPETITION_PENALTY", "repetition_penalty")):
            _raw = get_env(_env, "").strip()
            if _raw:
                try:
                    self._extra_body.setdefault(_key, float(_raw))
                except ValueError:
                    logger.warning("%s is not a number (%r) — ignoring", _env, _raw)
        # LLM_FORCE_TEMPERATURE overrides every request for gateways that accept only a fixed
        # temperature.
        self._no_vendor_keys = False
        # Real OpenAI omits stream usage unless stream_options.include_usage
        # is set. A gateway that rejects the field is remembered here.
        self._no_stream_usage = False
        self._force_temperature: float | None = None
        _ft = get_env("LLM_FORCE_TEMPERATURE", "").strip()
        if _ft:
            try:
                self._force_temperature = float(_ft)
            except ValueError:
                logger.warning("LLM_FORCE_TEMPERATURE is not a number (%r) — ignoring", _ft)
        # Create httpx.AsyncClient on the first request; constructor failures must not leave an
        # uncloseable transport behind.
        self._http_timeout: float = (
            http_timeout if http_timeout is not None
            else _env_float_default("LLM_HTTP_TIMEOUT", 600.0, logger=logger)
        )
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _env_first(names: tuple[str, ...]) -> str:
        for name in names:
            val = get_env(name)
            if val:
                return val
        return ""

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        return build_text_messages(request)

    def _clip_max_tokens(self, requested: int) -> int:
        """Apply the optional token floor, then the hard ceiling. The ceiling wins if configuration
        conflicts. Warn once per instance when adjusting the budget.
        """
        if self._min_output_tokens is not None and requested < self._min_output_tokens:
            requested = self._min_output_tokens
        if self._max_output_tokens is None or requested <= self._max_output_tokens:
            return requested
        if not self._max_tokens_warned:
            logger.warning(
                "%s: max_tokens=%d exceeds LLM_MAX_OUTPUT_TOKENS=%d, clipping.",
                self.PROVIDER, requested, self._max_output_tokens,
            )
            self._max_tokens_warned = True
        return self._max_output_tokens

    def _prepare_outbound_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply the same vendor-body merge the interactive POST uses.

        Batch submission never calls ``_post``, so the merge has to live in one
        place. Otherwise extra_body, thinking-off, and a forced temperature would
        diverge between a live call and a JSONL line.
        """
        for k, v in self._extra_body.items():
            if self._no_vendor_keys and k in VENDOR_BODY_KEYS:
                continue
            body.setdefault(k, v)
        if body.get("tools") and not self._no_vendor_keys:
            self._apply_thinking_off_for_tools(body)
        if self._force_temperature is not None and "temperature" in body:
            body["temperature"] = self._force_temperature
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST chat/completions under the optional concurrency semaphore."""
        self._prepare_outbound_body(body)
        sem = self._ensure_semaphore()
        if sem is None:
            return await self._post_inner(body)
        async with sem:
            return await self._post_inner(body)

    async def _post_with_length_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """Retry length-truncated responses with a doubled budget up to the configured cap and
        model limits. Normal responses and budgets already at the ceiling are unchanged.
        """
        return await post_with_length_retry(
            body,
            post=self._post,
            payload_of=lambda data: data,
            retries=self._length_retries,
            next_max_tokens=lambda current: self._clip_max_tokens(
                min(current * 2, self._length_retry_cap)
                if current
                else self._length_retry_cap
            ),
            provider=self.PROVIDER,
            logger=logger,
        )

    def _apply_stream_usage(self, body: dict[str, Any]) -> None:
        """Ask for a trailing usage chunk.

        Real OpenAI omits ``usage`` unless ``stream_options.include_usage`` is
        set, so ``Complete.usage`` stays None. ``setdefault`` leaves a caller
        supplied value alone. A gateway that rejects the field is remembered
        on this instance by ``_drop_stream_usage``.
        """
        if self._no_stream_usage:
            return
        body.setdefault("stream_options", {"include_usage": True})

    def _drop_stream_usage(self, status: int, detail: str, body: dict[str, Any]) -> bool:
        """Learn a stream_options rejection and strip it for one retry.

        Same shape as the vendor-key fallback: only an HTTP 400 that names an
        unknown field, and only when this request actually sent the field.
        """
        if (
            status != 400
            or self._no_stream_usage
            or "stream_options" not in body
            or not _is_unknown_body_field_error(detail)
        ):
            return False
        logger.info(
            "%s: gateway rejects stream_options (%s) — retrying without "
            "include_usage (learned per instance)",
            self.PROVIDER, detail[:120],
        )
        self._no_stream_usage = True
        body.pop("stream_options", None)
        return True

    async def _stream_and_reconstruct(
        self, headers: dict[str, str], body: dict[str, Any]
    ) -> _BufferedStreamResponse:
        """Send stream=true and reconstruct an equivalent non-stream response from SSE. Aggregate
        content, reasoning, tool-call arguments, finish reason, and usage while keeping
        reverse-proxy connections active during generation.
        """
        stream_body = {**body, "stream": True}
        self._apply_stream_usage(stream_body)
        for _attempt in range(2):
            result = await self._consume_reconstructed_stream(headers, stream_body, body)
            if result.status_code < 400 or not self._drop_stream_usage(
                result.status_code, result.text, stream_body,
            ):
                return result
        return result

    async def _consume_reconstructed_stream(
        self, headers: dict[str, str], stream_body: dict[str, Any], body: dict[str, Any],
    ) -> _BufferedStreamResponse:
        async with self._ensure_http().stream(
            "POST", self.URL, headers=headers, json=stream_body
        ) as r:
            if r.status_code >= 400:
                raw = (await r.aread()).decode("utf-8", "replace")
                return _BufferedStreamResponse(r.status_code, raw, None)
            content_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason = ""
            usage_raw: dict[str, Any] | None = None
            role = "assistant"
            resp_id = ""
            async for payload in iter_sse_payloads(r.aiter_lines()):
                if payload.get("id"):
                    resp_id = payload["id"]
                if payload.get("usage"):
                    usage_raw = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0] or {}
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = fr
                delta = choice0.get("delta") or {}
                if delta.get("role"):
                    role = delta["role"]
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        message: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        data: dict[str, Any] = {
            "id": resp_id,
            "model": body.get("model", ""),
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish_reason or "stop", "message": message}],
        }
        if usage_raw:
            data["usage"] = usage_raw
        return _BufferedStreamResponse(200, json.dumps(data), data)

    async def _post_inner(self, body: dict[str, Any]) -> dict[str, Any]:
        """Shared text/structured transport with backoff for rate limits, transient server errors,
        and network failures. Return a response containing choices or raise a typed error.
        """
        headers = {
            "Authorization": f"Bearer {self._key}",
            **self._extra_headers,
        }
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                r = (
                    await self._stream_and_reconstruct(headers, body)
                    if self._stream_transport
                    else await self._ensure_http().post(self.URL, headers=headers, json=body)
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                    httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    # Translate timeouts to LLMTimeoutError and other failures to OpenAIError.
                    err_cls = LLMTimeoutError if isinstance(exc, httpx.TimeoutException) else OpenAIError
                    raise err_cls(
                        f"{self.PROVIDER} network error after "
                        f"{attempt + 1} attempts: {exc}"
                    ) from exc
                # Add jitter to network retries so workers affected by the same outage do not
                # retry in lockstep.
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
                continue

            if (r.status_code in RETRYABLE_STATUS and attempt < self._max_retries
                    and not (r.status_code == 500 and r.text
                             and _PEG_FORMAT_ERROR_RE.search(r.text))):
                # Do not retry a deterministic llama.cpp PEG-format 500 with the same tool
                # request. Let structured-tier handling bypass the parser without paying for
                # repeated full generations.
                delay = (
                    retry_after_delay(r, self._retry_backoff_s, attempt)
                    if r.status_code == 429
                    else backoff_with_jitter(self._retry_backoff_s, attempt)
                )
                logger.warning(
                    "%s %s on attempt %d/%d — retry in %.1fs",
                    self.PROVIDER, r.status_code, attempt + 1, self._max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue

            # Retain up to 4000 response characters: intermediary HTML error pages may place
            # useful block identifiers well beyond the first 500 characters.
            detail = r.text[:4000] if r.text else "<empty>"
            if r.status_code in (401, 403):
                # Authentication failures are terminal and use a distinct exception type.
                raise LLMAuthError(f"{self.PROVIDER} HTTP {r.status_code}: {detail}")
            if r.status_code >= 400:
                # Learn temperature=1 per client after a specific rejection of temperature=0,
                # then retry this request. Keep the cache instance-local because models and
                # gateways in the same process may have different requirements.
                if (r.status_code == 400
                        and body.get("temperature") not in (None, 1, 1.0)
                        and _is_temperature_unsupported_error(detail)):
                    logger.info(
                        "%s: temperature=%s rejected (%s), fallback → "
                        "temperature=1 (learned per instance)",
                        self.PROVIDER, body.get("temperature"), detail[:120],
                    )
                    self._force_temperature = 1.0
                    temp_body = {**body, "temperature": 1.0}
                    try:
                        r_t = await self._ensure_http().post(
                            self.URL, headers=headers, json=temp_body
                        )
                    except (httpx.TimeoutException, httpx.ConnectError,
                            httpx.ReadError, httpx.NetworkError,
                            httpx.RemoteProtocolError) as exc_t:
                        raise OpenAIError(
                            f"{self.PROVIDER} network error on temperature "
                            f"fallback: {exc_t}"
                        ) from exc_t
                    if r_t.status_code < 400:
                        try:
                            data_t: dict[str, Any] = r_t.json()
                        except ValueError as exc_t:
                            raise OpenAIError(
                                f"{self.PROVIDER} malformed JSON on temperature "
                                f"fallback: {exc_t}"
                            ) from exc_t
                        if not (data_t.get("choices") or []):
                            raise OpenAIError(
                                f"{self.PROVIDER} no choices on temperature "
                                f"fallback: {data_t!r}"
                            )
                        return data_t
                    # If the fallback also fails, raise the original error.
                # When reasoning is mandatory, retry once with exclude=true rather than
                # reasoning.enabled=false.
                reasoning_block = body.get("reasoning") or {}
                if (r.status_code == 400
                        and reasoning_block.get("enabled") is False
                        and _is_reasoning_mandatory_error(detail)):
                    logger.info(
                        "%s: reasoning.enabled=false rejected (%s), "
                        "fallback → reasoning.exclude=true",
                        self.PROVIDER, detail[:120],
                    )
                    fallback_body = {**body, "reasoning": {"exclude": True}}
                    try:
                        r2 = await self._ensure_http().post(
                            self.URL, headers=headers, json=fallback_body
                        )
                    except (httpx.TimeoutException, httpx.ConnectError,
                            httpx.ReadError, httpx.NetworkError,
                            httpx.RemoteProtocolError) as exc2:
                        raise OpenAIError(
                            f"{self.PROVIDER} network error on reasoning fallback: {exc2}"
                        ) from exc2
                    if r2.status_code < 400:
                        try:
                            data2: dict[str, Any] = r2.json()
                        except ValueError as exc2:
                            raise OpenAIError(
                                f"{self.PROVIDER} malformed JSON on reasoning fallback: {exc2}"
                            ) from exc2
                        if not (data2.get("choices") or []):
                            raise OpenAIError(
                                f"{self.PROVIDER} no choices on reasoning fallback: {data2!r}"
                            )
                        return data2
                    # If the fallback also fails, raise the original error.
                # After an unknown vendor-field rejection, retry without vendor keys and
                # remember the restriction on this instance. Reasoning may remain enabled, but
                # the request can proceed; never apply this restriction process-wide to other
                # routes.
                if (r.status_code == 400
                        and any(k in body for k in VENDOR_BODY_KEYS)
                        and _is_unknown_body_field_error(detail)):
                    logger.info(
                        "%s: gateway rejects vendor body keys (%s) — "
                        "retrying without them; reasoning will remain enabled "
                        "(learned per instance)",
                        self.PROVIDER, detail[:120],
                    )
                    self._no_vendor_keys = True
                    stripped = {k: v for k, v in body.items()
                                if k not in VENDOR_BODY_KEYS}
                    try:
                        r_v = await self._ensure_http().post(
                            self.URL, headers=headers, json=stripped
                        )
                    except (httpx.TimeoutException, httpx.ConnectError,
                            httpx.ReadError, httpx.NetworkError,
                            httpx.RemoteProtocolError) as exc_v:
                        raise OpenAIError(
                            f"{self.PROVIDER} network error on vendor-keys "
                            f"fallback: {exc_v}"
                        ) from exc_v
                    if r_v.status_code < 400:
                        try:
                            data_v: dict[str, Any] = r_v.json()
                        except ValueError as exc_v:
                            raise OpenAIError(
                                f"{self.PROVIDER} malformed JSON on vendor-keys "
                                f"fallback: {exc_v}"
                            ) from exc_v
                        if not (data_v.get("choices") or []):
                            raise OpenAIError(
                                f"{self.PROVIDER} no choices on vendor-keys "
                                f"fallback: {data_v!r}"
                            )
                        return data_v
                    # If the fallback also fails, raise the original error.
                # Raise a structured terminal HTTP error with status_code and detail so callers
                # can distinguish tool-choice failures from unrelated errors.
                raise OpenAIError(
                    f"{self.PROVIDER} HTTP {r.status_code}: {detail}",
                    status_code=r.status_code,
                    detail=detail,
                )

            try:
                data: dict[str, Any] = r.json()
            except ValueError as exc:
                raise OpenAIError(
                    f"{self.PROVIDER} malformed JSON response: {exc} "
                    f"(body={r.text[:300]!r})"
                ) from exc

            if not (data.get("choices") or []):
                # Handle retryable upstream error codes embedded in an HTTP 200 body, which
                # ordinary HTTP-status retry logic would miss.
                _err = data.get("error")
                _raw_code = _err.get("code") if isinstance(_err, dict) else None
                try:
                    _err_code = int(_raw_code)  # Accept both integer codes and digit strings.
                except (TypeError, ValueError):
                    _err_code = None
                if _err_code in RETRYABLE_STATUS and attempt < self._max_retries:
                    delay = backoff_with_jitter(self._retry_backoff_s, attempt)
                    logger.warning(
                        "%s body-level error code=%s on attempt %d/%d — retry in %.1fs",
                        self.PROVIDER, _err_code, attempt + 1,
                        self._max_retries + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise OpenAIError(f"{self.PROVIDER} response has no choices: {data!r}")

            # Retry transient empty content only when no native tool_calls/function_call is
            # present. Empty content is normal for a successful tool response.
            choice0 = data["choices"][0] or {}
            msg0 = choice0.get("message") or {}
            empty_content = not (msg0.get("content") or "").strip()
            has_call = bool(msg0.get("tool_calls") or msg0.get("function_call"))
            fr0 = str(choice0.get("finish_reason") or choice0.get("finishReason") or "")
            comp_toks = int((data.get("usage") or {}).get("completion_tokens") or 0)
            # Do not repeat deterministically empty responses at the same budget: length
            # exhaustion, content filtering, or a reasoning-only stop with consumed completion
            # tokens. Budget escalation belongs to the outer length-retry layer, avoiding
            # multiplicative retry costs.
            if empty_content and not has_call and fr0 == "length":
                logger.warning(
                    "%s: empty content with finish_reason=length (completion_tokens=%d) — "
                    "budget exhausted by reasoning, retrying with the same budget will not "
                    "help, returning as is",
                    self.PROVIDER, comp_toks,
                )
                return data
            if empty_content and not has_call and fr0 == "content_filter":
                logger.warning(
                    "%s: finish_reason=content_filter (completion_tokens=%d) — "
                    "response blocked by moderation, retrying will not help, returning as is",
                    self.PROVIDER, comp_toks,
                )
                return data
            if empty_content and not has_call and fr0 == "stop" and comp_toks > 0:
                logger.warning(
                    "%s: finish_reason=stop with empty content, but %d completion_tokens "
                    "(reasoning without action) — retrying will not help, returning as is",
                    self.PROVIDER, comp_toks,
                )
                return data
            if empty_content and not has_call and attempt < self._max_retries:
                logger.warning(
                    "%s empty content (200, completion_tokens=%s) on attempt "
                    "%d/%d — retry",
                    self.PROVIDER,
                    (data.get("usage") or {}).get("completion_tokens"),
                    attempt + 1, self._max_retries + 1,
                )
                # Jitter empty-response retries too, avoiding synchronized load from reasoning
                # models under pressure.
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
                continue
            return data
        # Unreachable after the loop returns or raises; retained for static type checking.
        raise OpenAIError(f"{self.PROVIDER} retry exhausted: {last_exc}")

    def _usage(self, data: dict[str, Any]) -> LLMUsage:
        return LLMUsage.from_raw(
            data.get("usage"),
            cache_hit_field=self._cache_hit_field,
            cache_miss_field=self._cache_miss_field,
            cache_nested_field=self._cache_nested_field,
        )

    def _effective_model(self, request: LLMRequest) -> str:
        """Request model, else the model configured on this client."""
        return request.model or self._model

    def _route_id_for(self, request: LLMRequest) -> str:
        """Route label for this call. Do not write it back onto the client."""
        return f"{self.PROVIDER}/{self._effective_model(request)}"

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate plain text without tools. Return message.content as LLMResponse.text with empty
        arguments; callers may parse structured text themselves.
        """
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
        }
        # Small best-effort outputs can opt out of length retries; escalating repetitive
        # metadata responses only increases cost and latency.
        data = await (
            self._post_with_length_retry(body)
            if request.length_retry
            else self._post(body)
        )
        return self._text_response(data, request)

    def _text_response(
        self, data: dict[str, Any], request: LLMRequest, *, scan_canary: bool = True,
    ) -> LLMResponse:
        """Parse a plain chat completion. ``scan_canary=False`` is the batch path:
        stored batch output is not a live turn, and GigaChat batch does not scan either.
        """
        choice = data["choices"][0] or {}
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        # Strip leaked function-call control markers and surrounding whitespace before returning
        # text to consumers with their own JSON-fence parsers.
        content = _CONTROL_MARKER_RE.sub("", content).strip()
        warn_if_truncated(
            choice,
            request,
            content,
            sent_max_tokens=self._clip_max_tokens(request.max_tokens),
            provider=self.PROVIDER,
            logger=logger,
        )
        if scan_canary:
            self._check_response_canary(content, context="generate_text")
        return LLMResponse(
            arguments={},
            text=content,
            reasoning_content=self._reasoning_content_of(msg),
            finish_reason=finish_reason_opt(data),
            model=str(data.get("model", self._effective_model(request))),
            usage=self._usage(data),
            raw=data,
        )

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream plain text from OpenAI SSE content deltas until [DONE]. Use generate_structured
        for function-calling output.
        """
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
            "stream": True,
        }
        for k, v in self._extra_body.items():
            if self._no_vendor_keys and k in VENDOR_BODY_KEYS:
                continue
            body.setdefault(k, v)
        if self._force_temperature is not None and "temperature" in body:
            body["temperature"] = self._force_temperature
        sem = self._ensure_semaphore()
        if sem is None:
            async for chunk in self._do_stream(body):
                yield chunk
            return
        async with sem:
            async for chunk in self._do_stream(body):
                yield chunk

    async def _do_stream(self, body: dict[str, Any]) -> AsyncIterator[LLMStreamChunk]:
        headers = {"Authorization": f"Bearer {self._key}", **self._extra_headers}
        self._apply_stream_usage(body)
        agg: list[str] = []
        retry_without_usage = False
        try:
            async with self._ensure_http().stream(
                "POST", self.URL, headers=headers, json=body
            ) as r:
                if r.status_code >= 400:
                    raw = (await r.aread()).decode("utf-8", "replace")[:300]
                    if self._drop_stream_usage(r.status_code, raw, body):
                        retry_without_usage = True
                    elif r.status_code in (401, 403):
                        raise LLMAuthError(f"{self.PROVIDER} stream {r.status_code}: {raw}")
                    else:
                        raise OpenAIError(f"{self.PROVIDER} stream {r.status_code}: {raw}")
                else:
                    request_id = r.headers.get("x-request-id")
                    first = True
                    async for payload in iter_sse_payloads(r.aiter_lines()):
                        chunk = chunk_from_sse_payload(
                            payload, request_id=request_id, first=first,
                            reasoning_field=self._reasoning_field or None,
                        )
                        if chunk.delta_text:
                            agg.append(chunk.delta_text)
                        yield chunk
                        first = False
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise LLMTimeoutError(f"{self.PROVIDER} stream transient error: {exc}") from exc
        finally:
            # Scan partial output even when the stream is interrupted; incomplete text can
            # contain a canary leak.
            self._check_response_canary("".join(agg), context="generate_stream")
        if retry_without_usage:
            async for chunk in self._do_stream(body):
                yield chunk

    def _open_objects_source(self) -> str:
        """Describe whether the route restriction was declared or learned from a 400 response."""
        return (
            "declared open_object_schemas: unsupported in models.json"
            if self._open_objects_declared
            else "learned reactively from this gateway's 400 response; route "
                 "not declared in models.json"
        )

    def _fc_body(
        self,
        request: LLMRequest,
        fn: str,
        schema: dict[str, Any],
        tool_choice: Any,
    ) -> dict[str, Any]:
        """Build a native single-function request with either named-function tool_choice or
        required. Both force the sole tool, but required is supported by more gateways.
        """
        schema = self._adapt_schema(schema)
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": self._messages(request),
            "tools": [{
                "type": "function",
                "function": {
                    "name": fn,
                    "description": request.function_description
                    or f"Build a {fn} object.",
                    "parameters": schema,
                },
            }],
            "tool_choice": tool_choice,
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
        }
        # When configured, disable thinking only in native function-call bodies, overriding
        # conflicting extra-body reasoning settings. Preserve thinking for text generation and
        # text fallback.
        self._apply_thinking_off_for_tools(body)
        return body

    def _batch_tool_choice(self, fn: str) -> Any:
        """Tool choice for one batch line. A batch cannot walk the interactive tier
        ladder, so this is the first forced form that path would send: a named
        function, or ``required``/``auto`` when the route has already settled there.
        """
        if self._tool_choice_pref == "required":
            return "required"
        if self._tool_choice_pref == "auto":
            return "auto"
        return {"type": "function", "function": {"name": fn}}

    def build_completion_body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        """Body a non-batch call would POST for this request.

        ``structured=False`` is the ``generate_text`` body, not ``_text_json_body``.
        That helper is the text-emulation tier for structured output and would
        change a plain-text batch. ``structured=True`` is the forced single-function
        tier (``_fc_body``). Both then pass through ``_prepare_outbound_body``, which
        is what ``_post`` applies before the wire.
        """
        if structured:
            fn = request.function_name
            schema = request.schema_ if isinstance(request.schema_, dict) else {
                "type": "object", "properties": {},
            }
            body = self._fc_body(request, fn, schema, self._batch_tool_choice(fn))
        else:
            body = {
                "model": self._effective_model(request),
                "messages": self._messages(request),
                "temperature": request.temperature,
                "max_tokens": self._clip_max_tokens(request.max_tokens),
                **self._reasoning_body_kwargs(request),
            }
        return self._prepare_outbound_body(body)

    def response_from_payload(
        self, payload: dict[str, Any], request: LLMRequest, structured: bool,
    ) -> LLMResponse:
        """Parse a chat completion the interactive path would return, without the canary."""
        if structured:
            return self._structured_response(payload, request, scan_canary=False)
        return self._text_response(payload, request, scan_canary=False)

    def _text_json_body(self, request: LLMRequest, fn: str) -> dict[str, Any]:
        """Build text-based structured-output emulation without tools. Include the schema in the
        prompt so the model knows the required keys; parse JSON from content afterwards.
        """
        schema_hint = ""
        if request.schema_:
            try:
                schema_hint = "\nSCHEMA (use exactly these keys and types):\n" + json.dumps(
                    request.schema_, ensure_ascii=False,
                )
            except (TypeError, ValueError):
                schema_hint = ""
        instructed = request.model_copy(update={
            "system": request.system + (
                f"\n\nReturn ONLY a valid JSON object for function `{fn}` "
                "according to its schema, without Markdown wrapping or explanations." + schema_hint
            ),
        })
        return {
            "model": self._effective_model(request),
            "messages": self._messages(instructed),
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
        }

    def _adapt_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Adapt schemas only when configured. Currently this applies Google-compatible enum/type
        normalization; otherwise return the original schema object unchanged.
        """
        if not self._sanitize_enums:
            return schema
        return cast("dict[str, Any]", _stringify_numeric_enums(schema))

    @staticmethod
    def _reasoning_body_kwargs(request: LLMRequest) -> dict[str, Any]:
        """Map request.reasoning_effort to its standard request-body field; omit the key when no
        effort is requested.
        """
        if request.reasoning_effort:
            return {"reasoning_effort": request.reasoning_effort}
        return {}

    def _reasoning_content_of(self, message: dict[str, Any]) -> str | None:
        """Extract the reasoning channel separately from visible content so callers can distinguish
        empty output from a budget spent on reasoning.
        """
        value = (message.get(self._reasoning_field) if self._reasoning_field
                 else (message.get("reasoning_content") or message.get("reasoning")))
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _structured_response(
        self, data: dict[str, Any], request: LLMRequest, *, scan_canary: bool = True,
    ) -> LLMResponse:
        """Parse a function-call response into LLMResponse and, on the live path, scan it."""
        choice = data["choices"][0] or {}
        msg = choice.get("message") or {}
        warn_if_truncated(
            choice,
            request,
            msg.get("content") or "",
            sent_max_tokens=self._clip_max_tokens(request.max_tokens),
            provider=self.PROVIDER,
            logger=logger,
        )
        arguments = self._parse_tool_arguments(msg, request.schema_)
        # Arguments must be a mapping. Unwrap a single-element mapping list as a final
        # safeguard; reject other shapes with LLMValidationError rather than leaking a Pydantic
        # validation error.
        if not isinstance(arguments, dict):
            coerced = self._unwrap_singleton_list_arg(arguments)
            if not isinstance(coerced, dict):
                logger.warning(
                    "%s: tool-call arguments are not a dict (%s) — %r",
                    self.PROVIDER, type(arguments).__name__, str(arguments)[:300],
                )
                raise LLMValidationError(
                    f"{self.PROVIDER}: tool-call arguments are not a dict "
                    f"({type(arguments).__name__}), expected a function argument object"
                )
            arguments = coerced
        # Scan both serialized tool arguments and textual content for canary leakage.
        # Batch results skip this: the scan belongs to the live conversation, and a
        # stored line must not grow a warning the interactive client would not have
        # attached to that payload on the batch path.
        if scan_canary:
            self._check_response_canary(msg.get("content") or "", context="generate_structured.text")
            try:
                self._check_response_canary(
                    json.dumps(arguments, ensure_ascii=False, default=str),
                    context="generate_structured.args",
                )
            except Exception:
                pass
        return LLMResponse(
            arguments=arguments,
            text=msg.get("content") or None,
            reasoning_content=self._reasoning_content_of(msg),
            finish_reason=finish_reason_opt(data),
            model=str(data.get("model", self._effective_model(request))),
            usage=self._usage(data),
            raw=data,
        )

    async def _serve_text_tier(
        self, request: LLMRequest, fn: str, *, reason: str, explicit: bool = False,
    ) -> LLMResponse:
        """The single entry point for text emulation. explicit=True is reserved for requested text
        mode via LLM_DISABLE_TOOLS (or an explicit text preference in preserve mode). Implicit fallback must
        fail visibly in no-degrade measurements.
        """
        if (self._no_degrade or self._preserve_responses) and not explicit:
            raise LLMValidationError(
                f"{self.PROVIDER}: native function-calling failed ({reason}); "
                "LLM_NO_DEGRADE or fallback_policy=preserve forbids silent text fallback "
                "(strict measurement mode)"
            )
        if not explicit:
            logger.info(
                # Use one consistent served-tier marker across all modes; keep the mode a
                # separate token for log consumers.
                "%s: structured served by tier text (%s)", self._route_id_for(request), reason,
            )
        self._last_served_tier = "text"
        return self._structured_response(
            await self._post(self._text_json_body(request, fn)), request
        )

    async def _degrade_on_gateway_5xx(
        self, request: LLMRequest, fn: str, exc: "OpenAIError", *, tier: str,
    ) -> LLMResponse | None:
        """Try text once after a tool request gateway 5xx. Repeating the same tool payload cannot
        distinguish an unavailable gateway from payload-specific rejection. Preserve the
        no-degrade policy and avoid permanently caching a transient 5xx as a capability
        restriction.
        """
        if self._no_degrade or self._preserve_responses or not _is_gateway_5xx(exc):
            return None
        _code = getattr(exc, "status_code", "?")
        logger.warning(
            "%s: tier %s — gateway returned %s for a tool request, retries "
            "exhausted with the same payload. One attempt through the text tier: it omits "
            "tools and therefore distinguishes «gateway outage» and «gateway rejected "
            "tool payload» (route %s)",
            self.PROVIDER, tier, _code, self._route_id_for(request),
        )
        return await self._serve_text_tier(
            request, fn, reason=f"gateway {_code} on tool request (tier {tier})",
        )

    def _response_format_body(self, request: LLMRequest, fn: str) -> dict[str, Any]:
        """Build the native response_format tier. json_schema sends the schema in the body;
        json_object guarantees only valid JSON, so its prompt must still describe the required
        object.
        """
        json_schema = self._response_format == "json_schema" and bool(request.schema_)
        if json_schema:
            # Do not duplicate the schema in the prompt; response_format already carries it.
            hint = "\n\nReturn JSON according to the supplied schema."
        else:
            hint = (f"\n\nReturn ONLY a JSON object for function `{fn}` according to its schema, "
                    "without Markdown wrapping or explanations.")
            if request.schema_:
                try:
                    hint += "\nSCHEMA (use exactly these keys and types):\n" + json.dumps(
                        request.schema_, ensure_ascii=False,
                    )
                except (TypeError, ValueError):
                    pass  # If schema serialization fails, retain the instruction without the
                          # schema.
        instructed = request.model_copy(update={"system": request.system + hint})
        if json_schema:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": fn, "strict": True,
                                "schema": self._adapt_schema(request.schema_)},
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self._effective_model(request),
            "messages": self._messages(instructed),
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            "response_format": response_format,
            **self._reasoning_body_kwargs(request),
        }

    async def _serve_response_format_tier(
        self, request: LLMRequest, fn: str, *, reason: str,
    ) -> LLMResponse:
        """Serve structured output through the declared response_format dialect. Parse
        message.content and still validate required schema fields before considering the tier
        successful.
        """
        # Log entry into this tier separately from successful service. A schema-validation
        # failure may fall through to text and must not be counted as a response_format success.
        logger.info(
            "%s: structured entering tier response_format (dialect %s) (%s)",
            self._route_id_for(request), self._response_format, reason,
        )
        data = await self._post_with_length_retry(
            self._response_format_body(request, fn)
        )
        result = self._structured_response(data, request)
        if self._validate_schema and not _args_satisfy_schema(result.arguments, request.schema_):
            raise LLMValidationError(
                f"response response_format={self._response_format} does not conform to "
                f"the function schema '{fn}'"
            )
        self._last_served_tier = "response_format"
        logger.info(
            # Keep the tier name separate from its dialect so all served-tier log lines share
            # one format.
            "%s: structured served by tier response_format (dialect %s)",
            self._route_id_for(request), self._response_format,
        )
        return result

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate structured arguments using the strongest supported tier. Try native multi-tool
        or single-function forms as appropriate, then declared response_format and permitted
        text emulation. Preserve explicit mode selection, no-degrade measurements, schema
        validation, and the tier that actually served the response.
        """
        # In multi-tool mode, let the model choose from request.tools. If it instead ignores
        # tools and emits JSON content, learn that behavior and fall back to the forced
        # single-function schema.
        if request.tools and request.tools_required:
            # Reject a required native tool loop immediately when LLM_DISABLE_TOOLS is set; let
            # the caller select text emulation without a wasted HTTP request.
            if not self._tools_enabled:
                raise LLMValidationError(
                    f"{self.PROVIDER}: native tool-loop (tools_required) "
                    "is unavailable — tools are disabled (LLM_DISABLE_TOOLS)"
                )
            # Reject native tool loops whose schemas contain unsupported open objects before
            # HTTP. The caller must own emulation so it cannot mistake a client fallback for the
            # model's tool choice.
            if (not self._preserve_responses and self._open_objects_unsupported
                    and _request_has_open_object(request)):
                _paths = ", ".join(_request_open_object_paths(request)) or "?"
                raise LLMValidationError(
                    f"{self.PROVIDER}: native tool-loop (tools_required) "
                    f"is unavailable — route {self._route_id_for(request)} does not accept "
                    f"open object schemas at nodes [{_paths}] "
                    f"({self._open_objects_source()})"
                )
            # On a native-loop open-object 400, learn the restriction and raise
            # LLMValidationError with offending paths. Do not silently replace that turn with
            # text emulation.
            try:
                resp = await self._generate_structured_multi(
                    request, strict_tools=True,
                )
            except OpenAIError as exc:
                if self._preserve_responses or not _is_open_object_schema_rejected(exc):
                    raise
                self._open_objects_unsupported = True
                _paths = ", ".join(_request_open_object_paths(request)) or "?"
                logger.warning(
                    "%s: native tool-loop — gateway rejected an open object in the "
                    "schema (400) at nodes [%s]; rejecting the request, no substitution with "
                    "text emulation (decision cached for route %s)",
                    self.PROVIDER, _paths, self._route_id_for(request),
                )
                raise LLMValidationError(
                    f"{self.PROVIDER}: native tool-loop (tools_required) "
                    f"is unavailable — route {self._route_id_for(request)} does not accept "
                    f"open object schemas at nodes [{_paths}]"
                ) from exc
            # Tool calls indicate a native function-call turn; a response without them is the
            # model's text choice, not the client's text fallback tier.
            self._last_served_tier = "multi" if resp.tool_calls else "multi-text-turn"
            return resp

        # Check declared or learned open-object restrictions before the multi-tool tier. Every
        # schema-bearing tier would otherwise pay for the same unsupported schema.
        if (not self._preserve_responses and self._open_objects_unsupported
                    and _request_has_open_object(request)):
            # Explicit LLM_DISABLE_TOOLS takes precedence over the open-object bypass and must
            # remain valid under LLM_NO_DEGRADE.
            if not self._tools_enabled:
                return await self._serve_text_tier(
                    request, request.function_name or "build_artifact",
                    reason="LLM_DISABLE_TOOLS", explicit=True,
                )
            # Warn even when a declaration avoids the failing round trip: bypassing the schema
            # does not fix it, and operators still need the offending paths.
            logger.warning(
                "%s: open object schemas at nodes [%s] are unsupported by route %s "
                "(%s); request uses the text tier without attempting "
                "tool tiers",
                self.PROVIDER,
                ", ".join(_request_open_object_paths(request)) or "?",
                self._route_id_for(request),
                self._open_objects_source(),
            )
            # Do not pass explicit=True for an implicit schema fallback. That flag belongs only
            # to LLM_DISABLE_TOOLS; otherwise catalog declarations would silently bypass
            # LLM_NO_DEGRADE.
            return await self._serve_text_tier(
                request, request.function_name or "build_artifact",
                reason=("route does not accept open object schemas "
                        f"({self._open_objects_source()})"),
            )

        if request.tools and self._multitool_supported is not False:
            # Corrupt arguments can be recovered through a different tier: some gateways corrupt
            # auto calls but return clean forced calls. Preserve the cause so diagnostics
            # distinguish corruption from a missing tool call.
            corrupted: ToolArgsCorruptedError | None = None
            try:
                resp = await self._generate_structured_multi(
                    request, strict_tools=self._preserve_responses,
                )
            except ToolArgsCorruptedError as exc:
                # Repeating the same deterministic request reproduces the corrupt response; try
                # the forced tier instead. Under LLM_NO_DEGRADE, propagate corruption to keep
                # capability measurements honest. Unlike strict-to-required schema negotiation,
                # recovery here would conceal a damaged response. Apply this rule consistently
                # to auto and forced tiers.
                if self._no_degrade or self._preserve_responses:
                    raise
                corrupted = exc
                resp = None
            except OpenAIError as exc:
                if _is_open_object_schema_rejected(exc):
                    if self._preserve_responses:
                        raise
                    # Cache the open-object restriction only for schemas containing these nodes.
                    # Disabling multi-tool globally would penalize unrelated schemas.
                    self._open_objects_unsupported = True
                    # Warn with the route and specific schema nodes: silently bypassing an
                    # invalid strict schema would conceal the original defect.
                    logger.warning(
                        "%s: gateway rejected an open object schema (400) on "
                        "nodes [%s] — fallback structured -> text (decision "
                        "cached for route %s). Schema sanitization is NOT "
                        "applied: measured output was an empty object. "
                        "Persist this restriction by declaring "
                        "open_object_schemas: unsupported for the route in "
                        "models.json",
                        self.PROVIDER,
                        ", ".join(_request_open_object_paths(request)) or "?",
                        self._route_id_for(request),
                    )
                    return await self._serve_text_tier(
                        request, request.function_name or "build_artifact",
                        reason="gateway rejected an open object schema (multi-tool)",
                    )
                _degraded = await self._degrade_on_gateway_5xx(
                    request, request.function_name or "build_artifact", exc,
                    tier="multi-tool",
                )
                if _degraded is not None:
                    return _degraded
                if not _is_tool_parser_unsupported_error(exc):
                    raise
                # The server cannot build a tool parser from its template. Both forced tiers
                # send tools and would fail too, so cache the restriction and switch directly to
                # text.
                self._multitool_supported = False
                self._tool_choice_pref = "text"
                logger.info(
                    "%s: tool template parser cannot be built (400 jinja grammar) — "
                    "fallback structured → text mode",
                    self._route_id_for(request),
                )
                resp = None
            if resp is not None:
                if self._multitool_supported is None:
                    self._multitool_supported = True
                self._last_served_tier = "multi" if resp.tool_calls else "multi-text-turn"
                return resp
            if self._tool_choice_pref == "text":
                # Record the actual cause: template parser construction and argument corruption
                # are different failures and must remain distinguishable in logs and
                # measurements.
                return await self._serve_text_tier(
                    request, request.function_name or "build_artifact",
                    reason=(f"corrupt arguments on multi-tool ({corrupted})"
                            if corrupted
                            else "tool template parser cannot be built (multi-tool)"),
                )
            # Do not cache a missing auto tool call as unsupported multi-tool capability: it may
            # be specific to this prompt. Fall back for this call only; the next request must
            # try multi-tool again. Deterministic parser failures are detected and cached
            # separately above.
            logger.info(
                "%s: multi-tool (tool_choice=auto) %s — "
                "falling back to a single forced function '%s' (this call only, "
                "not cached)",
                self._route_id_for(request),
                (f"returned unparseable arguments ({corrupted})" if corrupted
                 else "did not return tool_call"),
                request.function_name,
            )

        fn = request.function_name or "build_artifact"
        schema = request.schema_ or {"type": "object"}

        # LLM_DISABLE_TOOLS=true explicitly selects text mode without traversing the fallback
        # ladder.
        if not self._tools_enabled:
            return await self._serve_text_tier(
                request, fn, reason="LLM_DISABLE_TOOLS", explicit=True,
            )

        # Start at the learned tier, or strict. Insert response_format only when the route
        # declares its dialect; sending it speculatively can make an otherwise valid request
        # fail.
        order = (["strict", "required", "response_format", "text"]
                 if self._response_format else ["strict", "required", "text"])
        if self._tool_choice_pref == "auto":
            order.insert(0, "auto")
        if self._tool_choice_pref in order:
            order = order[order.index(self._tool_choice_pref):]

        for idx, mode in enumerate(order):
            if mode == "response_format":
                try:
                    return await self._serve_response_format_tier(
                        request, fn, reason="fallback from tool tiers",
                    )
                except (OpenAIError, LLMValidationError) as exc:
                    if self._preserve_responses:
                        raise
                    logger.info(
                        "%s: tier response_format did not produce a parseable object "
                        "(%s) — fallback → text", self._route_id_for(request), exc,
                    )
                    self._tool_choice_pref = "text"
                    return await self._serve_text_tier(
                        request, fn,
                        reason=f"response_format did not produce a parseable object ({exc})",
                    )
            if mode == "text":
                if self._tool_choice_pref != "text":
                    self._tool_choice_pref = "text"
                return await self._serve_text_tier(
                    request, fn, reason="tool_choice is unsupported in all forms",
                    explicit=self._preserve_responses and self._text_requested,
                )
            tool_choice: Any = (
                {"type": "function", "function": {"name": fn}}
                if mode == "strict" else mode
            )
            try:
                data = await self._post_with_length_retry(self._fc_body(request, fn, schema, tool_choice))
            except OpenAIError as exc:
                if _is_open_object_schema_rejected(exc):
                    if self._preserve_responses:
                        raise
                    self._open_objects_unsupported = True
                    logger.warning(
                        "%s: tier %s — gateway rejected an open object schema "
                        "(400) at nodes [%s], fallback -> text (decision "
                        "cached for route %s)",
                        self.PROVIDER, mode,
                        ", ".join(_request_open_object_paths(request)) or "?",
                        self._route_id_for(request),
                    )
                    return await self._serve_text_tier(
                        request, fn,
                        reason="gateway rejected an open object schema "
                               f"(tier {mode})",
                    )
                if (_is_tool_choice_unsupported_error(exc)
                        or _is_tool_choice_param_reject(exc)
                        or _is_tool_choice_thinking_conflict_error(exc)):
                    next_mode = order[idx + 1] if idx + 1 < len(order) else "text"
                    logger.info(
                        "%s: tool_choice=%s rejected (%s), fallback → %s",
                        self._route_id_for(request), mode,
                        "unsupported tool_choice value" if (
                            _is_tool_choice_unsupported_error(exc) or _is_tool_choice_param_reject(exc)
                        )
                        else "400: incompatible with thinking enabled",
                        next_mode,
                    )
                    self._tool_choice_pref = next_mode
                    continue
                if _is_tool_parser_unsupported_error(exc):
                    # A template parser failure affects both tool tiers; switch directly to
                    # text.
                    logger.info(
                        "%s: tool_choice=%s — tool template parser cannot be built "
                        "(400 jinja grammar), fallback → text mode",
                        self._route_id_for(request), mode,
                    )
                    self._tool_choice_pref = "text"
                    return await self._serve_text_tier(
                        request, fn,
                        reason="tool template parser cannot be built (400 jinja grammar)",
                    )
                if _is_peg_format_error(exc):
                    # The server's PEG parser failed while processing tool output. Both forced
                    # tiers use tools, so cache the restriction to avoid repeating the same 500
                    # and retry cycle.
                    logger.info(
                        "%s: tool_choice=%s — PEG parser could not handle the volume of "
                        "tool-call output (500 peg-format), fallback → text mode",
                        self._route_id_for(request), mode,
                    )
                    self._tool_choice_pref = "text"
                    return await self._serve_text_tier(
                        request, fn,
                        reason="PEG parser could not handle the output volume (500 peg-format)",
                    )
                _degraded = await self._degrade_on_gateway_5xx(
                    request, fn, exc, tier=mode,
                )
                if _degraded is not None:
                    # Do not cache text preference after an ambiguous 5xx. Unlike deterministic
                    # template or PEG parser failures, a transient outage does not establish a
                    # permanent payload restriction.
                    return _degraded
                raise  # Propagate substantive errors to the caller.
            try:
                result = self._structured_response(data, request)
                if self._validate_schema and not _args_satisfy_schema(result.arguments, request.schema_):
                    # Parsed JSON may still violate the function schema. Follow the same
                    # fallback path as unparseable output instead of returning invalid arguments
                    # as a successful response.
                    raise LLMValidationError(
                        f"structured args do not conform to the function schema "
                        f"'{fn}' (tool_choice={mode})"
                    )
            except LLMValidationError as exc:
                # Propagate corrupt gateway arguments under LLM_NO_DEGRADE on forced tiers as
                # well as auto. Changing tool_choice would conceal response corruption in
                # capability measurements.
                if self._preserve_responses or (self._no_degrade and isinstance(exc, ToolArgsCorruptedError)):
                    raise
                # A 200 response can silently ignore forced tool_choice and contain neither tool
                # calls nor valid JSON. The HTTP fallback ladder cannot detect this, so advance
                # the same cached tier preference here.
                next_mode = order[idx + 1] if idx + 1 < len(order) else "text"
                logger.info(
                    "%s: tool_choice=%s accepted (200), but the model did not return "
                    "recognizable tool_call/JSON (%s) — fallback → %s",
                    self._route_id_for(request), mode, exc, next_mode,
                )
                if next_mode == "text":
                    self._tool_choice_pref = "text"
                    return await self._serve_text_tier(
                        request, fn,
                        reason=f"tool_choice={mode} accepted, but tool_call/JSON "
                               f"not recognized ({exc})",
                    )
                self._tool_choice_pref = next_mode
                continue
            # Log the serving tier on every successful request, even when its preference was
            # already updated before continue. Otherwise a silent fallback hides the tier
            # actually used from measurements.
            self._tool_choice_pref = mode
            self._last_served_tier = mode
            # Identify the full provider/model route: a provider label alone cannot distinguish
            # models sharing a gateway.
            logger.info("%s: structured served by tier %s", self._route_id_for(request), mode)
            return result

        # Unreachable because text is terminal; retained defensively.
        return await self._serve_text_tier(request, fn, reason="fallback ladder exhausted")

    @staticmethod
    def _unwrap_json_strings(obj: Any) -> Any:
        """Recursively decode string values containing JSON objects or arrays. Some
        function-calling models serialize nested objects as escaped JSON strings, causing
        validation to fail. Leave scalar JSON and ordinary text strings unchanged.
        """
        if isinstance(obj, dict):
            return {k: OpenAIClient._unwrap_json_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [OpenAIClient._unwrap_json_strings(v) for v in obj]
        if isinstance(obj, str):
            s = obj.strip()
            if s and s[0] in "{[":
                try:
                    parsed = json.loads(s)
                except (ValueError, TypeError):
                    return obj
                if isinstance(parsed, (dict, list)):
                    return OpenAIClient._unwrap_json_strings(parsed)
        return obj

    @staticmethod
    def _envelope_function_name(obj: Any) -> str | None:
        """Extract the function name from a narrow {name, arguments} content envelope. Preserve the
        model's selected tool when a gateway places its call in content instead of tool_calls.
        Accept only keys from this pair and dictionary arguments.
        """
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("arguments"), dict)
            and set(obj.keys()) <= {"name", "arguments"}
        ):
            name = obj.get("name")
            return str(name) if name else None
        return None

    @staticmethod
    def _unwrap_function_call_envelope(obj: Any) -> Any:
        """Unwrap a function-call content envelope into its arguments. Servers without a tool
        parser may emit {name, arguments} in content. Accept only this narrow key shape with
        dictionary arguments; leave ordinary argument objects unchanged.
        """
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("arguments"), dict)
            and set(obj.keys()) <= {"name", "arguments"}
        ):
            return obj["arguments"]
        return obj

    @staticmethod
    def _unwrap_singleton_list_arg(obj: Any) -> Any:
        """Unwrap a singleton list containing one argument object. Leave multi-element lists and
        lists of non-objects unchanged so genuine shape errors still fail validation.
        """
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        return obj

    async def _generate_structured_multi(
        self, request: LLMRequest, *, strict_tools: bool = False,
    ) -> LLMResponse | None:
        """Let the model select a function from request.tools with tool_choice=auto. Return the
        chosen name and parsed arguments, or None when no call is returned so the caller can try
        a forced function. With strict_tools, return the original text and empty tool_calls
        instead; the native loop owns nudging and fallback.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", f"Build {t['name']}."),
                    "parameters": self._adapt_schema(
                        t.get("parameters") or {"type": "object"}
                    ),
                },
            }
            for t in request.tools or []
        ]
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": self._messages(request),
            "tools": tools,
            "tool_choice": "required" if self._tool_choice_pref == "required" else "auto",
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
        }
        # As in _fc_body, disable hybrid thinking for the same native function-calling
        # compatibility constraint.
        self._apply_thinking_off_for_tools(body)
        data = await self._post_with_length_retry(body)
        choice = data["choices"][0] or {}
        msg = choice.get("message") or {}
        warn_if_truncated(
            choice,
            request,
            msg.get("content") or "",
            sent_max_tokens=self._clip_max_tokens(request.max_tokens),
            provider=self.PROVIDER,
            logger=logger,
        )
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Before treating absent tool_calls as a missing selection, inspect a named content
            # envelope. Otherwise forcing the fallback function can replace the tool the model
            # actually selected and conceal a routing error.
            envelope_name: str | None = None
            envelope_args: dict[str, Any] = {}
            if not strict_tools:
                try:
                    parsed_envelope = extract_json_from_text(msg.get("content") or "")
                    envelope_name = self._envelope_function_name(parsed_envelope)
                    if envelope_name:
                        unwrapped = self._unwrap_function_call_envelope(parsed_envelope)
                        envelope_args = (
                            self._unwrap_json_strings(unwrapped)
                            if isinstance(unwrapped, dict) else {}
                        )
                except Exception:
                    envelope_name = None
            if envelope_name:
                logger.info(
                    "%s: multi-tool: function name '%s' extracted from "
                    "content envelope (tool_calls is empty)",
                    self.PROVIDER, envelope_name,
                )
                self._check_response_canary(
                    msg.get("content") or "",
                    context="generate_structured.multi.envelope",
                )
                return LLMResponse(
                    arguments=envelope_args,
                    text=msg.get("content") or None,
                    function_name=envelope_name,
                    tool_calls=[{
                        "id": "", "name": envelope_name, "arguments": envelope_args,
                    }],
                    reasoning_content=self._reasoning_content_of(msg),
                    finish_reason=finish_reason_opt(data),
                    model=str(data.get("model", self._effective_model(request))),
                    usage=self._usage(data),
                    raw=data,
                )
            # No function was selected; signal fallback.
            logger.info(
                "%s: multi-tool: model did not call a function "
                "(finish=%r, content_len=%d)",
                self.PROVIDER, choice.get("finish_reason"),
                len(msg.get("content") or ""),
            )
            if not strict_tools:
                return None
            # With strict_tools, preserve the original turn without synthesizing tool calls from
            # content. The native loop decides whether to nudge or fall back.
            self._check_response_canary(
                msg.get("content") or "", context="generate_structured.multi.text",
            )
            return LLMResponse(
                text=msg.get("content") or None,
                reasoning_content=self._reasoning_content_of(msg),
                finish_reason=finish_reason_opt(data),
                model=str(data.get("model", self._effective_model(request))),
                usage=self._usage(data),
                raw=data,
            )
        chosen = (tool_calls[0] or {}).get("function") or {}
        fn_name = chosen.get("name") or ""
        if not fn_name:
            # Some incomplete tool parsers return a call with an empty function name and put the
            # name in content. Recover it so dispatch can distinguish correct and incorrect
            # selections.
            try:
                fn_name = self._envelope_function_name(
                    extract_json_from_text(msg.get("content") or "")
                ) or ""
            except Exception:
                fn_name = ""
        # Preserve every parallel tool call so the native loop can execute each and respond by
        # id. Dropping calls makes the next turn's history inconsistent. Keep arguments and
        # function_name as the first call for single-shot compatibility.
        all_calls: list[dict[str, Any]] = [
            {
                "id": (tc or {}).get("id") or "",
                "name": ((tc or {}).get("function") or {}).get("name") or "",
                "arguments": self._tool_call_arguments(tc),
            }
            for tc in tool_calls
        ]
        # For native loops, reuse the leniently parsed first call: empty arguments are valid for
        # tools without required parameters and must not fail the whole turn. Other callers
        # retain strict parsing and content JSON recovery.
        arguments = (
            all_calls[0]["arguments"]
            if strict_tools
            else self._unwrap_json_strings(self._parse_tool_arguments(msg))
        )
        self._check_response_canary(msg.get("content") or "", context="generate_structured.multi.text")
        try:
            self._check_response_canary(
                json.dumps(arguments, ensure_ascii=False, default=str),
                context="generate_structured.multi.args",
            )
        except Exception:
            pass
        return LLMResponse(
            arguments=arguments,
            text=msg.get("content") or None,
            function_name=fn_name,
            tool_calls=all_calls,
            reasoning_content=self._reasoning_content_of(msg),
            finish_reason=finish_reason_opt(data),
            model=str(data.get("model", self._effective_model(request))),
            usage=self._usage(data),
            raw=data,
        )

    def _tool_call_arguments(self, tool_call: dict[str, Any] | None) -> dict[str, Any]:
        """Parse a parallel call; preservation never substitutes corrupt arguments with {}."""
        raw_args = ((tool_call or {}).get("function") or {}).get("arguments")
        if raw_args is None or raw_args == "":
            return {}
        if isinstance(raw_args, dict):
            parsed: Any = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = extract_json_from_text(raw_args)
            except Exception as exc:
                if self._preserve_responses:
                    raise ToolArgsCorruptedError(
                        f"{self.PROVIDER}: invalid JSON in tool-call arguments: {exc}"
                    ) from exc
                return {}
        else:
            parsed = raw_args
        unwrapped = self._unwrap_json_strings(parsed)
        if isinstance(unwrapped, dict):
            return cast("dict[str, Any]", unwrapped)
        if self._preserve_responses:
            raise ToolArgsCorruptedError(f"{self.PROVIDER}: tool-call arguments must be an object")
        return {}

    def _merge_multi_tool_call_arguments(
        self, tool_calls: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Merge parallel calls to the same function into a single schema array. Some providers
        emit one array item per call; reading only the first loses valid output. Apply only when
        the schema has exactly one top-level array field, otherwise return None rather than
        guess a destination.
        """
        props = schema.get("properties") or {}
        array_keys = [k for k, v in props.items()
                      if isinstance(v, dict) and v.get("type") == "array"]
        if len(array_keys) != 1:
            return None
        key = array_keys[0]
        items: list[Any] = []
        for tc in tool_calls:
            raw = ((tc or {}).get("function") or {}).get("arguments")
            if isinstance(raw, str):
                try:
                    raw = extract_json_from_text(raw)
                except Exception:
                    continue
            if not isinstance(raw, dict):
                continue
            val = raw.get(key)
            if isinstance(val, list):
                items.extend(val)
            else:
                items.append(raw)
        return {key: items} if items else None

    def _recover_array_from_content(
        self, parsed_args: dict[str, Any], schema: dict[str, Any] | None,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Recover a full JSON array from content when tool arguments contain only its first
        unwrapped item. Apply only when the schema has exactly one top-level array field, that
        field is absent from arguments, and content parses as a list. Preserve already wrapped
        arguments.
        """
        if not schema:
            return None
        props = schema.get("properties") or {}
        array_keys = [k for k, v in props.items()
                      if isinstance(v, dict) and v.get("type") == "array"]
        if len(array_keys) != 1:
            return None
        key = array_keys[0]
        if key in parsed_args:
            return None
        content = message.get("content") or ""
        if not content:
            return None
        try:
            candidate = extract_json_array_from_text(content)
        except Exception:
            return None
        if not candidate:
            return None
        return {key: candidate}

    def _parse_tool_arguments(
        self, message: dict[str, Any], schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract arguments from modern tool_calls, legacy function_call, or JSON content, in that
        order. Raise LLMValidationError for invalid or missing arguments. Recursively decode
        nested JSON object/array strings emitted by some function-calling models.
        """
        raw_args: Any = None
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1 and schema:
            merged = self._merge_multi_tool_call_arguments(tool_calls, schema)
            if merged is not None:
                return cast("dict[str, Any]", self._unwrap_json_strings(merged))
        if tool_calls:
            raw_args = ((tool_calls[0] or {}).get("function") or {}).get("arguments")
        elif message.get("function_call"):
            raw_args = (message.get("function_call") or {}).get("arguments")

        if raw_args is not None:
            if isinstance(raw_args, dict):
                result = cast("dict[str, Any]", self._unwrap_json_strings(raw_args))
                recovered = self._recover_array_from_content(result, schema, message)
                return recovered if recovered is not None else result
            # The provider already parsed raw_args as a list; unwrap a singleton object.
            if isinstance(raw_args, list):
                raw_args = self._unwrap_singleton_list_arg(raw_args)
                if isinstance(raw_args, dict):
                    return cast("dict[str, Any]", self._unwrap_json_strings(raw_args))
            if isinstance(raw_args, str):
                try:
                    parsed = self._unwrap_singleton_list_arg(extract_json_from_text(raw_args))
                    result = cast("dict[str, Any]", self._unwrap_json_strings(parsed))
                    recovered = self._recover_array_from_content(result, schema, message)
                    return recovered if recovered is not None else result
                except Exception as exc:
                    raise ToolArgsCorruptedError(
                        f"{self.PROVIDER}: invalid JSON in tool-call arguments: {exc}"
                    ) from exc

        # Recover JSON from content when no function call was returned, including a narrow
        # {name, arguments} envelope from servers without tool parsers.
        content = message.get("content") or ""
        try:
            parsed = self._unwrap_function_call_envelope(extract_json_from_text(content))
            parsed = self._unwrap_singleton_list_arg(parsed)
            result = cast("dict[str, Any]", self._unwrap_json_strings(parsed))
            recovered = self._recover_array_from_content(result, schema, message)
            return recovered if recovered is not None else result
        except Exception as exc:
            # When the schema has exactly one top-level array field, wrap a bare content array
            # under that field. Use the same narrow trigger as native argument recovery.
            recovered = self._recover_array_from_content({}, schema, message)
            if recovered is not None:
                return cast("dict[str, Any]", self._unwrap_json_strings(recovered))
            raise LLMValidationError(
                f"{self.PROVIDER}: missing tool_call/function_call and could not "
                f"extract JSON from content ({content[:200]!r}): {exc}"
            ) from exc

    async def generate_stream_events(
        self, request: LLMRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed content, reasoning, tool, completion, and error events. Preserve delta
        order. Emit one ToolUseStart per tool index, zero or more argument deltas, then
        ToolUseStop events and exactly one Complete after DONE. Emit Error before re-raising
        transport exceptions so consumers can observe stream interruption. Request construction
        follows the text streaming path.
        """
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            **self._reasoning_body_kwargs(request),
            "stream": True,
        }
        for k, v in self._extra_body.items():
            if self._no_vendor_keys and k in VENDOR_BODY_KEYS:
                continue
            body.setdefault(k, v)
        if self._force_temperature is not None:
            body["temperature"] = self._force_temperature
        sem = self._ensure_semaphore()
        if sem is None:
            async for ev in self._do_stream_events(body):
                yield ev
            return
        async with sem:
            async for ev in self._do_stream_events(body):
                yield ev

    async def _do_stream_events(
        self, body: dict[str, Any]
    ) -> AsyncIterator[StreamEvent]:
        """Transport for generate_stream_events. Emit typed deltas and a terminal Complete; on
        network or HTTP failure, emit Error and re-raise the corresponding transport exception.
        """
        headers = {"Authorization": f"Bearer {self._key}", **self._extra_headers}
        self._apply_stream_usage(body)
        agg: list[str] = []
        retry_without_usage = False
        try:
            async with self._ensure_http().stream(
                "POST", self.URL, headers=headers, json=body
            ) as r:
                if r.status_code >= 400:
                    raw = (await r.aread()).decode("utf-8", "replace")[:300]
                    if self._drop_stream_usage(r.status_code, raw, body):
                        retry_without_usage = True
                    else:
                        if r.status_code in (401, 403):
                            err = LLMAuthError(
                                f"{self.PROVIDER} stream {r.status_code}: {raw}"
                            )
                        else:
                            err = OpenAIError(
                                f"{self.PROVIDER} stream {r.status_code}: {raw}"
                            )
                        yield Error(error=str(err), error_type=type(err).__name__)
                        raise err
                else:
                    request_id = r.headers.get("x-request-id")
                    tool_acc = ToolCallAccumulator()
                    finish_reason: str | None = None
                    usage: LLMUsage | None = None
                    async for payload in iter_sse_payloads(r.aiter_lines()):
                        # Remember finish_reason and usage from final chunks; emit exactly one
                        # terminal event after DONE.
                        choices = payload.get("choices")
                        choice = (
                            choices[0]
                            if isinstance(choices, list) and choices
                            else {}
                        )
                        if isinstance(choice, dict):
                            fr = choice.get("finish_reason")
                            if fr:
                                finish_reason = fr
                        if payload.get("usage"):
                            usage = LLMUsage.from_raw(
                                payload["usage"],
                                cache_hit_field=self._cache_hit_field,
                                cache_miss_field=self._cache_miss_field,
                                cache_nested_field=self._cache_nested_field,
                            )
                        for ev in events_from_sse_payload(
                            payload,
                            request_id=request_id,
                            tool_acc=tool_acc,
                            reasoning_field=self._reasoning_field,
                        ):
                            if isinstance(ev, ContentDelta):
                                agg.append(ev.delta_text)
                            yield ev
                    # After DONE, stop tool calls in first-seen index order, then emit Complete.
                    for stop in tool_acc.finalize(request_id=request_id):
                        yield stop
                    yield Complete(
                        finish_reason=finish_reason,
                        usage=usage,
                        request_id=request_id,
                    )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            err = LLMTimeoutError(f"{self.PROVIDER} stream transient error: {exc}")
            yield Error(error=str(err), error_type=type(err).__name__)
            raise err from exc
        finally:
            # Scan partial output for canaries even when the stream is interrupted.
            self._check_response_canary("".join(agg), context="generate_stream_events")
        if retry_without_usage:
            async for ev in self._do_stream_events(body):
                yield ev

    def _apply_thinking_off_for_tools(self, body: dict[str, Any]) -> None:
        """Apply reasoning_off to a function-calling body in place when disable_thinking_for_tools
        is set. Use only the route's declared dialect. If absent, warn once per instance and
        leave the body unchanged rather than guessing provider keys.
        """
        if not self._disable_thinking_for_tools:
            return
        if not self._reasoning_off_declared:
            if not self._dtft_no_off_warned:
                logger.warning(
                    "%s: disable_thinking_for_tools is set, but reasoning_off "
                    "is not declared for the route — skipping thinking-off",
                    self.PROVIDER,
                )
                self._dtft_no_off_warned = True
            return
        for key in self._reasoning_on_body:
            body.pop(key, None)
        body.update(self._reasoning_off_body)
