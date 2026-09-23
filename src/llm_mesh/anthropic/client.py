"""Async client for the Anthropic Messages API.

Text, streaming, and structured output share one transport. Structured calls use
native tool use; ``mode="json_schema"`` uses ``output_config.format``. Strict
tool use is sent only for schemas the grammar accepts, and a strict rejection
is retried once without it. Extended thinking keeps the requested answer
budget instead of leaving a single visible token. The default endpoint is
https://api.anthropic.com and the ``anthropic-version`` header is ``2023-06-01``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Literal

import httpx

from llm_mesh._common import (
    _env_flag,
    _env_float_default,
    _env_int_default,
    _env_is_disabled,
    _env_nonneg_int,
    _env_positive_int,
    _parse_json_dict_env,
    _args_satisfy_schema,
    apply_canary,
    warn_if_truncated,
)
from llm_mesh.base import BaseLLMClient, Capability
from llm_mesh._retry import (
    RETRYABLE_SERVER_STATUS,
    backoff_with_jitter,
    parse_retry_after,
    retry_after_delay,
)
from llm_mesh._streaming import iter_sse_payloads
from llm_mesh.config import get_env
from llm_mesh.stream_events import (
    Complete,
    ContentDelta,
    Error,
    ReasoningDelta,
    StreamEvent,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)
from llm_mesh.text_parsing import extract_json_from_text, looks_degenerate_repetition
from llm_mesh.types import (
    LLMAuthError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMTimeoutError,
    LLMUsage,
    LLMValidationError,
)

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# Anthropic rejects a thinking budget below this, and the budget must be
# strictly smaller than max_tokens.
_MIN_THINKING_BUDGET = 1024
_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 16000}

# Map Messages API stop reasons onto the shared finish_reason vocabulary.
# max_tokens becomes length so the shared truncation retry recognizes it.
# model_context_window_exceeded stays distinct: raising max_tokens cannot help.
_STOP_REASONS = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_RETRYABLE_STATUS = (429, 529, *RETRYABLE_SERVER_STATUS)
# A 400 is a structured-output rejection only when the body says the feature or
# the grammar is unsupported. A mention of "tools" or "strict" is not enough:
# that also appears on unrelated request errors.
_STRUCTURED_REJECTED_RE = re.compile(
    r"output_config|json_schema|input_schema|additionalProperties|"
    r"structured output|grammar|does not support|not supported",
    re.IGNORECASE,
)
_STRICT_REJECTED_RE = re.compile(
    r"\bstrict\b|additionalProperties|input_schema|grammar",
    re.IGNORECASE,
)
# Messages must start with a user turn. This is the shortest non-empty text
# the API accepts when history itself starts with the assistant.
_LEADING_USER_TEXT = "."
_MESSAGE_BLOCK_TYPES = {
    "text", "thinking", "redacted_thinking", "tool_use", "tool_result",
}


class AnthropicError(LLMError):
    """Messages API failure, with the HTTP status and response body when present."""

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
        self.retry_after: float | None = None


def messages_url(base_url: str) -> str:
    """Build ``<base>/v1/messages``, keeping a base that already ends in ``/v1``."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def map_stop_reason(reason: str | None) -> str | None:
    """Translate a Messages API stop reason. Unknown values pass through."""
    if not reason:
        return None
    return _STOP_REASONS.get(reason, reason)


def usage_from_anthropic(raw: Any) -> LLMUsage:
    """Normalize Messages API usage.

    ``input_tokens`` excludes cache reads and cache writes. Those counts are
    added into ``prompt_tokens``. A reported cache read becomes cache hits;
    the rest of the prompt is cache misses. Absent cache fields stay unknown
    (``-1``), distinct from a reported empty cache.
    """
    if not isinstance(raw, dict):
        return LLMUsage()

    def _int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    uncached = _int(raw.get("input_tokens"))
    output = _int(raw.get("output_tokens"))
    read_raw = raw.get("cache_read_input_tokens")
    create_raw = raw.get("cache_creation_input_tokens")
    read = _int(read_raw) if read_raw is not None else None
    created = _int(create_raw) if create_raw is not None else None
    prompt = uncached + (read or 0) + (created or 0)
    if read is None and created is None:
        hit, miss = -1, -1
    else:
        hit = read or 0
        miss = max(prompt - hit, 0)
    return LLMUsage(
        prompt_tokens=prompt,
        completion_tokens=output,
        total_tokens=prompt + output,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def text_of(payload: dict[str, Any]) -> str:
    """Join user-visible text blocks. Thinking blocks are not included."""
    return "".join(
        str(block.get("text") or "")
        for block in _content_blocks(payload)
        if block.get("type") == "text"
    )


def thinking_of(payload: dict[str, Any]) -> str | None:
    """Join extended-thinking blocks, or None when the response has none."""
    parts = [
        str(block.get("thinking") or "")
        for block in _content_blocks(payload)
        if block.get("type") == "thinking" and block.get("thinking")
    ]
    return "".join(parts) or None


def tool_calls_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Public tool calls: id, name, and a parsed arguments object.

    Unparseable ``input`` is a failed tool call, not an empty argument object.
    """
    calls: list[dict[str, Any]] = []
    for block in _content_blocks(payload):
        if block.get("type") != "tool_use":
            continue
        calls.append({
            "id": str(block.get("id") or ""),
            "name": str(block.get("name") or ""),
            "arguments": _parse_model_tool_input(block.get("input")),
        })
    return calls


def _parse_model_tool_input(raw: Any) -> dict[str, Any]:
    """Parse tool input from a model response. Absent input is an empty object."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMValidationError(
                f"tool input is not valid JSON: {raw[:200]!r}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise LLMValidationError(f"tool input is not a JSON object: {raw!r}")


def has_thinking_blocks(payload: dict[str, Any]) -> bool:
    """True when the message carries thinking or redacted thinking."""
    return any(
        block.get("type") in ("thinking", "redacted_thinking")
        for block in _content_blocks(payload)
    )


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_message_blocks(content: Any) -> bool:
    """True when content is already a list of Messages API blocks."""
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(block, dict) and block.get("type") in _MESSAGE_BLOCK_TYPES
            for block in content
        )
    )


def _assistant_tool_blocks(turn: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, call in enumerate(turn.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        raw_args = function.get("arguments", call.get("arguments", {}))
        blocks.append({
            "type": "tool_use",
            "id": str(call.get("id") or f"toolu_{index}"),
            "name": name,
            "input": _tool_input(raw_args),
        })
    return blocks


def _assistant_content(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant blocks for one history turn, preserving thinking signatures.

    A content list of Messages blocks is copied unchanged, including
    ``signature`` and ``redacted_thinking``. OpenAI-shaped tool calls are
    appended only when the blocks do not already contain ``tool_use``.
    """
    content = turn.get("content")
    if _is_message_blocks(content):
        blocks = [dict(block) for block in content]
    else:
        blocks = []
        text = content if isinstance(content, str) else str(content or "")
        if text:
            blocks.append({"type": "text", "text": text})
    if turn.get("tool_calls") and not any(block.get("type") == "tool_use" for block in blocks):
        blocks.extend(_assistant_tool_blocks(turn))
    return blocks


def build_messages(
    request: LLMRequest, *, user_suffix: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Split a shared request into a system string and alternating Messages.

    The system prompt is a top-level field, not a message. OpenAI tool turns
    become ``tool_use`` blocks followed by ``tool_result`` blocks on the next
    user message. An assistant ``content`` list of Messages blocks is sent
    unchanged, so a previous thinking block keeps its signature. Consecutive
    turns of the same role are merged. History that starts with the assistant
    gets a leading user turn, which the Messages API requires. Other history
    roles are dropped.
    """
    system = apply_canary(request.system)
    messages: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    def append(role: str, content: Any) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = (
                _as_blocks(messages[-1]["content"]) + _as_blocks(content)
            )
            return
        messages.append({"role": role, "content": content})

    def flush_tool_results() -> None:
        nonlocal tool_results
        if tool_results:
            append("user", tool_results)
            tool_results = []

    for turn in request.history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role == "tool":
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": str(turn.get("tool_call_id") or ""),
                "content": str(turn.get("content", "")),
            })
            continue
        flush_tool_results()
        if role == "assistant" and (
            turn.get("tool_calls") or _is_message_blocks(turn.get("content"))
        ):
            append("assistant", _assistant_content(turn))
        elif role in ("user", "assistant"):
            content = turn.get("content")
            if _is_message_blocks(content):
                append(role, [dict(block) for block in content])
            else:
                append(role, str(content or ""))
    flush_tool_results()
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": _LEADING_USER_TEXT})

    user = request.user
    if user_suffix:
        user = f"{user}\n\n{user_suffix}" if user else user_suffix
    if user or not messages or messages[-1]["role"] != "user":
        append("user", user)
    return system, messages


class AnthropicClient(BaseLLMClient):
    """Messages API client for text, streaming, and structured output.

    ``api_key`` authenticates with ``x-api-key``. An omitted key is read from
    ``LLM_API_KEY``. An omitted ``base_url`` is read from ``LLM_BASE_URL``,
    then the public Anthropic endpoint. ``fallback_policy="preserve"`` returns provider
    failures instead of emulating structured output as text. The default is
    ``"recover"``.
    """

    CAPABILITIES = frozenset({
        Capability.TEXT,
        Capability.STREAM,
        Capability.STREAM_EVENTS,
        Capability.STRUCTURED,
        # Native tool use. Strict grammar is sent only when the schema accepts it.
        Capability.TOOLS,
        # tool_choice any/auto selects among the request's tools.
        Capability.MULTI_TOOL,
        # tools_required keeps the native tool loop and does not text-emulate.
        Capability.TOOLS_REQUIRED,
        # mode="json_schema" uses output_config.format, not the text ladder.
        Capability.JSON_SCHEMA_MODE,
        Capability.LENGTH_RETRY,
        # POST /v1/messages/count_tokens. The probe uses this instead of a generation.
        Capability.COUNT_TOKENS,
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
        anthropic_version: str = ANTHROPIC_VERSION,
        no_degrade: bool | None = None,
        disable_thinking_for_tools: bool | None = None,
        tool_choice_pref: str | None = None,
        validate_schema: bool = True,
        fallback_policy: Literal["recover", "preserve"] = "recover",
        length_retry_cap: int = 32768,
    ) -> None:
        if fallback_policy not in ("recover", "preserve"):
            raise ValueError("fallback_policy must be 'recover' or 'preserve'")
        self._preserve_responses = fallback_policy == "preserve"
        base = base_url or get_env("LLM_BASE_URL") or ANTHROPIC_BASE_URL
        self.URL = messages_url(base)
        self.PROVIDER = label or get_env("LLM_PROVIDER_LABEL") or "anthropic"
        self._model = model or get_env("LLM_MODEL")
        self._key = api_key or get_env("LLM_API_KEY")
        if not self._key:
            raise AnthropicError(
                f"{self.PROVIDER}: missing API key (LLM_API_KEY is not set)"
            )
        self._version = anthropic_version
        self._extra_headers = dict(extra_headers or {})
        parsed_headers = _parse_json_dict_env(
            "LLM_EXTRA_HEADERS", logger=logger, include_error=False,
        )
        if parsed_headers is not None:
            self._extra_headers.update({str(k): str(v) for k, v in parsed_headers.items()})
        self._extra_body = _parse_json_dict_env(
            "LLM_EXTRA_BODY", logger=logger, include_error=False,
        ) or {}
        self._max_retries = _env_int_default("LLM_MAX_RETRIES", 3, logger=logger)
        self._retry_backoff_s = float(get_env("LLM_RETRY_BACKOFF_S", "1.0") or "1.0")
        self._max_concurrent = _env_positive_int("LLM_MAX_CONCURRENT")
        self._semaphore: asyncio.Semaphore | None = None
        self._max_output_tokens = _env_positive_int("LLM_MAX_OUTPUT_TOKENS")
        self._min_output_tokens = _env_positive_int("LLM_MIN_OUTPUT_TOKENS")
        # Zero is a valid retry budget. An invalid value keeps the default of 2.
        self._length_retries = _env_nonneg_int("LLM_LENGTH_RETRIES", 2)
        self._length_retry_cap = length_retry_cap
        cap = _env_positive_int("LLM_LENGTH_RETRY_CAP")
        if cap is not None:
            self._length_retry_cap = cap
        if verify is not None:
            self._verify = verify
        else:
            # No strip: a padded value was never treated as off.
            self._verify = not _env_is_disabled("LLM_VERIFY_SSL", default="1")
        self._tools_enabled = not _env_flag("LLM_DISABLE_TOOLS")
        self._disable_thinking_for_tools = (
            disable_thinking_for_tools
            if disable_thinking_for_tools is not None
            else _env_flag("LLM_DISABLE_THINKING_FOR_TOOLS")
        )
        self._reasoning_disabled = _env_flag("LLM_DISABLE_REASONING")
        self._default_effort = get_env("LLM_REASONING_EFFORT", "").strip().lower()
        pref = (
            tool_choice_pref
            if tool_choice_pref is not None
            else get_env("LLM_TOOL_CHOICE_PREF", "")
        ).strip().lower()
        self._tool_choice_pref = pref if pref in ("auto", "required", "text") else None
        self._validate_schema = validate_schema
        self._no_degrade = (
            no_degrade if no_degrade is not None else _env_flag("LLM_NO_DEGRADE")
        )
        self._force_temperature: float | None = None
        forced = get_env("LLM_FORCE_TEMPERATURE", "").strip()
        if forced:
            try:
                self._force_temperature = float(forced)
            except ValueError:
                logger.warning(
                    "LLM_FORCE_TEMPERATURE is not a number (%r) — ignoring", forced,
                )
        self._http_timeout = (
            http_timeout
            if http_timeout is not None
            else _env_float_default("LLM_HTTP_TIMEOUT", 600.0, logger=logger)
        )
        self._client: httpx.AsyncClient | None = None
        self._last_served_tier: str | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._key,
            "anthropic-version": self._version,
            "content-type": "application/json",
            **self._extra_headers,
        }

    def _clip_max_tokens(self, requested: int) -> int:
        value = max(int(requested), 1)
        if self._min_output_tokens is not None:
            value = max(value, self._min_output_tokens)
        if self._max_output_tokens is not None:
            value = min(value, self._max_output_tokens)
        return max(value, 1)

    def _finish(self, payload: dict[str, Any]) -> str | None:
        return map_stop_reason(
            str(payload.get("stop_reason") or "") or None
        )

    def _usage(self, payload: dict[str, Any]) -> LLMUsage:
        return usage_from_anthropic(payload.get("usage"))

    def _response(
        self,
        payload: dict[str, Any],
        *,
        arguments: dict[str, Any] | None = None,
        function_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        request: LLMRequest | None = None,
    ) -> LLMResponse:
        text = text_of(payload)
        payload.pop("_sent_max_tokens", None)
        # Provider-reported model wins. The fallback is the model this call
        # sent, not the instance model, when the caller passes the request.
        fallback = self._effective_model(request) if request is not None else self._model
        return LLMResponse(
            arguments=arguments or {},
            text=text or None,
            function_name=function_name,
            tool_calls=tool_calls or [],
            reasoning_content=thinking_of(payload),
            request_id=str(payload.get("id") or "") or None,
            finish_reason=self._finish(payload),
            model=str(payload.get("model") or fallback),
            usage=self._usage(payload),
            raw=payload,
        )

    def _effective_model(self, request: LLMRequest) -> str:
        """Request model, else the model configured on this client.

        The client model already includes LLM_MODEL when the constructor
        argument was empty.
        """
        return request.model or self._model

    def _base_body(
        self,
        request: LLMRequest,
        *,
        user_suffix: str = "",
        tools: bool = False,
    ) -> dict[str, Any]:
        system, messages = build_messages(request, user_suffix=user_suffix)
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "max_tokens": self._clip_max_tokens(request.max_tokens),
            "messages": messages,
        }
        if system:
            body["system"] = system
        if self._force_temperature is not None:
            body["temperature"] = self._force_temperature
        else:
            body["temperature"] = request.temperature
        for key, value in self._extra_body.items():
            body.setdefault(key, value)
        self._apply_thinking(body, request, tools=tools)
        return body

    def _output_limit(self) -> int:
        """Highest max_tokens thinking may raise the request to."""
        limit = self._length_retry_cap
        if self._max_output_tokens is not None:
            limit = min(limit, self._max_output_tokens)
        return limit

    def _place_thinking(self, body: dict[str, Any], budget: int) -> bool:
        """Keep a real answer beside the thinking budget.

        ``max_tokens`` is the caller's output budget and already includes any
        ceiling. Thinking tokens count against it, and the budget must be
        strictly smaller. When the budget already fits, it is left as requested.
        When it does not, ``max_tokens`` is raised by the thinking budget so
        the original answer size remains, up to the output limit. If that
        limit cannot hold the answer plus the minimum thinking budget,
        thinking stays off. The answer budget is not reduced to one token.
        """
        answer = int(body["max_tokens"])
        requested = budget
        if budget < _MIN_THINKING_BUDGET:
            budget = _MIN_THINKING_BUDGET
        if budget < answer:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return True
        limit = self._output_limit()
        if answer + budget <= limit:
            body["max_tokens"] = answer + budget
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return True
        room = limit - answer
        if room < _MIN_THINKING_BUDGET:
            logger.warning(
                "%s: max_tokens=%d cannot hold a thinking budget of %d "
                "beside the requested answer — extended thinking left off",
                self.PROVIDER, answer, _MIN_THINKING_BUDGET,
            )
            return False
        logger.info(
            "%s: thinking budget %d does not fit under the output limit "
            "— using %d and keeping %d tokens for the answer",
            self.PROVIDER, requested, room, answer,
        )
        body["max_tokens"] = answer + room
        body["thinking"] = {"type": "enabled", "budget_tokens": room}
        return True

    def _apply_thinking(
        self, body: dict[str, Any], request: LLMRequest, *, tools: bool,
    ) -> None:
        """Enable extended thinking for a configured effort.

        The Messages API requires temperature 1 while thinking is enabled.
        A ceiling that cannot hold the answer and the minimum budget leaves
        thinking off instead of shrinking the answer to one token.
        """
        if self._reasoning_disabled or (tools and self._disable_thinking_for_tools):
            body.pop("thinking", None)
            return
        effort = (request.reasoning_effort or self._default_effort or "").strip().lower()
        # An explicit thinking block (LLM_EXTRA_BODY) wins over the effort table.
        if isinstance(body.get("thinking"), dict):
            raw_budget = body["thinking"].get("budget_tokens")
            try:
                budget = int(raw_budget)
            except (TypeError, ValueError):
                budget = _MIN_THINKING_BUDGET
        else:
            budget = _THINKING_BUDGET.get(effort)
            if budget is None:
                return
        if not self._place_thinking(body, budget):
            body.pop("thinking", None)
            return
        if body.get("temperature") not in (None, 1, 1.0):
            logger.info(
                "%s: reasoning effort %s forces temperature=1 "
                "(requested %s)",
                self.PROVIDER, effort or "custom", body.get("temperature"),
            )
        body["temperature"] = 1

    def _warn_truncated(self, payload: dict[str, Any], request: LLMRequest) -> None:
        reason = self._finish(payload)
        if reason != "length":
            return
        sent = int(payload.get("_sent_max_tokens") or request.max_tokens)
        warn_if_truncated(
            {"finish_reason": "length"},
            request,
            text_of(payload),
            sent_max_tokens=sent,
            provider=self.PROVIDER,
            logger=logger,
        )

    async def _post(
        self,
        body: dict[str, Any],
        *,
        url: str | None = None,
        expect_message: bool = True,
    ) -> dict[str, Any]:
        headers = self._headers()
        sent = int(body.get("max_tokens") or 0)
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._ensure_http().post(
                    url or self.URL, headers=headers, json=body,
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                    httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    err_cls = (
                        LLMTimeoutError
                        if isinstance(exc, httpx.TimeoutException)
                        else AnthropicError
                    )
                    raise err_cls(
                        f"{self.PROVIDER} network error after "
                        f"{attempt + 1} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                delay = (
                    retry_after_delay(response, self._retry_backoff_s, attempt)
                    if response.status_code == 429
                    else backoff_with_jitter(self._retry_backoff_s, attempt)
                )
                logger.warning(
                    "%s %s on attempt %d/%d — retry in %.1fs",
                    self.PROVIDER, response.status_code, attempt + 1,
                    self._max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue

            detail = response.text[:4000] if response.text else "<empty>"
            if response.status_code in (401, 403):
                raise LLMAuthError(f"{self.PROVIDER} HTTP {response.status_code}: {detail}")
            if response.status_code >= 400:
                raise AnthropicError(
                    f"{self.PROVIDER} HTTP {response.status_code}: {detail}",
                    status_code=response.status_code,
                    detail=detail,
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise AnthropicError(
                    f"{self.PROVIDER} malformed JSON response: {exc} "
                    f"(body={response.text[:300]!r})"
                ) from exc
            if data.get("type") == "error" or (
                expect_message and not isinstance(data.get("content"), list)
            ):
                raise AnthropicError(
                    f"{self.PROVIDER} response is not a message: {data!r}"
                )
            if not expect_message:
                return data
            data["_sent_max_tokens"] = sent
            if self._should_retry_empty(data, attempt):
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
                continue
            return data
        raise AnthropicError(f"{self.PROVIDER} retry exhausted: {last_exc}")

    def _should_retry_empty(self, data: dict[str, Any], attempt: int) -> bool:
        # Thinking is content. A turn that only reasoned is not an empty response.
        if text_of(data).strip() or tool_calls_of(data) or has_thinking_blocks(data):
            return False
        reason = self._finish(data)
        if reason in ("length", "content_filter"):
            logger.warning(
                "%s: empty content with finish_reason=%s — returning as is",
                self.PROVIDER, reason,
            )
            return False
        if attempt >= self._max_retries:
            return False
        logger.warning(
            "%s empty content on attempt %d/%d — retry",
            self.PROVIDER, attempt + 1, self._max_retries + 1,
        )
        return True

    async def count_tokens(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[int]:
        """Count input tokens for each string, in order.

        The Messages count endpoint scores one message list per request, so
        each string is its own user message. An empty list does not call
        the API.
        """
        if not texts:
            return []
        chosen = model or self._model
        counts: list[int] = []
        for text in texts:
            data = await self._post_limited(
                {
                    "model": chosen,
                    "messages": [{"role": "user", "content": text}],
                },
                url=f"{self.URL}/count_tokens",
                expect_message=False,
            )
            raw = data.get("input_tokens")
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise LLMValidationError(
                    f"{self.PROVIDER}: count_tokens response has no integer "
                    f"input_tokens ({data!r})"
                )
            counts.append(raw)
        return counts

    async def _post_limited(
        self,
        body: dict[str, Any],
        **post_kwargs: Any,
    ) -> dict[str, Any]:
        sem = self._ensure_semaphore()
        if sem is None:
            return await self._post(body, **post_kwargs)
        async with sem:
            return await self._post(body, **post_kwargs)

    async def _post_with_length_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """Raise max_tokens after a genuine max_tokens stop. One empty length
        response may be a thinking budget; a second empty response stops.
        """
        result = await self._post_limited(body)
        current = int(body.get("max_tokens") or 0)
        empty_length_retries = 0
        for _ in range(self._length_retries):
            if self._finish(result) != "length":
                return result
            visible = text_of(result)
            if not visible.strip() and not tool_calls_of(result):
                if empty_length_retries >= 1:
                    logger.warning(
                        "%s: empty content with finish_reason=length after "
                        "raising the budget — stopping budget escalation",
                        self.PROVIDER,
                    )
                    return result
                empty_length_retries += 1
            if looks_degenerate_repetition(visible):
                logger.warning(
                    "%s: finish_reason=length, but output is degenerate "
                    "(repetition) — skipping length-retry",
                    self.PROVIDER,
                )
                return result
            target = min(current * 2, self._length_retry_cap)
            if self._max_output_tokens is not None:
                target = min(target, self._max_output_tokens)
            if target <= current:
                break
            logger.warning(
                "%s: finish_reason=length → retry with max_tokens %d→%d",
                self.PROVIDER, current, target,
            )
            current = target
            result = await self._post_limited({**body, "max_tokens": target})
        return result

    async def _send(
        self, body: dict[str, Any], *, length_retry: bool,
    ) -> dict[str, Any]:
        if length_retry:
            return await self._post_with_length_retry(body)
        return await self._post_limited(body)

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate plain text. Thinking, when enabled, is returned separately."""
        self._last_served_tier = "text"
        body = self._base_body(request)
        payload = await self._send(body, length_retry=request.length_retry)
        self._warn_truncated(payload, request)
        text = text_of(payload)
        self._check_response_canary(text, context="generate_text")
        return self._response(payload, request=request)

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate a schema-constrained response.

        ``json_schema`` uses ``output_config.format``. A single function uses
        forced tool choice. ``tools`` lets the model select, with ``any`` when
        a tool call is required. ``tools_required`` never falls back to text.
        """
        if request.mode == "text" or self._tool_choice_pref == "text":
            return await self.generate_text(request)
        if request.mode == "json_schema":
            return await self._generate_json_schema(request)
        if not self._tools_enabled:
            return await self._emulate_json(request)
        if request.tools:
            return await self._generate_tools(request)
        return await self._generate_forced_tool(request)

    async def _generate_json_schema(self, request: LLMRequest) -> LLMResponse:
        schema = request.schema_ or {"type": "object"}
        if not schema_accepts_strict(schema):
            if self._no_degrade or self._preserve_responses:
                raise LLMValidationError(
                    f"{self.PROVIDER}: schema is not valid for Anthropic structured "
                    "output (objects need additionalProperties: false and every "
                    "property listed in required)"
                )
            logger.warning(
                "%s: schema is outside the strict grammar — falling back to text",
                self.PROVIDER,
            )
            return await self._emulate_json(request)
        body = self._base_body(request)
        body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            payload = await self._send(body, length_retry=request.length_retry)
        except AnthropicError as exc:
            if not self._can_degrade(exc, request):
                raise
            logger.warning(
                "%s: json_schema rejected (%s) — falling back to text",
                self.PROVIDER, exc.detail[:160],
            )
            return await self._emulate_json(request)
        self._last_served_tier = "json_schema"
        return self._structured_from_text(payload, request, schema)

    async def _generate_forced_tool(self, request: LLMRequest) -> LLMResponse:
        name = request.function_name or "build_artifact"
        schema = request.schema_ or {"type": "object"}
        body = self._base_body(request, tools=True)
        body["tools"] = [_tool_def(name, request.function_description, schema)]
        body["tool_choice"] = {"type": "tool", "name": name}
        return await self._complete_tool_request(body, request, schema, forced=name)

    async def _generate_tools(self, request: LLMRequest) -> LLMResponse:
        tools = [
            _tool_def(
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
                tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
            )
            for tool in request.tools or []
            if isinstance(tool, dict) and tool.get("name")
        ]
        if not tools:
            raise LLMValidationError(f"{self.PROVIDER}: tools_required but no tools were provided")
        required = request.tools_required or self._tool_choice_pref == "required"
        body = self._base_body(request, tools=True)
        body["tools"] = tools
        body["tool_choice"] = {"type": "any"} if required else {"type": "auto"}
        return await self._complete_tool_request(
            body, request, schema=None, forced=None,
        )

    async def _complete_tool_request(
        self,
        body: dict[str, Any],
        request: LLMRequest,
        schema: dict[str, Any] | None,
        *,
        forced: str | None,
    ) -> LLMResponse:
        try:
            payload = await self._send_tool_body(body, request)
        except AnthropicError as exc:
            if request.tools_required or not self._can_degrade(exc, request):
                raise
            logger.warning(
                "%s: tool use rejected (%s) — falling back to text",
                self.PROVIDER, exc.detail[:160],
            )
            return await self._emulate_json(request)
        calls = tool_calls_of(payload)
        self._warn_truncated(payload, request)
        if not calls:
            # A successful turn that did not call a tool is text. Required tool
            # use is a contract failure; do not spend another generation on it.
            if request.tools_required or forced or self._tool_choice_pref == "required":
                raise LLMValidationError(
                    f"{self.PROVIDER}: response did not include a tool_use block "
                    f"(stop_reason={payload.get('stop_reason')!r})",
                    payload=payload,
                )
            self._last_served_tier = "text"
            return self._response(payload, request=request)
        self._last_served_tier = "tool"
        chosen = calls[0]
        if schema is not None and not self._accept_arguments(chosen["arguments"], schema, request):
            return await self._emulate_json(request)
        text = text_of(payload)
        self._check_response_canary(text, context="generate_structured.text")
        self._check_response_canary(
            json.dumps(chosen["arguments"], ensure_ascii=False, default=str),
            context="generate_structured.args",
        )
        return self._response(
            payload,
            arguments=chosen["arguments"],
            function_name=str(chosen["name"] or forced or ""),
            tool_calls=calls,
            request=request,
        )

    def _accept_arguments(
        self,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
        request: LLMRequest,
    ) -> bool:
        if not self._validate_schema or _args_satisfy_schema(arguments, schema):
            return True
        if request.tools_required or self._no_degrade or self._preserve_responses:
            raise LLMValidationError(
                f"{self.PROVIDER}: tool arguments do not satisfy the schema",
                payload=arguments,
            )
        logger.warning(
            "%s: tool arguments failed schema validation — falling back to text",
            self.PROVIDER,
        )
        return False

    async def _send_tool_body(
        self, body: dict[str, Any], request: LLMRequest,
    ) -> dict[str, Any]:
        """Send a tool request. A strict-grammar 400 is retried once without strict.

        The retry is still native tool use. ``tools_required``, ``no_degrade``,
        and ``fallback_policy="preserve"`` do not turn it into a text call.
        """
        try:
            return await self._send(body, length_retry=request.length_retry)
        except AnthropicError as exc:
            if not _body_uses_strict(body) or not _strict_rejection(exc):
                raise
            logger.warning(
                "%s: strict tool use rejected (%s) — retrying without strict",
                self.PROVIDER, exc.detail[:160],
            )
            return await self._send(
                _without_strict(body), length_retry=request.length_retry,
            )

    def _can_degrade(self, exc: AnthropicError, request: LLMRequest) -> bool:
        if self._no_degrade or self._preserve_responses or request.tools_required:
            return False
        if exc.status_code != 400:
            return False
        return bool(_STRUCTURED_REJECTED_RE.search(exc.detail or str(exc)))

    async def _emulate_json(self, request: LLMRequest) -> LLMResponse:
        schema = request.schema_ or {"type": "object"}
        suffix = (
            "Respond with one JSON object and no other text. "
            f"Schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        self._last_served_tier = "text"
        body = self._base_body(request, user_suffix=suffix)
        payload = await self._send(body, length_retry=request.length_retry)
        return self._structured_from_text(payload, request, schema)

    def _structured_from_text(
        self,
        payload: dict[str, Any],
        request: LLMRequest,
        schema: dict[str, Any],
    ) -> LLMResponse:
        text = text_of(payload)
        self._check_response_canary(text, context="generate_structured.text")
        try:
            arguments = extract_json_from_text(text)
        except Exception as exc:
            raise LLMValidationError(
                f"{self.PROVIDER}: could not extract JSON from content ({text[:200]!r}): {exc}",
                payload=payload,
            ) from exc
        if not isinstance(arguments, dict):
            raise LLMValidationError(
                f"{self.PROVIDER}: structured content was not a JSON object",
                payload=payload,
            )
        if self._validate_schema and not _args_satisfy_schema(arguments, schema):
            raise LLMValidationError(
                f"{self.PROVIDER}: JSON content does not satisfy the schema",
                payload=arguments,
            )
        self._warn_truncated(payload, request)
        return self._response(payload, arguments=arguments, request=request)

    async def generate_stream(
        self, request: LLMRequest,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream text and thinking deltas, then a terminal chunk with usage."""
        body = self._base_body(request)
        body["stream"] = True
        visible: list[str] = []
        try:
            sem = self._ensure_semaphore()
            if sem is None:
                async for chunk in self._do_stream(body):
                    if chunk.delta_text:
                        visible.append(chunk.delta_text)
                    yield chunk
            else:
                async with sem:
                    async for chunk in self._do_stream(body):
                        if chunk.delta_text:
                            visible.append(chunk.delta_text)
                        yield chunk
        finally:
            # Same generator the caller closes. A nested generator's finally
            # would run only when that generator is collected.
            self._check_response_canary("".join(visible), context="generate_stream")

    async def _do_stream(self, body: dict[str, Any]) -> AsyncIterator[LLMStreamChunk]:
        usage_raw: dict[str, Any] = {}
        request_id: str | None = None
        finish: str | None = None
        first = True
        blocks = _StreamBlocks()
        async for event in self._iter_events(body):
            kind = event.get("type")
            if kind == "message_start":
                message = event.get("message") if isinstance(event.get("message"), dict) else {}
                request_id = str(message.get("id") or "") or request_id
                if isinstance(message.get("usage"), dict):
                    usage_raw.update(message["usage"])
            elif kind == "content_block_start":
                blocks.start(event)
            elif kind == "content_block_delta":
                blocks.delta(event)
                delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                text = delta.get("text") if delta.get("type") == "text_delta" else ""
                thinking = (
                    delta.get("thinking") if delta.get("type") == "thinking_delta" else ""
                )
                if text or thinking:
                    yield LLMStreamChunk(
                        delta_text=str(text or ""),
                        delta_reasoning=str(thinking or ""),
                        request_id=request_id if first else None,
                    )
                    first = False
            elif kind == "message_delta":
                delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                finish = map_stop_reason(str(delta.get("stop_reason") or "") or None)
                if isinstance(event.get("usage"), dict):
                    usage_raw.update(event["usage"])
        yield LLMStreamChunk(
            finish_reason=finish,
            usage=usage_from_anthropic(usage_raw) if usage_raw else None,
            request_id=request_id if first else None,
            content_blocks=blocks.replay() or None,
        )

    async def generate_stream_events(
        self, request: LLMRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed content, thinking, and tool events, then one Complete.

        Tool definitions are sent only when the request carries ``tools``.
        A request without tools is streamed as plain text. The canary is
        scanned on this generator, including when the caller stops early.
        """
        if request.tools:
            body = self._base_body(request, tools=True)
            body["tools"] = [
                _tool_def(
                    str(tool.get("name") or ""),
                    str(tool.get("description") or ""),
                    tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
                )
                for tool in request.tools or []
                if isinstance(tool, dict) and tool.get("name")
            ]
            required = request.tools_required or self._tool_choice_pref == "required"
            body["tool_choice"] = {"type": "any"} if required else {"type": "auto"}
        else:
            body = self._base_body(request)
        body["stream"] = True
        visible: list[str] = []
        try:
            sem = self._ensure_semaphore()
            if sem is None:
                async for event in self._do_stream_events(body):
                    if isinstance(event, ContentDelta) and event.delta_text:
                        visible.append(event.delta_text)
                    yield event
            else:
                async with sem:
                    async for event in self._do_stream_events(body):
                        if isinstance(event, ContentDelta) and event.delta_text:
                            visible.append(event.delta_text)
                        yield event
        finally:
            self._check_response_canary(
                "".join(visible), context="generate_stream_events",
            )

    async def _do_stream_events(self, body: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        usage_raw: dict[str, Any] = {}
        request_id: str | None = None
        finish: str | None = None
        open_tools: dict[int, str] = {}
        blocks = _StreamBlocks()
        try:
            async for event in self._iter_events(body):
                kind = event.get("type")
                if kind == "message_start":
                    message = event.get("message") if isinstance(event.get("message"), dict) else {}
                    request_id = str(message.get("id") or "") or request_id
                    if isinstance(message.get("usage"), dict):
                        usage_raw.update(message["usage"])
                elif kind == "content_block_start":
                    blocks.start(event)
                    index = int(event.get("index") or 0)
                    block = (
                        event.get("content_block")
                        if isinstance(event.get("content_block"), dict) else {}
                    )
                    if block.get("type") == "tool_use":
                        name = str(block.get("name") or "")
                        open_tools[index] = name
                        yield ToolUseStart(
                            index=index,
                            id=str(block.get("id") or "") or None,
                            name=name,
                            request_id=request_id,
                        )
                elif kind == "content_block_delta":
                    blocks.delta(event)
                    index = int(event.get("index") or 0)
                    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta" and delta.get("text"):
                        yield ContentDelta(
                            delta_text=str(delta["text"]), request_id=request_id,
                        )
                    elif delta_type == "thinking_delta" and delta.get("thinking"):
                        yield ReasoningDelta(
                            delta_reasoning=str(delta["thinking"]), request_id=request_id,
                        )
                    elif delta_type == "input_json_delta" and delta.get("partial_json"):
                        yield ToolUseDelta(
                            index=index,
                            arguments_delta=str(delta["partial_json"]),
                            request_id=request_id,
                        )
                elif kind == "content_block_stop":
                    index = int(event.get("index") or 0)
                    if index in open_tools:
                        yield ToolUseStop(index=index, request_id=request_id)
                        del open_tools[index]
                elif kind == "message_delta":
                    delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
                    finish = map_stop_reason(str(delta.get("stop_reason") or "") or None)
                    if isinstance(event.get("usage"), dict):
                        usage_raw.update(event["usage"])
            for index in list(open_tools):
                yield ToolUseStop(index=index, request_id=request_id)
            yield Complete(
                finish_reason=finish,
                usage=usage_from_anthropic(usage_raw) if usage_raw else None,
                request_id=request_id,
                content_blocks=blocks.replay() or None,
            )
        except (AnthropicError, LLMAuthError, LLMTimeoutError) as exc:
            yield Error(error=str(exc), error_type=type(exc).__name__, request_id=request_id)
            raise

    async def _iter_events(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Read one SSE response, retrying 429/529/5xx before the first event.

        The response is closed with ``aclose`` rather than ``async with``. A
        nested streaming context manager does not finish when the consumer
        stops early, so the canary scan on the caller would be skipped.
        """
        client = self._ensure_http()
        for attempt in range(self._max_retries + 1):
            yielded = False
            response: httpx.Response | None = None
            try:
                response = await client.send(
                    client.build_request(
                        "POST", self.URL, headers=self._headers(), json=body,
                    ),
                    stream=True,
                )
                if response.status_code in (401, 403):
                    raw = (await response.aread()).decode("utf-8", "replace")[:4000]
                    raise LLMAuthError(
                        f"{self.PROVIDER} stream {response.status_code}: {raw}"
                    )
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", "replace")[:4000]
                    err = AnthropicError(
                        f"{self.PROVIDER} stream {response.status_code}: {raw}",
                        status_code=response.status_code,
                        detail=raw,
                    )
                    if response.status_code == 429:
                        err.retry_after = parse_retry_after(response)
                    raise err
                async for payload in iter_sse_payloads(response.aiter_lines()):
                    if payload.get("type") == "error":
                        message = payload.get("error")
                        detail = (
                            message.get("message")
                            if isinstance(message, dict) else str(message)
                        )
                        raise AnthropicError(
                            f"{self.PROVIDER} stream error: {detail}", detail=str(detail),
                        )
                    yielded = True
                    yield payload
                return
            except LLMAuthError:
                raise
            except AnthropicError as exc:
                if (
                    yielded
                    or exc.status_code not in _RETRYABLE_STATUS
                    or attempt >= self._max_retries
                ):
                    raise
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else backoff_with_jitter(self._retry_backoff_s, attempt)
                )
                logger.warning(
                    "%s stream %s on attempt %d/%d — retry in %.1fs",
                    self.PROVIDER, exc.status_code, attempt + 1,
                    self._max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                    httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if yielded or attempt >= self._max_retries:
                    err_cls = (
                        LLMTimeoutError
                        if isinstance(exc, httpx.TimeoutException)
                        else AnthropicError
                    )
                    raise err_cls(
                        f"{self.PROVIDER} stream network error: {exc}"
                    ) from exc
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
            finally:
                if response is not None:
                    await response.aclose()

class _StreamBlocks:
    """Rebuild Messages blocks from SSE, including thinking signatures."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._tool_json: dict[int, list[str]] = {}

    def start(self, event: dict[str, Any]) -> None:
        index = int(event.get("index") or 0)
        block = event.get("content_block")
        if not isinstance(block, dict):
            return
        kind = block.get("type")
        if kind == "thinking":
            stored: dict[str, Any] = {
                "type": "thinking",
                "thinking": str(block.get("thinking") or ""),
            }
            if block.get("signature"):
                stored["signature"] = str(block["signature"])
            self._blocks[index] = stored
        elif kind == "redacted_thinking":
            self._blocks[index] = {
                "type": "redacted_thinking",
                "data": str(block.get("data") or ""),
            }
        elif kind == "text":
            self._blocks[index] = {"type": "text", "text": str(block.get("text") or "")}
        elif kind == "tool_use":
            raw_input = block.get("input")
            self._blocks[index] = {
                "type": "tool_use",
                "id": str(block.get("id") or ""),
                "name": str(block.get("name") or ""),
                "input": raw_input if isinstance(raw_input, dict) else {},
            }
            self._tool_json[index] = []
        elif kind:
            self._blocks[index] = dict(block)

    def delta(self, event: dict[str, Any]) -> None:
        index = int(event.get("index") or 0)
        delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
        kind = delta.get("type")
        block = self._blocks.setdefault(index, {})
        if kind == "text_delta":
            text = str(delta.get("text") or "")
            if block.get("type") != "text":
                block.clear()
                block.update({"type": "text", "text": ""})
            block["text"] = str(block.get("text") or "") + text
        elif kind == "thinking_delta":
            piece = str(delta.get("thinking") or "")
            if block.get("type") != "thinking":
                block.clear()
                block.update({"type": "thinking", "thinking": ""})
            block["thinking"] = str(block.get("thinking") or "") + piece
        elif kind == "signature_delta":
            signature = str(delta.get("signature") or "")
            if not signature:
                return
            if block.get("type") != "thinking":
                block.clear()
                block.update({"type": "thinking", "thinking": ""})
            block["signature"] = str(block.get("signature") or "") + signature
        elif kind == "input_json_delta":
            self._tool_json.setdefault(index, []).append(str(delta.get("partial_json") or ""))

    def replay(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = dict(self._blocks[index])
            if block.get("type") == "tool_use":
                raw = "".join(self._tool_json.get(index) or [])
                if raw:
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = block.get("input")
                    if isinstance(parsed, dict):
                        block["input"] = parsed
            if block.get("type") == "thinking" and not block.get("signature"):
                block.pop("signature", None)
            if block.get("type"):
                blocks.append(block)
        return blocks


def schema_accepts_strict(schema: Any) -> bool:
    """True when every object matches Anthropic's strict-tool grammar.

    Each object must set ``additionalProperties`` to false and list every
    property in ``required``. A ``$ref`` cannot be checked locally, so it is
    not sent as strict.
    """
    if not isinstance(schema, dict) or "$ref" in schema:
        return False
    for key in ("anyOf", "oneOf", "allOf"):
        parts = schema.get(key)
        if parts is None:
            continue
        if not isinstance(parts, list) or not parts:
            return False
        if not all(schema_accepts_strict(part) for part in parts):
            return False
    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        if schema.get("additionalProperties") is not False:
            return False
        props = schema.get("properties")
        if not isinstance(props, dict):
            return False
        required = schema.get("required")
        if not isinstance(required, list) or set(props) - set(required):
            return False
        return all(schema_accepts_strict(child) for child in props.values())
    if schema_type == "array" or "items" in schema:
        items = schema.get("items")
        if isinstance(items, dict):
            return schema_accepts_strict(items)
        if isinstance(items, list) and items:
            return all(schema_accepts_strict(item) for item in items)
        return False
    return schema_type in ("string", "number", "integer", "boolean", "null")


def _tool_def(name: str, description: str, schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        schema = {"type": "object"}
    tool: dict[str, Any] = {
        "name": name,
        "description": description or f"Build {name}.",
        "input_schema": schema,
    }
    if schema_accepts_strict(schema):
        tool["strict"] = True
    return tool


def _body_uses_strict(body: dict[str, Any]) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("strict") is True for tool in tools
    )


def _without_strict(body: dict[str, Any]) -> dict[str, Any]:
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict):
            tools.append({key: value for key, value in tool.items() if key != "strict"})
        else:
            tools.append(tool)
    return {**body, "tools": tools}


def _strict_rejection(exc: AnthropicError) -> bool:
    if exc.status_code != 400:
        return False
    return bool(_STRICT_REJECTED_RE.search(exc.detail or str(exc)))


__all__ = [
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_VERSION",
    "AnthropicClient",
    "AnthropicError",
    "build_messages",
    "map_stop_reason",
    "messages_url",
    "usage_from_anthropic",
]
