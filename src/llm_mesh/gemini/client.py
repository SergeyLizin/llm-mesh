"""Async client for the Gemini generateContent API.

The model name is in the URL (``models/{model}:generateContent``), not in
the body. System text is ``systemInstruction``. Tool history is
``functionCall`` / ``functionResponse``. ``countTokens`` is a separate
method on the same model. OpenAI-compatible Gemini proxies stay on
``OpenAIClient``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Literal
from urllib.parse import quote

import httpx

from llm_mesh._common import (
    _env_flag,
    _env_float_default,
    _env_int_default,
    _env_is_disabled,
    _env_nonneg_int,
    _env_positive_int,
    _args_satisfy_schema,
    _parse_json_dict_env,
    apply_canary,
    warn_if_truncated,
)
from llm_mesh._retry import (
    RETRYABLE_SERVER_STATUS,
    backoff_with_jitter,
    retry_after_delay,
)
from llm_mesh._streaming import iter_sse_payloads
from llm_mesh.base import BaseLLMClient, Capability
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_LEADING_USER_TEXT = "."
_RETRYABLE_STATUS = frozenset({429, *RETRYABLE_SERVER_STATUS})
_SCHEMA_REJECTED = ("responseSchema", "response_schema", "responseMimeType", "response_mime_type")
_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
}


class GeminiError(LLMError):
    """HTTP or protocol error from a generateContent endpoint."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def generate_content_url(
    base_url: str,
    model: str,
    *,
    stream: bool = False,
    count: bool = False,
) -> str:
    """Build ``<base>/models/{model}:<method>``.

    A base that already ends in ``/v1beta`` or ``/v1`` is kept. Other bases
    get ``/v1beta``. Streaming adds ``alt=sse``.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1beta") or base.endswith("/v1"):
        root = base
    else:
        root = f"{base}/v1beta"
    if count:
        action = "countTokens"
    elif stream:
        action = "streamGenerateContent"
    else:
        action = "generateContent"
    url = f"{root}/models/{quote(model, safe='')}:{action}"
    if stream:
        url += "?alt=sse"
    return url


def map_finish_reason(reason: str | None, *, has_function: bool) -> str | None:
    """Translate a candidate finish reason. A function call is ``tool_calls``."""
    if has_function and reason not in ("MAX_TOKENS",):
        return "tool_calls"
    if not reason:
        return None
    return _FINISH.get(reason, reason.lower())


def usage_from_gemini(raw: Any) -> LLMUsage:
    """Normalize ``usageMetadata``.

    ``candidatesTokenCount`` is visible output. ``thoughtsTokenCount`` is
    added into ``completion_tokens`` and also reported as reasoning.
    ``cachedContentTokenCount``, when present, is cache hits; the rest of
    the prompt is misses. An absent cache field stays unknown.
    """
    if not isinstance(raw, dict):
        return LLMUsage()

    def _int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt = _int(raw.get("promptTokenCount"))
    visible = _int(raw.get("candidatesTokenCount"))
    thoughts = _int(raw.get("thoughtsTokenCount"))
    total = _int(raw.get("totalTokenCount")) or (prompt + visible + thoughts)
    cached = raw.get("cachedContentTokenCount")
    if cached is None:
        hit, miss = -1, -1
    else:
        hit = _int(cached)
        miss = max(prompt - hit, 0)
    return LLMUsage(
        prompt_tokens=prompt,
        completion_tokens=visible + thoughts,
        total_tokens=total,
        reasoning_tokens=thoughts,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _parts_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, dict)]


def text_of(payload: dict[str, Any]) -> str:
    """Join user-visible text parts. Thought parts are not included."""
    return "".join(
        str(part.get("text") or "")
        for part in _parts_of(payload)
        if part.get("text") and not part.get("thought")
    )


def thinking_of(payload: dict[str, Any]) -> str | None:
    """Join thought parts, or None when the response has none."""
    parts = [
        str(part.get("text") or "")
        for part in _parts_of(payload)
        if part.get("thought") and part.get("text")
    ]
    return "".join(parts) or None


def function_calls_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Function calls in order, each with a synthetic id, name, and args."""
    calls: list[dict[str, Any]] = []
    for part in _parts_of(payload):
        call = part.get("functionCall")
        if not isinstance(call, dict) or not call.get("name"):
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            args = {}
        calls.append({
            "id": f"call_{len(calls)}",
            "name": str(call["name"]),
            "arguments": args,
        })
    return calls


def _candidate_reason(payload: dict[str, Any]) -> str | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return None
    reason = candidates[0].get("finishReason")
    return str(reason) if reason else None


def _block_reason(payload: dict[str, Any]) -> str | None:
    feedback = payload.get("promptFeedback")
    if not isinstance(feedback, dict):
        return None
    reason = feedback.get("blockReason")
    return str(reason) if reason else None


def _is_gemini_parts(content: Any) -> bool:
    """True when content is already a list of generateContent parts."""
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(part, dict)
            and "type" not in part
            and any(key in part for key in ("text", "functionCall", "functionResponse", "thought"))
            for part in content
        )
    )


def _tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _response_object(raw: Any) -> dict[str, Any]:
    """functionResponse.response must be an object."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return {"result": raw}
    if raw is None:
        return {"result": ""}
    return {"result": raw}


def _parse_json_object(content: str, *, salvage: bool) -> dict[str, Any] | None:
    """Parse a JSON object. Salvage accepts a fence or one object in prose."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        if not salvage:
            return None
    if not salvage or not content:
        return None
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        try:
            parsed = json.loads("\n".join(lines))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def build_contents(request: LLMRequest) -> tuple[str, list[dict[str, Any]]]:
    """Split a shared request into system text and alternating contents.

    OpenAI tool turns become a model ``functionCall`` and a following user
    ``functionResponse``. The response name is the tool message's ``name``,
    or the name of the earlier call with the same id. A content list that is
    already generateContent parts is copied, including ``thoughtSignature``.
    Consecutive turns of the same role are merged. History that starts with
    the model gets a leading user part.
    """
    system = apply_canary(request.system)
    contents: list[dict[str, Any]] = []
    names_by_id: dict[str, str] = {}
    pending: list[dict[str, Any]] = []

    def append(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
            return
        contents.append({"role": role, "parts": parts})

    def flush_responses() -> None:
        nonlocal pending
        if pending:
            append("user", pending)
            pending = []

    for turn in request.history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role == "tool":
            call_id = str(turn.get("tool_call_id") or "")
            name = str(turn.get("name") or names_by_id.get(call_id) or "")
            if name:
                pending.append({
                    "functionResponse": {
                        "name": name,
                        "response": _response_object(turn.get("content")),
                    },
                })
            continue
        flush_responses()
        if role not in ("user", "assistant", "model"):
            continue
        wire_role = "model" if role in ("assistant", "model") else "user"
        content = turn.get("content")
        if _is_gemini_parts(content):
            parts = [dict(part) for part in content]
        else:
            text = content if isinstance(content, str) else str(content or "")
            parts = [{"text": text}] if text else []
        for index, call in enumerate(turn.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            if any(isinstance(part.get("functionCall"), dict) for part in parts):
                break
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or call.get("name") or "")
            if not name:
                continue
            call_id = str(call.get("id") or f"call_{index}")
            names_by_id[call_id] = name
            parts.append({
                "functionCall": {
                    "name": name,
                    "args": _tool_args(function.get("arguments", call.get("arguments", {}))),
                },
            })
        append(wire_role, parts)
    flush_responses()
    if contents and contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": _LEADING_USER_TEXT}]})
    user = request.user
    if user or not contents or contents[-1]["role"] != "user":
        append("user", [{"text": user}] if user else [])
    return system, contents


class GeminiClient(BaseLLMClient):
    """Gemini generateContent client.

    ``GEMINI_API_KEY`` is read before ``LLM_API_KEY``. An omitted base URL
    is ``GEMINI_BASE_URL``, then ``LLM_BASE_URL``, then the public
    Generative Language endpoint. ``fallback_policy="preserve"`` and
    ``no_degrade`` refuse salvage of JSON Schema text and a silent retry
    after a schema rejection. The default policy is ``"recover"``.
    """

    CAPABILITIES = frozenset({
        Capability.TEXT,
        Capability.STREAM,
        Capability.STREAM_EVENTS,
        Capability.STRUCTURED,
        Capability.TOOLS,
        Capability.MULTI_TOOL,
        Capability.TOOLS_REQUIRED,
        Capability.JSON_SCHEMA_MODE,
        Capability.LENGTH_RETRY,
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
        no_degrade: bool | None = None,
        fallback_policy: Literal["recover", "preserve"] = "recover",
        validate_schema: bool = True,
    ) -> None:
        if fallback_policy not in ("recover", "preserve"):
            raise ValueError("fallback_policy must be 'recover' or 'preserve'")
        self._base = (
            base_url
            or get_env("GEMINI_BASE_URL")
            or get_env("LLM_BASE_URL")
            or GEMINI_BASE_URL
        )
        self.PROVIDER = label or get_env("LLM_PROVIDER_LABEL") or "gemini"
        self._model = model or get_env("LLM_MODEL")
        self._key = api_key or get_env("GEMINI_API_KEY") or get_env("LLM_API_KEY")
        if not self._key:
            raise GeminiError(
                f"{self.PROVIDER}: missing API key "
                "(GEMINI_API_KEY/LLM_API_KEY are not set)"
            )
        self._extra_headers = dict(extra_headers or {})
        parsed_headers = _parse_json_dict_env(
            "LLM_EXTRA_HEADERS", logger=logger, include_error=False,
        )
        if parsed_headers is not None:
            self._extra_headers.update({str(key): str(value) for key, value in parsed_headers.items()})
        self._extra_body = _parse_json_dict_env(
            "LLM_EXTRA_BODY", logger=logger, include_error=False,
        ) or {}
        self._max_retries = _env_int_default("LLM_MAX_RETRIES", 3, logger=logger)
        self._retry_backoff_s = float(get_env("LLM_RETRY_BACKOFF_S", "1.0") or "1.0")
        self._max_concurrent = _env_positive_int("LLM_MAX_CONCURRENT")
        self._semaphore: asyncio.Semaphore | None = None
        self._max_output_tokens = _env_positive_int("LLM_MAX_OUTPUT_TOKENS")
        self._min_output_tokens = _env_positive_int("LLM_MIN_OUTPUT_TOKENS")
        self._length_retries = _env_nonneg_int("LLM_LENGTH_RETRIES", 1)
        self._preserve_responses = fallback_policy == "preserve"
        self._no_degrade = (
            no_degrade if no_degrade is not None else _env_flag("LLM_NO_DEGRADE")
        )
        self._validate_schema = validate_schema
        self._disable_reasoning = _env_flag("LLM_DISABLE_REASONING")
        self._reasoning_effort = get_env("LLM_REASONING_EFFORT", "").strip().lower()
        forced = get_env("LLM_FORCE_TEMPERATURE", "").strip()
        self._force_temperature: float | None = None
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
        if verify is not None:
            self._verify = verify
        else:
            self._verify = not _env_is_disabled("LLM_VERIFY_SSL", default="1")
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            **self._extra_headers,
        }
        headers["x-goog-api-key"] = self._key
        return headers

    def _model_of(self, request: LLMRequest | None = None, model: str | None = None) -> str:
        chosen = model or (request.model if request is not None else None) or self._model
        if not chosen:
            raise GeminiError(f"{self.PROVIDER}: model is not set")
        return chosen

    def _clip_max_tokens(self, requested: int) -> int:
        value = max(int(requested), 1)
        if self._min_output_tokens is not None:
            value = max(value, self._min_output_tokens)
        if self._max_output_tokens is not None:
            value = min(value, self._max_output_tokens)
        return max(value, 1)

    def _can_degrade(self, request: LLMRequest) -> bool:
        return not (
            self._no_degrade or self._preserve_responses or request.tools_required
        )

    def _schema_for(self, request: LLMRequest, name: str | None) -> dict[str, Any] | None:
        """Schema that constrains this result. A named tool uses its parameters."""
        if request.tools and name:
            for tool in request.tools:
                if isinstance(tool, dict) and tool.get("name") == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
            return None
        return request.schema_

    def _require_schema(
        self,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> None:
        """Reject arguments jsonschema confirms are outside the schema.

        ``validate_schema=False`` returns the model output unchanged. A missing
        schema, or a schema the validator cannot load, is not a rejection.
        """
        if not self._validate_schema or _args_satisfy_schema(arguments, schema):
            return
        raise LLMValidationError(
            f"{self.PROVIDER}: arguments do not satisfy the schema",
            payload=arguments,
        )

    def _apply_extra(self, body: dict[str, Any]) -> None:
        for key, value in self._extra_body.items():
            body.setdefault(key, value)

    def _thinking_config(self, request: LLMRequest) -> dict[str, Any] | None:
        if self._disable_reasoning:
            return {"thinkingBudget": 0}
        effort = (request.reasoning_effort or self._reasoning_effort or "").lower()
        if effort in ("low", "medium", "high"):
            return {"thinkingLevel": effort}
        return None

    def _generation_config(self, request: LLMRequest) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": (
                self._force_temperature
                if self._force_temperature is not None
                else request.temperature
            ),
            "maxOutputTokens": self._clip_max_tokens(request.max_tokens),
        }
        thinking = self._thinking_config(request)
        if thinking is not None:
            config["thinkingConfig"] = thinking
        return config

    def _declarations(self, request: LLMRequest) -> list[dict[str, Any]]:
        if request.tools:
            return [
                {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {
                        "type": "object", "properties": {},
                    },
                }
                for tool in request.tools
                if tool.get("name")
            ]
        if request.mode == "json_schema":
            return []
        return [{
            "name": request.function_name,
            "description": request.function_description or "",
            "parameters": request.schema_ or {"type": "object", "properties": {}},
        }]

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        system, contents = build_contents(request)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": self._generation_config(request),
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if structured and request.mode == "json_schema":
            body["generationConfig"] = {
                **body["generationConfig"],
                "responseMimeType": "application/json",
                "responseSchema": request.schema_ or {"type": "object", "properties": {}},
            }
        elif structured:
            declarations = self._declarations(request)
            if declarations:
                body["tools"] = [{"functionDeclarations": declarations}]
                calling: dict[str, Any] = {}
                if request.tools and not request.tools_required:
                    calling["mode"] = "AUTO"
                else:
                    calling["mode"] = "ANY"
                    if not request.tools:
                        calling["allowedFunctionNames"] = [request.function_name]
                body["toolConfig"] = {"functionCallingConfig": calling}
        self._apply_extra(body)
        return body

    def _finish(self, payload: dict[str, Any]) -> str | None:
        return map_finish_reason(
            _candidate_reason(payload),
            has_function=bool(function_calls_of(payload)),
        )

    def _response(
        self,
        payload: dict[str, Any],
        request: LLMRequest,
        *,
        request_id: str | None,
        arguments: dict[str, Any] | None = None,
        function_name: str | None = None,
    ) -> LLMResponse:
        calls = function_calls_of(payload)
        text = text_of(payload)
        if arguments is None and calls:
            arguments = calls[0]["arguments"]
            function_name = calls[0]["name"]
        return LLMResponse(
            arguments=arguments or {},
            text=text,
            function_name=function_name,
            tool_calls=calls,
            reasoning_content=thinking_of(payload),
            finish_reason=self._finish(payload),
            request_id=request_id,
            model=str(payload.get("modelVersion") or self._model_of(request)),
            usage=usage_from_gemini(payload.get("usageMetadata")),
            raw=payload,
        )

    def _require_candidate(self, payload: dict[str, Any]) -> None:
        blocked = _block_reason(payload)
        if blocked:
            raise LLMValidationError(
                f"{self.PROVIDER}: prompt blocked ({blocked})",
                payload=payload,
            )
        if not _parts_of(payload) and _candidate_reason(payload) is None:
            raise GeminiError(
                f"{self.PROVIDER}: response has no candidate: {payload!r}"
            )

    async def _post(
        self,
        body: dict[str, Any],
        *,
        model: str,
        count: bool = False,
    ) -> tuple[dict[str, Any], str | None]:
        url = generate_content_url(self._base, model, count=count)
        headers = self._headers()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._ensure_http().post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                    httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    err_cls = (
                        LLMTimeoutError
                        if isinstance(exc, httpx.TimeoutException)
                        else GeminiError
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
                raise GeminiError(
                    f"{self.PROVIDER} HTTP {response.status_code}: {detail}",
                    status_code=response.status_code,
                    detail=detail,
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise GeminiError(
                    f"{self.PROVIDER} malformed JSON response: {exc} "
                    f"(body={response.text[:300]!r})"
                ) from exc
            if not isinstance(data, dict):
                raise GeminiError(f"{self.PROVIDER} response is not an object: {data!r}")
            if isinstance(data.get("error"), dict):
                message = data["error"].get("message") or data["error"]
                raise GeminiError(f"{self.PROVIDER}: {message}", detail=str(message))
            request_id = (
                response.headers.get("x-request-id")
                or response.headers.get("x-goog-request-id")
                or (str(data["responseId"]) if data.get("responseId") else None)
            )
            return data, request_id
        raise GeminiError(f"{self.PROVIDER} retry exhausted: {last_exc}")

    async def _post_limited(
        self,
        body: dict[str, Any],
        *,
        model: str,
        count: bool = False,
    ) -> tuple[dict[str, Any], str | None]:
        sem = self._ensure_semaphore()
        if sem is None:
            return await self._post(body, model=model, count=count)
        async with sem:
            return await self._post(body, model=model, count=count)

    async def _send(
        self,
        body: dict[str, Any],
        *,
        model: str,
        length_retry: bool,
    ) -> tuple[dict[str, Any], str | None]:
        result = await self._post_limited(body, model=model)
        if not length_retry:
            return result
        payload, _request_id = result
        current = int(body["generationConfig"]["maxOutputTokens"])
        empty_length_retries = 0
        for _ in range(self._length_retries):
            if self._finish(payload) != "length":
                return result
            visible = text_of(payload)
            if not visible.strip() and not function_calls_of(payload):
                if empty_length_retries >= 1:
                    logger.warning(
                        "%s: empty content with finish_reason=length after "
                        "raising the budget — stopping budget escalation",
                        self.PROVIDER,
                    )
                    return result
                empty_length_retries += 1
            nxt = self._clip_max_tokens(current * 2)
            if nxt <= current:
                return result
            logger.warning(
                "%s: finish_reason=length → retry with maxOutputTokens %d→%d",
                self.PROVIDER, current, nxt,
            )
            current = nxt
            config = {**body["generationConfig"], "maxOutputTokens": nxt}
            body = {**body, "generationConfig": config}
            result = await self._post_limited(body, model=model)
            payload, _request_id = result
        return result

    def _schema_rejected(self, exc: GeminiError) -> bool:
        if exc.status_code != 400:
            return False
        detail = exc.detail or str(exc)
        return any(token in detail for token in _SCHEMA_REJECTED)

    async def count_tokens(
        self, texts: list[str], *, model: str | None = None,
    ) -> list[int]:
        """Count input tokens for each string, in order.

        ``countTokens`` returns one total per request, so each string is its
        own user content. An empty list does not call the API.
        """
        if not texts:
            return []
        chosen = self._model_of(model=model)
        counts: list[int] = []
        for text in texts:
            data, _request_id = await self._post_limited(
                {"contents": [{"role": "user", "parts": [{"text": text}]}]},
                model=chosen,
                count=True,
            )
            raw = data.get("totalTokens")
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise LLMValidationError(
                    f"{self.PROVIDER}: countTokens response has no integer "
                    f"totalTokens ({data!r})"
                )
            counts.append(raw)
        return counts

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate plain text. Thought parts are returned separately."""
        model = self._model_of(request)
        body = self._body(request, structured=False)
        payload, request_id = await self._send(
            body, model=model, length_retry=request.length_retry,
        )
        self._require_candidate(payload)
        text = text_of(payload)
        warn_if_truncated(
            {"finish_reason": self._finish(payload) or ""},
            request,
            text,
            sent_max_tokens=int(body["generationConfig"]["maxOutputTokens"]),
            provider=self.PROVIDER,
            logger=logger,
        )
        self._check_response_canary(text, context="generate_text")
        return self._response(payload, request, request_id=request_id)

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate a schema-constrained response or a function call.

        ``json_schema`` uses ``responseSchema``. One function uses
        ``functionCallingConfig.mode=ANY`` and ``allowedFunctionNames``.
        ``tools`` uses ``AUTO`` unless ``tools_required``, which uses
        ``ANY``. A rejected schema is retried once as text when degradation
        is allowed. ``tools_required`` never falls back to text. Arguments
        that violate the schema raise ``LLMValidationError`` unless
        ``validate_schema`` is false.
        """
        if request.mode == "text":
            return await self.generate_text(request)
        model = self._model_of(request)
        body = self._body(request, structured=True)
        try:
            payload, request_id = await self._send(
                body, model=model, length_retry=request.length_retry,
            )
        except GeminiError as exc:
            if request.mode != "json_schema" or not self._schema_rejected(exc) or not self._can_degrade(request):
                raise
            logger.warning(
                "%s: response schema rejected — retrying as text", self.PROVIDER,
            )
            payload, request_id = await self._send(
                self._body(request, structured=False),
                model=model,
                length_retry=request.length_retry,
            )
        self._require_candidate(payload)
        if request.mode == "json_schema":
            content = text_of(payload)
            arguments = _parse_json_object(
                content, salvage=self._can_degrade(request),
            )
            if arguments is None:
                why = (
                    "non-JSON content; LLM_NO_DEGRADE or "
                    "fallback_policy=preserve forbids salvage"
                    if not self._can_degrade(request)
                    else "unparseable JSON"
                )
                raise LLMValidationError(
                    f"{self.PROVIDER}: response schema returned {why} "
                    f"(content={content[:200]!r})",
                    payload=payload,
                )
            self._require_schema(arguments, request.schema_)
            self._check_response_canary(content, context="generate_structured")
            return self._response(
                payload, request, request_id=request_id,
                arguments=arguments, function_name=request.function_name,
            )
        calls = function_calls_of(payload)
        if not calls:
            if request.tools and not request.tools_required:
                text = text_of(payload)
                self._check_response_canary(text, context="generate_structured")
                return self._response(payload, request, request_id=request_id)
            raise LLMValidationError(
                f"{self.PROVIDER}: missing functionCall "
                f"(content={text_of(payload)[:200]!r})",
                payload=payload,
            )
        for call in calls:
            self._require_schema(
                call["arguments"], self._schema_for(request, str(call["name"])),
            )
        encoded = json.dumps(calls[0]["arguments"], ensure_ascii=False)
        self._check_response_canary(encoded, context="generate_structured")
        return self._response(payload, request, request_id=request_id)

    async def _iter_payloads(
        self, body: dict[str, Any], *, model: str,
    ) -> AsyncIterator[dict[str, Any]]:
        url = generate_content_url(self._base, model, stream=True)
        headers = self._headers()
        client = self._ensure_http()
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                request = client.build_request("POST", url, headers=headers, json=body)
                response = await client.send(request, stream=True)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                    httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    err_cls = (
                        LLMTimeoutError
                        if isinstance(exc, httpx.TimeoutException)
                        else GeminiError
                    )
                    raise err_cls(
                        f"{self.PROVIDER} network error after "
                        f"{attempt + 1} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(backoff_with_jitter(self._retry_backoff_s, attempt))
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                status = response.status_code
                delay = (
                    retry_after_delay(response, self._retry_backoff_s, attempt)
                    if status == 429
                    else backoff_with_jitter(self._retry_backoff_s, attempt)
                )
                await response.aclose()
                response = None
                logger.warning(
                    "%s stream %s on attempt %d/%d — retry in %.1fs",
                    self.PROVIDER, status, attempt + 1,
                    self._max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code >= 400:
                detail = (await response.aread())[:4000].decode("utf-8", "replace")
                status = response.status_code
                await response.aclose()
                if status in (401, 403):
                    raise LLMAuthError(f"{self.PROVIDER} HTTP {status}: {detail}")
                raise GeminiError(
                    f"{self.PROVIDER} HTTP {status}: {detail}",
                    status_code=status,
                    detail=detail,
                )
            break
        else:
            raise GeminiError(f"{self.PROVIDER} retry exhausted: {last_exc}")
        assert response is not None
        try:
            async for payload in iter_sse_payloads(response.aiter_lines()):
                if isinstance(payload.get("error"), dict):
                    message = payload["error"].get("message") or payload["error"]
                    raise GeminiError(f"{self.PROVIDER}: {message}", detail=str(message))
                yield payload
        finally:
            await response.aclose()

    async def _under_semaphore(self, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Hold the instance concurrency slot for the whole stream, including early close."""
        sem = self._ensure_semaphore()
        if sem is None:
            async for item in source:
                yield item
            return
        async with sem:
            async for item in source:
                yield item

    async def generate_stream(
        self, request: LLMRequest,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream text and thought deltas, then one terminal chunk with usage."""
        visible: list[str] = []
        try:
            async for chunk in self._under_semaphore(self._stream_chunks(request)):
                if chunk.delta_text:
                    visible.append(chunk.delta_text)
                yield chunk
        finally:
            self._check_response_canary("".join(visible), context="generate_stream")

    async def _stream_chunks(
        self, request: LLMRequest,
    ) -> AsyncIterator[LLMStreamChunk]:
        model = self._model_of(request)
        body = self._body(request, structured=False)
        blocks: list[dict[str, Any]] = []
        request_id: str | None = None
        reason: str | None = None
        usage: Any = None
        async for payload in self._iter_payloads(body, model=model):
            if request_id is None and payload.get("responseId"):
                request_id = str(payload["responseId"])
            thought = thinking_of(payload) or ""
            text = text_of(payload)
            blocks.extend(_parts_of(payload))
            if payload.get("usageMetadata"):
                usage = payload["usageMetadata"]
            mapped = self._finish(payload)
            if mapped:
                reason = mapped
            if thought:
                yield LLMStreamChunk(
                    delta_reasoning=thought, request_id=request_id,
                )
            if text:
                yield LLMStreamChunk(delta_text=text, request_id=request_id)
        yield LLMStreamChunk(
            finish_reason=reason,
            usage=usage_from_gemini(usage) if usage else None,
            request_id=request_id,
            content_blocks=blocks or None,
        )

    async def generate_stream_events(
        self, request: LLMRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed deltas, then one Complete. Errors are emitted and re-raised."""
        visible: list[str] = []
        try:
            async for event in self._under_semaphore(self._stream_events(request)):
                if isinstance(event, ContentDelta) and event.delta_text:
                    visible.append(event.delta_text)
                yield event
        finally:
            self._check_response_canary(
                "".join(visible), context="generate_stream_events",
            )

    async def _stream_events(
        self, request: LLMRequest,
    ) -> AsyncIterator[StreamEvent]:
        model = self._model_of(request)
        body = self._body(request, structured=False)
        blocks: list[dict[str, Any]] = []
        request_id: str | None = None
        reason: str | None = None
        usage: Any = None
        tool_index = 0
        try:
            async for payload in self._iter_payloads(body, model=model):
                if request_id is None and payload.get("responseId"):
                    request_id = str(payload["responseId"])
                parts = _parts_of(payload)
                blocks.extend(parts)
                if payload.get("usageMetadata"):
                    usage = payload["usageMetadata"]
                mapped = self._finish(payload)
                if mapped:
                    reason = mapped
                for part in parts:
                    if part.get("thought") and part.get("text"):
                        yield ReasoningDelta(
                            delta_reasoning=str(part["text"]), request_id=request_id,
                        )
                    elif part.get("text"):
                        text = str(part["text"])
                        yield ContentDelta(delta_text=text, request_id=request_id)
                    call = part.get("functionCall")
                    if isinstance(call, dict) and call.get("name"):
                        args = call.get("args") if isinstance(call.get("args"), dict) else {}
                        yield ToolUseStart(
                            index=tool_index, id=f"call_{tool_index}",
                            name=str(call["name"]), request_id=request_id,
                        )
                        yield ToolUseDelta(
                            index=tool_index,
                            arguments_delta=json.dumps(args, ensure_ascii=False),
                            request_id=request_id,
                        )
                        yield ToolUseStop(index=tool_index, request_id=request_id)
                        tool_index += 1
            yield Complete(
                finish_reason=reason,
                usage=usage_from_gemini(usage) if usage else None,
                request_id=request_id,
                content_blocks=blocks or None,
            )
        except Exception as exc:
            yield Error(
                error=str(exc), error_type=type(exc).__name__, request_id=request_id,
            )
            raise
