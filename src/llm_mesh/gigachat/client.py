"""Async GigaChat client with OAuth and legacy functions/function_call structured output. OAuth
exchanges base64 client credentials for a bearer token; a 401 triggers one refresh and retry.
TLS verification defaults to False for installations without the required CA roots; callers can
configure verification explicitly. Output budgets are clipped to per-model limits and logged.
The default timeout is 600 seconds, overridable with LLM_HTTP_TIMEOUT for long generation
requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from llm_mesh.config import get_env
import re
import time
import uuid
from typing import Literal, Any, AsyncIterator, cast

import httpx

from llm_mesh._common import (
    _env_flag,
    _env_float_default,
    _env_is_disabled,
    _env_positive_int,
    _parse_json_dict_env,
    apply_canary as _apply_canary,
    build_text_messages as _build_text_messages,
    check_response_canary,
    finish_reason_opt,
    post_with_length_retry,
    warn_if_truncated,
)
from llm_mesh.base import BaseLLMClient, Capability
from llm_mesh.gigachat._common import (
    _env_nonneg_int,
    _parse_env_float,
    _parse_expires_at,
    _resolve_scope,
    ReasoningContentParser,
    simplify_schema_for_gigachat,
    split_reasoning_content,
)
from llm_mesh._retry import (
    RETRYABLE_SERVER_STATUS,
    backoff_with_jitter,
    retry_after_delay,
)
from llm_mesh._streaming import chunk_from_sse_payload, iter_sse_payloads
from llm_mesh.stream_events import (
    Complete,
    ContentDelta,
    Error,
    ReasoningDelta,
    StreamEvent,
)
from llm_mesh._streaming import events_from_sse_payload, ToolCallAccumulator

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


def _check_response_canary(response_text: str, context: str = "") -> None:
    """Scan generated text and serialized function arguments for the active canary. With no token
    this is a no-op; detection logs CRITICAL through the application hook without raising.
    Scanning covers individual generations, not only final user-facing output.
    """
    check_response_canary(response_text, context=context)


# Per-model completion-token limits, not context windows. Send max_tokens explicitly rather than
# relying on the provider default.
_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "GigaChat": 4096,
    "GigaChat-2": 4096,
    "GigaChat-2-Pro": 8192,
    "GigaChat-2-Max": 16384,
    "GigaChat-Pro": 8192,
    "GigaChat-Max": 16384,
    # Ultra requires its own exact match and tier fallback. The 32768 limit is an estimate, not
    # a confirmed public specification; a lower cap silently truncates output, while an
    # excessive request produces a visible gateway error.
    "GigaChat-3-Ultra": 32768,
}
_DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Deduplicate clipping warnings per model at module scope. Clients may be recreated for each
# call, so instance-local flags would flood logs.
_MAX_TOKENS_CLIP_WARNED: set[str] = set()


def _model_max_tokens(model: str) -> int:
    """Resolve the output-token limit by exact model match, then highest matching tier (Ultra
    32768, Max 16384, Pro 8192), then a conservative default. Tier fallback supports preview and
    future model names; check higher tiers first.
    """
    if model in _MODEL_MAX_OUTPUT_TOKENS:
        return _MODEL_MAX_OUTPUT_TOKENS[model]
    m = model.lower()
    if "ultra" in m:
        return 32768
    if "max" in m:
        return 16384
    if "pro" in m:
        return 8192
    return _DEFAULT_MAX_OUTPUT_TOKENS


# Override default endpoints with LLM_AUTH_URL and LLM_BASE_URL for proxy deployments.
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Try registered scopes in sequence if the requested or default scope fails.
_SCOPE_FALLBACKS: tuple[str, ...] = (
    "GIGACHAT_API_CORP",
    "GIGACHAT_API_B2B",
    "GIGACHAT_API_PERS",
)


def _resolve_credentials(explicit: str | None) -> str | None:
    """Resolve credentials from the explicit argument, then LLM_API_KEY."""
    if explicit:
        return explicit
    return get_env("LLM_API_KEY")


# Client construction and lifecycle.






class GigaChatAsyncClient(BaseLLMClient):
    """Async GigaChat client supporting structured output through legacy function calling.

    The instance model is the default. ``LLMRequest.model`` overrides it for
    one call. A separate instance is for a different capability set (output
    ceiling, tool choice), not for another model name.

    ``no_degrade`` and ``fallback_policy="preserve"`` refuse salvage of
    fenced JSON and prose function envelopes. ``LLM_EXTRA_BODY`` and
    ``LLM_EXTRA_HEADERS`` merge onto chat requests; fields the client
    already set win.
    """

    # TOOLS_REQUIRED is absent. generate_structured raises LLMValidationError
    # when request.tools_required is set: the legacy functions API has no
    # tool_calls array and no tool role, so a native tool loop cannot run.
    # BATCH is absent here too. make_client returns BatchingLLMClient when
    # LLM_BATCH_MODE is set; that adapter is not a BaseLLMClient.
    CAPABILITIES = frozenset({
        Capability.TEXT,
        Capability.STREAM,
        Capability.STREAM_EVENTS,
        Capability.STRUCTURED,
        # Legacy functions/function_call, not the modern tools/tool_calls API.
        Capability.TOOLS,
        # function_call="auto" selects one of request.tools. There is no
        # parallel tool_calls loop.
        Capability.MULTI_TOOL,
        # mode="json_schema" sends native response_format json_schema.
        Capability.JSON_SCHEMA_MODE,
        Capability.LENGTH_RETRY,
        Capability.COUNT_TOKENS,
        Capability.EMBEDDINGS,
    })

    def __init__(
        self,
        *,
        credentials: str | None = None,
        token: str | None = None,
        scope: str | None = None,
        model: str = "GigaChat",
        label: str | None = None,
        api_url: str | None = None,
        auth_url: str | None = None,
        verify: bool | None = None,
        timeout_s: float | None = None,
        max_token_refresh_attempts: int = 1,
        max_transient_retries: int = 2,
        transient_backoff_s: float = 1.0,
        max_concurrent: int | None = None,
        tool_choice: Literal["single", "auto"] = "single",
        use_model_token_limits: bool = True,
        no_degrade: bool | None = None,
        fallback_policy: Literal["recover", "preserve"] = "recover",
    ) -> None:
        if tool_choice not in ("single", "auto"):
            raise ValueError("tool_choice must be 'single' or 'auto'")
        if fallback_policy not in ("recover", "preserve"):
            raise ValueError("fallback_policy must be 'recover' or 'preserve'")
        self._tool_choice = tool_choice
        self._credentials = _resolve_credentials(credentials)
        self._token = token
        # Absolute token expiry in epoch seconds. None means unknown expiry and reactive refresh
        # on 401. Refresh known tokens early by _token_expiry_skew_s.
        self._token_expires_at: float | None = None
        self._token_expiry_skew_s = 60.0
        self._scope = _resolve_scope(scope)
        self.model = model
        # Same role as OpenAI/Anthropic PROVIDER: the probe label. The canary
        # override does not read it; those call sites already qualify context.
        self.PROVIDER = label or get_env("LLM_PROVIDER_LABEL") or "gigachat"
        self._api_url = (api_url or get_env("LLM_BASE_URL", GIGACHAT_BASE_URL)).rstrip("/")
        self._auth_url = auth_url or get_env("LLM_AUTH_URL", GIGACHAT_AUTH_URL)
        self._verify = verify if verify is not None else not _env_is_disabled(
            "LLM_VERIFY_SSL", default="false",
        )
        self._timeout = httpx.Timeout(
            timeout_s if timeout_s is not None
            else _env_float_default("LLM_HTTP_TIMEOUT", 600.0, logger=logger),
            connect=30.0,
        )
        self._max_refresh = max_token_refresh_attempts
        # Retry transient network failures and server errors. Rate limits use the separately
        # configured Retry-After/backoff handling.
        self._max_transient_retries = max_transient_retries
        self._transient_backoff_s = transient_backoff_s
        # On length truncation, double max_tokens up to the model ceiling.
        # LLM_LENGTH_RETRIES controls the retry budget; default is one.
        self._length_retries = _env_nonneg_int("LLM_LENGTH_RETRIES", 1)
        self._refresh_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

        # Limit outbound chat requests per instance. Configuration precedence is explicit
        # argument, LLM_MAX_CONCURRENT, then no limit. Lazily create the semaphore inside
        # the active event loop.
        if max_concurrent is None:
            max_concurrent = _env_positive_int("LLM_MAX_CONCURRENT")
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None

        # Use a catalog-configurable reasoning field, defaulting to reasoning_content.
        # Deployments with a different response field can declare it without changing parsing
        # code. When that field is empty, <think> tags inside content are the fallback.
        self._reasoning_field: str = (
            get_env("LLM_REASONING_FIELD", "").strip() or "reasoning_content"
        )

        # LLM_MAX_OUTPUT_TOKENS is an explicit ceiling for every model on this
        # client. Otherwise the ceiling is resolved per request: a per-call
        # model override must not inherit the instance model's limit.
        # _max_output_tokens remains the instance-model ceiling for readers
        # that inspect the client before a request.
        self._use_model_token_limits = use_model_token_limits
        self._forced_max_output_tokens = _env_positive_int("LLM_MAX_OUTPUT_TOKENS")
        if self._forced_max_output_tokens is not None:
            self._max_output_tokens = self._forced_max_output_tokens
        else:
            self._max_output_tokens = (
                _model_max_tokens(model) if use_model_token_limits else None
            )

        # Environment sampling overrides allow A/B measurements without changing caller
        # defaults, including temperature=1 with top_p=0. With no override, preserve the
        # caller's temperature and omit top_p.
        self._force_temperature = _parse_env_float("LLM_FORCE_TEMPERATURE")
        self._force_top_p = _parse_env_float("LLM_FORCE_TOP_P")

        # LLM_DISABLE_REASONING suppresses reasoning_effort and sends
        # chat_template_kwargs.enable_thinking=false for GigaChat reasoning models. The flag is
        # off by default, preserving existing generation behavior.
        self._disable_reasoning = _env_flag("LLM_DISABLE_REASONING")
        self._reasoning_effort = get_env("LLM_REASONING_EFFORT", "").strip()
        self._preserve_responses = fallback_policy == "preserve"
        self._no_degrade = (
            no_degrade if no_degrade is not None else _env_flag("LLM_NO_DEGRADE")
        )
        parsed_headers = _parse_json_dict_env(
            "LLM_EXTRA_HEADERS", logger=logger, include_error=False,
        )
        self._extra_headers = (
            {str(key): str(value) for key, value in parsed_headers.items()}
            if parsed_headers else {}
        )
        self._extra_body = _parse_json_dict_env(
            "LLM_EXTRA_BODY", logger=logger, include_error=False,
        ) or {}

        if not self._token and not self._credentials:
            raise LLMAuthError(
                "GigaChat: provide `credentials` (base64 ClientID:Secret) "
                "or `token`, or set env LLM_API_KEY"
            )

    async def __aenter__(self) -> GigaChatAsyncClient:
        # Reuse an already opened transport. Replacing it here used to drop
        # the previous client without closing it.
        self._ensure_http()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def _check_response_canary(self, response_text: str, *, context: str) -> None:
        """Scan text and serialized function arguments for the active canary.

        Call sites pass a fully qualified context such as gigachat.generate_text.
        The base helper would prefix PROVIDER and change the warning. Forward
        to the module function: stream tests replace that name, and a direct
        call to check_response_canary would hide the replacement.
        """
        _check_response_canary(response_text, context)

    async def count_tokens(
        self, texts: "list[str]", *, model: str | None = None
    ) -> list[int]:
        """Count tokens using POST /tokens/count, preserving input order. This uses the model
        tokenizer rather than a character heuristic and is intended for preflight context-budget
        checks, not a hot agent loop. Respect concurrency limits and refresh once on 401.
        """
        if not texts:
            return []
        body = {"model": model or self.model, "input": list(texts)}
        sem = self._ensure_semaphore()
        if sem is None:
            return await self._do_count_tokens(body)
        async with sem:
            return await self._do_count_tokens(body)

    async def _do_count_tokens(self, body: dict[str, Any]) -> list[int]:
        token = await self._ensure_token()
        client = self._ensure_http()
        url = f"{self._api_url}/tokens/count"
        for refresh_attempt in range(self._max_refresh + 1):
            resp = await client.post(
                url,
                json=body,
                headers=self._chat_headers(token, str(uuid.uuid4())),
            )
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list):
                    raise LLMValidationError(
                        f"GigaChat tokens/count: expected a list, got {type(data).__name__}"
                    )
                return [int((item or {}).get("tokens", 0)) for item in data]
            if resp.status_code == 401 and refresh_attempt < self._max_refresh:
                token = await self._refresh_token()
                continue
            raise LLMError(
                f"GigaChat tokens/count {resp.status_code}: {resp.text[:200]}"
            )
        raise LLMError("GigaChat tokens/count: token refresh attempts exhausted")

    def _apply_sampling(self, body: dict[str, Any], request_temperature: float) -> None:
        """Apply temperature/top_p environment overrides. Otherwise retain the caller's temperature
        and omit top_p.
        """
        body["temperature"] = (
            self._force_temperature if self._force_temperature is not None
            else request_temperature
        )
        if self._force_top_p is not None:
            body["top_p"] = self._force_top_p

    def _apply_extra_body(self, body: dict[str, Any]) -> None:
        """Fill holes from ``LLM_EXTRA_BODY``. Keys already on the body win."""
        for key, value in self._extra_body.items():
            body.setdefault(key, value)

    def _chat_headers(self, token: str, rquid: str, **more: str) -> dict[str, str]:
        """Bearer token and ``RqUID`` stay the client's. Extra headers fill the rest."""
        headers = {**self._extra_headers, **more}
        headers["Authorization"] = f"Bearer {token}"
        headers["RqUID"] = rquid
        headers.setdefault("Content-Type", "application/json")
        return headers

    @property
    def _forbids_silent_recovery(self) -> bool:
        """True when fenced JSON and prose function envelopes must not be accepted."""
        return self._no_degrade or self._preserve_responses

    def _resolve_reasoning_effort(self, request: "LLMRequest") -> str | None:
        """Resolve reasoning effort from the request, then LLM_REASONING_EFFORT. Accept low,
        medium, or high; ignore invalid values. When reasoning is disabled, omit this field and
        disable thinking separately in the request body.
        """
        if self._disable_reasoning:
            return None
        eff = request.reasoning_effort or self._reasoning_effort
        eff = (eff or "").lower()
        return eff if eff in ("low", "medium", "high") else None

    def _apply_reasoning_disable(self, body: dict[str, Any]) -> None:
        """When reasoning is disabled, merge enable_thinking=false into chat_template_kwargs
        without replacing other entries. Apply consistently to text, streaming, and structured
        request bodies.
        """
        if not self._disable_reasoning:
            return
        ctk = dict(body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        body["chat_template_kwargs"] = ctk

    def _effective_model(self, request: LLMRequest) -> str:
        """Request model, else the model configured on this client."""
        return request.model or self.model

    def _output_ceiling(self, model: str) -> int | None:
        """Explicit env ceiling, else the named model's limit, else none.

        use_model_token_limits=False with no LLM_MAX_OUTPUT_TOKENS means the
        caller owns the budget. A later length retry must not treat that as
        a numeric cap.
        """
        if self._forced_max_output_tokens is not None:
            return self._forced_max_output_tokens
        if not self._use_model_token_limits:
            return None
        return _model_max_tokens(model)

    def _clip_max_tokens(self, requested: int, model: str | None = None) -> int:
        """Clip max_tokens to the model ceiling and warn once per model per process when clipping
        occurs. Omit model to use the instance model, which is what callers
        did before per-call overrides existed.
        """
        name = model or self.model
        ceiling = self._output_ceiling(name)
        if ceiling is None or requested <= ceiling:
            return requested
        if name not in _MAX_TOKENS_CLIP_WARNED:
            logger.warning(
                "GigaChat(%s): max_tokens=%d exceeds per-model limit %d, "
                "clipping. Set max_tokens=%d or less for this model.",
                name, requested, ceiling,
                ceiling,
            )
            _MAX_TOKENS_CLIP_WARNED.add(name)
        return ceiling

    # --- OAuth -----------------------------------------------------------

    async def _fetch_token(self) -> tuple[str, float | None]:
        """Fetch a bearer token and optional absolute expiry using OAuth client credentials. Within
        each scope, retry rate limits with Retry-After, transient server/network errors with
        backoff, and recognized DNS-proxy errors with a short delay. Exhausted network retries
        raise LLMTimeoutError; terminal authorization failures advance to the next scope.
        """
        if not self._credentials:
            raise LLMAuthError("GigaChat: missing credentials for token refresh")

        scopes: tuple[str, ...] = (self._scope,) if self._scope else _SCOPE_FALLBACKS
        last_status: int | None = None
        last_text: str = ""
        client = self._ensure_http()
        max_attempts = max(5, self._max_transient_retries + 1)

        for scope in scopes:
            for attempt in range(max_attempts):
                try:
                    resp = await client.post(
                        self._auth_url,
                        headers={
                            "Authorization": f"Basic {self._credentials}",
                            "RqUID": str(uuid.uuid4()),
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                        data={"scope": scope},
                    )
                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:
                    if attempt + 1 < max_attempts:
                        delay = backoff_with_jitter(self._transient_backoff_s, attempt)
                        logger.warning(
                            "GigaChat auth: %s → retry in %.1fs (attempt %d/%d, scope=%s)",
                            type(exc).__name__,
                            delay,
                            attempt + 1,
                            max_attempts,
                            scope,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise LLMTimeoutError(
                        f"GigaChat auth timeout after {attempt + 1} attempts: {exc}"
                    ) from exc

                if resp.status_code == 200:
                    data = resp.json()
                    return str(data["access_token"]), _parse_expires_at(data.get("expires_at"))

                last_status, last_text = resp.status_code, resp.text[:200]
                if resp.status_code == 429 and attempt + 1 < max_attempts:
                    # Honor Retry-After for rate limits; otherwise use jittered backoff.
                    delay = retry_after_delay(resp, self._transient_backoff_s, attempt)
                    logger.warning(
                        "GigaChat auth: 429 rate limit → retry in %.1fs (attempt %d/%d, scope=%s)",
                        delay, attempt + 1, max_attempts, scope,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Retry transient OAuth server errors with jittered backoff within the current
                # scope, matching chat-request handling.
                if resp.status_code in RETRYABLE_SERVER_STATUS and attempt + 1 < max_attempts:
                    delay = backoff_with_jitter(self._transient_backoff_s, attempt)
                    logger.warning(
                        "GigaChat auth: %d → transient retry in %.1fs (attempt %d/%d, scope=%s)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                        max_attempts,
                        scope,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Retry recognized transient DNS/proxy resolution errors in the same scope;
                # these may recover after a short delay.
                if resp.status_code == 403 and (
                    "resolve_no_records" in resp.text
                    or "Host resolves" in resp.text
                ):
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                break  # This scope has failed terminally; try the next one.

        raise LLMAuthError(f"GigaChat auth failed (last={last_status}): {last_text}")

    def _token_expired(self) -> bool:
        """Return whether a cached token expires within the refresh margin. Unknown expiry,
        including explicitly supplied static tokens, is not treated as expired; handle those
        reactively on 401.
        """
        if self._token_expires_at is None:
            return False
        return time.time() >= self._token_expires_at - self._token_expiry_skew_s

    async def _ensure_token(self) -> str:
        if self._token and not self._token_expired():
            return self._token
        # Fetch missing or expired tokens only when credentials are available. A static token
        # cannot be refreshed without credentials.
        if not self._credentials:
            return self._token  # type: ignore[return-value]
        async with self._refresh_lock:
            if self._token and not self._token_expired():  # double-check (race)
                return self._token
            self._token, self._token_expires_at = await self._fetch_token()
        return self._token

    async def _refresh_token(self) -> str:
        """Force a token refresh after a 401 response."""
        async with self._refresh_lock:
            self._token = None
            self._token, self._token_expires_at = await self._fetch_token()
        return self._token

    # --- chat/completions -----------------------------------------------

    async def _post_chat_with_length_retry(
        self, body: dict[str, Any], *, model: str,
    ) -> tuple[dict[str, Any], str | None]:
        """Wrap transport retries with adaptive length retries. Double max_tokens up to the model
        ceiling, at most _length_retries times. Normal responses and requests already at the
        ceiling are unchanged. This avoids returning truncated text or incomplete structured
        JSON. The ceiling is the effective model's, not the instance model's.
        """
        ceiling = self._output_ceiling(model)

        def next_max_tokens(current: int) -> int:
            # No ceiling: double without a cap. min(current * 2, None) raises
            # TypeError, which used to fail the call when model limits were off.
            if ceiling is None:
                return current * 2 if current else 0
            if not current:
                return ceiling
            return min(current * 2, ceiling)

        return await post_with_length_retry(
            body,
            post=self._post_chat_with_retry,
            payload_of=lambda result: result[0],
            retries=self._length_retries,
            next_max_tokens=next_max_tokens,
            provider=f"GigaChat({model})",
            logger=logger,
        )

    async def _post_chat_with_retry(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """POST chat/completions under the concurrency semaphore, returning (payload, request_id).
        Prefer the gateway x-request-id header, otherwise use the generated RqUID for tracing.
        """
        sem = self._ensure_semaphore()
        if sem is None:
            return await self._do_post_chat_with_retry(body)
        async with sem:
            return await self._do_post_chat_with_retry(body)

    async def _do_post_chat_with_retry(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """POST chat/completions with bounded transient network/server retries and backoff.
        Authentication refresh has a separate retry budget. Rate-limit handling honors
        Retry-After. Successful responses return parsed JSON; exhausted failures propagate as
        the appropriate LLM error.
        """
        token = await self._ensure_token()
        client = self._ensure_http()
        url = f"{self._api_url}/chat/completions"

        # --- transient retry-loop ---
        for transient_attempt in range(self._max_transient_retries + 1):
            retry_delay: float | None = None  # Set after observing a retryable HTTP status.
            try:
                # --- token-refresh loop (orthogonal) ---
                for refresh_attempt in range(self._max_refresh + 1):
                    rquid = str(uuid.uuid4())
                    try:
                        resp = await client.post(
                            url,
                            json=body,
                            headers=self._chat_headers(token, rquid),
                        )
                    except (
                        httpx.TimeoutException,
                        httpx.ConnectError,
                        httpx.ReadError,
                        httpx.NetworkError,
                        httpx.RemoteProtocolError,
                    ):
                        # Propagate to the outer transient retry loop.
                        raise

                    if resp.status_code == 200:
                        request_id = resp.headers.get("x-request-id") or rquid
                        return cast("dict[str, Any]", resp.json()), request_id

                    if resp.status_code == 401 and refresh_attempt < self._max_refresh:
                        logger.info(
                            "GigaChat: 401 → token refresh attempt %d",
                            refresh_attempt + 1,
                        )
                        token = await self._refresh_token()
                        continue

                    # Honor Retry-After on 429, otherwise use jittered backoff.
                    if resp.status_code == 429:
                        retry_delay = retry_after_delay(
                            resp, self._transient_backoff_s, transient_attempt
                        )
                        logger.warning(
                            "GigaChat: 429 rate limit → retry in %.1fs (attempt %d)",
                            retry_delay, transient_attempt + 1,
                        )
                        break  # Exit the refresh loop and back off in the transient retry loop.

                    # transient server-side errors (500/502/503/504) → retry
                    if resp.status_code in RETRYABLE_SERVER_STATUS:
                        retry_delay = backoff_with_jitter(
                            self._transient_backoff_s, transient_attempt
                        )
                        logger.warning(
                            "GigaChat: %d → transient retry in %.1fs (attempt %d)",
                            resp.status_code, retry_delay, transient_attempt + 1,
                        )
                        break

                    # Terminal errors.
                    text = resp.text[:300]
                    if resp.status_code == 401:
                        raise LLMAuthError(f"GigaChat 401 after refresh: {text}")
                    raise LLMError(f"GigaChat {resp.status_code}: {text}")
                else:
                    # The refresh loop exhausted its budget without returning or breaking.
                    raise LLMError("GigaChat: token refresh attempts exhausted without a response")

                # Back off and retry after a retryable 5xx or 429 response.
                if retry_delay is not None:
                    if transient_attempt < self._max_transient_retries:
                        await asyncio.sleep(retry_delay)
                        continue
                    raise LLMError(
                        f"GigaChat: transient error (5xx/429) after "
                        f"{transient_attempt + 1} attempts"
                    )

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                if transient_attempt < self._max_transient_retries:
                    delay = backoff_with_jitter(self._transient_backoff_s, transient_attempt)
                    logger.warning(
                        "GigaChat: %s → retry in %.1fs (attempt %d/%d)",
                        type(exc).__name__,
                        delay,
                        transient_attempt + 1,
                        self._max_transient_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMTimeoutError(
                    f"GigaChat transient error after {transient_attempt + 1} attempts: {exc}"
                ) from exc

        # Unreachable.
        raise LLMError("GigaChat: internal error in retry logic")

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task: str = "document",
        dimensions: int | None = None,
        sparse: bool | None = None,
    ) -> list[Any]:
        """POST ``/embeddings``. GigaChat returns dense vectors only."""
        from llm_mesh.embeddings import (
            embedding_body,
            embedding_options,
            parse_embedding_response,
            prepare_inputs,
        )

        if not texts:
            return []
        instruction, use_sparse, use_dimensions = embedding_options(
            self, dimensions=dimensions, sparse=sparse, allow_sparse=False,
        )
        prepared = prepare_inputs(texts, task, instruction)
        if use_dimensions is not None:
            raise LLMValidationError(
                "GigaChat embeddings do not accept dimensions; "
                "the vector width is the model's"
            )
        body = embedding_body(model or self.model, prepared, sparse=False)
        sem = self._ensure_semaphore()
        if sem is None:
            payload = await self._post_embeddings(body)
        else:
            async with sem:
                payload = await self._post_embeddings(body)
        return parse_embedding_response(payload, len(prepared), sparse=False)

    async def _post_embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        token = await self._ensure_token()
        client = self._ensure_http()
        url = f"{self._api_url}/embeddings"
        max_attempts = self._max_transient_retries + 1
        for attempt in range(max_attempts):
            refreshed = False
            while True:
                try:
                    resp = await client.post(
                        url,
                        json=body,
                        headers=self._chat_headers(token, str(uuid.uuid4())),
                    )
                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:
                    if attempt + 1 >= max_attempts:
                        raise LLMTimeoutError(
                            f"GigaChat embeddings transient error after "
                            f"{attempt + 1} attempts: {exc}"
                        ) from exc
                    delay = backoff_with_jitter(self._transient_backoff_s, attempt)
                    logger.warning(
                        "GigaChat embeddings: %s → retry in %.1fs (attempt %d/%d)",
                        type(exc).__name__, delay, attempt + 1, max_attempts,
                    )
                    await asyncio.sleep(delay)
                    break
                if resp.status_code == 200:
                    data = resp.json()
                    if not isinstance(data, dict):
                        raise LLMValidationError(
                            "GigaChat embeddings: response is not an object"
                        )
                    return data
                if resp.status_code == 401 and not refreshed and self._max_refresh > 0:
                    token = await self._refresh_token()
                    refreshed = True
                    continue
                retryable = (
                    resp.status_code == 429
                    or resp.status_code in RETRYABLE_SERVER_STATUS
                )
                if retryable and attempt + 1 < max_attempts:
                    delay = (
                        retry_after_delay(resp, self._transient_backoff_s, attempt)
                        if resp.status_code == 429
                        else backoff_with_jitter(self._transient_backoff_s, attempt)
                    )
                    logger.warning(
                        "GigaChat embeddings: %d → retry in %.1fs (attempt %d/%d)",
                        resp.status_code, delay, attempt + 1, max_attempts,
                    )
                    await asyncio.sleep(delay)
                    break
                raise LLMError(
                    f"GigaChat embeddings {resp.status_code}: {resp.text[:200]}"
                )
        raise LLMError("GigaChat embeddings: retries exhausted")

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate plain text without functions or tools. Suitable for code or prose generation
        where function calling is unnecessary; parse message.content as text.
        """
        request = self._guarded_request(request)
        body = {
            "model": self._effective_model(request),
            "messages": _build_text_messages(request, tool_turns=False),
            "max_tokens": self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
        }
        self._apply_sampling(body, request.temperature)
        effort = self._resolve_reasoning_effort(request)
        if effort:
            body["reasoning_effort"] = effort
        self._apply_reasoning_disable(body)
        self._apply_extra_body(body)
        # When length_retry=False, preserve the small best-effort budget instead of escalating a
        # truncated or repetitive response.
        if request.length_retry:
            payload, request_id = await self._post_chat_with_length_retry(
                body, model=self._effective_model(request),
            )
        else:
            payload, request_id = await self._post_chat_with_retry(body)
        if request.mode == "json_schema" or (request.tools and self._tool_choice == "auto"):
            try:
                return self._parse_extended_response(payload, request, request_id)
            except LLMValidationError as exc:
                exc.payload = payload
                raise
        try:
            choice = payload["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError) as exc:
            raise LLMValidationError(f"GigaChat: malformed response: {exc}", payload=payload) from exc
        content = message.get("content", "") or ""
        if not isinstance(content, str):
            content = ""
        visible, reasoning = self._visible_and_reasoning(message)
        warn_if_truncated(
            choice,
            request,
            content,
            sent_max_tokens=self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
            provider="GigaChat",
            logger=logger,
        )
        usage_raw = payload.get("usage", {})
        # Normalize usage in LLMUsage.from_raw: GigaChat reports flat reasoning counts, while
        # OpenAI-compatible providers may nest them.
        usage = LLMUsage.from_raw(usage_raw)
        response = LLMResponse(
            arguments={},
            text=visible,
            reasoning_content=reasoning,
            finish_reason=finish_reason_opt(payload),
            request_id=request_id,
            model=str(payload.get("model", self._effective_model(request))),
            usage=usage,
            raw=payload,
        )
        self._check_response_canary(content, context="gigachat.generate_text")
        return response

    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream plain text as LLMStreamChunk deltas, including optional reasoning, final
        finish_reason, and usage. Attach request_id to the first chunk. Legacy GigaChat function
        calls are not streamed as argument deltas; use generate_structured for structured
        output.
        """
        request = self._guarded_request(request)
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": _build_text_messages(request, tool_turns=False),
            "max_tokens": self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
            "stream": True,
        }
        self._apply_sampling(body, request.temperature)
        effort = self._resolve_reasoning_effort(request)
        if effort:
            body["reasoning_effort"] = effort
        self._apply_reasoning_disable(body)
        self._apply_extra_body(body)

        sem = self._ensure_semaphore()
        if sem is None:
            async for chunk in self._do_stream(body):
                yield chunk
            return
        async with sem:
            async for chunk in self._do_stream(body):
                yield chunk

    async def _do_stream(self, body: dict[str, Any]) -> AsyncIterator[LLMStreamChunk]:
        token = await self._ensure_token()
        client = self._ensure_http()
        url = f"{self._api_url}/chat/completions"
        refreshed = False

        while True:
            rquid = str(uuid.uuid4())
            headers = self._chat_headers(
                token, rquid, Accept="text/event-stream",
            )
            agg: list[str] = []  # Per-attempt state; finally scans any partial output.
            try:
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as resp:
                    if resp.status_code == 401 and not refreshed:
                        await resp.aread()
                        logger.info("GigaChat stream: 401 → token refresh")
                        token = await self._refresh_token()
                        refreshed = True
                        continue
                    if resp.status_code != 200:
                        raw = (await resp.aread()).decode("utf-8", "replace")[:300]
                        if resp.status_code == 401:
                            raise LLMAuthError(
                                f"GigaChat stream 401 after refresh: {raw}"
                            )
                        raise LLMError(f"GigaChat stream {resp.status_code}: {raw}")

                    request_id = resp.headers.get("x-request-id") or rquid
                    first = True
                    parser = ReasoningContentParser()
                    closed = False
                    async for payload in iter_sse_payloads(resp.aiter_lines()):
                        chunk = chunk_from_sse_payload(
                            payload, request_id=request_id, first=first,
                            reasoning_field=self._reasoning_field,
                        )
                        if chunk.delta_text:
                            agg.append(chunk.delta_text)
                        flush = chunk.finish_reason is not None
                        if flush:
                            closed = True
                        yield self._apply_think_chunk(parser, chunk, flush=flush)
                        first = False
                    if not closed:
                        tail = parser.flush()
                        if tail.content or tail.reasoning_content:
                            yield LLMStreamChunk(
                                delta_text=tail.content,
                                delta_reasoning=tail.reasoning_content,
                            )
                    return
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                raise LLMTimeoutError(f"GigaChat stream transient error: {exc}") from exc
            finally:
                # Scan on completion, network interruption, and early consumer termination. An
                # empty aggregate on a 401 retry is a no-op.
                self._check_response_canary(
                    "".join(agg), context="gigachat.generate_stream"
                )

    # Bound retries when GigaChat returns neither content nor function_call despite an
    # explicitly forced function.
    _MAX_NO_FC_RETRIES: int = 2

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate parsed function arguments and usage, with transport retry/refresh handling and
        bounded retries for a missing function_call. Reject tools_required explicitly: legacy
        functions cannot provide the modern required-tool selection contract. The caller can
        then choose text emulation rather than mistaking a forced single function for a
        model-selected tool.
        """
        request = self._guarded_request(request)
        if request.tools_required:
            raise LLMValidationError(
                "GigaChat: native tool-loop (tools_required) is not supported — "
                "legacy functions API without tool_calls/tool role"
            )
        body = self._build_body(request)
        last_exc: LLMValidationError | None = None
        for attempt in range(self._MAX_NO_FC_RETRIES + 1):
            if attempt:
                logger.warning(
                    "GigaChat: missing function_call → retry %d/%d (fn=%s)",
                    attempt,
                    self._MAX_NO_FC_RETRIES,
                    request.function_name,
                )
            payload = None
            request_id = None
            try:
                payload, request_id = await self._post_chat_with_length_retry(
                    body, model=self._effective_model(request),
                )
                response = self._parse_response(payload, request, request_id)
            except LLMValidationError as exc:
                # Capture every rejected response, including retries that later
                # succeed. Response bodies only: no request/auth headers. DEBUG
                # is opt-in because generated content may contain user data.
                logger.debug(
                    "GigaChat structured response rejected fn=%s attempt=%d request_id=%s "
                    "exception=%s raw_response=%r",
                    request.function_name, attempt + 1, request_id,
                    type(exc).__name__, payload, exc_info=True,
                )
                if "missing function_call" in str(exc) and attempt < self._MAX_NO_FC_RETRIES:
                    last_exc = exc
                    continue
                raise
            # Scan both text and serialized function arguments: structured output can leak the
            # canary too.
            self._check_response_canary(
                response.text or "", context="gigachat.generate_structured.text",
            )
            if response.arguments:
                try:
                    import json as _json
                    args_str = _json.dumps(response.arguments, ensure_ascii=False, default=str)
                    self._check_response_canary(
                        args_str, context="gigachat.generate_structured.args",
                    )
                except Exception:
                    pass
            return response
        raise last_exc  # type: ignore[misc]

    # --- helpers --------------------------------------------------------

    def _build_body(self, request: LLMRequest) -> dict[str, Any]:
        """Build /chat/completions payloads by request.mode. json_schema uses native
        response_format with the raw schema; GigaChat does not support json_object. The default
        function_call mode uses legacy functions: auto selection with request.tools or one
        forced request.function_name. Convert history to the same legacy wire format.
        """
        messages = (self._legacy_messages(request) if self._tool_choice == "auto"
                    else _build_text_messages(request, tool_turns=False))
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": messages,
            "max_tokens": self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
        }
        self._apply_sampling(body, request.temperature)
        effort = self._resolve_reasoning_effort(request)
        if effort:
            body["reasoning_effort"] = effort
        self._apply_reasoning_disable(body)
        self._apply_extra_body(body)

        if request.mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "schema": request.schema_ or {"type": "object", "properties": {}},
                "strict": True,
            }
            return body

        if request.tools and self._tool_choice == "auto":
            body["functions"] = [
                {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": simplify_schema_for_gigachat(
                        t.get("parameters") or {}
                    ),
                }
                for t in request.tools
            ]
            body["function_call"] = "auto"
            return body

        body["functions"] = [
            {
                "name": request.function_name,
                "description": request.function_description
                or f"Build a {request.function_name} object.",
                "parameters": simplify_schema_for_gigachat(request.schema_ or {}),
            }
        ]
        body["function_call"] = {"name": request.function_name}
        return body


    def _parse_response(
        self, payload: dict[str, Any], request: LLMRequest, request_id: str | None = None
    ) -> LLMResponse:
        if request.mode == "json_schema" or (request.tools and self._tool_choice == "auto"):
            try:
                return self._parse_extended_response(payload, request, request_id)
            except LLMValidationError as exc:
                exc.payload = payload
                raise
        try:
            choice = payload["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError) as exc:
            raise LLMValidationError(f"GigaChat: malformed response: {exc}", payload=payload) from exc

        warn_if_truncated(
            choice,
            request,
            message.get("content", "") or "",
            sent_max_tokens=self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
            provider="GigaChat",
            logger=logger,
        )
        fc = message.get("function_call")
        if not fc:
            content = message.get("content", "")
            raise LLMValidationError(
                f"GigaChat: missing function_call in response (content={content[:200]!r})",
                payload=payload,
            )

        raw_args = fc.get("arguments", {})
        # GigaChat may return arguments as either a mapping or a JSON string.
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LLMValidationError(
                    f"GigaChat: invalid JSON in arguments: {exc}",
                    payload=payload,
                ) from exc
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            raise LLMValidationError(
                f"GigaChat: arguments have unexpected type {type(raw_args).__name__}"
            )

        usage_raw = payload.get("usage", {})
        # Normalize usage in LLMUsage.from_raw: GigaChat reports flat reasoning counts, while
        # OpenAI-compatible providers may nest them.
        usage = LLMUsage.from_raw(usage_raw)

        # Retain request for future argument validation against request.schema_, although
        # response construction does not currently use it.
        _ = request
        # Expose the model-selected function name so the caller can route a multi-tool response.
        chosen_fn = fc.get("name") if isinstance(fc, dict) else None
        _, reasoning = self._visible_and_reasoning(message)
        return LLMResponse(
            arguments=arguments,
            function_name=chosen_fn,
            reasoning_content=reasoning,
            finish_reason=finish_reason_opt(payload),
            request_id=request_id,
            model=str(payload.get("model", self._effective_model(request))),
            usage=usage,
            raw=payload,
        )

    def _parse_extended_response(
        self, payload: dict[str, Any], request: LLMRequest, request_id: str | None = None
    ) -> LLMResponse:
        try:
            choice = payload["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError) as exc:
            raise LLMValidationError(f"GigaChat: malformed response: {exc}") from exc

        usage_raw = payload.get("usage", {})
        # Centralize usage parsing in LLMUsage.from_raw: GigaChat reports reasoning tokens at
        # the top level, while OpenAI-compatible providers may nest them in
        # completion_tokens_details.
        usage = LLMUsage.from_raw(usage_raw)

        # Native response_format returns JSON in content rather than function_call. Allow
        # Markdown fences or surrounding prose through salvage parsing.
        if request.mode == "json_schema":
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = ""
            visible, reasoning = self._visible_and_reasoning(message)
            warn_if_truncated(choice, request, content, sent_max_tokens=self._clip_max_tokens(request.max_tokens, self._effective_model(request)), provider="GigaChat", logger=logger)
            arguments = self._json_schema_arguments(visible)
            if arguments is None:
                why = (
                    "non-JSON content; LLM_NO_DEGRADE or "
                    "fallback_policy=preserve forbids salvage"
                    if self._forbids_silent_recovery
                    else "unparseable JSON"
                )
                raise LLMValidationError(
                    f"GigaChat: response_format returned {why} "
                    f"(content={content[:200]!r})"
                )
            return LLMResponse(
                arguments=arguments,
                function_name=request.function_name,
                text=visible,
                reasoning_content=reasoning,
                request_id=request_id,
                model=str(payload.get("model", self._effective_model(request))),
                usage=usage,
                finish_reason=finish_reason_opt(payload),
                raw=payload,
            )

        # Legacy output may contain function_call, tool_calls, or a textual call envelope in
        # content.
        content = message.get("content", "") or ""
        if not isinstance(content, str):
            content = ""
        visible, reasoning = self._visible_and_reasoning(message)
        warn_if_truncated(choice, request, content, sent_max_tokens=self._clip_max_tokens(request.max_tokens, self._effective_model(request)), provider="GigaChat", logger=logger)
        stripped = dict(message)
        stripped["content"] = visible
        chosen_fn, arguments = self._extract_function_call(stripped, request)
        if chosen_fn is None:
            if request.tools:
                # In auto mode, return an empty selection when the model answers in text; do not
                # retry it as a missing forced function call.
                return LLMResponse(
                    arguments={},
                    function_name=None,
                    text=visible,
                    reasoning_content=reasoning,
                    request_id=request_id,
                    model=str(payload.get("model", self._effective_model(request))),
                    usage=usage,
                    finish_reason=finish_reason_opt(payload),
                    raw=payload,
                )
            raise LLMValidationError(
                f"GigaChat: missing function_call in response (content={content[:200]!r})"
            )
        return LLMResponse(
            arguments=arguments,
            function_name=chosen_fn,
            text=visible or None,
            reasoning_content=reasoning,
            request_id=request_id,
            model=str(payload.get("model", self._effective_model(request))),
            usage=usage,
            finish_reason=finish_reason_opt(payload),
            raw=payload,
        )

    async def generate_stream_events(
        self, request: LLMRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed content/reasoning deltas followed by exactly one Complete with
        finish_reason and usage after DONE. GigaChat returns function calls whole, so no tool
        events are emitted; use generate_structured for those. Emit Error and re-raise transport
        failures. Refresh credentials once on 401, as in generate_stream.
        """
        request = self._guarded_request(request)
        body: dict[str, Any] = {
            "model": self._effective_model(request),
            "messages": _build_text_messages(request, tool_turns=False),
            "max_tokens": self._clip_max_tokens(request.max_tokens, self._effective_model(request)),
            "stream": True,
        }
        self._apply_sampling(body, request.temperature)
        effort = self._resolve_reasoning_effort(request)
        if effort:
            body["reasoning_effort"] = effort
        self._apply_reasoning_disable(body)
        self._apply_extra_body(body)

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
        """Stream provider events and assemble any tool deltas supplied by the endpoint."""

        token = await self._ensure_token()
        client = self._ensure_http()
        url = f"{self._api_url}/chat/completions"
        refreshed = False
        agg: list[str] = []  # Per-attempt state; finally scans any partial output.

        while True:
            rquid = str(uuid.uuid4())
            headers = self._chat_headers(
                token, rquid, Accept="text/event-stream",
            )
            try:
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as resp:
                    if resp.status_code == 401 and not refreshed:
                        await resp.aread()
                        logger.info("GigaChat stream_events: 401 → token refresh")
                        token = await self._refresh_token()
                        refreshed = True
                        continue
                    if resp.status_code != 200:
                        raw = (await resp.aread()).decode("utf-8", "replace")[:300]
                        if resp.status_code == 401:
                            err: Exception = LLMAuthError(
                                f"GigaChat stream 401 after refresh: {raw}"
                            )
                        else:
                            err = LLMError(f"GigaChat stream {resp.status_code}: {raw}")
                        yield Error(error=str(err), error_type=type(err).__name__)
                        raise err

                    request_id = resp.headers.get("x-request-id") or rquid
                    tool_acc = ToolCallAccumulator()
                    parser = ReasoningContentParser()
                    finish_reason: str | None = None
                    usage: LLMUsage | None = None
                    async for payload in iter_sse_payloads(resp.aiter_lines()):
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
                            usage = LLMUsage.from_raw(payload["usage"])
                        events = events_from_sse_payload(
                            payload,
                            request_id=request_id,
                            tool_acc=tool_acc,
                            reasoning_field=self._reasoning_field,
                        )
                        for ev in events:
                            if isinstance(ev, ContentDelta):
                                agg.append(ev.delta_text)
                        for ev in self._events_with_think(parser, events):
                            yield ev
                    tail = parser.flush()
                    if tail.reasoning_content:
                        yield ReasoningDelta(
                            delta_reasoning=tail.reasoning_content,
                            request_id=request_id,
                        )
                    if tail.content:
                        yield ContentDelta(delta_text=tail.content, request_id=request_id)
                    for stop in tool_acc.finalize(request_id=request_id):
                        yield stop
                    yield Complete(
                        finish_reason=finish_reason,
                        usage=usage,
                        request_id=request_id,
                    )
                    return
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                err = LLMTimeoutError(f"GigaChat stream transient error: {exc}")
                yield Error(error=str(err), error_type=type(err).__name__)
                raise err from exc
            finally:
                # Scan partial output for canaries as in generate_stream. The aggregate is empty
                # during a 401 retry, so the scan is a no-op.
                self._check_response_canary(
                    "".join(agg), context="gigachat.generate_stream_events"
                )

    @staticmethod
    def _history_fc_arguments(raw_args: Any) -> dict[str, Any]:
        """Normalize history function_call arguments to an object. GigaChat rejects the JSON
        strings commonly used in OpenAI history. Return an empty object for malformed JSON.
        """
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _parse_function_arguments(raw_args: Any) -> dict[str, Any]:
        """Parse function-call arguments supplied as a JSON string or dictionary into a dictionary."""
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LLMValidationError(
                    f"GigaChat: invalid JSON in arguments: {exc}"
                ) from exc
        elif isinstance(raw_args, dict):
            parsed = raw_args
        else:
            raise LLMValidationError(
                f"GigaChat: arguments have unexpected type {type(raw_args).__name__}"
            )
        if not isinstance(parsed, dict):
            raise LLMValidationError(
                f"GigaChat: arguments are not an object ({type(parsed).__name__})"
            )
        return parsed

    def _extract_function_call(
        self, message: dict[str, Any], request: LLMRequest
    ) -> tuple[str | None, dict[str, Any]]:
        """Extract the function name and arguments from legacy calls, tool_calls, or content
        envelopes.
        """
        fc = message.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            return str(fc["name"]), self._parse_function_arguments(
                fc.get("arguments", {})
            )

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
            fn_obj = tc0.get("function") if isinstance(tc0.get("function"), dict) else {}
            name = fn_obj.get("name")
            if name:
                return str(name), self._parse_function_arguments(
                    fn_obj.get("arguments", {})
                )

        if request.tools and not self._forbids_silent_recovery:
            allowed = {t.get("name") for t in request.tools if t.get("name")}
            content = message.get("content") or ""
            m = self._PSEUDO_FN_RE.match(content)
            if m and m.group(1) in allowed:
                try:
                    args = json.loads(m.group(2))
                except (ValueError, TypeError):
                    args = {}
                return m.group(1), args if isinstance(args, dict) else {}
        return None, {}

    def _visible_and_reasoning(self, message: dict[str, Any]) -> tuple[str, str | None]:
        """Read the configured reasoning field, falling back to ``<think>`` tags in content."""
        field = message.get(self._reasoning_field)
        return split_reasoning_content(
            message.get("content") or "",
            field if isinstance(field, str) else None,
        )

    @staticmethod
    def _apply_think_chunk(
        parser: ReasoningContentParser,
        chunk: LLMStreamChunk,
        *,
        flush: bool,
    ) -> LLMStreamChunk:
        """Move think tags from a text delta into reasoning. A provider reasoning delta on the
        same chunk wins. flush releases a tag prefix held across earlier deltas.
        """
        parsed = parser.feed(chunk.delta_text)
        visible = parsed.content
        extracted = parsed.reasoning_content
        if flush:
            tail = parser.flush()
            visible += tail.content
            extracted += tail.reasoning_content
        reasoning = chunk.delta_reasoning or extracted
        if visible == chunk.delta_text and reasoning == chunk.delta_reasoning:
            return chunk
        return chunk.model_copy(update={"delta_text": visible, "delta_reasoning": reasoning})

    @staticmethod
    def _events_with_think(
        parser: ReasoningContentParser,
        events: list[StreamEvent],
    ) -> list[StreamEvent]:
        """Rewrite content events so think tags become reasoning deltas. A reasoning delta
        already present in this payload stays the source for that payload.
        """
        has_field = any(isinstance(ev, ReasoningDelta) and ev.delta_reasoning for ev in events)
        rewritten: list[StreamEvent] = []
        for ev in events:
            if not isinstance(ev, ContentDelta):
                rewritten.append(ev)
                continue
            parsed = parser.feed(ev.delta_text)
            if parsed.reasoning_content and not has_field:
                rewritten.append(
                    ReasoningDelta(
                        delta_reasoning=parsed.reasoning_content,
                        request_id=ev.request_id,
                    )
                )
            if parsed.content:
                rewritten.append(
                    ContentDelta(delta_text=parsed.content, request_id=ev.request_id)
                )
        return rewritten

    def _json_schema_arguments(self, content: str) -> dict[str, Any] | None:
        """Parse native JSON Schema content.

        Recovery accepts fences and a JSON object buried in prose. With
        ``no_degrade`` or ``fallback_policy="preserve"`` only a JSON object
        that is the whole content counts.
        """
        if not self._forbids_silent_recovery:
            return self._parse_json_content(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any] | None:
        """Parse JSON content directly, from a Markdown fence, or from the first object block, in
        that order. Return None if all attempts fail.
        """
        if not content:
            return None
        # Try direct JSON parsing.
        try:
            v = json.loads(content)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            pass
        # Try a Markdown fence, optionally labeled json.
        s = content.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            try:
                v = json.loads("\n".join(lines))
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                pass
        # Salvage the first object block.
        first = content.find("{")
        last = content.rfind("}")
        if first != -1 and last > first:
            try:
                v = json.loads(content[first:last + 1])
                return v if isinstance(v, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    _PSEUDO_FN_RE = re.compile(r"^\s*(\w+)\s*=?\s*\n?\s*(\{.*\})\s*$", re.DOTALL)

    def _legacy_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Convert conversation history to legacy function turns."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _apply_canary(request.system)},
        ]
        # Remember the latest function name when converting role=tool to role=function.
        last_tool_name = ""
        for turn in request.history or []:
            role = turn.get("role")
            # Convert modern tool_calls/tool history into legacy assistant.function_call and
            # role=function turns.
            if role == "assistant" and turn.get("function_call"):
                fc = turn["function_call"]
                name = ""
                args: Any = {}
                if isinstance(fc, dict):
                    name = str(fc.get("name") or "")
                    args = self._history_fc_arguments(fc.get("arguments"))
                messages.append({
                    "role": "assistant",
                    "content": turn.get("content"),
                    "function_call": {"name": name, "arguments": args},
                })
                if name:
                    last_tool_name = name
            elif role == "assistant" and turn.get("tool_calls"):
                tc0 = (turn["tool_calls"] or [{}])[0]
                fn = (tc0.get("function") if isinstance(tc0, dict) else None) or {}
                name = fn.get("name") or ""
                # GigaChat history requires object arguments; JSON strings cause a 400 invalid
                # JSON syntax error.
                args = self._history_fc_arguments(fn.get("arguments"))
                messages.append({
                    "role": "assistant",
                    "content": turn.get("content"),
                    "function_call": {"name": name, "arguments": args},
                })
                last_tool_name = str(name)
            elif role == "function":
                messages.append({
                    "role": "function",
                    "name": turn.get("name") or last_tool_name or "",
                    "content": str(turn.get("content", "")),
                })
            elif role == "tool":
                messages.append({
                    "role": "function",
                    "name": turn.get("name") or last_tool_name or "",
                    "content": str(turn.get("content", "")),
                })
            elif role in ("user", "assistant"):
                messages.append(
                    {"role": role, "content": str(turn.get("content", ""))}
                )
        # An empty user message continues after role=function without adding a new user turn.
        if request.user:
            messages.append({"role": "user", "content": request.user})
        return messages
