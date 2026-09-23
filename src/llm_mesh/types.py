"""Shared LLM request, response, usage, streaming, and error types."""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- Errors -----------------------------------------------------------------


class LLMError(RuntimeError):
    """Base exception for LLM calls."""


class LLMRequestBlocked(LLMError):
    """The request hook refused the call before it reached the provider.

    This is not LLMValidationError. Tier ladders, length-retry, and
    degradation catch LLMValidationError and provider errors such as
    OpenAIError, and may retry or fall through to another tier. A guard
    refusal is the application's decision. It must not be recovered and must
    not count as degradation. no_degrade and preserve do not apply.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class LLMBudgetExceeded(LLMError):
    """Accumulated spend reached a ceiling before this call was sent.

    This is not LLMValidationError. Tier ladders and length-retry catch
    validation errors and may try again. A budget refusal is final: it is
    not retried and it is not degradation.

    A ceiling in one currency does not convert or account the other. A
    route that reports both ``cost_usd`` and ``cost_rub`` needs both
    fields set, or the unconfigured currency is not a limit.
    """


class LLMAuthError(LLMError):
    """OAuth or bearer authentication failure, including failed token refresh."""


class LLMTimeoutError(LLMError):
    """LLM API timeout or transient transport exhaustion."""


class LLMValidationError(LLMError):
    """A model response violates the contract, such as missing function calls or invalid JSON."""

    def __init__(self, *args: object, payload: object = None) -> None:
        super().__init__(*args)
        self.payload = payload


# --- request / response ----------------------------------------------------


class LLMRequest(BaseModel):
    """Unified request for function-calling or text mode structured output. schema_ supplies the
    JSON Schema used in the API payload or text prompt; function_name identifies the forced
    function when applicable.
    """

    model_config = ConfigDict(extra="forbid")

    system: str
    user: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")  # The schema alias would otherwise shadow BaseModel.schema().
    function_name: str = "build_artifact"
    function_description: str = ""
    # Optional functions with name, description, and parameters. The model selects a function
    # with auto tool choice, taking precedence over the single function/schema. Separate
    # functions let models satisfy required parameters without resolving conditional nested
    # unions. The selection is returned in LLMResponse.function_name.
    tools: list[dict[str, Any]] | None = None
    # For native tool loops, preserve an empty tool-call turn instead of forcing a fallback
    # function. The caller owns nudging and fallback; do not disable multi-tool for neighboring
    # calls. Providers without the required native multi-tool contract raise LLMValidationError
    # rather than silently substituting another interaction mode.
    tools_required: bool = False
    # Optional multi-turn history inserted as real role messages between system and current
    # user. Basic history uses user/assistant text; clients supporting native tool loops also
    # preserve their tool-call history. None keeps the stateless behavior; unsupported roles are
    # ignored.
    history: list[dict[str, Any]] | None = None
    # Per-call model override. None, and an empty string, keep the client's
    # configured model, including any environment default that client already
    # applied. Resolution is ``request.model or client model``.
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    # Retry length-truncated output with a larger token budget by default. Disable for small
    # best-effort outputs where truncation typically indicates repetitive generation rather than
    # insufficient budget.
    length_retry: bool = True
    # Optional reasoning effort: low, medium, or high. When unset, omit the request field unless
    # the client has a configured default such as LLM_REASONING_EFFORT. Response reasoning
    # is separate from user-visible content.
    reasoning_effort: str | None = None
    # Output mode: function_call uses native function arguments; json_schema uses native
    # response_format with the raw schema where supported; text returns ordinary message content
    # without tools. GigaChat legacy functions simplify schemas, while its native json_schema
    # path retains them.
    mode: str = "function_call"
    # Transport knobs for this call. None keeps the client constructor, which
    # itself falls back to the env default. A set field wins over both.
    # Negative values are rejected at dispatch. Batch clients and
    # count_tokens stay on the constructor timeout except where the method
    # they call already accepts one. Retries on those paths are
    # constructor-scoped.
    timeout_s: float | None = None
    max_retries: int | None = None
    # Images on the current user turn. An empty list is text-only and does
    # not change the request body. History turns are not image carriers.
    # The canary scans system text, user text, and model output. It does
    # not read image bytes. Image bytes are not written to logs.
    images: list["ImageAttachment"] = Field(default_factory=list)


_IMAGE_MEDIA_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
})


class ImageAttachment(BaseModel):
    """One image on the current user turn. Exactly one of ``url`` or ``data``.

    ``media_type`` defaults to ``image/png`` when ``data`` is set. A set
    type must be png, jpeg, gif, or webp. The canary does not scan the
    bytes, and nothing in the library logs them.
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "ImageAttachment":
        has_url = self.url is not None
        has_data = self.data is not None
        if has_url == has_data:
            raise ValueError("ImageAttachment requires exactly one of url or data")
        if has_data and not self.media_type:
            self.media_type = "image/png"
        if self.media_type is not None and self.media_type not in _IMAGE_MEDIA_TYPES:
            allowed = ", ".join(sorted(_IMAGE_MEDIA_TYPES))
            raise ValueError(f"media_type must be one of {allowed}")
        return self

    def openai_url(self) -> str:
        """URL for an OpenAI image_url part. Bytes become a data URL."""
        if self.url is not None:
            return self.url
        encoded = base64.b64encode(self.data or b"").decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"

    def anthropic_block(self) -> dict[str, Any]:
        """Messages API image block.

        ``anthropic-version`` ``2023-06-01`` accepts both base64 and url
        sources. There is no newer version header; url was added on this one.
        """
        if self.data is not None:
            encoded = base64.b64encode(self.data).decode("ascii")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": encoded,
                },
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": self.url},
        }

    def gemini_part(self) -> dict[str, Any]:
        """generateContent part. Public URLs are not a part kind here.

        The REST body is camelCase, the same as ``functionCall`` and
        ``generationConfig``. ``inline_data`` is ignored by the API, so the
        call would succeed as text and the image would never arrive.
        """
        if self.url is not None:
            raise LLMValidationError(
                "Gemini: generateContent has no public-URL image input; "
                "pass image bytes"
            )
        return {
            "inlineData": {
                "mimeType": self.media_type,
                "data": base64.b64encode(self.data or b"").decode("ascii"),
            },
        }


LLMRequest.model_rebuild()


class LLMUsage(BaseModel):
    """Token usage for one LLM call."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Provider-reported reasoning tokens, read from nested completion details or a top-level
    # field. Keeping this visible distinguishes reasoning-only budget exhaustion from useful
    # content generation. Defaults to zero when no reasoning usage is reported.
    reasoning_tokens: int = 0
    # Prompt cache hits and misses. Providers report either explicit flat counts or nested hits
    # with misses derived from prompt_tokens. Use -1 for unavailable statistics and zero for a
    # reported empty cache; these must remain distinguishable.
    cache_hit_tokens: int = -1
    cache_miss_tokens: int = -1
    # Provider-reported cost, separated by currency. A ruble-denominated cost_rub may be
    # duplicated in cost, while other gateways use cost for dollars. Read bare cost as dollars
    # only when cost_rub is absent. None means unreported, not free.
    cost_rub: float | None = None
    cost_usd: float | None = None

    # Default cache-field names for flat hit/miss and nested hit-only dialects. Route
    # configuration can override these names. ClassVar prevents Pydantic from treating constants
    # as model fields.
    DEFAULT_CACHE_HIT_FIELD: ClassVar[str] = "prompt_cache_hit_tokens"
    DEFAULT_CACHE_MISS_FIELD: ClassVar[str] = "prompt_cache_miss_tokens"
    DEFAULT_CACHE_NESTED_FIELD: ClassVar[str] = "prompt_tokens_details.cached_tokens"

    @classmethod
    def from_raw(
        cls,
        usage_raw: Any,
        *,
        cache_hit_field: str | None = None,
        cache_miss_field: str | None = None,
        cache_nested_field: str | None = None,
    ) -> "LLMUsage":
        """Central parser for provider usage, shared by blocking and streaming clients. Nested
        reasoning tokens take precedence over flat values. Configurable flat cache counts take
        precedence over nested hits because they explicitly report misses. Nested field names
        support arbitrary dotted paths. Clients should use this method rather than duplicate
        usage parsing.
        """
        if not isinstance(usage_raw, dict):
            return cls()
        details = usage_raw.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if reasoning is None:
            reasoning = usage_raw.get("reasoning_tokens")

        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        def _dig(d: dict[str, Any], dotted: str) -> Any:
            """Read a dotted dictionary path, returning None when a segment is missing."""
            cur: Any = d
            for part in dotted.split("."):
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
            return cur

        prompt = _int(usage_raw.get("prompt_tokens"))

        # Configured field semantics: None selects the default, an empty string disables the
        # field, and another value names the field or nested dotted path.
        def _resolve(name: str | None, default: str) -> str | None:
            if name is None:
                return default
            return name or None  # An empty field name disables lookup.

        hit_name = _resolve(cache_hit_field, cls.DEFAULT_CACHE_HIT_FIELD)
        miss_name = _resolve(cache_miss_field, cls.DEFAULT_CACHE_MISS_FIELD)
        nested_name = _resolve(cache_nested_field, cls.DEFAULT_CACHE_NESTED_FIELD)

        # Prefer explicit flat cache counts over derived nested statistics.
        hit_raw = usage_raw.get(hit_name) if hit_name else None
        miss_raw = usage_raw.get(miss_name) if miss_name else None
        if hit_raw is None and nested_name:
            hit_raw = _dig(usage_raw, nested_name)
            # Nested cached_tokens reports hits only. Derive misses when prompt_tokens is
            # available; otherwise keep misses unknown at -1.
            if hit_raw is not None and miss_raw is None and prompt:
                miss_raw = max(prompt - _int(hit_raw), 0)

        def _money(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        rub = _money(usage_raw.get("cost_rub"))
        # Treat bare cost as dollars only when cost_rub is absent, avoiding double-counting a
        # duplicated ruble charge.
        usd = None if rub is not None else _money(usage_raw.get("cost"))
        return cls(
            prompt_tokens=prompt,
            completion_tokens=_int(usage_raw.get("completion_tokens")),
            total_tokens=_int(usage_raw.get("total_tokens")),
            reasoning_tokens=_int(reasoning),
            cache_hit_tokens=_int(hit_raw) if hit_raw is not None else -1,
            cache_miss_tokens=_int(miss_raw) if miss_raw is not None else -1,
            cost_rub=rub,
            cost_usd=usd,
        )


class Budget(BaseModel):
    """Independent ceilings for one client instance.

    ``None`` means that dimension is not limited. Dollars and rubles are
    not converted into each other. Tokens come from ``LLMUsage.total_tokens``.
    """

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float | None = None
    max_cost_rub: float | None = None
    max_total_tokens: int | None = None


class BudgetState(BaseModel):
    """Accumulated spend. ``budget_state`` returns a copy, not the ledger."""

    model_config = ConfigDict(extra="forbid")

    cost_usd: float = 0.0
    cost_rub: float = 0.0
    total_tokens: int = 0


class CallRecord(BaseModel):
    """One interactive call, emitted after it finishes.

    Batch clients are not instrumented: a file batch is not one call, and
    the coalescer would double-count work the inner client already records
    when it is used directly. ``count_tokens`` is included because it is a
    client call, even though it does not send an ``LLMRequest``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    method: str
    tier: str | None = None
    latency_ms: int
    ok: bool
    error_type: str | None = None
    request_id: str | None = None
    usage: LLMUsage | None = None


class LLMResponse(BaseModel):
    """An LLM response with parsed function arguments and metadata."""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None  # Raw content for text mode.
    # Function selected in multi-tool mode; None when the single structured function is
    # predetermined.
    function_name: str | None = None
    # All calls returned in one model turn, each with id, name, and arguments. Preserve parallel
    # calls so the caller can respond to each without inconsistent history. The first call is
    # also exposed through function_name and arguments for single-shot compatibility.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    # Model reasoning, separate from user-visible content. None when the provider does not
    # return it.
    reasoning_content: str | None = None
    # Request identifier for tracing: a provider x-request-id or a generated RqUID where
    # applicable.
    request_id: str | None = None
    # Provider finish reason, such as stop, length, or tool_calls; None when absent. Expose it
    # on blocking responses so callers can distinguish budget truncation from a capability
    # failure.
    finish_reason: str | None = None
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw: dict[str, Any] | None = None  # Raw response for diagnostics when needed.
    # How many corrective schema re-asks this response includes. Zero is the
    # first answer. One means the model was shown the validation error and
    # answered again. Measurement modes (no_degrade, preserve) stay at zero.
    validation_reasks: int = 0


class LLMStreamChunk(BaseModel):
    """One generate_stream increment. delta_text carries user-visible content and delta_reasoning
    carries thinking output. Terminal chunks carry finish_reason and may include usage;
    request_id identifies the provider request when available. A completed Anthropic
    message may also carry content_blocks for the next turn.
    """

    model_config = ConfigDict(extra="forbid")

    delta_text: str = ""
    delta_reasoning: str = ""
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    request_id: str | None = None
    # Assistant content blocks to replay on the next turn. Anthropic thinking
    # requires the signature and any redacted block to be sent back. None when
    # the chunk is not a completed message.
    content_blocks: list[dict[str, Any]] | None = None
