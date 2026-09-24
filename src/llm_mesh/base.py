"""Shared client base and the explicit capability model.

Public contracts stay the Protocols in protocol.py. This class owns the
transport plumbing those clients used to copy, and states which optional
behaviors each provider actually implements. Third-party providers subclass
BaseLLMClient, declare CAPABILITIES, and register a builder in the catalog.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, AsyncIterator

import httpx

from llm_mesh._common import check_response_canary
from llm_mesh.stream_events import StreamEvent
from llm_mesh.types import (
    Budget,
    BudgetState,
    CallRecord,
    LLMBudgetExceeded,
    LLMRequest,
    LLMResponse,
    LLMValidationError,
    LLMStreamChunk,
    LLMUsage,
)

__all__ = [
    "BaseLLMClient",
    "Capability",
]

# The public method the caller invoked. A structured call that delegates to
# generate_text must not emit a second record for the inner method.
_ACTIVE_CALL: ContextVar["_CallObservation | None"] = ContextVar(
    "llm_mesh_active_call", default=None,
)
# Per-call transport overrides. None means the constructor (or its env
# default) still applies. Set from the request at dispatch and restored
# when the outer call ends, including when the call raises.
_CALL_RETRIES: ContextVar[int | None] = ContextVar(
    "llm_mesh_call_retries", default=None,
)
_CALL_TIMEOUT: ContextVar[httpx.Timeout | None] = ContextVar(
    "llm_mesh_call_timeout", default=None,
)


class _CallObservation:
    """Fields filled in while a call runs. The context manager emits them."""

    def __init__(self, *, provider: str, method: str, model: str) -> None:
        self.provider = provider
        self.method = method
        self.model = model
        self.tier: str | None = None
        self.request_id: str | None = None
        self.usage: Any = None
        self.ok = True
        self.error_type: str | None = None

    def note(self, value: Any) -> Any:
        """Copy usage, request id, and model off a response, chunk, or event."""
        usage = getattr(value, "usage", None)
        if usage is not None:
            self.usage = usage
        request_id = getattr(value, "request_id", None)
        if request_id:
            self.request_id = request_id
        model = getattr(value, "model", None)
        if isinstance(model, str) and model:
            self.model = model
        return value


def note_active_usage(usage: Any) -> None:
    """Attach usage to the call in progress so a later raise still records it.

    The public wrapper notes a returned response. A schema failure that
    raises after a re-ask never returns, and the summed usage would be
    dropped. Telemetry reads this slot in ``finally``.
    """
    observed = _ACTIVE_CALL.get()
    if observed is not None and usage is not None:
        observed.usage = usage


class BudgetLedger:
    """Spend counters for one client. Currencies are never converted."""

    def __init__(self, budget: Budget | None) -> None:
        self.budget = budget
        self.cost_usd = 0.0
        self.cost_rub = 0.0
        self.total_tokens = 0

    def check(self, provider: str) -> None:
        budget = self.budget
        if budget is None:
            return
        if budget.max_cost_usd is not None and self.cost_usd >= budget.max_cost_usd:
            raise LLMBudgetExceeded(
                f"{provider}: budget exceeded: spent {self.cost_usd} USD, "
                f"limit {budget.max_cost_usd} USD"
            )
        if budget.max_cost_rub is not None and self.cost_rub >= budget.max_cost_rub:
            raise LLMBudgetExceeded(
                f"{provider}: budget exceeded: spent {self.cost_rub} RUB, "
                f"limit {budget.max_cost_rub} RUB"
            )
        if (
            budget.max_total_tokens is not None
            and self.total_tokens >= budget.max_total_tokens
        ):
            raise LLMBudgetExceeded(
                f"{provider}: budget exceeded: spent {self.total_tokens} tokens, "
                f"limit {budget.max_total_tokens} tokens"
            )

    def add(self, usage: LLMUsage | None) -> None:
        if self.budget is None or usage is None:
            return
        # Unreported money stays unset. Adding zero would make "not reported"
        # look like a free call and hide a missing field.
        if usage.cost_usd is not None:
            self.cost_usd += usage.cost_usd
        if usage.cost_rub is not None:
            self.cost_rub += usage.cost_rub
        self.total_tokens += usage.total_tokens

    def reset(self) -> None:
        self.cost_usd = 0.0
        self.cost_rub = 0.0
        self.total_tokens = 0

    def state(self) -> BudgetState:
        return BudgetState(
            cost_usd=self.cost_usd,
            cost_rub=self.cost_rub,
            total_tokens=self.total_tokens,
        )


class Capability(str, Enum):
    """A behavior that differs between providers and can be queried.

    Declaring a capability means the class implements it. A missing one is
    not a silent fallback: optional methods on the base raise
    NotImplementedError and name the capability.
    """

    TEXT = "text"
    STREAM = "stream"
    STREAM_EVENTS = "stream_events"
    STRUCTURED = "structured"
    # Native tool or function calling, as opposed to text emulation.
    TOOLS = "tools"
    # The model may choose among several tools in one request.
    MULTI_TOOL = "multi_tool"
    # A native tool loop driven by tools_required. Not text emulation.
    TOOLS_REQUIRED = "tools_required"
    # A provider-native JSON schema mode, not the text-emulation ladder.
    JSON_SCHEMA_MODE = "json_schema_mode"
    LENGTH_RETRY = "length_retry"
    COUNT_TOKENS = "count_tokens"
    # Dense embeddings, and sparse vectors when the provider returns them.
    EMBEDDINGS = "embeddings"
    # Gateway /score or /v1/rerank, or a local cross-encoder.
    RERANK = "rerank"
    # Batch submission. Interactive clients do not declare this: GigaChat
    # batching is a separate transport, not a method of GigaChatClient.
    BATCH = "batch"


class BaseLLMClient(ABC):
    """Transport plumbing and the four methods every interactive client has.

    Constructors stay on the subclasses. This class does not take connection
    arguments. The subclass constructor must set these attributes before the
    first request, or the shared helpers raise AttributeError:

    - ``_client``: ``None`` until ``_ensure_http`` opens it.
    - ``_verify``: TLS verification flag passed to httpx.
    - ``_max_concurrent``: a positive int, or ``None`` for no limit.
    - ``_semaphore``: ``None`` until ``_ensure_semaphore`` builds it in the
      running loop.
    - ``_http_timeout``: a float used for every phase. If it is missing or
      ``None``, ``_ensure_http`` uses ``_timeout`` instead (GigaChat stores
      an ``httpx.Timeout`` there, with a separate connect budget).
    - ``PROVIDER``: probe label, and the prefix for the base canary context.
      Override ``_check_response_canary`` when the context is already
      qualified. GigaChat does that; it still sets ``PROVIDER`` so a
      hand-built client reports the same kind of label as the others.

    Env parsers live in ``llm_mesh._common``, not on this class.
    """

    CAPABILITIES: frozenset[Capability] = frozenset()

    # Filled by bind_catalog_route. A hand-built client stays a chat client
    # with no query instruction and the model's own vector width.
    _catalog_task = "chat"
    _embedding_sparse = False
    _embedding_dimensions: int | None = None
    _rerank_protocol = "score"
    _rerank_top_k: int | None = None
    _rerank_min_score = 0.0

    def supports(self, capability: Capability) -> bool:
        """Return whether this client implements the capability."""
        return capability in type(self).CAPABILITIES

    def _ensure_http(self) -> httpx.AsyncClient:
        """Open the transport on first use.

        Construction must not create a socket. A client that failed in
        __init__ would otherwise leak a connection, and tests replace this
        method to inject a transport. GigaChat stores an httpx.Timeout on
        _timeout because its read and connect limits differ; OpenAI and
        Anthropic store a float on _http_timeout.
        """
        if self._client is None:
            timeout = getattr(self, "_http_timeout", None)
            if timeout is None:
                timeout = self._timeout
            self._client = httpx.AsyncClient(timeout=timeout, verify=self._verify)
        return self._client

    def _ensure_semaphore(self) -> asyncio.Semaphore | None:
        """Create the semaphore inside the running loop.

        The client is often constructed before any loop exists. Building the
        semaphore in __init__ binds it to the wrong loop.
        """
        if self._max_concurrent is None:
            return None
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def aclose(self) -> None:
        """Close the transport. A second call is a no-op."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _check_response_canary(self, response_text: str, *, context: str) -> None:
        """Scan output under a provider-qualified context.

        GigaChat overrides this. Its call sites already pass a fully
        qualified context, and prefixing again would change the warning.
        """
        check_response_canary(
            response_text,
            context=f"{self.PROVIDER}.{context}",
        )

    def _validation_reask_allowed(self) -> bool:
        """Whether a confirmed schema failure may be shown back to the model.

        The retry is the same tier and the same mechanism, not a fallback.
        ``no_degrade`` and ``fallback_policy="preserve"`` must observe the
        raw first response, so they never re-ask. Validation has to be on:
        a client that does not check arguments (GigaChat) never calls this.
        """
        return bool(getattr(self, "_validate_schema", False)) and not (
            getattr(self, "_no_degrade", False)
            or getattr(self, "_preserve_responses", False)
        )

    def _guarded_request(self, request: LLMRequest) -> LLMRequest:
        """Run the process-wide request hook before this call reaches the wire.

        The default hook returns the same object, so an unconfigured client
        behaves as before. count_tokens is not a call site: it takes raw
        strings and is a diagnostic.
        """
        from llm_mesh.hooks import apply_request_guard

        request = apply_request_guard(request, provider=self.PROVIDER)
        self._bind_call_transport(request)
        return request

    def _bind_call_transport(self, request: LLMRequest) -> None:
        """Apply this request's timeout and retry budget for the call.

        Request fields win over the constructor, which already won over the
        env default. Negative values are a caller error, not a provider
        failure, so they raise before any HTTP. ``None`` leaves the
        constructor value in place. Batch uploads and ``count_tokens`` do
        not read these fields: their transports are constructor-scoped
        except where a method already takes a timeout of its own.
        """
        if request.timeout_s is not None and request.timeout_s < 0:
            raise LLMValidationError(
                f"{getattr(self, 'PROVIDER', type(self).__name__)}: "
                f"timeout_s must be >= 0, got {request.timeout_s}"
            )
        if request.max_retries is not None and request.max_retries < 0:
            raise LLMValidationError(
                f"{getattr(self, 'PROVIDER', type(self).__name__)}: "
                f"max_retries must be >= 0, got {request.max_retries}"
            )
        _CALL_RETRIES.set(request.max_retries)
        _CALL_TIMEOUT.set(
            None if request.timeout_s is None else self._timeout_of(request.timeout_s)
        )

    def _timeout_of(self, seconds: float) -> httpx.Timeout:
        """httpx timeout for one call. GigaChat keeps a separate connect budget."""
        return httpx.Timeout(seconds)

    def _retry_limit(self, configured: int) -> int:
        """Attempt budget for a retry loop. The request overrides the constructor."""
        override = _CALL_RETRIES.get()
        return configured if override is None else override

    def _timeout_kw(self) -> dict[str, httpx.Timeout]:
        """``timeout=`` for one httpx call, or nothing when the client default stands."""
        timeout = _CALL_TIMEOUT.get()
        if timeout is None:
            return {}
        return {"timeout": timeout}

    def _metrics_model(
        self, request: LLMRequest | None = None, model: str | None = None,
    ) -> str:
        if model:
            return model
        if request is not None and request.model:
            return request.model
        for attr in ("_model", "model"):
            value = getattr(self, attr, None)
            if isinstance(value, str) and value:
                return value
        return ""

    @asynccontextmanager
    async def _observe(
        self,
        method: str,
        request: LLMRequest | None = None,
        *,
        model: str | None = None,
    ) -> AsyncIterator[_CallObservation]:
        """Time one public call and emit a single CallRecord when it ends.

        A nested public method (structured output delegating to text) updates
        the outer record instead of emitting its own. The hook runs in
        ``finally``, so a guard refusal and a stream error are both recorded.
        Latency includes retries inside the call.
        """
        outer = _ACTIVE_CALL.get()
        if outer is not None:
            yield outer
            return
        observed = _CallObservation(
            provider=getattr(self, "PROVIDER", type(self).__name__),
            method=method,
            model=self._metrics_model(request, model),
        )
        token = _ACTIVE_CALL.set(observed)
        prev_retries = _CALL_RETRIES.get()
        prev_timeout = _CALL_TIMEOUT.get()
        if hasattr(self, "_last_served_tier"):
            self._last_served_tier = None
        started = time.perf_counter()
        try:
            ledger = getattr(self, "_budget_ledger", None)
            if ledger is not None:
                ledger.check(observed.provider)
            yield observed
        except GeneratorExit:
            # The caller closed the stream. That is not a failed call.
            raise
        except BaseException as exc:
            observed.ok = False
            observed.error_type = type(exc).__name__
            raise
        finally:
            try:
                if observed.tier is None and hasattr(self, "_last_served_tier"):
                    tier = self._last_served_tier
                    if isinstance(tier, str) and tier:
                        observed.tier = tier
                latency_ms = int((time.perf_counter() - started) * 1000)
                from llm_mesh.hooks import emit_call_record

                ledger = getattr(self, "_budget_ledger", None)
                if ledger is not None:
                    ledger.add(observed.usage)
                emit_call_record(CallRecord(
                    provider=observed.provider,
                    model=observed.model,
                    method=observed.method,
                    tier=observed.tier,
                    latency_ms=latency_ms,
                    ok=observed.ok,
                    error_type=observed.error_type,
                    request_id=observed.request_id,
                    usage=observed.usage,
                ))
            finally:
                _CALL_RETRIES.set(prev_retries)
                _CALL_TIMEOUT.set(prev_timeout)
                _ACTIVE_CALL.reset(token)

    @asynccontextmanager
    async def _close_agen(self, source: Any) -> AsyncIterator[Any]:
        """Close a nested async generator when this generator is closed.

        A ``finally`` on the nested generator does not run when the caller
        closes the public one. The canary scan has to see that close.
        """
        try:
            yield source
        finally:
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                await aclose()

    def _bind_budget(self, budget: Budget | None) -> None:
        self._budget_ledger = BudgetLedger(budget)

    def reset_budget(self) -> None:
        """Zero the accumulated spend. The ceilings stay."""
        self._budget_ledger.reset()

    def budget_state(self) -> BudgetState:
        """A copy of the accumulated dollars, rubles, and tokens."""
        return self._budget_ledger.state()

    async def count_tokens(
        self, texts: list[str], *, model: str | None = None
    ) -> list[int]:
        """Count tokens with the provider tokenizer.

        Raw strings are not passed through the request hook: this is a
        diagnostic, not a prompt sent to a model. There is no character
        heuristic here. A client that cannot count
        tokens says so instead of returning a guessed budget. The method is
        present on every subclass, so probe supports(Capability.COUNT_TOKENS)
        rather than hasattr.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support {Capability.COUNT_TOKENS.value}"
        )

    def bind_catalog_route(self, route: dict) -> None:
        """Copy embedding fields off a catalog route onto this instance.

        ``task`` is ``chat`` when the route omits it. ``query_instruction``
        wraps later ``embed(..., task="query")`` calls. ``sparse`` and
        ``dimensions`` are the defaults for those calls.
        """
        from llm_mesh.embeddings import QueryInstruction, checked_dimensions, normalize_task
        from llm_mesh.rerank import (
            checked_min_score,
            checked_top_k,
            normalize_rerank_protocol,
        )

        self._catalog_task = normalize_task(route.get("task"))
        self._query_instruction = QueryInstruction.parse(route.get("query_instruction"))
        self._embedding_sparse = bool(route.get("sparse"))
        self._embedding_dimensions = checked_dimensions(route.get("dimensions"))
        self._rerank_protocol = normalize_rerank_protocol(route.get("rerank_protocol"))
        self._rerank_top_k = checked_top_k(route.get("rerank_top_k"))
        self._rerank_min_score = checked_min_score(route.get("rerank_min_score"))

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task: str = "document",
        dimensions: int | None = None,
        sparse: bool | None = None,
    ) -> list:
        """Embed each text. ``task="query"`` applies the model's query instruction.

        The returned list has one entry per input, in input order. A provider
        that cannot embed raises instead of calling a chat completion.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support {Capability.EMBEDDINGS.value}"
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list:
        """Score documents against ``query`` and return the best first.

        A provider that cannot rerank raises. A failed call does not return
        the input order: that list would look like a successful ranking.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support {Capability.RERANK.value}"
        )

    @abstractmethod
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion."""

    @abstractmethod
    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Generate one structured response."""

    @abstractmethod
    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream text chunks. Implementations are async generators."""
        raise NotImplementedError
        yield  # pragma: no cover

    @abstractmethod
    async def generate_stream_events(
        self, request: LLMRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream typed events. Implementations are async generators."""
        raise NotImplementedError
        yield  # pragma: no cover
