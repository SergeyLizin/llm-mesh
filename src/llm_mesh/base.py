"""Shared client base and the explicit capability model.

Public contracts stay the Protocols in protocol.py. This class owns the
transport plumbing those clients used to copy, and states which optional
behaviors each provider actually implements. Third-party providers subclass
BaseLLMClient, declare CAPABILITIES, and register a builder in the catalog.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator

import httpx

from llm_mesh._common import check_response_canary
from llm_mesh.stream_events import StreamEvent
from llm_mesh.types import LLMRequest, LLMResponse, LLMStreamChunk

__all__ = [
    "BaseLLMClient",
    "Capability",
]


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
    # batching is a separate transport, not a method of GigaChatAsyncClient.
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

    def _guarded_request(self, request: LLMRequest) -> LLMRequest:
        """Run the process-wide request hook before this call reaches the wire.

        The default hook returns the same object, so an unconfigured client
        behaves as before. count_tokens is not a call site: it takes raw
        strings and is a diagnostic.
        """
        from llm_mesh.hooks import apply_request_guard

        return apply_request_guard(request, provider=self.PROVIDER)

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
