"""Structural protocols for public LLM client capabilities."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from llm_mesh.stream_events import StreamEvent
from llm_mesh.types import LLMRequest, LLMResponse, LLMStreamChunk


class TextGenerator(Protocol):
    """A client capable of generating plain text."""

    async def generate_text(self, request: LLMRequest) -> LLMResponse: ...


class StreamGenerator(Protocol):
    """A client capable of streaming text generation asynchronously."""

    def generate_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]: ...


class EventStreamGenerator(Protocol):
    """A client capable of emitting typed content, reasoning, and tool events."""

    def generate_stream_events(self, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...


class StructuredGenerator(Protocol):
    """A client capable of generating structured responses."""

    async def generate_structured(self, request: LLMRequest) -> LLMResponse: ...


class AsyncClosable(Protocol):
    """A client with asynchronous transport cleanup."""

    async def aclose(self) -> None: ...


class LLMClient(
    TextGenerator,
    StreamGenerator,
    EventStreamGenerator,
    StructuredGenerator,
    AsyncClosable,
    Protocol,
):
    """The complete public protocol for an interactive LLM client."""


class BatchLLMClient(TextGenerator, StructuredGenerator, AsyncClosable, Protocol):
    """The batch client protocol, without unsupported streaming capabilities."""


__all__ = [
    "AsyncClosable",
    "BatchLLMClient",
    "EventStreamGenerator",
    "LLMClient",
    "StreamGenerator",
    "StructuredGenerator",
    "TextGenerator",
]
