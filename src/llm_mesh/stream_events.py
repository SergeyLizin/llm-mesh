"""Typed events for generate_stream_events, with a type discriminator and separate content,
reasoning, tool, completion, and error classes for consuming provider SSE responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .types import LLMUsage


class StreamEventType(str, Enum):
    """Stream event discriminator. Inherit str for readable JSON serialization and direct
    comparisons such as event.type == "content_delta".
    """

    CONTENT_DELTA = "content_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_STOP = "tool_use_stop"
    COMPLETE = "complete"
    ERROR = "error"


class StreamEvent(BaseModel):
    """Base stream event. Every event carries the stream's request_id when available; it remains
    None when the provider supplies no identifier.
    """

    model_config = ConfigDict(extra="forbid")

    type: StreamEventType
    request_id: str | None = None


class ContentDelta(StreamEvent):
    """An increment of user-visible delta.content."""

    type: StreamEventType = StreamEventType.CONTENT_DELTA
    delta_text: str


class ReasoningDelta(StreamEvent):
    """An increment of model reasoning. The SSE parser selects the configured reasoning field so
    consumers do not need provider-specific key handling.
    """

    type: StreamEventType = StreamEventType.REASONING_DELTA
    delta_reasoning: str


class ToolUseStart(StreamEvent):
    """The beginning of a tool call. Emitted once per index when an id or function name first
    appears. Some providers omit ids, so consumers must accept None.
    """

    type: StreamEventType = StreamEventType.TOOL_USE_START
    index: int
    id: str | None = None
    name: str


class ToolUseDelta(StreamEvent):
    """A fragment of tool-call arguments. Accumulate by index and parse JSON only after
    ToolUseStop.
    """

    type: StreamEventType = StreamEventType.TOOL_USE_DELTA
    index: int
    arguments_delta: str


class ToolUseStop(StreamEvent):
    """A completed tool call. OpenAI SSE has no per-call completion marker, so emit one stop per
    observed index at stream end, immediately before Complete.
    """

    type: StreamEventType = StreamEventType.TOOL_USE_STOP
    index: int


class Complete(StreamEvent):
    """Terminal successful stream event, emitted after DONE with finish_reason and any reported
    usage. No events follow. Interrupted streams emit Error instead.
    """

    type: StreamEventType = StreamEventType.COMPLETE
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    # Replayable assistant blocks for a follow-up turn. Anthropic fills this
    # when the stream carried thinking signatures, redacted thinking, or tool
    # use. Other providers leave it empty.
    content_blocks: list[dict[str, Any]] | None = None


class Error(StreamEvent):
    """Stream interruption, emitted before re-raising a network or HTTP exception. Consumers can
    record the interruption even if an outer layer catches the exception. error_type is the
    exception class name when available.
    """

    type: StreamEventType = StreamEventType.ERROR
    error: str
    error_type: str | None = None


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "ContentDelta",
    "ReasoningDelta",
    "ToolUseStart",
    "ToolUseDelta",
    "ToolUseStop",
    "Complete",
    "Error",
]
