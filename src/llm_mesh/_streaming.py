"""Shared OpenAI-compatible SSE parsing for data payloads and [DONE]. Clients own authentication,
refresh, semaphores, and transport; this module constructs chunks and normalizes usage
consistently across streaming and non-streaming paths.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .types import LLMStreamChunk, LLMUsage


def parse_usage(usage_raw: Any) -> LLMUsage | None:
    """Build LLMUsage from a raw chunk usage mapping, or return None if absent."""
    if not isinstance(usage_raw, dict):
        return None
    # Keep usage parsing centralized in LLMUsage.from_raw so streaming and blocking calls report
    # the same reasoning-token counts.
    return LLMUsage.from_raw(usage_raw)


def chunk_from_sse_payload(
    payload: dict[str, Any],
    *,
    request_id: str | None,
    first: bool,
    reasoning_field: str | None = None,
) -> LLMStreamChunk:
    """Build a stream chunk from an SSE payload. Malformed fields produce empty deltas rather than
    exceptions. Attach request_id only to the first chunk. An explicit reasoning_field selects
    that field; None checks both reasoning_content and reasoning, including Cerebras responses.
    """
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = {}
    delta_text = delta.get("content") or ""
    delta_reasoning = (delta.get(reasoning_field) if reasoning_field
                       else (delta.get("reasoning_content") or delta.get("reasoning"))) or ""
    return LLMStreamChunk(
        delta_text=delta_text if isinstance(delta_text, str) else "",
        delta_reasoning=delta_reasoning if isinstance(delta_reasoning, str) else "",
        finish_reason=choice.get("finish_reason"),
        usage=parse_usage(payload.get("usage")),
        request_id=request_id if first else None,
    )


async def iter_sse_payloads(
    line_iter: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield payload mappings from SSE lines. Skip non-data lines and invalid JSON; stop at [DONE].
    Propagate network exceptions to the client transport layer.
    """
    async for line in line_iter:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


# Typed events for generate_stream_events.

from .stream_events import (  # noqa: E402 -- intentionally grouped after the basic SSE helpers
    ContentDelta,
    ReasoningDelta,
    StreamEvent,
    ToolUseDelta,
    ToolUseStart,
)


class ToolCallAccumulator:
    """Accumulate delta.tool_calls by index, emitting one ToolUseStart followed by ToolUseDelta
    events.
    """

    def __init__(self) -> None:
        # Observed indices, retained for ordered stop events during finalization.
        self._seen_order: list[int] = []
        # Track whether a start event has been emitted for each index across the entire stream.
        self._started: set[int] = set()
        # Latest known (id, name) for each index; id may be None.
        self._ids: dict[int, str | None] = {}
        self._names: dict[int, str] = {}

    def feed(
        self, delta: dict[str, Any], *, request_id: str | None = None
    ) -> list[StreamEvent]:
        """Process one choices[0].delta mapping into zero or more events. Tool-call indices default
        to zero when omitted. Accumulate id/name before emitting start, then emit argument
        deltas. Attach request_id as events are constructed rather than mutating Pydantic
        objects afterwards.
        """
        events: list[StreamEvent] = []
        raw = delta.get("tool_calls")
        if not isinstance(raw, list):
            return events
        for tc in raw:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = 0

            # Ignore records without id, name, or arguments instead of emitting empty tool-start
            # events.
            fn = tc.get("function")
            if not isinstance(fn, dict):
                fn = {}
            if tc.get("id") is None and not fn.get("name") and not fn.get("arguments"):
                continue

            # Accumulate id/name before deciding whether to emit a start event so it contains
            # the newest metadata.
            new_id = tc.get("id")
            if new_id is not None:
                self._ids.setdefault(idx, str(new_id))
            new_name = fn.get("name")
            if isinstance(new_name, str) and new_name:
                self._names.setdefault(idx, new_name)

            # Emit one start per index for the whole stream, before any argument delta for that
            # index.
            if idx not in self._started:
                self._started.add(idx)
                self._seen_order.append(idx)
                events.append(
                    ToolUseStart(
                        index=idx,
                        id=self._ids.get(idx),
                        name=self._names.get(idx, ""),
                        request_id=request_id,
                    )
                )

            args_delta = fn.get("arguments")
            if isinstance(args_delta, str) and args_delta:
                events.append(
                    ToolUseDelta(
                        index=idx,
                        arguments_delta=args_delta,
                        request_id=request_id,
                    )
                )
        return events

    def finalize(self, *, request_id: str | None = None) -> list[StreamEvent]:
        """Close all observed tool calls after [DONE], in first-seen order. OpenAI SSE does not
        provide an explicit per-tool completion signal; see ToolUseStop.
        """
        from .stream_events import ToolUseStop

        return [
            ToolUseStop(index=idx, request_id=request_id) for idx in self._seen_order
        ]


def events_from_sse_payload(
    payload: dict[str, Any],
    *,
    request_id: str | None,
    tool_acc: ToolCallAccumulator,
    reasoning_field: str | None = None,
) -> list[StreamEvent]:
    """Build zero or more content, reasoning, and tool events from one SSE payload. Reuse one
    external accumulator for the entire stream. The client collects finish_reason and usage and
    emits exactly one Complete after [DONE]. reasoning_field follows the same explicit-field or
    known-field fallback policy as chunk_from_sse_payload.
    """
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = {}

    events: list[StreamEvent] = []

    delta_text = delta.get("content")
    if isinstance(delta_text, str) and delta_text:
        events.append(ContentDelta(delta_text=delta_text, request_id=request_id))

    delta_reasoning = (delta.get(reasoning_field) if reasoning_field
                       else (delta.get("reasoning_content") or delta.get("reasoning")))
    if isinstance(delta_reasoning, str) and delta_reasoning:
        events.append(
            ReasoningDelta(delta_reasoning=delta_reasoning, request_id=request_id)
        )

    events.extend(tool_acc.feed(delta, request_id=request_id))

    return events
