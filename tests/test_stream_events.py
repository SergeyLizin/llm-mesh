"""Tests for typed streaming events and legacy stream compatibility. Replace _ensure_http with a
fake streaming context manager supplying SSE lines. Exercise SSE parsing and event lifecycles
while leaving client transport handling in place.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_mesh._streaming import ToolCallAccumulator, events_from_sse_payload
from llm_mesh.stream_events import (
    Complete,
    ContentDelta,
    Error,
    ReasoningDelta,
    StreamEvent,
    StreamEventType,
    ToolUseDelta,
    ToolUseStart,
    ToolUseStop,
)
from llm_mesh.types import LLMRequest, LLMStreamChunk, LLMTimeoutError


# ───────────────────────── helpers ─────────────────────────


def _sse(payload: dict[str, Any]) -> str:
    """Serialize a dictionary as one SSE data line."""
    return "data: " + json.dumps(payload)


def _sse_lines(*payloads: dict[str, Any], done: bool = True) -> list[str]:
    """Build one SSE line per payload, optionally followed by DONE."""
    lines = [_sse(p) for p in payloads]
    if done:
        lines.append("data: [DONE]")
    return lines


class _FakeStreamResponse:
    """Minimal streamed httpx response with status, headers, and async lines. Expose error bodies
    as bytes through aread.
    """

    def __init__(
        self,
        lines: list[str] | None = None,
        *,
        status_code: int = 200,
        body_text: str = "",
        request_id: str | None = "req-123",
    ) -> None:
        self._lines = lines or []
        self.status_code = status_code
        self._body_text = body_text
        self.headers = MagicMock()
        self.headers.get = lambda name, default=None: (
            request_id if name == "x-request-id" else default
        )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return self._body_text.encode("utf-8", "replace")


class _FakeStreamCM:
    """Async context manager returning a fake stream response."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc) -> None:
        return None


class _FakeHttp:
    """Minimal async HTTP double whose stream method returns the prepared response."""

    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    def stream(self, *args, **kwargs) -> _FakeStreamCM:
        return _FakeStreamCM(self._response)


def _make_client(monkeypatch, *, lines: list[str], status_code: int = 200,
                 body_text: str = "", request_id: str | None = "req-123", **env) -> Any:
    """Build an OpenAI client with fake streaming HTTP transport."""
    from llm_mesh.openai.client import OpenAIClient

    for k in ("LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.setenv(k, env.pop(k, "http://x" if k.endswith("URL") else "k"))
    for name, val in env.items():
        monkeypatch.setenv(name, val if isinstance(val, str) else json.dumps(val))

    c = OpenAIClient(model="m", base_url="http://x", api_key="k")
    response = _FakeStreamResponse(
        lines, status_code=status_code, body_text=body_text, request_id=request_id
    )
    c._ensure_http = lambda: _FakeHttp(response)  # type: ignore[method-assign]
    return c


def _run(coro):
    """Run a coroutine with asyncio.run in a fresh loop that is closed afterward, avoiding leaked
    or stale event-loop state between tests.
    """
    return asyncio.run(coro)


async def _alist(gen) -> list[StreamEvent]:
    """Collect an async generator and always close it, including after an exception, to avoid
    pending tasks between test runs.
    """
    out: list[StreamEvent] = []
    try:
        async for ev in gen:
            out.append(ev)
    finally:
        # aclose also accepts an already exhausted generator.
        aclose = getattr(gen, "aclose", None)
        if aclose is not None:
            await aclose()
    return out


# ───────────────────────── events_from_sse_payload ─────────────────────────


class TestEventsFromSsePayload:
    """Test payload-to-event parsing. A payload may produce multiple content, reasoning, and tool
    events, unlike the legacy single-chunk parser.
    """

    def test_content_delta(self):
        acc = ToolCallAccumulator()
        ev = events_from_sse_payload(
            {"choices": [{"delta": {"content": "Hello"}}]},
            request_id="r1",
            tool_acc=acc,
        )
        assert len(ev) == 1
        assert isinstance(ev[0], ContentDelta)
        assert ev[0].delta_text == "Hello"
        assert ev[0].request_id == "r1"

    def test_reasoning_content_key_default(self):
        """By default, read the reasoning_content field."""
        acc = ToolCallAccumulator()
        ev = events_from_sse_payload(
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            request_id="r1",
            tool_acc=acc,
        )
        assert len(ev) == 1
        assert isinstance(ev[0], ReasoningDelta)
        assert ev[0].delta_reasoning == "think"

    def test_reasoning_field_config_driven(self):
        """Use the explicitly configured reasoning field without falling back to other provider
        keys.
        """
        # Read only the explicitly configured reasoning field.
        acc = ToolCallAccumulator()
        ev = events_from_sse_payload(
            {"choices": [{"delta": {"reasoning": "think2"}}]},
            request_id="r1",
            tool_acc=acc,
            reasoning_field="reasoning",
        )
        assert isinstance(ev[0], ReasoningDelta)
        assert ev[0].delta_reasoning == "think2"

        # With reasoning_content configured, the other field produces no event.
        acc2 = ToolCallAccumulator()
        ev2 = events_from_sse_payload(
            {"choices": [{"delta": {"reasoning": "think2"}}]},
            request_id="r1",
            tool_acc=acc2,
            reasoning_field="reasoning_content",
        )
        assert ev2 == []

    def test_empty_payload_yields_nothing(self):
        """Empty deltas or absent choices produce no events."""
        acc = ToolCallAccumulator()
        assert events_from_sse_payload({}, request_id="r", tool_acc=acc) == []
        assert events_from_sse_payload(
            {"choices": [{"delta": {}}]}, request_id="r", tool_acc=acc
        ) == []
        # A finish-only chunk produces no delta event; completion is emitted separately.
        assert events_from_sse_payload(
            {"choices": [{"finish_reason": "stop"}]}, request_id="r", tool_acc=acc
        ) == []

    def test_malformed_payload_is_safe(self):
        """Malformed delta shapes and non-string content must not crash parsing."""
        acc = ToolCallAccumulator()
        # non-dict choices
        assert events_from_sse_payload(
            {"choices": "garbage"}, request_id="r", tool_acc=acc
        ) == []
        # non-dict delta
        assert events_from_sse_payload(
            {"choices": [{"delta": "nope"}]}, request_id="r", tool_acc=acc
        ) == []
        # Skip non-string content instead of raising.
        assert events_from_sse_payload(
            {"choices": [{"delta": {"content": 42}}]}, request_id="r", tool_acc=acc
        ) == []

    def test_content_and_reasoning_in_one_chunk(self):
        """A single payload can carry both content and reasoning."""
        acc = ToolCallAccumulator()
        ev = events_from_sse_payload(
            {"choices": [{"delta": {"content": "a", "reasoning_content": "b"}}]},
            request_id="r",
            tool_acc=acc,
        )
        assert len(ev) == 2
        assert isinstance(ev[0], ContentDelta)
        assert isinstance(ev[1], ReasoningDelta)


# ───────────────────────── ToolCallAccumulator ─────────────────────────


class TestToolCallAccumulator:
    """Test tool-call accumulation by index: one start per index, deltas in arrival order, and
    stops for every observed index at finalization.
    """

    def test_single_tool_call_start_then_delta(self):
        """Start with id and name, then append argument fragments."""
        acc = ToolCallAccumulator()
        ev1 = acc.feed({"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": '{"loc"'}}
        ]})
        assert len(ev1) == 2
        assert isinstance(ev1[0], ToolUseStart)
        assert ev1[0].index == 0 and ev1[0].id == "c1" and ev1[0].name == "get_weather"
        assert isinstance(ev1[1], ToolUseDelta)
        assert ev1[1].arguments_delta == '{"loc"'

        ev2 = acc.feed({"tool_calls": [
            {"index": 0, "function": {"arguments": ':"SF"}'}}
        ]})
        assert len(ev2) == 1
        assert isinstance(ev2[0], ToolUseDelta)
        assert ev2[0].arguments_delta == ':"SF"}'

        stops = acc.finalize()
        assert [s.type for s in stops] == [StreamEventType.TOOL_USE_STOP]
        assert stops[0].index == 0

    def test_start_emitted_exactly_once_per_index(self):
        """Emit exactly one start across repeated payloads for the same index."""
        acc = ToolCallAccumulator()
        acc.feed({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f"}}]})
        ev2 = acc.feed({"tool_calls": [{"index": 0, "function": {"arguments": "x"}}]})
        # The second batch contains only a delta, without another start.
        assert all(not isinstance(e, ToolUseStart) for e in ev2)
        assert len(ev2) == 1 and isinstance(ev2[0], ToolUseDelta)

    def test_parallel_tool_calls_two_indices(self):
        """Track parallel calls independently and preserve their first-seen order."""
        acc = ToolCallAccumulator()
        acc.feed({"tool_calls": [
            {"index": 0, "id": "c0", "function": {"name": "f0"}},
            {"index": 1, "id": "c1", "function": {"name": "f1"}},
        ]})
        ev = acc.feed({"tool_calls": [
            {"index": 1, "function": {"arguments": "y"}},
            {"index": 0, "function": {"arguments": "x"}},
        ]})
        # Preserve the provider's delta order: index 1, then index 0.
        assert [(e.index, e.arguments_delta) for e in ev] == [(1, "y"), (0, "x")]
        stops = acc.finalize()
        assert [s.index for s in stops] == [0, 1]  # Use first-seen order.

    def test_tool_call_without_id(self):
        """Emit a tool start even when the provider omits its id."""
        acc = ToolCallAccumulator()
        ev = acc.feed({"tool_calls": [
            {"index": 0, "function": {"name": "f", "arguments": "{}"}}
        ]})
        assert ev[0].id is None
        assert ev[0].name == "f"

    def test_tool_call_without_index_defaults_to_zero(self):
        """Default an omitted tool index to zero without crashing."""
        acc = ToolCallAccumulator()
        ev = acc.feed({"tool_calls": [{"id": "c", "function": {"name": "f"}}]})
        assert ev[0].index == 0

    def test_non_list_tool_calls_safe(self):
        """Non-list tool_calls yields no events."""
        acc = ToolCallAccumulator()
        assert acc.feed({}) == []
        assert acc.feed({"tool_calls": None}) == []
        assert acc.feed({"tool_calls": "nope"}) == []
        assert acc.feed({"tool_calls": [{"oops": 1}]}) == []

    def test_request_id_propagated_to_tool_events(self):
        acc = ToolCallAccumulator()
        ev = acc.feed(
            {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]},
            request_id="r42",
        )
        assert all(e.request_id == "r42" for e in ev)
        stops = acc.finalize(request_id="r42")
        assert stops[0].request_id == "r42"


# ───────────────────────── generate_stream_events (OpenAI) ─────────────────────────


class TestGenerateStreamEvents:
    """Integration tests for the full fake-HTTP stream lifecycle: deltas, tool stops, and Complete."""

    def test_content_stream_terminates_with_complete(self, monkeypatch):
        c = _make_client(
            monkeypatch,
            lines=_sse_lines(
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"content": " there"}}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
                {"choices": [{"finish_reason": "stop"}]},
            ),
        )
        events = _run(_alist(c.generate_stream_events(LLMRequest(system="s", user="u"))))
        types = [e.type for e in events]
        assert types[:-1] == [
            StreamEventType.CONTENT_DELTA, StreamEventType.CONTENT_DELTA,
        ]
        assert events[0].delta_text == "Hi"
        assert events[1].delta_text == " there"
        assert isinstance(events[-1], Complete)
        assert events[-1].finish_reason == "stop"
        assert events[-1].usage is not None
        assert events[-1].usage.prompt_tokens == 3

    def test_reasoning_stream(self, monkeypatch):
        c = _make_client(
            monkeypatch,
            lines=_sse_lines(
                {"choices": [{"delta": {"reasoning_content": "think"}}]},
                {"choices": [{"delta": {"content": "answer"}}, {"finish_reason": None}]},
                {"choices": [{"finish_reason": "stop"}]},
            ),
        )
        events = _run(_alist(c.generate_stream_events(LLMRequest(system="s", user="u"))))
        assert isinstance(events[0], ReasoningDelta)
        assert isinstance(events[1], ContentDelta)
        assert isinstance(events[-1], Complete)

    def test_tool_call_stream(self, monkeypatch):
        """Verify the complete tool lifecycle: Start, Delta, then Stop and Complete after DONE."""
        c = _make_client(
            monkeypatch,
            lines=_sse_lines(
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "search", "arguments": '{"q":"a'}}
                ]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": 'bc"}'}}
                ]}}]},
                {"choices": [{"finish_reason": "tool_calls"}]},
            ),
        )
        events = _run(_alist(c.generate_stream_events(LLMRequest(system="s", user="u"))))
        types = [e.type for e in events]
        assert types == [
            StreamEventType.TOOL_USE_START,
            StreamEventType.TOOL_USE_DELTA,
            StreamEventType.TOOL_USE_DELTA,
            StreamEventType.TOOL_USE_STOP,
            StreamEventType.COMPLETE,
        ]
        assert events[0].id == "call_1" and events[0].name == "search"
        assert events[-1].finish_reason == "tool_calls"

    def test_http_error_emits_error_then_raises(self, monkeypatch):
        """Emit Error before raising an HTTP failure so consumers can observe the interruption."""
        from llm_mesh.openai.client import OpenAIError

        c = _make_client(
            monkeypatch,
            lines=[],
            status_code=500,
            body_text='{"error":"boom"}',
        )
        events: list[StreamEvent] = []
        with pytest.raises(OpenAIError):
            events = _run(_alist(c.generate_stream_events(LLMRequest(system="s", user="u"))))
        # Collect events up to the exception in a separate run to verify Error was emitted
        # first.
        events2: list[StreamEvent] = []

        async def collect_until_error():
            try:
                async for ev in c.generate_stream_events(LLMRequest(system="s", user="u")):
                    events2.append(ev)
            except OpenAIError:
                pass

        _run(collect_until_error())
        assert any(isinstance(e, Error) for e in events2)

    def test_network_error_emits_error_and_raises_timeout(self, monkeypatch):
        """Emit Error and raise LLMTimeoutError after a network interruption."""
        from llm_mesh.openai.client import OpenAIClient
        import httpx

        c = OpenAIClient(model="m", base_url="http://x", api_key="k")

        class _BoomHttp:
            def stream(self, *a, **kw):
                async def _cm():
                    raise httpx.ReadError("disconnected")
                    yield  # pragma: no cover - unreachable

                # A context-manager stub that raises on entry.
                class _CM:
                    async def __aenter__(self):
                        raise httpx.ReadError("disconnected")

                    async def __aexit__(self, *e):
                        return None

                return _CM()

        c._ensure_http = lambda: _BoomHttp()  # type: ignore[method-assign]

        events: list[StreamEvent] = []

        async def collect():
            try:
                async for ev in c.generate_stream_events(LLMRequest(system="s", user="u")):
                    events.append(ev)
            except LLMTimeoutError:
                pass

        _run(collect())
        assert any(isinstance(e, Error) for e in events)
        assert events[-1].error_type == "LLMTimeoutError"

    def test_request_id_on_all_events(self, monkeypatch):
        """Attach request_id to every event so consumers need not retain the first event
        separately.
        """
        c = _make_client(
            monkeypatch,
            lines=_sse_lines(
                {"choices": [{"delta": {"content": "a"}}]},
                {"choices": [{"delta": {"content": "b"}}]},
                {"choices": [{"finish_reason": "stop"}]},
            ),
            request_id="req-xyz",
        )
        events = _run(_alist(c.generate_stream_events(LLMRequest(system="s", user="u"))))
        assert all(e.request_id == "req-xyz" for e in events)


# ───────────────────────── GigaChat ─────────────────────────


class TestGigaChatStreamEvents:
    """GigaChat shares the stream contract but does not expose tool events through its legacy
    function interface.
    """

    def test_content_stream_without_tool_events(self, monkeypatch):
        """Verify GigaChat content streaming without tool events. Construct the client inside the
        coroutine to keep async resource setup within the active loop.
        """

        async def run() -> list[StreamEvent]:
            from llm_mesh.gigachat.client import GigaChatClient as GigaChatClient

            monkeypatch.setenv("LLM_BASE_URL", "http://g")
            monkeypatch.setenv("LLM_API_KEY", "k")
            monkeypatch.setenv("LLM_AUTH_SCOPE", "GIGACHAT_API_PERS")
            monkeypatch.setenv("LLM_AUTH_URL", "http://g/auth")

            c = GigaChatClient(tool_choice="auto", use_model_token_limits=False, model="m", credentials="k")
            c._ensure_token = lambda: _async_const("tok")  # type: ignore[method-assign]
            response = _FakeStreamResponse(
                _sse_lines(
                    {"choices": [{"delta": {"content": "Hi"}}]},
                    {"choices": [{"finish_reason": "stop"}]},
                )
            )
            c._ensure_http = lambda: _FakeHttp(response)  # type: ignore[method-assign]
            return await _alist(c.generate_stream_events(LLMRequest(system="s", user="u")))

        events = _run(run())
        types = [e.type for e in events]
        assert types == [StreamEventType.CONTENT_DELTA, StreamEventType.COMPLETE]
        assert events[-1].finish_reason == "stop"


# ───────────────────────── Backward-compat regression ─────────────────────────


class TestBackwardCompatGenerateStream:
    """Preserve the legacy generate_stream LLMStreamChunk contract alongside typed events."""

    def test_generate_stream_still_yields_llm_stream_chunk(self, monkeypatch):
        c = _make_client(
            monkeypatch,
            lines=_sse_lines(
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"content": " there"}}]},
                {"choices": [{"finish_reason": "stop"}]},
            ),
        )

        async def collect():
            return [ch async for ch in c.generate_stream(LLMRequest(system="s", user="u"))]

        chunks = _run(collect())
        assert all(isinstance(ch, LLMStreamChunk) for ch in chunks)
        assert "".join(ch.delta_text for ch in chunks) == "Hi there"
        # The terminal legacy chunk carries finish_reason.
        assert chunks[-1].finish_reason == "stop"


# ───────────────────────── fixtures ─────────────────────────


async def _async_const(v: Any) -> Any:
    """Return a constant asynchronously for token-method mocks."""
    return v
