"""Anthropic Messages API client: wire format, usage, tools, and streaming."""

from __future__ import annotations

import json
import logging
from contextlib import aclosing

import httpx
import pytest
import respx

from llm_mesh import AnthropicClient, AnthropicError, LLMRequest, LLMValidationError, canary
from llm_mesh.anthropic.client import (
    ANTHROPIC_BASE_URL,
    build_messages,
    map_stop_reason,
    messages_url,
    usage_from_anthropic,
)
from llm_mesh.models_catalog import _MANAGED_ENV, make_client, missing_credentials
from llm_mesh.stream_events import StreamEventType
from llm_mesh.types import LLMAuthError, LLMUsage

BASE = "https://example.test"
URL = BASE + "/v1/messages"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
LOOSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
}


def _message(text="Hello", *, stop="end_turn", usage=None, blocks=None, model="claude-sonnet-5"):
    content = blocks if blocks is not None else [{"type": "text", "text": text}]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": stop,
        "content": content,
        "usage": usage or {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 6,
            "cache_creation_input_tokens": 1,
        },
    }


def _client(**kwargs):
    values = {"model": "claude-sonnet-5", "base_url": BASE, "api_key": "test-key"}
    values.update(kwargs)
    return AnthropicClient(**values)


def _request(**updates):
    values = {"system": "SYS", "user": "NOW", "max_tokens": 128}
    values.update(updates)
    return LLMRequest(**values)


def _sse(events: list[dict]) -> str:
    lines: list[str] = []
    for event in events:
        lines.append(f"event: {event['type']}")
        lines.append("data: " + json.dumps(event))
        lines.append("")
    return "\n".join(lines) + "\n"


@pytest.fixture
def clean_managed(monkeypatch):
    for key in (*_MANAGED_ENV, "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


def test_messages_url_accepts_a_versioned_base():
    assert messages_url("https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"
    assert messages_url("https://proxy.example/v1/") == "https://proxy.example/v1/messages"


def test_stop_reasons_use_the_shared_vocabulary():
    assert map_stop_reason("end_turn") == "stop"
    assert map_stop_reason("max_tokens") == "length"
    assert map_stop_reason("tool_use") == "tool_calls"
    assert map_stop_reason("refusal") == "content_filter"
    assert map_stop_reason("model_context_window_exceeded") == "model_context_window_exceeded"
    assert map_stop_reason(None) is None


def test_usage_folds_cache_reads_and_writes_into_prompt_tokens():
    usage = usage_from_anthropic({
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 6,
        "cache_creation_input_tokens": 1,
    })
    assert usage == LLMUsage(
        prompt_tokens=17,
        completion_tokens=4,
        total_tokens=21,
        cache_hit_tokens=6,
        cache_miss_tokens=11,
    )
    unknown = usage_from_anthropic({"input_tokens": 2, "output_tokens": 3})
    assert unknown.cache_hit_tokens == -1
    assert unknown.cache_miss_tokens == -1
    assert unknown.prompt_tokens == 2


def test_messages_alternate_roles_and_translate_tool_turns():
    system, messages = build_messages(_request(history=[
        {"role": "user", "content": "OLD-Q"},
        {"role": "assistant", "content": "OLD-A"},
        {"role": "developer", "content": "DROP"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "toolu_1",
            "function": {"name": "emit", "arguments": "{\"answer\": \"ok\"}"},
        }]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "OBS"},
    ]))
    assert system == "SYS"
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assistant = messages[1]["content"]
    assert assistant[0] == {"type": "text", "text": "OLD-A"}
    assert assistant[1]["type"] == "tool_use"
    assert assistant[1]["name"] == "emit"
    assert assistant[1]["input"] == {"answer": "ok"}
    assert messages[2]["content"][0] == {
        "type": "tool_result", "tool_use_id": "toolu_1", "content": "OBS",
    }
    assert messages[2]["content"][1] == {"type": "text", "text": "NOW"}


def test_missing_api_key_is_reported(clean_managed):
    with pytest.raises(AnthropicError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(model="claude-sonnet-5")


def test_dedicated_key_and_base_url_precede_neutral_names(monkeypatch, clean_managed):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("LLM_API_KEY", "other-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/v1")
    monkeypatch.setenv("LLM_BASE_URL", "https://other.example/v1")
    client = AnthropicClient(model="claude-sonnet-5")
    assert client._key == "anthropic-key"
    assert client.URL == "https://anthropic.example/v1/messages"


@pytest.mark.asyncio
async def test_generate_text_sends_messages_api_headers_and_usage():
    seen = {}

    def reply(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message("Hello"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_text(_request(temperature=0.2))
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "SYS"
    assert seen["body"]["messages"] == [{"role": "user", "content": "NOW"}]
    assert seen["body"]["max_tokens"] == 128
    assert seen["body"]["temperature"] == 0.2
    assert "tools" not in seen["body"]
    assert result.text == "Hello"
    assert result.finish_reason == "stop"
    assert result.request_id == "msg_1"
    assert result.usage.prompt_tokens == 17
    assert result.usage.cache_hit_tokens == 6
    assert "_sent_max_tokens" not in (result.raw or {})


@pytest.mark.asyncio
async def test_length_stop_retries_with_a_larger_budget():
    budgets = []

    def reply(request):
        body = json.loads(request.content)
        budgets.append(body["max_tokens"])
        if len(budgets) == 1:
            return httpx.Response(200, json=_message("partial", stop="max_tokens"))
        return httpx.Response(200, json=_message("done"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_text(_request(max_tokens=100))
    assert budgets == [100, 200]
    assert result.text == "done"
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_thinking_budget_forces_temperature_one():
    seen = {}

    def reply(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message(
            "Hi",
            blocks=[
                {"type": "thinking", "thinking": "because"},
                {"type": "text", "text": "Hi"},
            ],
        ))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_text(
                _request(max_tokens=32000, reasoning_effort="high", temperature=0),
            )
    assert seen["body"]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert seen["body"]["temperature"] == 1
    assert result.text == "Hi"
    assert result.reasoning_content == "because"


@pytest.mark.asyncio
async def test_disable_reasoning_suppresses_thinking(monkeypatch):
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"thinking": {"type": "enabled", "budget_tokens": 1024}}')
    seen = {}

    def reply(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message("Hi"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            await client.generate_text(_request(reasoning_effort="high", max_tokens=32000))
    assert "thinking" not in seen["body"]
    assert seen["body"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_output_ceiling_too_small_for_thinking_leaves_it_off(monkeypatch):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "256")
    seen = {}

    def reply(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message("Hi"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            await client.generate_text(_request(max_tokens=4000, reasoning_effort="low"))
    assert "thinking" not in seen["body"]
    assert seen["body"]["max_tokens"] == 256
    assert seen["body"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_forced_tool_and_json_schema_use_native_fields():
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        if "tools" in bodies[-1]:
            return httpx.Response(200, json=_message("", stop="tool_use", blocks=[{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "emit",
                "input": {"answer": "ok"},
            }]))
        return httpx.Response(200, json=_message('{"answer": "ok"}'))

    request = _request(function_name="emit", schema=SCHEMA)
    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            tool = await client.generate_structured(request)
            schema = await client.generate_structured(request.model_copy(update={"mode": "json_schema"}))
    assert bodies[0]["tool_choice"] == {"type": "tool", "name": "emit"}
    assert bodies[0]["tools"][0]["input_schema"] == SCHEMA
    assert bodies[0]["tools"][0]["strict"] is True
    assert tool.arguments == {"answer": "ok"}
    assert tool.function_name == "emit"
    assert tool.tool_calls[0]["id"] == "toolu_1"
    assert tool.finish_reason == "tool_calls"
    assert bodies[1]["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert "tools" not in bodies[1]
    assert schema.arguments == {"answer": "ok"}


@pytest.mark.asyncio
async def test_required_tools_do_not_fall_back_to_text():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_message("plain text"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            with pytest.raises(LLMValidationError, match="tool_use"):
                await client.generate_structured(_request(
                    tools=[{"name": "emit", "parameters": SCHEMA}],
                    tools_required=True,
                ))
    assert calls == 1


@pytest.mark.asyncio
async def test_rejected_json_schema_falls_back_once_unless_degrade_is_disabled():
    calls = {"n": 0}

    def reply(request):
        calls["n"] += 1
        body = json.loads(request.content)
        if "output_config" in body:
            return httpx.Response(400, json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "output_config is unsupported"},
            })
        return httpx.Response(200, json=_message('{"answer": "ok"}'))

    request = _request(mode="json_schema", schema=SCHEMA)
    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_structured(request)
        calls["n"] = 0
        async with aclosing(_client(no_degrade=True)) as client:
            with pytest.raises(AnthropicError, match="output_config"):
                await client.generate_structured(request)
    assert result.arguments == {"answer": "ok"}
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_authentication_failure_is_not_retried():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"type": "error", "error": {"message": "bad key"}})

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            with pytest.raises(LLMAuthError):
                await client.generate_text(_request())
    assert calls == 1


@pytest.mark.asyncio
async def test_stream_reports_text_thinking_and_merged_usage():
    payload = _sse([
        {"type": "message_start", "message": {
            "id": "msg_9",
            "usage": {"input_tokens": 3, "cache_read_input_tokens": 2},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    with respx.mock as router:
        router.post(URL).mock(return_value=httpx.Response(200, text=payload))
        async with aclosing(_client()) as client:
            chunks = [chunk async for chunk in client.generate_stream(_request())]
    assert chunks[0].delta_reasoning == "hmm"
    assert chunks[0].request_id == "msg_9"
    assert chunks[1].delta_text == "Hi"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens,
            chunks[-1].usage.cache_hit_tokens) == (5, 5, 2)


@pytest.mark.asyncio
async def test_stream_events_emit_tool_deltas_then_complete():
    payload = _sse([
        {"type": "message_start", "message": {"id": "msg_2", "usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "toolu_1", "name": "emit", "input": {},
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": "{\"answer\":",
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": "\"ok\"}",
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    request = _request(tools=[{"name": "emit", "parameters": SCHEMA}])
    with respx.mock as router:
        route = router.post(URL).mock(return_value=httpx.Response(200, text=payload))
        async with aclosing(_client()) as client:
            events = [event async for event in client.generate_stream_events(request)]
    kinds = [event.type for event in events]
    assert kinds == [
        StreamEventType.TOOL_USE_START,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.TOOL_USE_STOP,
        StreamEventType.COMPLETE,
    ]
    assert events[0].name == "emit"
    assert events[0].id == "toolu_1"
    assert "".join(event.arguments_delta for event in events[1:3]) == '{"answer":"ok"}'
    assert events[-1].finish_reason == "tool_calls"
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"][0]["name"] == "emit"
    assert sent["stream"] is True


@pytest.mark.asyncio
async def test_stream_http_error_is_visible_before_it_is_raised():
    with respx.mock as router:
        router.post(URL).mock(return_value=httpx.Response(401, text="nope"))
        async with aclosing(_client()) as client:
            events = []
            with pytest.raises(LLMAuthError):
                async for event in client.generate_stream_events(_request()):
                    events.append(event)
    assert [event.type for event in events] == [StreamEventType.ERROR]


def test_thinking_blocks_round_trip_and_history_starts_with_user():
    system, messages = build_messages(_request(history=[
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "step", "signature": "sig_abc"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "answer"},
        ]},
    ]))
    assert system == "SYS"
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "."
    thinking, redacted, text = messages[1]["content"]
    assert thinking == {"type": "thinking", "thinking": "step", "signature": "sig_abc"}
    assert redacted == {"type": "redacted_thinking", "data": "opaque"}
    assert text == {"type": "text", "text": "answer"}
    assert messages[2]["content"] == "NOW"


@pytest.mark.asyncio
async def test_thinking_budget_keeps_the_requested_answer():
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_message("Hi"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            await client.generate_text(_request(max_tokens=4096, reasoning_effort="high"))
            await client.generate_text(_request(max_tokens=4096, reasoning_effort="medium"))
    assert bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert bodies[0]["max_tokens"] == 20096
    assert bodies[1]["thinking"]["budget_tokens"] == 4096
    assert bodies[1]["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_thinking_budget_shrinks_under_the_output_ceiling(monkeypatch):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "8192")
    seen = {}

    def reply(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message("Hi"))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            await client.generate_text(_request(max_tokens=4096, reasoning_effort="high"))
    assert seen["body"]["max_tokens"] == 8192
    assert seen["body"]["thinking"]["budget_tokens"] == 4096
    assert seen["body"]["max_tokens"] - seen["body"]["thinking"]["budget_tokens"] == 4096


@pytest.mark.asyncio
async def test_thinking_only_response_is_not_retried_as_empty():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_message("", blocks=[
            {"type": "thinking", "thinking": "because", "signature": "sig"},
        ]))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_text(_request())
    assert calls == 1
    assert result.reasoning_content == "because"
    assert result.text is None


@pytest.mark.asyncio
async def test_loose_schema_is_not_sent_as_strict_tool_use():
    seen = {}

    def reply(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_message("", stop="tool_use", blocks=[{
            "type": "tool_use", "id": "toolu_1", "name": "emit", "input": {"answer": "ok"},
        }]))

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_structured(_request(
                function_name="emit", schema=LOOSE_SCHEMA,
            ))
    assert "strict" not in seen["body"]["tools"][0]
    assert seen["body"]["tools"][0]["input_schema"] == LOOSE_SCHEMA
    assert result.arguments == {"answer": "ok"}


@pytest.mark.asyncio
async def test_strict_rejection_retries_without_strict_before_text():
    bodies = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("tools", [{}])[0].get("strict") is True:
            return httpx.Response(400, json={
                "type": "error",
                "error": {"message": "strict grammar requires additionalProperties: false"},
            })
        return httpx.Response(200, json=_message("", stop="tool_use", blocks=[{
            "type": "tool_use", "id": "toolu_1", "name": "emit", "input": {"answer": "ok"},
        }]))

    request = _request(
        tools=[{"name": "emit", "parameters": SCHEMA}],
        tools_required=True,
    )
    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            result = await client.generate_structured(request)
    assert [body["tool_choice"] for body in bodies] == [
        {"type": "any"}, {"type": "any"},
    ]
    assert bodies[0]["tools"][0]["strict"] is True
    assert "strict" not in bodies[1]["tools"][0]
    assert "output_config" not in bodies[1]
    assert result.arguments == {"answer": "ok"}


@pytest.mark.asyncio
async def test_preserve_does_not_emulate_text_after_a_strict_rejection():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={
            "type": "error",
            "error": {"message": "strict grammar rejected the input_schema"},
        })

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client(fallback_policy="preserve")) as client:
            with pytest.raises(AnthropicError, match="input_schema"):
                await client.generate_structured(_request(
                    tools=[{"name": "emit", "parameters": SCHEMA}],
                    tools_required=True,
                ))
    assert calls == 2


@pytest.mark.asyncio
async def test_unrelated_400_is_not_turned_into_a_text_call():
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={
            "type": "error",
            "error": {"message": "credit balance is too low"},
        })

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            with pytest.raises(AnthropicError, match="credit balance"):
                await client.generate_structured(_request(mode="json_schema", schema=SCHEMA))
    assert calls == 1


@pytest.mark.asyncio
async def test_loose_json_schema_is_an_explicit_error_when_degrade_is_off():
    with respx.mock as router:
        route = router.post(URL).mock(
            return_value=httpx.Response(200, json=_message('{"answer": "ok"}')),
        )
        async with aclosing(_client(fallback_policy="preserve")) as client:
            with pytest.raises(LLMValidationError, match="additionalProperties"):
                await client.generate_structured(_request(
                    mode="json_schema", schema=LOOSE_SCHEMA,
                ))
        assert route.call_count == 0
        async with aclosing(_client()) as client:
            result = await client.generate_structured(_request(
                mode="json_schema", schema=LOOSE_SCHEMA,
            ))
    assert route.call_count == 1
    assert "output_config" not in json.loads(route.calls[0].request.content)
    assert result.arguments == {"answer": "ok"}


@pytest.mark.asyncio
async def test_unparseable_tool_input_is_not_an_empty_object():
    with respx.mock as router:
        router.post(URL).mock(return_value=httpx.Response(200, json=_message(
            "", stop="tool_use", blocks=[{
                "type": "tool_use", "id": "toolu_1", "name": "emit", "input": "not-json",
            }],
        )))
        async with aclosing(_client()) as client:
            with pytest.raises(LLMValidationError, match="not valid JSON"):
                await client.generate_structured(_request(
                    function_name="emit", schema=SCHEMA,
                ))


@pytest.mark.asyncio
async def test_stream_retries_a_rate_limit_before_the_first_event():
    calls = 0
    payload = _sse([
        {"type": "message_start", "message": {"id": "msg_3", "usage": {"input_tokens": 1}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow")
        return httpx.Response(200, text=payload)

    with respx.mock as router:
        router.post(URL).mock(side_effect=reply)
        async with aclosing(_client()) as client:
            chunks = [chunk async for chunk in client.generate_stream(_request())]
    assert calls == 2
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].content_blocks == [{"type": "text", "text": "Hi"}]


@pytest.mark.asyncio
async def test_stream_keeps_the_thinking_signature():
    payload = _sse([
        {"type": "message_start", "message": {"id": "msg_4", "usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "thinking", "thinking": "",
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "thinking_delta", "thinking": "hmm",
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "signature_delta", "signature": "sig_abc",
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ])
    with respx.mock as router:
        router.post(URL).mock(return_value=httpx.Response(200, text=payload))
        async with aclosing(_client()) as client:
            chunks = [chunk async for chunk in client.generate_stream(_request())]
            events = [event async for event in client.generate_stream_events(_request())]
    assert chunks[-1].content_blocks == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig_abc"},
        {"type": "text", "text": "Hi"},
    ]
    assert events[-1].content_blocks == chunks[-1].content_blocks


@pytest.mark.asyncio
async def test_stream_scans_a_canary_when_the_consumer_stops(caplog):
    payload = _sse([
        {"type": "message_start", "message": {"id": "msg_5", "usage": {"input_tokens": 1}}},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "text_delta", "text": "leaked test-canary",
        }},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])
    token = canary.set_canary_context_token("test-canary")
    try:
        with respx.mock as router:
            router.post(URL).mock(return_value=httpx.Response(200, text=payload))
            async with aclosing(_client()) as client:
                stream = client.generate_stream(_request())
                with caplog.at_level(logging.CRITICAL, logger="llm_mesh.canary"):
                    chunk = await anext(stream)
                    await stream.aclose()
        assert chunk.delta_text == "leaked test-canary"
        assert "CANARY_DETECTED" in caplog.text
    finally:
        canary.reset_canary_context_token(token)


def test_catalog_anthropic_route_does_not_require_a_base_url(monkeypatch, clean_managed):
    route = {
        "id": "claude-sonnet-5",
        "kind": "anthropic",
        "model": "claude-sonnet-5",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    }
    assert missing_credentials(route) == ["ANTHROPIC_API_KEY"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert missing_credentials(route) == []
    client = make_client(route)
    assert isinstance(client, AnthropicClient)
    assert client.URL == ANTHROPIC_BASE_URL + "/v1/messages"
    assert client._model == "claude-sonnet-5"
    assert client._key == "test-key"


@pytest.mark.asyncio
@respx.mock
async def test_count_tokens_posts_one_message_per_string():
    route = respx.post(URL + "/count_tokens").mock(side_effect=[
        httpx.Response(200, json={"input_tokens": 2}),
        httpx.Response(200, json={"input_tokens": 5}),
    ])
    client = _client(model="claude-count")
    try:
        counts = await client.count_tokens(["aa", "bbbbb"], model="claude-override")
    finally:
        await client.aclose()
    assert counts == [2, 5]
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert bodies == [
        {"model": "claude-override", "messages": [{"role": "user", "content": "aa"}]},
        {"model": "claude-override", "messages": [{"role": "user", "content": "bbbbb"}]},
    ]


@pytest.mark.asyncio
async def test_count_tokens_empty_list_does_not_call(monkeypatch):
    client = _client()
    try:
        assert await client.count_tokens([]) == []
    finally:
        await client.aclose()
