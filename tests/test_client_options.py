"""Regression coverage for explicit client policies shared across applications."""
import json
import httpx

import pytest
import respx

from llm_mesh import OpenAIClient, GigaChatClient, LLMRequest, LLMValidationError
from llm_mesh.stream_events import Complete


@pytest.fixture
def client(monkeypatch):
    for name in ("LLM_REASONING_ON", "LLM_REASONING_OFF", "LLM_EXTRA_BODY",
                 "LLM_DISABLE_REASONING", "LLM_DISABLE_THINKING_FOR_TOOLS",
                 "LLM_FORCE_TEMPERATURE", "LLM_NO_DEGRADE"):
        monkeypatch.delenv(name, raising=False)


def test_explicit_no_degrade_overrides_environment(client, monkeypatch):
    monkeypatch.setenv("LLM_NO_DEGRADE", "true")
    c = OpenAIClient(base_url="https://example.test/v1", api_key="key", no_degrade=False)
    assert c._no_degrade is False


@pytest.mark.asyncio
@respx.mock
async def test_empty_tools_off_dialect_does_not_restore_baseline(client, monkeypatch):
    monkeypatch.setenv("LLM_REASONING_ON", '{"thinking": {"type": "enabled"}}')
    monkeypatch.setenv("LLM_REASONING_OFF", '{}')
    c = OpenAIClient(model="m", base_url="https://example.test/v1", api_key="key",
                     disable_thinking_for_tools=True)
    route = respx.post("https://example.test/v1/chat/completions").respond(200, json={
        "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"type": "function", "function": {"name": "answer", "arguments": '{"value": 1}'}}
        ]}}]})
    try:
        await c.generate_structured(LLMRequest(system="s", user="u", function_name="answer"))
        assert "thinking" not in json.loads(route.calls.last.request.content)
        assert c._extra_body["thinking"] == {"type": "enabled"}
    finally:
        await c.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate_stream", "generate_stream_events"])
async def test_stream_request_options_are_preserved(client, monkeypatch, method):
    monkeypatch.setenv("LLM_FORCE_TEMPERATURE", "1")
    c = OpenAIClient(model="m", base_url="https://example.test/v1", api_key="key")
    seen = {}

    async def stream(body):
        seen.update(body)
        if False:
            yield Complete()

    monkeypatch.setattr(c, "_do_stream" if method == "generate_stream" else "_do_stream_events", stream)
    request = LLMRequest(system="s", user="u", reasoning_effort="high", temperature=0)
    assert [event async for event in getattr(c, method)(request)] == []
    assert seen["reasoning_effort"] == "high"
    assert seen["temperature"] == 1


def test_gigachat_token_limit_can_be_supplied_by_caller(monkeypatch):
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    c = GigaChatClient(token="test-token", use_model_token_limits=False)
    assert c._clip_max_tokens(100000) == 100000
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1000")
    c = GigaChatClient(token="test-token", use_model_token_limits=False)
    assert c._clip_max_tokens(100000) == 1000


def test_gigachat_json_schema_errors_retain_payload():
    c = GigaChatClient(token="test-token")
    payload = {"choices": [{"message": {"content": "invalid"}}]}
    request = LLMRequest(system="s", user="u", mode="json_schema")
    with pytest.raises(LLMValidationError) as exc:
        c._parse_response(payload, request)
    assert exc.value.payload is payload


@pytest.mark.asyncio
@respx.mock
async def test_named_tool_parameter_rejection_keeps_native_output(client):
    c = OpenAIClient(model="m", base_url="https://example.test/v1", api_key="key", no_degrade=True)
    route = respx.post("https://example.test/v1/chat/completions").mock(side_effect=[
        httpx.Response(400, json={"error": {"param": "tool_choice", "message": "Invalid value"}}),
        httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"type": "function", "function": {"name": "answer", "arguments": '{"value": 1}'}}]
        }}]}),
    ])
    try:
        result = await c.generate_structured(LLMRequest(system="s", user="u", function_name="answer"))
        assert result.arguments == {"value": 1}
        assert json.loads(route.calls[0].request.content)["tool_choice"]["type"] == "function"
        assert json.loads(route.calls[1].request.content)["tool_choice"] == "required"
        assert c._last_served_tier == "required"
    finally:
        await c.aclose()
