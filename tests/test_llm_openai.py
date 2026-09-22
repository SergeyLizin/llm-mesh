"""Unit tests for the production OpenAI-compatible client using respx HTTP mocks."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh.openai._common import VENDOR_BODY_KEYS
from llm_mesh.openai.client import (
    OpenAIClient,
    OpenAIError,
    _is_temperature_unsupported_error,
    _is_unknown_body_field_error,
    chat_completions_url,
)
from llm_mesh.types import (
    LLMAuthError,
    LLMRequest,
    LLMTimeoutError,
    LLMValidationError,
)


BASE = "https://host/v1"
URL = "https://host/v1/chat/completions"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate environment configuration; set values explicitly in each test."""
    for var in (
        "LLM_BASE_URL", "LLM_BASE_URL",
        "LLM_API_KEY", "LLM_API_KEY", "OPENROUTER_API_KEY",
        "LLM_MAX_RETRIES", "LLM_RETRY_BACKOFF_S",
        "LLM_DISABLE_TOOLS", "LLM_PROVIDER_LABEL",
        "LLM_MAX_CONCURRENT", "LLM_MAX_OUTPUT_TOKENS",
        "LLM_DISABLE_REASONING", "LLM_REASONING_OFF", "LLM_REASONING_ON",
        "LLM_REASONING_FIELD",
        "LLM_EXTRA_BODY",
        "LLM_MIN_OUTPUT_TOKENS", "LLM_NO_DEGRADE", "LLM_TOOL_CHOICE_PREF",
    ):
        monkeypatch.delenv(var, raising=False)


def _request() -> LLMRequest:
    return LLMRequest(system="You are an assistant", user="hello", mode="text")


def _structured_request() -> LLMRequest:
    return LLMRequest(
        system="You are a classifier",
        user="I want to delete a procedure",
        schema={"type": "object", "properties": {"action": {"type": "string"}}},
        function_name="classify_intent",
        mode="function_call",
    )


def _ok_response(content: str = "response", model: str = "Qwen3.5-397B-A17B") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _tool_call_response(args: dict, model: str = "Qwen3.5-397B-A17B") -> dict:
    return {
        "model": model,
        "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "classify_intent",
                             "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


# --- Configuration and construction -----------------------------------------


def test_helper_url():
    assert chat_completions_url("https://host/v1/") == URL
    assert chat_completions_url("https://host/v1") == URL


def test_requires_base_url(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    with pytest.raises(OpenAIError, match="base_url"):
        OpenAIClient(model="m")


def test_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    with pytest.raises(OpenAIError, match="API key"):
        OpenAIClient(model="m")


def test_explicit_args_override_env(monkeypatch):
    """Explicit base_url and api_key override environment configuration."""
    c = OpenAIClient(model="m", base_url="https://x/v1/", api_key="explicit")
    assert c.URL == "https://x/v1/chat/completions"
    assert c._key == "explicit"


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_429_honors_retry_after(monkeypatch):
    """Honor Retry-After when retrying 429, as in the GigaChat client."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "9"}, text="rate limited"),
        httpx.Response(200, json=_ok_response()),
    ])
    respx.post(URL).mock(side_effect=lambda _r: next(responses))
    slept: list[float] = []

    import asyncio as _aio
    orig = _aio.sleep
    async def _cap(d, *a, **k):
        slept.append(d)
        return await orig(0)
    monkeypatch.setattr(_aio, "sleep", _cap)

    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert 9.0 in slept  # Honor the server-provided delay.


@respx.mock
@pytest.mark.asyncio
async def test_generate_text_ok(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_PROVIDER_LABEL", "cloudru")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="Qwen3.5-397B-A17B")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert resp.model == "Qwen3.5-397B-A17B"
    assert resp.usage.total_tokens == 18
    assert resp.arguments == {}
    assert client.PROVIDER == "cloudru"
    assert route.calls.last.request.headers["authorization"] == "Bearer secret-key"


def _sse(*chunks: dict) -> str:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


@respx.mock
@pytest.mark.asyncio
async def test_generate_stream_yields_deltas(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    sse = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
    )
    respx.post(URL).mock(
        return_value=httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAIClient(model="m")
    try:
        chunks = [c async for c in client.generate_stream(_request())]
    finally:
        await client.aclose()
    assert "".join(c.delta_text for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"


@respx.mock
@pytest.mark.asyncio
async def test_provider_key_requires_explicit_mapping(monkeypatch):
    """Provider secret names are selected by catalog api_key_env, never inferred."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    with pytest.raises(OpenAIError, match="missing API key"):
        OpenAIClient(model="m")


@respx.mock
@pytest.mark.asyncio
async def test_extra_headers(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m", extra_headers={"X-Title": "bench"})
    try:
        await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.calls.last.request.headers["x-title"] == "bench"


@respx.mock
@pytest.mark.asyncio
async def test_http_error_terminal(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(OpenAIError, match="HTTP 500"):
            await client.generate_text(_request())
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_auth_error_typed(monkeypatch):
    """Raise LLMAuthError for 401 and 403."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(LLMAuthError, match="HTTP 401"):
            await client.generate_text(_request())
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_timeout_error_typed(monkeypatch):
    """Raise LLMTimeoutError when timeout retries are exhausted."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(LLMTimeoutError):
            await client.generate_text(_request())
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_empty_content_retried(monkeypatch):
    """Retry transient empty content with no generated tokens and return the next normal response."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    empty = {"model": "m", "choices": [{"message": {"role": "assistant", "content": ""}}],
             "usage": {"completion_tokens": 0}}
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(200, json=empty),
        httpx.Response(200, json=_ok_response()),
    ])
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert route.call_count == 2  # Retry the first empty response.


@respx.mock
@pytest.mark.asyncio
async def test_empty_content_with_tool_calls_not_retried(monkeypatch):
    """Empty content is valid alongside tool calls and must not be retried."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=_tool_call_response({"action": "X"}))
    )
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "X"}
    assert route.call_count == 1  # A tool call is present; do not retry.


def test_concurrency_semaphore(monkeypatch):
    """Create a concurrency semaphore from LLM_MAX_CONCURRENT; otherwise leave requests unlimited."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "3")
    c = OpenAIClient(model="m")
    assert c._max_concurrent == 3
    monkeypatch.delenv("LLM_MAX_CONCURRENT")
    assert OpenAIClient(model="m")._max_concurrent is None


@respx.mock
@pytest.mark.asyncio
async def test_max_tokens_clip(monkeypatch):
    """Clip request max_tokens to LLM_MAX_OUTPUT_TOKENS."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "100")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", mode="text", max_tokens=4096)
    try:
        await client.generate_text(req)
    finally:
        await client.aclose()
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 100


@respx.mock
@pytest.mark.asyncio
async def test_truncation_warning(monkeypatch, caplog):
    """Warn on finish_reason=length."""
    import logging
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    resp = _ok_response()
    resp["choices"][0]["finish_reason"] = "length"
    respx.post(URL).mock(return_value=httpx.Response(200, json=resp))
    client = OpenAIClient(model="m")
    try:
        with caplog.at_level(logging.WARNING, logger="llm_mesh.openai.client"):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert any("truncated by max_tokens" in r.message for r in caplog.records)


@respx.mock
@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(429, text="rate limited"),
        httpx.Response(200, json=_ok_response()),
    ])
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# generate_structured (function-calling)
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_calls(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=_tool_call_response({"action": "APP_REMOVE"}))
    )
    client = OpenAIClient(model="Qwen3.5-397B-A17B")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "APP_REMOVE"}
    body = json.loads(route.calls.last.request.content)
    assert body["tool_choice"]["function"]["name"] == "classify_intent"
    assert body["tools"][0]["function"]["name"] == "classify_intent"


@respx.mock
@pytest.mark.asyncio
async def test_structured_fallback_to_content_json(monkeypatch):
    """Parse content JSON when the model ignores tool_choice."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    resp_json = {
        "model": "m",
        "choices": [{"message": {"role": "assistant",
                                 "content": '```json\n{"action": "APP_MODIFY"}\n```'}}],
        "usage": {},
    }
    respx.post(URL).mock(return_value=httpx.Response(200, json=resp_json))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "APP_MODIFY"}


@respx.mock
@pytest.mark.asyncio
async def test_structured_disable_tools_emulation(monkeypatch):
    """With LLM_DISABLE_TOOLS, omit tools and parse content JSON."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
    resp_json = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": '{"action": "X"}'}}],
        "usage": {},
    }
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=resp_json))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "X"}
    assert "tools" not in json.loads(route.calls.last.request.content)


@respx.mock
@pytest.mark.asyncio
async def test_structured_no_json_raises(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    resp_json = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "plain text without json"}}],
        "usage": {},
    }
    respx.post(URL).mock(return_value=httpx.Response(200, json=resp_json))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(LLMValidationError):
            await client.generate_structured(_structured_request())
    finally:
        await client.aclose()


# --- Tool-choice fallback after unsupported routing forms -------------------

_TOOL_CHOICE_404 = (
    "No endpoints found that support the provided 'tool_choice' value."
)


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_choice_404_degrades_to_required(monkeypatch):
    """Retry an unsupported object-form tool_choice with required."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(404, text=_TOOL_CHOICE_404),                          # strict
        httpx.Response(200, json=_tool_call_response({"action": "APP_REMOVE"})),  # required
    ])
    client = OpenAIClient(model="meta-llama/llama-4-maverick")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "APP_REMOVE"}
    assert route.call_count == 2
    # The first request uses object-form tool_choice; the second uses required.
    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert first["tool_choice"] == {"type": "function", "function": {"name": "classify_intent"}}
    assert second["tool_choice"] == "required"
    assert second["tools"][0]["function"]["name"] == "classify_intent"
    # Cache the learned request form.
    assert client._tool_choice_pref == "required"


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_choice_pref_cached(monkeypatch):
    """After fallback, start the next call directly at required."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(404, text=_TOOL_CHOICE_404),                          # call 1: strict
        httpx.Response(200, json=_tool_call_response({"action": "A"})),       # call 1: required
        httpx.Response(200, json=_tool_call_response({"action": "B"})),       # The second call starts at required.
    ])
    client = OpenAIClient(model="meta-llama/llama-4-maverick")
    try:
        await client.generate_structured(_structured_request())
        resp2 = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp2.arguments == {"action": "B"}
    # Three total requests: two for the first call and one for the second.
    assert route.call_count == 3
    assert json.loads(route.calls[2].request.content)["tool_choice"] == "required"


_TOOL_CHOICE_THINKING_400 = (
    '{"error":{"message":"tool_choice \'specified\' is incompatible with '
    'thinking enabled","type":"invalid_request_error"}}'
)


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_choice_thinking_conflict_degrades_to_required(monkeypatch):
    """Treat Kimi's object-form thinking conflict like other unsupported tool-choice forms and
    retry required.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(400, text=_TOOL_CHOICE_THINKING_400),                  # strict
        httpx.Response(200, json=_tool_call_response({"action": "APP_REMOVE"})),  # required
    ])
    client = OpenAIClient(model="kimi-k3")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "APP_REMOVE"}
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert first["tool_choice"] == {"type": "function", "function": {"name": "classify_intent"}}
    assert second["tool_choice"] == "required"
    assert client._tool_choice_pref == "required"


@respx.mock
@pytest.mark.asyncio
async def test_structured_bare_400_propagates(monkeypatch):
    """Propagate an unrelated 400 without fallback."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(
        return_value=httpx.Response(400, text='{"error":{"message":"bad request"}}')
    )
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(OpenAIError):
            await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_structured_bare_404_propagates(monkeypatch):
    """Propagate an unrelated model-not-found 404 without fallback."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(
        return_value=httpx.Response(404, text="model not found")
    )
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(OpenAIError):
            await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    # Make one attempt and propagate the error to the caller.
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_structured_required_also_404_falls_to_text(monkeypatch):
    """Fall back to text JSON when both strict and required are unsupported."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(404, text=_TOOL_CHOICE_404),                          # strict
        httpx.Response(404, text=_TOOL_CHOICE_404),                          # required
        httpx.Response(200, json={                                           # text-JSON
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": '{"action": "T"}'}}],
            "usage": {},
        }),
    ])
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "T"}
    assert route.call_count == 3
    # The third request contains neither tools nor tool_choice.
    third = json.loads(route.calls[2].request.content)
    assert "tools" not in third
    assert "tool_choice" not in third
    assert client._tool_choice_pref == "text"


# --- Text fallback after tool-parser template construction failures ---------

_TOOL_PARSER_400 = (
    '{"error":{"code":400,"message":"Unable to generate parser for this '
    'template. Automatic parser generation failed: Cannot perform operation '
    'on null values","type":"invalid_request_error"}}'
)


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_parser_400_degrades_to_text(monkeypatch):
    """Fall back from a strict template-parser 400 to text JSON without tools."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(400, text=_TOOL_PARSER_400),                          # strict
        httpx.Response(200, json={                                           # text-JSON
            "model": "gigachat",
            "choices": [{"message": {"role": "assistant", "content": '{"action": "T"}'}}],
            "usage": {},
        }),
    ])
    client = OpenAIClient(model="gigachat")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "T"}
    assert route.call_count == 2
    second = json.loads(route.calls[1].request.content)
    assert "tools" not in second
    assert "tool_choice" not in second
    assert client._tool_choice_pref == "text"


@respx.mock
@pytest.mark.asyncio
async def test_structured_tool_parser_400_multi_tool_degrades_to_text(monkeypatch):
    """Fall back from a multi-tool template-parser 400 to text JSON."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(400, text=_TOOL_PARSER_400),                          # multi auto
        httpx.Response(200, json={                                           # text-JSON
            "model": "gigachat",
            "choices": [{"message": {"role": "assistant", "content": '{"action": "T"}'}}],
            "usage": {},
        }),
    ])
    req = LLMRequest(
        system="s", user="u", mode="function_call",
        function_name="classify_intent",
        schema={"type": "object", "properties": {"action": {"type": "string"}}},
        tools=[{
            "name": "classify_intent",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        }],
    )
    client = OpenAIClient(model="gigachat")
    try:
        resp = await client.generate_structured(req)
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "T"}
    assert route.call_count == 2
    assert client._multitool_supported is False
    assert client._tool_choice_pref == "text"
    second = json.loads(route.calls[1].request.content)
    assert "tools" not in second


@respx.mock
@pytest.mark.asyncio
async def test_structured_silent_no_tool_call_degrades_to_required(monkeypatch):
    """A 200 response containing prose instead of a forced call must advance to required, just like
    an explicit unsupported-form response.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(200, json=_ok_response("Sure, I can help with that!")),   # Strict returns prose rather than JSON.
        httpx.Response(200, json=_tool_call_response({"action": "APP_REMOVE"})),  # Required succeeds.
    ])
    client = OpenAIClient(model="t-lite-it-2.1")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "APP_REMOVE"}
    assert route.call_count == 2
    assert client._tool_choice_pref == "required"


@respx.mock
@pytest.mark.asyncio
async def test_structured_silent_no_tool_call_all_tiers_falls_to_text(monkeypatch):
    """When both forced tiers silently return prose, try terminal text with a JSON instruction."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(200, json=_ok_response("Sure, I can help with that!")),   # strict
        httpx.Response(200, json=_ok_response("Okay, I will look into it.")),  # required
        httpx.Response(200, json=_ok_response('{"action": "T"}')),           # text-JSON
    ])
    client = OpenAIClient(model="t-lite-it-2.1")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "T"}
    assert route.call_count == 3
    third = json.loads(route.calls[2].request.content)
    assert "tools" not in third
    assert "tool_choice" not in third
    assert client._tool_choice_pref == "text"


# --- Reasoning settings and extra request fields ----------------------------


def test_disable_reasoning_without_dialect_sends_no_vendor_fields(monkeypatch):
    """Use only explicitly configured provider fields."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    c = OpenAIClient(model="m")
    assert c._extra_body == {}


def test_reasoning_field_default_reads_both_known_names(monkeypatch):
    """Without an explicit field name, read either reasoning_content or reasoning so undeclared
    routes do not lose their thinking channel.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    assert c._reasoning_content_of({"reasoning_content": "thought"}) == "thought"
    assert c._reasoning_content_of({"reasoning": "thought"}) == "thought"


def test_declared_reasoning_field_narrows_the_read(monkeypatch):
    """An explicitly declared reasoning field restricts parsing to that name."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_REASONING_FIELD", "reasoning")
    c = OpenAIClient(model="m")
    assert c._reasoning_content_of({"reasoning": "thought"}) == "thought"
    assert c._reasoning_content_of({"reasoning_content": "thought"}) is None


def test_route_specific_reasoning_off_replaces_the_generic_set(monkeypatch):
    """A route-specific reasoning_off replaces the generic vendor field set. Strict gateways reject
    unfamiliar keys, so send only the declared dialect.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_OFF", '{"reasoning_effort": "none"}')
    c = OpenAIClient(model="m")
    assert c._extra_body == {"reasoning_effort": "none"}


def test_route_specific_reasoning_off_nested_dialect_sent_alone(monkeypatch):
    """Send the declared nested reasoning.effort=none dialect without unrelated generic fields."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_OFF", '{"reasoning": {"effort": "none"}}')
    c = OpenAIClient(model="m")
    assert c._extra_body == {"reasoning": {"effort": "none"}}


def test_missing_reasoning_off_preserves_declared_baseline(monkeypatch):
    """Use only explicitly configured provider fields."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_ON", '{"thinking": {"type": "enabled"}}')
    c = OpenAIClient(model="m")
    assert c._extra_body == {"thinking": {"type": "enabled"}}


def test_route_specific_reasoning_on_applied_as_baseline(monkeypatch):
    """Apply a declared reasoning_on dialect to all request bodies; otherwise reasoning-on routes
    could silently use a disabled gateway default.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_REASONING_ON", '{"thinking": {"type": "enabled"}}')
    c = OpenAIClient(model="m")
    assert c._extra_body["thinking"] == {"type": "enabled"}


def test_reasoning_on_merges_nested_object_without_erasing_extra_body(monkeypatch):
    """Merge nested reasoning_on keys without erasing unrelated extra_body fields."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"reasoning": {"exclude": false}}')
    monkeypatch.setenv("LLM_REASONING_ON", '{"reasoning": {"effort": "high"}}')
    c = OpenAIClient(model="m")
    assert c._extra_body["reasoning"] == {"exclude": False, "effort": "high"}


def test_reasoning_on_ignored_when_disable_reasoning_set(monkeypatch):
    """When disabling reasoning, send only the off dialect even if the route declares both on and
    off settings.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_ON", '{"thinking": {"type": "enabled"}}')
    monkeypatch.setenv("LLM_REASONING_OFF", '{"thinking": {"type": "disabled"}}')
    c = OpenAIClient(model="m")
    assert c._extra_body["thinking"] == {"type": "disabled"}
    assert c._reasoning_on_body == {}


def test_reasoning_on_absent_keeps_body_byte_identical(monkeypatch):
    """Without LLM_REASONING_ON, preserve the existing body."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    assert c._extra_body == {}


def test_reasoning_on_invalid_json_ignored(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_REASONING_ON", "{not json")
    c = OpenAIClient(model="m")
    assert c._extra_body == {}


def test_reasoning_off_invalid_json_does_not_guess_vendor_fields(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_OFF", "{not json")
    c = OpenAIClient(model="m")
    assert c._extra_body == {}


def test_reasoning_off_empty_object_sends_no_vendor_fields(monkeypatch):
    """Use only explicitly configured provider fields."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_REASONING_OFF", "{}")
    c = OpenAIClient(model="m")
    assert c._extra_body == {}


def test_disable_reasoning_unset_no_extra_body(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert OpenAIClient(model="m")._extra_body == {}


def test_extra_body_parsed_and_merged(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"top_p": 0.8, "chat_template_kwargs": {"enable_thinking": false}}')
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    c = OpenAIClient(model="m")
    assert c._extra_body["top_p"] == 0.8
    assert c._extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning" not in c._extra_body


def test_extra_body_invalid_json_ignored(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_EXTRA_BODY", "{not json")
    assert OpenAIClient(model="m")._extra_body == {}


@respx.mock
@pytest.mark.asyncio
async def test_extra_body_injected_into_request(monkeypatch):
    """Send configured reasoning and chat-template fields in the request body."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    monkeypatch.setenv("LLM_REASONING_OFF", '{"chat_template_kwargs": {"enable_thinking": false}, "reasoning": {"enabled": false}}')
    client = OpenAIClient(model="m")
    try:
        await client.generate_text(_request())
    finally:
        await client.aclose()
    body = json.loads(route.calls.last.request.content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning"] == {"enabled": False}


@respx.mock
@pytest.mark.asyncio
async def test_reasoning_mandatory_fallback(monkeypatch):
    """If reasoning cannot be disabled, retry with exclude=true."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    mandatory_err = httpx.Response(
        400, text='{"error": {"message": "Reasoning is mandatory and cannot be disabled"}}'
    )
    route = respx.post(URL).mock(side_effect=[mandatory_err, httpx.Response(200, json=_ok_response())])
    monkeypatch.setenv("LLM_REASONING_OFF", '{"chat_template_kwargs": {"enable_thinking": false}, "reasoning": {"enabled": false}}')
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert route.call_count == 2
    # The second request uses exclude=true instead of enabled=false.
    fallback_body = json.loads(route.calls.last.request.content)
    assert fallback_body["reasoning"] == {"exclude": True}


def _len_response(content: str = "truncated", finish: str = "length") -> dict:
    return {
        "model": "GigaChat3-10B",
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


@pytest.mark.asyncio
@respx.mock
async def test_length_retry_doubles_max_tokens(monkeypatch):
    """After length truncation, double max_tokens and return the complete second response."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[int] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body["max_tokens"])
        if len(seen) == 1:
            return httpx.Response(200, json=_len_response("truncated"))
        return httpx.Response(200, json={
            "model": "GigaChat3-10B",
            "choices": [{"message": {"role": "assistant", "content": "complete"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
        })

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="GigaChat3-10B")
    resp = await client.generate_text(LLMRequest(system="s", user="u", max_tokens=8192))
    assert resp.text == "complete"
    assert seen == [8192, 16384]  # Double the budget after length truncation.
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_length_retry_stops_at_cap(monkeypatch):
    """Stop retrying at the ceiling and return the latest response for caller-side validation."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_LENGTH_RETRY_CAP", "16384")
    seen: list[int] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content)["max_tokens"])
        return httpx.Response(200, json=_len_response("still truncated"))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="GigaChat3-10B")
    resp = await client.generate_text(LLMRequest(system="s", user="u", max_tokens=8192))
    assert resp.text == "still truncated"
    assert seen == [8192, 16384]  # Increase 8192 to the 16384 ceiling, then stop.
    await client.aclose()


def test_unwrap_function_call_envelope():
    """Unwrap narrow function-call content envelopes while preserving ordinary objects and
    non-dictionary arguments.
    """
    u = OpenAIClient._unwrap_function_call_envelope
    # Unwrap the envelope into its arguments.
    assert u({"name": "emit_x", "arguments": {"action": "BUILD"}}) == {"action": "BUILD"}
    # An arguments-only envelope also matches the permitted key set.
    assert u({"arguments": {"a": 1}}) == {"a": 1}
    # Preserve an ordinary argument object.
    assert u({"action": "BUILD", "scope": "app"}) == {"action": "BUILD", "scope": "app"}
    # A name with non-dictionary arguments can be a legitimate object; preserve it.
    assert u({"name": "John", "role": "manager"}) == {"name": "John", "role": "manager"}
    # An extra key means this is not a call envelope.
    assert u({"name": "x", "arguments": {"a": 1}, "extra": 2}) == {"name": "x", "arguments": {"a": 1}, "extra": 2}
    # Preserve list arguments rather than treating them as an object envelope.
    assert u({"name": "x", "arguments": [1, 2]}) == {"name": "x", "arguments": [1, 2]}


def test_parse_tool_arguments_content_envelope_mlx():
    """Strip the function-call marker and unwrap an MLX content envelope into schema arguments."""
    client = OpenAIClient(model="m", base_url="http://x/v1", api_key="k")
    msg = {"content": '<|function_call|>{"name":"classify","arguments":{"action":"REQ_GENERATE","scope":"app"}}'}
    args = client._parse_tool_arguments(msg)
    assert args == {"action": "REQ_GENERATE", "scope": "app"}


def test_unwrap_singleton_list_arg():
    """Unwrap a singleton list containing one object; preserve multi-element and non-object lists."""
    u = OpenAIClient._unwrap_singleton_list_arg
    assert u([{"artifact_kind": "forms"}]) == {"artifact_kind": "forms"}
    # Do not conceal a genuinely invalid multi-element shape.
    assert u([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]
    # Preserve lists of non-objects.
    assert u([1, 2, 3]) == [1, 2, 3]
    assert u([[1]]) == [[1]]
    # Preserve dictionaries unchanged.
    assert u({"artifact_kind": "forms"}) == {"artifact_kind": "forms"}
    assert u([]) == []


def test_parse_tool_arguments_singleton_list_gemma():
    """Decode a JSON-string singleton argument array into an object accepted by LLMResponse."""
    client = OpenAIClient(model="m", base_url="http://x/v1", api_key="k")
    msg = {"tool_calls": [{"function": {
        "name": "emit_artifact",
        "arguments": '[{"artifact_kind": "forms", "form_id": "F1"}]',
    }}]}
    args = client._parse_tool_arguments(msg)
    assert args == {"artifact_kind": "forms", "form_id": "F1"}


def _length_response(content: str = "repetition", model: str = "Qwen3.5-397B-A17B") -> dict:
    """Build a response truncated by max_tokens."""
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@respx.mock
@pytest.mark.asyncio
async def test_generate_text_length_retry_default_on(monkeypatch):
    """Length retries are enabled by default and resend with doubled max_tokens."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_length_response()))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", mode="text", max_tokens=4096)  # Use the default length-retry setting.
    await client.generate_text(req)
    # One initial request plus two default retries gives three POSTs.
    assert route.call_count == 3


@respx.mock
@pytest.mark.asyncio
async def test_generate_text_length_retry_opt_out(monkeypatch):
    """With length_retry=False, return truncated best-effort output after one POST instead of
    spending more time on a likely generation loop.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_length_response()))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", mode="text", max_tokens=4096, length_retry=False)
    resp = await client.generate_text(req)
    assert route.call_count == 1
    assert resp.text == "repetition"


@respx.mock
@pytest.mark.asyncio
async def test_length_retry_skipped_on_degenerate_output(monkeypatch):
    """Do not retry repetitive length-truncated output; issue one POST rather than three."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    degenerate = _length_response(content="}\n" * 3000)
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=degenerate))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", mode="text", max_tokens=4096)
    await client.generate_text(req)
    assert route.call_count == 1  # The repetition guard prevents budget doubling.


@respx.mock
@pytest.mark.asyncio
async def test_length_retry_proceeds_on_legit_truncation(monkeypatch):
    """Allow budget growth for legitimate varied truncation; repetition detection must not block
    useful retries.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    legit = "".join(
        f'<bpmn:userTask id="Activity_{i}" name="Task {i}"/>\n' for i in range(100)
    )
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_length_response(content=legit)))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", mode="text", max_tokens=4096)
    await client.generate_text(req)
    assert route.call_count == 3  # One initial request plus two default length retries.


# ---------------------------------------------------------------------------
# Native tool-loop (request.tools_required)
# ---------------------------------------------------------------------------


def _tools_response(*calls: tuple[str, str], content: str | None = None) -> dict:
    """Build a response with N named tool calls and JSON-string arguments."""
    return {
        "model": "m",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                    for i, (name, args) in enumerate(calls)
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }


_TOOLS = [
    {"name": "get_record", "description": "d", "parameters": {"type": "object"}},
    {"name": "finish", "description": "d", "parameters": {"type": "object"}},
]


@respx.mock
@pytest.mark.asyncio
async def test_multi_tool_returns_all_calls_of_a_turn(monkeypatch):
    """Return every parallel tool call so the caller can respond to each."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tools_response(
        ("get_record", '{"record_id": "r1"}'),
        ("finish", '{"diagnosis": "d"}'),
    )))
    client = OpenAIClient(model="m")
    resp = await client.generate_structured(
        LLMRequest(system="s", user="u", tools=_TOOLS, function_name="finish")
    )
    assert [c["name"] for c in resp.tool_calls] == ["get_record", "finish"]
    assert [c["id"] for c in resp.tool_calls] == ["call_0", "call_1"]
    assert resp.tool_calls[0]["arguments"] == {"record_id": "r1"}
    # For backward compatibility, expose the first call through function_name and arguments.
    assert resp.function_name == "get_record"
    assert resp.arguments == {"record_id": "r1"}


@respx.mock
@pytest.mark.asyncio
async def test_multi_tool_broken_args_of_one_call_do_not_kill_the_turn(monkeypatch):
    """Malformed JSON in one parallel call yields empty arguments without discarding other calls."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tools_response(
        ("get_record", '{"record_id": "r1"}'),
        ("finish", "not json"),
    )))
    client = OpenAIClient(model="m")
    resp = await client.generate_structured(
        LLMRequest(system="s", user="u", tools=_TOOLS, function_name="finish")
    )
    assert resp.tool_calls[1]["arguments"] == {}
    assert resp.tool_calls[0]["arguments"] == {"record_id": "r1"}


@respx.mock
@pytest.mark.asyncio
async def test_tools_required_returns_turn_as_is_without_degrading(monkeypatch):
    """With tools_required, preserve a text-only model turn rather than fabricate a forced finish
    call.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "Let me think"},
                     "finish_reason": "stop"}],
    }))
    client = OpenAIClient(model="m")
    resp = await client.generate_structured(LLMRequest(
        system="s", user="u", tools=_TOOLS, tools_required=True,
        function_name="finish",
    ))
    assert resp.tool_calls == []
    assert resp.text == "Let me think"
    assert route.call_count == 1  # No forced retry occurs.
    # Do not cache multi-tool rejection or affect neighboring calls.
    assert client._multitool_supported is None


@respx.mock
@pytest.mark.asyncio
async def test_tool_history_reaches_the_wire(monkeypatch):
    """Forward assistant tool calls and corresponding tool-result history to the provider."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(
        200, json=_tools_response(("finish", '{"diagnosis": "d"}')),
    ))
    client = OpenAIClient(model="m")
    await client.generate_structured(LLMRequest(
        system="s", user="u", tools=_TOOLS, tools_required=True,
        function_name="finish",
        history=[
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_record", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
        ],
    ))
    body = json.loads(route.calls[0].request.content)
    assert [m["role"] for m in body["messages"]] == [
        "system", "assistant", "tool", "user",
    ]
    assert body["messages"][1]["tool_calls"][0]["id"] == "c1"
    assert body["messages"][2] == {
        "role": "tool", "tool_call_id": "c1", "content": "OBS",
    }
    assert body["tool_choice"] == "auto"


@respx.mock
@pytest.mark.asyncio
async def test_tools_required_rejected_free_when_tools_disabled(monkeypatch):
    """Reject native loops without HTTP when LLM_DISABLE_TOOLS is set."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    with pytest.raises(LLMValidationError) as caught:
        await client.generate_structured(LLMRequest(
            system="s", user="u", tools=_TOOLS, tools_required=True,
            function_name="finish",
        ))
    assert "LLM_DISABLE_TOOLS" in str(caught.value)
    assert route.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_tools_required_tolerates_empty_arguments_on_first_call(monkeypatch):
    """Allow empty arguments in the first native-loop call; tools without required parameters must
    not force the entire loop into text fallback.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tools_response(
        ("validate_record", ""),
        ("get_record", '{"record_id": "r1"}'),
    )))
    client = OpenAIClient(model="m")
    resp = await client.generate_structured(LLMRequest(
        system="s", user="u", tools=_TOOLS, tools_required=True,
        function_name="finish",
    ))
    assert [c["name"] for c in resp.tool_calls] == ["validate_record", "get_record"]
    assert resp.tool_calls[0]["arguments"] == {}
    assert resp.function_name == "validate_record"
    assert resp.arguments == {}


@respx.mock
@pytest.mark.asyncio
async def test_non_strict_multi_tool_keeps_strict_argument_parsing(monkeypatch):
    """Without tools_required, preserve strict first-call parsing and content JSON recovery."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tools_response(
        ("get_record", "not json"),
    )))
    client = OpenAIClient(model="m")
    with pytest.raises(LLMValidationError):
        await client.generate_structured(
            LLMRequest(system="s", user="u", tools=_TOOLS, function_name="finish")
        )


@respx.mock
@pytest.mark.asyncio
async def test_body_level_5xx_error_retried(monkeypatch):
    """Retry an HTTP 200 body carrying a retryable upstream error code instead of failing as
    missing choices.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    responses = iter([
        httpx.Response(200, json={"error": {"message": "Internal Server Error", "code": 500}}),
        httpx.Response(200, json=_tool_call_response({"action": "APP_BUILD"})),
    ])
    route = respx.post(URL).mock(side_effect=lambda _r: next(responses))

    import asyncio as _aio
    orig = _aio.sleep
    monkeypatch.setattr(_aio, "sleep", lambda *a, **k: orig(0))

    client = OpenAIClient(model="m")
    resp = await client.generate_structured(_structured_request())
    await client.aclose()
    assert resp.arguments == {"action": "APP_BUILD"}  # The retry recovered successfully.
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_body_level_non_retryable_error_raises(monkeypatch):
    """Propagate non-retryable body-level errors without retrying."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json={"error": {"message": "bad request", "code": 400}})
    )
    client = OpenAIClient(model="m")
    with pytest.raises(OpenAIError, match="no choices"):
        await client.generate_structured(_structured_request())
    await client.aclose()
    assert route.call_count == 1  # No retry.


# --- Minimum output budget for reasoning headroom ---------------------------


def _empty_len_response() -> dict:
    """Build an empty length-truncated response whose budget was consumed by reasoning."""
    return {
        "model": "reasoner",
        "choices": [{"message": {"role": "assistant", "content": ""},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 500, "total_tokens": 505},
    }


def test_min_output_tokens_parsed_from_env(monkeypatch):
    """Read the optional budget floor from the environment; leave it unset by default."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "4000")
    assert OpenAIClient(model="m")._min_output_tokens == 4000
    monkeypatch.delenv("LLM_MIN_OUTPUT_TOKENS")
    assert OpenAIClient(model="m")._min_output_tokens is None
    # Ignore invalid or zero values without raising.
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "not-a-number")
    assert OpenAIClient(model="m")._min_output_tokens is None
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "0")
    assert OpenAIClient(model="m")._min_output_tokens is None


@respx.mock
@pytest.mark.asyncio
async def test_min_output_tokens_raises_small_budget(monkeypatch):
    """Raise a small request budget to the configured floor."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "4000")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    try:
        await client.generate_text(LLMRequest(system="s", user="u", max_tokens=500))
    finally:
        await client.aclose()
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 4000


@respx.mock
@pytest.mark.asyncio
async def test_min_output_tokens_does_not_lower_large_budget(monkeypatch):
    """Do not reduce budgets already above the floor."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "4000")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    try:
        await client.generate_text(LLMRequest(system="s", user="u", max_tokens=8192))
    finally:
        await client.aclose()
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 8192


@respx.mock
@pytest.mark.asyncio
async def test_min_output_tokens_ceiling_wins_over_floor(monkeypatch):
    """The hard ceiling wins when the configured floor exceeds it."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "8000")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2000")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    try:
        await client.generate_text(LLMRequest(system="s", user="u", max_tokens=500))
    finally:
        await client.aclose()
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 2000


@respx.mock
@pytest.mark.asyncio
async def test_min_output_tokens_default_off_is_byte_identical(monkeypatch):
    """Without a floor, preserve the requested initial budget. Do not repeat empty length responses
    at the same budget; allow one doubling, then stop if content remains empty. This bounds the
    failure to two POSTs.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.delenv("LLM_MIN_OUTPUT_TOKENS", raising=False)
    seen: list[int] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content)["max_tokens"])
        return httpx.Response(200, json=_empty_len_response())

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="reasoner")
    try:
        await client.generate_text(LLMRequest(system="s", user="u", max_tokens=500))
    finally:
        await client.aclose()
    assert seen[0] == 500  # Start with exactly the requested budget.
    assert seen == [500, 1000]  # Make exactly two attempts without a retry storm.


@respx.mock
@pytest.mark.asyncio
async def test_min_output_tokens_starts_high_and_avoids_storm(monkeypatch):
    """A configured floor supplies sufficient reasoning headroom on the first attempt and avoids
    the empty-response retry cycle.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "4000")
    seen: list[int] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content)["max_tokens"])
        if len(seen) == 1:
            return httpx.Response(200, json=_empty_len_response())
        return httpx.Response(200, json=_ok_response("final text"))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="reasoner")
    try:
        resp = await client.generate_text(
            LLMRequest(system="s", user="u", max_tokens=500)
        )
    finally:
        await client.aclose()
    assert resp.text == "final text"
    assert seen[0] == 4000  # Start at the floor instead of 500.
    assert 500 not in seen  # Never send the original insufficient budget.


# --- Finish reasons across all response construction paths ------------------


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "stop"])
async def test_finish_reason_propagated_text(monkeypatch, reason):
    """Propagate finish_reason through text generation."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "text"},
                     "finish_reason": reason}],
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(
            LLMRequest(system="s", user="u", mode="text", length_retry=False)
        )
    finally:
        await client.aclose()
    assert resp.finish_reason == reason


@respx.mock
@pytest.mark.asyncio
async def test_finish_reason_propagated_structured(monkeypatch):
    """Propagate finish_reason through a single forced function generation."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "classify_intent",
                             "arguments": '{"action": "delete"}'},
            }]},
            "finish_reason": "tool_calls",
        }],
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.finish_reason == "tool_calls"
    assert resp.arguments == {"action": "delete"}


@respx.mock
@pytest.mark.asyncio
async def test_finish_reason_propagated_multi_tool(monkeypatch):
    """Propagate finish_reason through multi-tool calls."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tools_response(
        ("get_record", '{"record_id": "r1"}'),
    )))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(
            LLMRequest(system="s", user="u", tools=_TOOLS, function_name="finish")
        )
    finally:
        await client.aclose()
    assert resp.finish_reason == "tool_calls"


@respx.mock
@pytest.mark.asyncio
async def test_finish_reason_propagated_multi_tool_text_turn(monkeypatch):
    """Preserve finish_reason on native-loop text turns to distinguish budget exhaustion from a
    deliberate text response.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "Let me think"},
                     "finish_reason": "length"}],
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(LLMRequest(
            system="s", user="u", tools=_TOOLS, tools_required=True,
            function_name="finish",
        ))
    finally:
        await client.aclose()
    assert resp.tool_calls == []
    assert resp.finish_reason == "length"


@respx.mock
@pytest.mark.asyncio
async def test_finish_reason_absent_is_none(monkeypatch):
    """Use None when the provider omits finish_reason."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(LLMRequest(system="s", user="u", mode="text"))
    finally:
        await client.aclose()
    assert resp.finish_reason is None


# --- Serving-tier visibility and no-degrade measurements ---------------------


def _no_tool_call_text_response(content: str = "plain text") -> dict:
    """Build a 200 response with neither a tool call nor recognizable JSON."""
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


@respx.mock
@pytest.mark.asyncio
async def test_served_tier_strict_on_success(monkeypatch):
    """Record strict as the serving tier when strict forcing succeeds."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_tool_call_response({"action": "delete"})))
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert client._last_served_tier == "strict"


@respx.mock
@pytest.mark.asyncio
async def test_served_tier_text_on_silent_degradation(monkeypatch):
    """Expose silent fallback to text through the serving-tier metric."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if "tools" in body:  # Strict and required return the same unparseable text.
            return httpx.Response(200, json=_no_tool_call_text_response())
        return httpx.Response(200, json=_ok_response('{"action": "delete"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "delete"}  # The call succeeds...
    assert client._last_served_tier == "text"      # ...through the text tier.


@respx.mock
@pytest.mark.asyncio
async def test_no_degrade_refuses_silent_text_fallback(monkeypatch):
    """Under LLM_NO_DEGRADE, reject silent text fallback."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_NO_DEGRADE", "1")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_no_tool_call_text_response()))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(LLMValidationError, match="LLM_NO_DEGRADE"):
            await client.generate_structured(_structured_request())
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_no_degrade_does_not_block_explicit_text_mode(monkeypatch):
    """Explicit LLM_DISABLE_TOOLS text mode remains valid under no-degrade because it is the
    requested mode.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_NO_DEGRADE", "1")
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response('{"action": "delete"}')))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "delete"}
    assert client._last_served_tier == "text"


@respx.mock
@pytest.mark.asyncio
async def test_no_degrade_default_off_keeps_prod_fallback(monkeypatch):
    """Without no-degrade, preserve the production fallback behavior."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_NO_DEGRADE", raising=False)

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if "tools" in body:
            return httpx.Response(200, json=_no_tool_call_text_response())
        return httpx.Response(200, json=_ok_response('{"action": "delete"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"action": "delete"}  # Return a response without raising.


@respx.mock
@pytest.mark.asyncio
async def test_served_tier_multi_and_text_turn(monkeypatch):
    """Distinguish native multi-tool calls from model-selected text turns; the latter are not a
    client text tier fallback.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    respx.post(URL).mock(return_value=httpx.Response(
        200, json=_tools_response(("get_record", '{"record_id": "r1"}')),
    ))
    client = OpenAIClient(model="m")
    req = LLMRequest(system="s", user="u", tools=_TOOLS, tools_required=True,
                     function_name="finish")
    try:
        await client.generate_structured(req)
        assert client._last_served_tier == "multi"
        respx.post(URL).mock(return_value=httpx.Response(
            200, json=_no_tool_call_text_response("Let me think"),
        ))
        await client.generate_structured(req)
    finally:
        await client.aclose()
    assert client._last_served_tier == "multi-text-turn"


def test_no_degrade_parsed_from_env(monkeypatch):
    """No-degrade defaults to off and accepts conventional truthy environment values."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_NO_DEGRADE", raising=False)
    assert OpenAIClient(model="m")._no_degrade is False
    assert OpenAIClient(model="m")._last_served_tier is None
    for val in ("1", "true", "yes", "TRUE"):
        monkeypatch.setenv("LLM_NO_DEGRADE", val)
        assert OpenAIClient(model="m")._no_degrade is True


# --- Learn temperature=1 restrictions per client instance -------------------

_TEMP_400 = {"error": {"message": "Unsupported value: 'temperature' does not "
                                  "support 0 with this model. Only the default "
                                  "(1) value is supported.", "code": 400}}


def test_temperature_unsupported_detector():
    """Recognize temperature restrictions without matching unrelated 400 errors."""
    assert _is_temperature_unsupported_error(
        "Unsupported value: 'temperature' does not support 0 with this model"
    )
    assert _is_temperature_unsupported_error("Only the default (1) temperature value")
    assert not _is_temperature_unsupported_error("tool_choice value is not supported")
    assert not _is_temperature_unsupported_error("Reasoning is mandatory")


@respx.mock
@pytest.mark.asyncio
async def test_temperature_fallback_retries_with_one(monkeypatch):
    """Retry a temperature restriction with temperature=1."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[float] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        temp = json.loads(req.content).get("temperature")
        seen.append(temp)
        if temp != 1.0:
            return httpx.Response(400, json=_TEMP_400)
        return httpx.Response(200, json=_ok_response("response"))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="gpt-5.5")
    try:
        resp = await client.generate_text(
            LLMRequest(system="s", user="u", mode="text", temperature=0.0)
        )
    finally:
        await client.aclose()
    assert resp.text == "response"
    assert seen == [0.0, 1.0]


@respx.mock
@pytest.mark.asyncio
async def test_temperature_fallback_is_learned_for_next_calls(monkeypatch):
    """Use the learned temperature directly on subsequent calls."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[float] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        temp = json.loads(req.content).get("temperature")
        seen.append(temp)
        if temp != 1.0:
            return httpx.Response(400, json=_TEMP_400)
        return httpx.Response(200, json=_ok_response("response"))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="gpt-5.5")
    req = LLMRequest(system="s", user="u", mode="text", temperature=0.0)
    try:
        await client.generate_text(req)
        await client.generate_text(req)
    finally:
        await client.aclose()
    assert seen == [0.0, 1.0, 1.0]  # Pay for the rejected temperature only once.


@respx.mock
@pytest.mark.asyncio
async def test_temperature_fallback_does_not_leak_between_instances(monkeypatch):
    """Keep learned temperature restrictions local to the instance so other models retain
    deterministic settings.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[float] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        temp = json.loads(req.content).get("temperature")
        seen.append(temp)
        if temp != 1.0:
            return httpx.Response(400, json=_TEMP_400)
        return httpx.Response(200, json=_ok_response("response"))

    respx.post(URL).mock(side_effect=_handler)
    picky = OpenAIClient(model="gpt-5.5")
    try:
        await picky.generate_text(
            LLMRequest(system="s", user="u", mode="text", temperature=0.0)
        )
    finally:
        await picky.aclose()
    assert picky._force_temperature == 1.0

    other = OpenAIClient(model="qwen")  # A different model in the same process.
    try:
        assert other._force_temperature is None
        respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response()))
        route_seen: list[float] = []

        def _h2(req: httpx.Request) -> httpx.Response:
            route_seen.append(json.loads(req.content).get("temperature"))
            return httpx.Response(200, json=_ok_response())

        respx.post(URL).mock(side_effect=_h2)
        await other.generate_text(
            LLMRequest(system="s", user="u", mode="text", temperature=0.0)
        )
    finally:
        await other.aclose()
    assert route_seen == [0.0]  # The neighboring model keeps its requested temperature.


@respx.mock
@pytest.mark.asyncio
async def test_temperature_fallback_not_triggered_by_other_400(monkeypatch):
    """An unrelated 400 neither retries nor teaches a temperature override."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(return_value=httpx.Response(
        400, json={"error": {"message": "context length exceeded"}},
    ))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(OpenAIError, match="HTTP 400"):
            await client.generate_text(
                LLMRequest(system="s", user="u", mode="text", temperature=0.0)
            )
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert client._force_temperature is None


# --- Regressions discovered by multi-provider probes ------------------------


@respx.mock
@pytest.mark.asyncio
async def test_content_filter_empty_is_returned_without_retry(monkeypatch):
    """Return moderation-filtered empty content without repeating a deterministic refusal."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": ""},
                     "finish_reason": "content_filter"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 1  # Do not retry.
    assert resp.text == ""
    assert resp.finish_reason == "content_filter"


@respx.mock
@pytest.mark.asyncio
async def test_reasoning_without_action_is_returned_without_retry(monkeypatch):
    """Return a reasoning-only stop without repeating the same non-action."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": ""},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 614, "total_tokens": 619},
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert resp.finish_reason == "stop"


@respx.mock
@pytest.mark.asyncio
async def test_empty_content_with_stop_and_zero_tokens_still_retries(monkeypatch):
    """Still retry empty stop responses with zero generated tokens as transient provider failures."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    calls = {"n": 0}

    def _handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={
                "model": "m",
                "choices": [{"message": {"role": "assistant", "content": ""},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0,
                          "total_tokens": 5},
            })
        return httpx.Response(200, json=_ok_response("recovered"))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert calls["n"] == 2
    assert resp.text == "recovered"


def test_usage_reads_reasoning_tokens_from_both_shapes():
    """Parse nested and flat reasoning token counts centrally, preferring nested values, so
    blocking and streaming usage agree.
    """
    from llm_mesh.types import LLMUsage

    nested = LLMUsage.from_raw({
        "prompt_tokens": 10, "completion_tokens": 4000, "total_tokens": 4010,
        "completion_tokens_details": {"reasoning_tokens": 3900},
    })
    assert nested.reasoning_tokens == 3900
    flat = LLMUsage.from_raw({"completion_tokens": 100, "reasoning_tokens": 90})
    assert flat.reasoning_tokens == 90
    # Default reasoning tokens to zero when absent.
    assert LLMUsage.from_raw({"completion_tokens": 7}).reasoning_tokens == 0
    # Malformed usage must not crash parsing.
    assert LLMUsage.from_raw(None).total_tokens == 0
    assert LLMUsage.from_raw({"prompt_tokens": "missing"}).prompt_tokens == 0


def test_alibaba_400_tool_choice_is_degradation_signal():
    """Treat Alibaba's tool-choice 400 as an unsupported-form signal, like OpenRouter's 404."""
    from llm_mesh.openai.client import _is_tool_choice_unsupported_error

    alibaba = OpenAIError(
        "HTTP 400", status_code=400,
        detail="The tool_choice parameter does not support being set to "
               "required or a specific function",
    )
    assert _is_tool_choice_unsupported_error(alibaba)
    openrouter = OpenAIError(
        "HTTP 404", status_code=404,
        detail="No endpoints found that support the provided 'tool_choice'",
    )
    assert _is_tool_choice_unsupported_error(openrouter)
    # Unrelated 400 and 404 responses must propagate instead of triggering tool-choice fallback.
    assert not _is_tool_choice_unsupported_error(
        OpenAIError("HTTP 400", status_code=400, detail="invalid schema")
    )
    assert not _is_tool_choice_unsupported_error(
        OpenAIError("HTTP 404", status_code=404, detail="model not found")
    )


def test_peg_format_500_is_degradation_signal():
    """Treat a server PEG parser 500 as a parser limitation and fall back to text."""
    from llm_mesh.openai.client import _is_peg_format_error

    assert _is_peg_format_error(OpenAIError(
        "HTTP 500", status_code=500,
        detail="The model produced output that does not match the expected "
               "peg-gemma4 format",
    ))
    # Ordinary 500 responses remain transient and are retried by transport.
    assert not _is_peg_format_error(
        OpenAIError("HTTP 500", status_code=500, detail="internal error")
    )


@respx.mock
@pytest.mark.asyncio
async def test_multitool_miss_is_not_cached(monkeypatch):
    """A missing auto tool call affects only that request; do not disable multi-tool for the
    client's remaining lifetime.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    tools_seen: list[bool] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        multi = len(body.get("tools") or []) > 1
        tools_seen.append(multi)
        if multi and len(tools_seen) == 1:
            # The first turn returns text instead of a tool call.
            return httpx.Response(200, json=_ok_response('{"action": "DELETE"}'))
        return httpx.Response(200, json=_tool_call_response({"action": "DELETE"}))

    respx.post(URL).mock(side_effect=_handler)
    request = _structured_request().model_copy(update={"tools": [
        {"name": "classify_intent", "parameters": {"type": "object"}},
        {"name": "other_intent", "parameters": {"type": "object"}},
    ]})
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(request)
        assert client._multitool_supported is not False, (
            "miss on auto must not be cached as unsupported capability"
        )
        tools_seen.clear()
        await client.generate_structured(request)
    finally:
        await client.aclose()
    assert tools_seen[0] is True, "next call must try again with multi-tool"


def test_sanitize_enums_keeps_type_and_stringifies_values():
    """Stringify Google enum literals while preserving their declared scalar type and the model's
    output contract.
    """
    from llm_mesh.openai.client import _stringify_numeric_enums

    src = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3]},
            "flag": {"type": "boolean", "enum": [True, False]},
            "note": {"type": ["string", "null"]},
        },
    }
    out = _stringify_numeric_enums(src)
    assert out["properties"]["score"] == {"type": "integer", "enum": ["1", "2", "3"]}
    assert out["properties"]["flag"]["enum"] == ["true", "false"]
    assert out["properties"]["note"]["type"] == "string"  # Collapse the nullable type union.
    assert src["properties"]["score"]["enum"] == [1, 2, 3]  # Do not mutate the original schema.


@respx.mock
@pytest.mark.asyncio
async def test_sanitize_enums_default_off(monkeypatch):
    """Without enum sanitization enabled, send the schema unchanged."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_tool_call_response({"action": "DELETE"}))

    respx.post(URL).mock(side_effect=_handler)
    request = _structured_request().model_copy(update={
        "schema_": {"type": "object",
                    "properties": {"score": {"type": "integer", "enum": [1, 2]}}},
    })
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(request)
    finally:
        await client.aclose()
    params = seen[0]["tools"][0]["function"]["parameters"]
    assert params["properties"]["score"]["enum"] == [1, 2]


@respx.mock
@pytest.mark.asyncio
async def test_disable_thinking_for_tools_only_touches_fc(monkeypatch):
    """Disable thinking only for incompatible function-calling payloads; preserve it for text
    generation.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_THINKING_FOR_TOOLS", "true")
    monkeypatch.setenv("LLM_REASONING_OFF", '{"thinking": {"type": "disabled"}}')
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if body.get("tools"):
            return httpx.Response(200, json=_tool_call_response({"action": "DELETE"}))
        return httpx.Response(200, json=_ok_response())

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(_structured_request())
        await client.generate_text(_request())
    finally:
        await client.aclose()
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in seen[0]
    assert "thinking" not in seen[1]  # The text path is unchanged.


@respx.mock
@pytest.mark.asyncio
async def test_text_tier_embeds_schema_in_prompt(monkeypatch):
    """Embed the schema in text emulation so the model knows the required field names without a
    function declaration.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_ok_response('{"action": "DELETE"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    system = seen[0]["messages"][0]["content"]
    assert "SCHEMA" in system
    assert '"action"' in system


@respx.mock
@pytest.mark.asyncio
async def test_empty_at_length_is_not_retried_at_same_budget(monkeypatch):
    """Do not repeat empty length-truncated content at the same budget. Only the outer length retry
    may increase the budget, preventing multiplicative retry cycles.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("LLM_LENGTH_RETRIES", "0")  # Isolate the inner retry layer.
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": ""},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 500, "total_tokens": 505},
    }))
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 1  # Exactly one POST, not four.
    assert resp.finish_reason == "length"


@respx.mock
@pytest.mark.asyncio
async def test_peg_500_degrades_without_burning_retries(monkeypatch):
    """Bypass transient retries for deterministic PEG parser failures so the fallback detector runs
    before repeated expensive generations.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    calls = {"fc": 0, "text": 0}

    def _handler(req: httpx.Request) -> httpx.Response:
        if json.loads(req.content).get("tools"):
            calls["fc"] += 1
            return httpx.Response(500, json={"error": {"message":
                "The model produced output that does not match the expected "
                "peg-gemma4 format"}})
        calls["text"] += 1
        return httpx.Response(200, json=_ok_response('{"action": "DELETE"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_structured_request())
    finally:
        await client.aclose()
    assert calls["fc"] == 1, "PEG-500 must not be retried as transient"
    assert calls["text"] == 1, "must fall back to text mode"
    assert resp.arguments == {"action": "DELETE"}
    assert client._tool_choice_pref == "text"


@respx.mock
@pytest.mark.asyncio
async def test_ordinary_500_still_retries(monkeypatch):
    """Keep transport retries for ordinary transient 500 responses."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    calls = {"n": 0}

    def _handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": {"message": "internal error"}})
        return httpx.Response(200, json=_ok_response())

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert calls["n"] == 2
    assert resp.text == "response"


@respx.mock
@pytest.mark.asyncio
async def test_multitool_recovers_function_name_from_content_envelope(monkeypatch):
    """Recover the chosen tool from a content envelope instead of forcing the fallback function and
    concealing a routing mistake.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    envelope = json.dumps(
        {"name": "other_intent", "arguments": {"action": "ARCHIVE"}},
        ensure_ascii=False,
    )
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok_response(envelope)))
    request = _structured_request().model_copy(update={"tools": [
        {"name": "classify_intent", "parameters": {"type": "object"}},
        {"name": "other_intent", "parameters": {"type": "object"}},
    ]})
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(request)
    finally:
        await client.aclose()
    # Return the model's selected name, not request.function_name.
    assert resp.function_name == "other_intent"
    assert resp.arguments == {"action": "ARCHIVE"}
    assert resp.tool_calls[0]["name"] == "other_intent"


@respx.mock
@pytest.mark.asyncio
async def test_multitool_without_envelope_still_degrades(monkeypatch):
    """Without a content envelope, retain the existing single-function fallback behavior."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if len(body.get("tools") or []) > 1:
            return httpx.Response(200, json=_ok_response('{"action": "DELETE"}'))
        return httpx.Response(200, json=_tool_call_response({"action": "DELETE"}))

    respx.post(URL).mock(side_effect=_handler)
    request = _structured_request().model_copy(update={"tools": [
        {"name": "classify_intent", "parameters": {"type": "object"}},
        {"name": "other_intent", "parameters": {"type": "object"}},
    ]})
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(request)
    finally:
        await client.aclose()
    assert len(seen) == 2, "must fall back to a single forced function"
    assert len(seen[1]["tools"]) == 1
    assert resp.arguments == {"action": "DELETE"}


def test_envelope_function_name_trigger_is_narrow():
    """Keep envelope detection narrow enough to preserve ordinary schema objects."""
    from llm_mesh.openai.client import OpenAIClient as C

    assert C._envelope_function_name(
        {"name": "emit_x", "arguments": {"a": 1}}
    ) == "emit_x"
    # Reject envelopes with extra keys, non-object arguments, or an empty name.
    assert C._envelope_function_name(
        {"name": "emit_x", "arguments": {"a": 1}, "extra": 1}
    ) is None
    assert C._envelope_function_name({"name": "emit_x", "arguments": "[]"}) is None
    assert C._envelope_function_name({"name": "", "arguments": {}}) is None
    assert C._envelope_function_name({"action": "DELETE"}) is None


def test_gigachat_ultra_gets_flagship_output_cap():
    """Give GigaChat Ultra the flagship output ceiling instead of the smaller generic default."""
    from llm_mesh.gigachat.client import _model_max_tokens

    assert _model_max_tokens("GigaChat-3-Ultra") == 32768
    # Cover future model generations through the tier heuristic, not only exact names.
    assert _model_max_tokens("GigaChat-4-Ultra") == 32768
    assert _model_max_tokens("GigaChat-Ultra-preview") == 32768
    # Preserve tier precedence.
    assert _model_max_tokens("GigaChat-3-Max") == 16384
    assert _model_max_tokens("GigaChat-3-Pro") == 8192
    assert _model_max_tokens("GigaChat-2") == 4096
    assert _model_max_tokens("unknown-model") == 4096


@respx.mock
@pytest.mark.asyncio
async def test_multitool_recovers_name_when_tool_call_name_empty(monkeypatch):
    """Recover a missing tool-call name from content when an incomplete server parser returns an
    unnamed call.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    payload = {
        "model": "m",
        "choices": [{"message": {
            "role": "assistant",
            "content": json.dumps(
                {"name": "other_intent", "arguments": {"action": "ARCHIVE"}},
                ensure_ascii=False,
            ),
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
    request = _structured_request().model_copy(update={"tools": [
        {"name": "classify_intent", "parameters": {"type": "object"}},
        {"name": "other_intent", "parameters": {"type": "object"}},
    ]})
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(request)
    finally:
        await client.aclose()
    assert resp.function_name == "other_intent"


@respx.mock
@pytest.mark.asyncio
async def test_multitool_normal_name_is_not_overridden(monkeypatch):
    """Do not overwrite an existing tool-call name with a content envelope."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    payload = {
        "model": "m",
        "choices": [{"message": {
            "role": "assistant",
            # Content deliberately names a different function; tool_call takes precedence.
            "content": json.dumps({"name": "other_intent", "arguments": {}}),
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "classify_intent",
                                         "arguments": '{"action": "DELETE"}'}}],
        }}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    respx.post(URL).mock(return_value=httpx.Response(200, json=payload))
    request = _structured_request().model_copy(update={"tools": [
        {"name": "classify_intent", "parameters": {"type": "object"}},
        {"name": "other_intent", "parameters": {"type": "object"}},
    ]})
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(request)
    finally:
        await client.aclose()
    assert resp.function_name == "classify_intent"


def test_floor_above_default_cap_raises_cap(monkeypatch):
    """When the floor exceeds the default retry ceiling, raise that ceiling to twice the floor so
    length retries remain possible.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "32768")
    monkeypatch.delenv("LLM_LENGTH_RETRY_CAP", raising=False)
    client = OpenAIClient(model="m")
    assert client._length_retry_cap == 65536
    start = client._clip_max_tokens(500)
    assert min(start * 2, client._length_retry_cap) > start, "retry must remain possible"


def test_floor_above_explicit_cap_is_respected_but_warned(monkeypatch, caplog):
    """Respect an explicitly configured ceiling but warn when it prevents budget growth."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "32768")
    monkeypatch.setenv("LLM_LENGTH_RETRY_CAP", "32768")
    with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
        client = OpenAIClient(model="m")
    assert client._length_retry_cap == 32768  # Respect the operator's explicit limit.
    assert any("length-retry disabled" in r.message for r in caplog.records)


def test_floor_below_cap_changes_nothing(monkeypatch):
    """Leave a consistent floor/ceiling combination unchanged."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MIN_OUTPUT_TOKENS", "16384")
    monkeypatch.delenv("LLM_LENGTH_RETRY_CAP", raising=False)
    assert OpenAIClient(model="m")._length_retry_cap == 32768
    monkeypatch.delenv("LLM_MIN_OUTPUT_TOKENS", raising=False)
    assert OpenAIClient(model="m")._length_retry_cap == 32768

def test_parallel_tool_calls_of_one_function_merge_into_one_array(monkeypatch):
    """Merge parallel calls containing individual array items instead of silently discarding all
    but the first.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}
    msg = {"tool_calls": [
        {"function": {"name": "f", "arguments": '{"items": [{"n": 1}]}'}},
        {"function": {"name": "f", "arguments": '{"items": [{"n": 2}]}'}},
    ]}
    assert c._parse_tool_arguments(msg, schema) == {"items": [{"n": 1}, {"n": 2}]}


def test_merge_is_skipped_when_schema_has_no_single_array(monkeypatch):
    """Do not merge without exactly one schema array destination; ambiguous merging can corrupt
    valid output.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    schema = {"type": "object", "properties": {"a": {"type": "array"}, "b": {"type": "array"}}}
    msg = {"tool_calls": [
        {"function": {"name": "f", "arguments": '{"a": [1]}'}},
        {"function": {"name": "f", "arguments": '{"a": [2]}'}},
    ]}
    assert c._parse_tool_arguments(msg, schema) == {"a": [1]}


def test_full_array_in_content_recovers_a_truncated_tool_call(monkeypatch):
    """Recover the full array from content when tool arguments contain only its first item."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}
    msg = {"content": '[{"n": 1}, {"n": 2}]',
           "tool_calls": [{"function": {"name": "f", "arguments": '{"n": 1}'}}]}
    assert c._parse_tool_arguments(msg, schema) == {"items": [{"n": 1}, {"n": 2}]}


def test_recovery_does_not_fire_without_schema(monkeypatch):
    """Without a schema, leave recovery behavior unchanged."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    c = OpenAIClient(model="m")
    msg = {"content": '[{"n": 1}, {"n": 2}]',
           "tool_calls": [{"function": {"name": "f", "arguments": '{"n": 1}'}}]}
    assert c._parse_tool_arguments(msg) == {"n": 1}


# --- Retry unknown vendor body keys rejected by strict gateways -------------
_UNKNOWN_KEY_400 = {
    "message": "feature 'extra arguments: "
               '{"chat_template_kwargs":{"enable_thinking":false}}\' is not available'
}


def test_unknown_body_field_error_is_recognized():
    assert _is_unknown_body_field_error(json.dumps(_UNKNOWN_KEY_400))
    assert _is_unknown_body_field_error("Unrecognized request argument supplied: thinking")
    assert _is_unknown_body_field_error("Unknown parameter: 'reasoning'")
    assert _is_unknown_body_field_error("This model does not support the thinking parameter")
    # A rejected parameter value is a different failure and must not remove vendor keys.
    assert not _is_unknown_body_field_error(
        "Unsupported value: 'temperature' does not support 0 with this model"
    )
    assert not _is_unknown_body_field_error("Reasoning is mandatory and cannot be disabled")
    assert not _is_unknown_body_field_error("")


@respx.mock
@pytest.mark.asyncio
async def test_unknown_body_field_retries_without_vendor_keys(monkeypatch):
    """Retry unknown-field errors without vendor body keys."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if any(k in body for k in VENDOR_BODY_KEYS):
            return httpx.Response(400, json=_UNKNOWN_KEY_400)
        return httpx.Response(200, json=_ok_response("response"))

    respx.post(URL).mock(side_effect=_handler)
    monkeypatch.setenv("LLM_REASONING_OFF", '{"chat_template_kwargs": {"enable_thinking": false}, "reasoning": {"enabled": false}}')
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_text(_request())
    finally:
        await client.aclose()

    assert resp.text == "response"
    assert len(seen) == 2
    assert any(k in seen[0] for k in VENDOR_BODY_KEYS), "first request used vendor keys"
    assert not any(k in seen[1] for k in VENDOR_BODY_KEYS), "repetition — without them"


@respx.mock
@pytest.mark.asyncio
async def test_unknown_body_field_is_learned_for_next_calls(monkeypatch):
    """Cache rejected vendor fields per instance to avoid repeated failing requests."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if any(k in body for k in VENDOR_BODY_KEYS):
            return httpx.Response(400, json=_UNKNOWN_KEY_400)
        return httpx.Response(200, json=_ok_response("response"))

    respx.post(URL).mock(side_effect=_handler)
    monkeypatch.setenv("LLM_REASONING_OFF", '{"chat_template_kwargs": {"enable_thinking": false}, "reasoning": {"enabled": false}}')
    client = OpenAIClient(model="m")
    try:
        await client.generate_text(_request())
        await client.generate_text(_request())
    finally:
        await client.aclose()

    assert len(seen) == 3, "second call must not incur another 400"
    assert not any(k in seen[2] for k in VENDOR_BODY_KEYS)


@respx.mock
@pytest.mark.asyncio
async def test_unknown_body_field_does_not_fire_on_value_rejection(monkeypatch):
    """Do not remove supported vendor fields merely because their value is rejected."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    respx.post(URL).mock(return_value=httpx.Response(
        400, json={"error": {"message": "Invalid value for 'thinking.type'"}}))
    client = OpenAIClient(model="m")
    try:
        with pytest.raises(OpenAIError):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert client._no_vendor_keys is False


# --- Explicitly declared native response_format tier -------------------------


def _rf_response(payload: str) -> dict:
    return {"model": "m",
            "choices": [{"message": {"role": "assistant", "content": payload},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


_RF_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}},
              "required": ["city"]}


def _rf_request() -> LLMRequest:
    return LLMRequest(system="s", user="u", schema=_RF_SCHEMA, function_name="fn")


@respx.mock
@pytest.mark.asyncio
async def test_response_format_tier_absent_without_declaration(monkeypatch):
    """Without a declared dialect, preserve the existing ladder instead of speculatively sending
    response_format.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_RESPONSE_FORMAT", raising=False)
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if "tools" in body:  # Reject both tool tiers as unsupported.
            return httpx.Response(404, json={"error": {"message":
                "No endpoints found that support the provided 'tool_choice'"}})
        return httpx.Response(200, json=_rf_response('{"city": "Paris"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_rf_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"city": "Paris"}
    assert client._last_served_tier == "text"
    assert not any("response_format" in b for b in seen)


@respx.mock
@pytest.mark.asyncio
async def test_response_format_json_schema_sends_schema_not_prompt(monkeypatch):
    """Send json_schema in the API body without duplicating it in the prompt; retain the JSON
    instruction required by some gateways.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_schema")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if "tools" in body:
            return httpx.Response(404, json={"error": {"message":
                "No endpoints found that support the provided 'tool_choice'"}})
        return httpx.Response(200, json=_rf_response('{"city": "Paris"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_rf_request())
    finally:
        await client.aclose()

    assert resp.arguments == {"city": "Paris"}
    assert client._last_served_tier == "response_format"
    rf_body = next(b for b in seen if "response_format" in b)
    assert rf_body["response_format"]["type"] == "json_schema"
    assert rf_body["response_format"]["json_schema"]["schema"] == _RF_SCHEMA
    system = rf_body["messages"][0]["content"]
    assert "JSON" in system.upper(), "the word json must appear in the messages"
    assert "properties" not in system, "do not duplicate the schema in the prompt"


@respx.mock
@pytest.mark.asyncio
async def test_response_format_json_object_dictates_schema_in_prompt(monkeypatch):
    """For json_object, describe the schema in the prompt because the gateway guarantees JSON
    syntax only.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")
    seen: list[dict] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        if "tools" in body:
            return httpx.Response(404, json={"error": {"message":
                "No endpoints found that support the provided 'tool_choice'"}})
        return httpx.Response(200, json=_rf_response('{"city": "Paris"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        await client.generate_structured(_rf_request())
    finally:
        await client.aclose()

    rf_body = next(b for b in seen if "response_format" in b)
    assert rf_body["response_format"] == {"type": "json_object"}
    assert "properties" in rf_body["messages"][0]["content"], "schema — in the prompt"


@respx.mock
@pytest.mark.asyncio
async def test_response_format_degrades_to_text_on_unparsable(monkeypatch):
    """Fall back to text when response_format output cannot be parsed or validated against the
    requested object.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if "tools" in body:
            return httpx.Response(404, json={"error": {"message":
                "No endpoints found that support the provided 'tool_choice'"}})
        if "response_format" in body:
            return httpx.Response(200, json=_rf_response('{"wrong": 1}'))
        return httpx.Response(200, json=_rf_response('{"city": "Paris"}'))

    respx.post(URL).mock(side_effect=_handler)
    client = OpenAIClient(model="m")
    try:
        resp = await client.generate_structured(_rf_request())
    finally:
        await client.aclose()
    assert resp.arguments == {"city": "Paris"}
    assert client._last_served_tier == "text"
