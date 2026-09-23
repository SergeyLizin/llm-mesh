"""Gemini generateContent client: wire format, tools, schema, and streaming."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest
import respx

from llm_mesh import GeminiClient, GeminiError, LLMRequest, LLMValidationError
from llm_mesh.gemini.client import generate_content_url
from llm_mesh.models_catalog import make_client, missing_credentials
from llm_mesh.probe import check_client
from llm_mesh.stream_events import StreamEventType

BASE = "https://gemini.test/v1beta"
MODEL = "gemini-2.5-flash"
URL = f"{BASE}/models/{MODEL}:generateContent"
STREAM_URL = URL.replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
COUNT_URL = f"{BASE}/models/{MODEL}:countTokens"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _client(**kwargs):
    values = {"model": MODEL, "base_url": BASE, "api_key": "test-key"}
    values.update(kwargs)
    return GeminiClient(**values)


def _request(**updates):
    values = {"system": "SYS", "user": "NOW", "max_tokens": 128}
    values.update(updates)
    return LLMRequest(**values)


def _answer(text: str = "ok", *, finish: str = "STOP", thoughts: str | None = None,
            function: dict | None = None, usage: dict | None = None) -> dict:
    parts: list[dict] = []
    if thoughts:
        parts.append({"text": thoughts, "thought": True})
    if text:
        parts.append({"text": text})
    if function is not None:
        parts.append({"functionCall": function})
    return {
        "responseId": "resp-1",
        "modelVersion": MODEL,
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": finish,
        }],
        "usageMetadata": usage or {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 3,
            "totalTokenCount": 16,
        },
    }


def _body(request) -> dict:
    return json.loads(request.content)


def test_url_keeps_a_versioned_base_and_adds_v1beta():
    assert generate_content_url(
        "https://generativelanguage.googleapis.com", "gemini-2.5-flash",
    ) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    assert generate_content_url(BASE + "/", MODEL, stream=True).endswith(
        f"/models/{MODEL}:streamGenerateContent?alt=sse"
    )
    assert generate_content_url(BASE, MODEL, count=True).endswith(":countTokens")


def test_missing_key_names_the_variables(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(model=MODEL, base_url=BASE)


@pytest.mark.asyncio
@respx.mock
async def test_text_request_uses_system_instruction_and_contents():
    route = respx.post(URL).respond(200, json=_answer("Hi"))
    async with aclosing(_client()) as client:
        response = await client.generate_text(_request())
    sent = _body(route.calls.last.request)
    assert sent["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "NOW"}]}]
    assert sent["generationConfig"]["maxOutputTokens"] == 128
    assert sent["generationConfig"]["temperature"] == 0
    assert "tools" not in sent
    assert route.calls.last.request.headers["x-goog-api-key"] == "test-key"
    assert response.text == "Hi"
    assert response.reasoning_content is None
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 6
    assert response.usage.reasoning_tokens == 2
    assert response.usage.cache_hit_tokens == 3
    assert response.usage.cache_miss_tokens == 7
    assert response.request_id == "resp-1"


@pytest.mark.asyncio
@respx.mock
async def test_thoughts_are_not_visible_text():
    respx.post(URL).respond(200, json=_answer("Hi", thoughts="hmm"))
    async with aclosing(_client()) as client:
        response = await client.generate_text(_request(reasoning_effort="low"))
    assert response.text == "Hi"
    assert response.reasoning_content == "hmm"


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_effort_sets_thinking_level():
    route = respx.post(URL).respond(200, json=_answer())
    async with aclosing(_client()) as client:
        await client.generate_text(_request(reasoning_effort="high"))
    config = _body(route.calls.last.request)["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "high"}


@pytest.mark.asyncio
@respx.mock
async def test_disable_reasoning_sends_thinking_budget_zero(monkeypatch):
    monkeypatch.setenv("LLM_DISABLE_REASONING", "1")
    route = respx.post(URL).respond(200, json=_answer())
    async with aclosing(_client()) as client:
        await client.generate_text(_request(reasoning_effort="high"))
    config = _body(route.calls.last.request)["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.asyncio
@respx.mock
async def test_forced_function_uses_any_and_allowed_names():
    route = respx.post(URL).respond(200, json=_answer(
        "", function={"name": "build_artifact", "args": {"answer": "x"}},
    ))
    async with aclosing(_client()) as client:
        response = await client.generate_structured(
            _request(schema=SCHEMA, function_name="build_artifact"),
        )
    sent = _body(route.calls.last.request)
    assert sent["tools"][0]["functionDeclarations"][0]["name"] == "build_artifact"
    assert sent["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["build_artifact"],
    }
    assert "responseSchema" not in sent["generationConfig"]
    assert response.function_name == "build_artifact"
    assert response.arguments == {"answer": "x"}
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["id"] == "call_0"


@pytest.mark.asyncio
@respx.mock
async def test_tools_auto_and_required():
    route = respx.post(URL).respond(200, json=_answer("plain"))
    tools = [{"name": "lookup", "description": "find", "parameters": SCHEMA}]
    async with aclosing(_client()) as client:
        chosen = await client.generate_structured(_request(tools=tools))
        assert chosen.function_name is None
        assert chosen.text == "plain"
        assert _body(route.calls.last.request)["toolConfig"]["functionCallingConfig"] == {
            "mode": "AUTO",
        }
        with pytest.raises(LLMValidationError, match="missing functionCall"):
            await client.generate_structured(_request(tools=tools, tools_required=True))
    assert _body(route.calls.last.request)["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


@pytest.mark.asyncio
@respx.mock
async def test_json_schema_and_fenced_salvage():
    route = respx.post(URL).respond(
        200, json=_answer('```json\n{"answer": "yes"}\n```'),
    )
    async with aclosing(_client()) as client:
        response = await client.generate_structured(
            _request(mode="json_schema", schema=SCHEMA),
        )
    config = _body(route.calls.last.request)["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == SCHEMA
    assert "tools" not in _body(route.calls.last.request)
    assert response.arguments == {"answer": "yes"}


@pytest.mark.asyncio
@respx.mock
async def test_no_degrade_rejects_fenced_json():
    respx.post(URL).respond(200, json=_answer('```json\n{"answer": "yes"}\n```'))
    async with aclosing(_client(no_degrade=True)) as client:
        with pytest.raises(LLMValidationError, match="forbids salvage"):
            await client.generate_structured(
                _request(mode="json_schema", schema=SCHEMA),
            )


@pytest.mark.asyncio
@respx.mock
async def test_schema_rejection_retries_as_text_once():
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(400, text='{"error":{"message":"Unknown name responseSchema"}}'),
        httpx.Response(200, json=_answer('{"answer": "ok"}')),
    ])
    async with aclosing(_client()) as client:
        response = await client.generate_structured(
            _request(mode="json_schema", schema=SCHEMA),
        )
    assert response.arguments == {"answer": "ok"}
    assert len(route.calls) == 2
    second = _body(route.calls[1].request)
    assert "responseSchema" not in second["generationConfig"]


@pytest.mark.asyncio
@respx.mock
async def test_preserve_does_not_retry_a_rejected_schema():
    route = respx.post(URL).respond(
        400, text='{"error":{"message":"Unknown name responseSchema"}}',
    )
    async with aclosing(_client(fallback_policy="preserve")) as client:
        with pytest.raises(GeminiError, match="400"):
            await client.generate_structured(
                _request(mode="json_schema", schema=SCHEMA),
            )
    assert len(route.calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_history_round_trips_function_calls():
    route = respx.post(URL).respond(200, json=_answer("done"))
    request = _request(history=[
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_0",
                "function": {"name": "lookup", "arguments": '{"q": "a"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "found"},
    ])
    async with aclosing(_client()) as client:
        await client.generate_text(request)
    contents = _body(route.calls.last.request)["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "lookup", "args": {"q": "a"}}}],
    }
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "lookup"
    assert contents[2]["parts"][0]["functionResponse"]["response"] == {"result": "found"}
    assert contents[2]["parts"][1] == {"text": "NOW"}


@pytest.mark.asyncio
@respx.mock
async def test_gemini_parts_keep_thought_signature():
    route = respx.post(URL).respond(200, json=_answer("next"))
    signature = "sig"
    request = _request(history=[{
        "role": "model",
        "content": [{
            "functionCall": {"name": "lookup", "args": {}},
            "thoughtSignature": signature,
        }],
    }])
    async with aclosing(_client()) as client:
        await client.generate_text(request)
    model_turn = _body(route.calls.last.request)["contents"][1]
    assert model_turn["parts"][0]["thoughtSignature"] == signature
    assert _body(route.calls.last.request)["contents"][0]["parts"][0]["text"] == "."


@pytest.mark.asyncio
@respx.mock
async def test_length_retry_doubles_max_output_tokens():
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(200, json=_answer("partial", finish="MAX_TOKENS")),
        httpx.Response(200, json=_answer("done")),
    ])
    async with aclosing(_client()) as client:
        response = await client.generate_text(_request(max_tokens=32))
    assert response.text == "done"
    assert _body(route.calls[0].request)["generationConfig"]["maxOutputTokens"] == 32
    assert _body(route.calls[1].request)["generationConfig"]["maxOutputTokens"] == 64


@pytest.mark.asyncio
@respx.mock
async def test_prompt_block_is_a_validation_error():
    respx.post(URL).respond(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    async with aclosing(_client()) as client:
        with pytest.raises(LLMValidationError, match="SAFETY"):
            await client.generate_text(_request())


@pytest.mark.asyncio
@respx.mock
async def test_extra_body_does_not_replace_contents(monkeypatch):
    monkeypatch.setenv(
        "LLM_EXTRA_BODY",
        '{"safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT"}], "contents": []}',
    )
    monkeypatch.setenv("LLM_EXTRA_HEADERS", '{"X-Project": "p", "x-goog-api-key": "stolen"}')
    route = respx.post(URL).respond(200, json=_answer())
    async with aclosing(_client()) as client:
        await client.generate_text(_request())
    sent = _body(route.calls.last.request)
    assert sent["contents"][0]["parts"][0]["text"] == "NOW"
    assert sent["safetySettings"][0]["category"] == "HARM_CATEGORY_HARASSMENT"
    assert route.calls.last.request.headers["x-goog-api-key"] == "test-key"
    assert route.calls.last.request.headers["X-Project"] == "p"


@pytest.mark.asyncio
@respx.mock
async def test_stream_separates_thoughts_and_reports_usage():
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
         "responseId": "resp-9"},
    ]
    payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, text=payload))
    async with aclosing(_client()) as client:
        streamed = [chunk async for chunk in client.generate_stream(_request())]
    assert streamed[0].delta_reasoning == "hmm"
    assert streamed[1].delta_text == "Hi"
    assert streamed[-1].finish_reason == "stop"
    assert streamed[-1].usage is not None
    assert streamed[-1].usage.prompt_tokens == 2
    assert streamed[-1].request_id == "resp-9"


@pytest.mark.asyncio
@respx.mock
async def test_stream_events_emit_one_complete_for_a_function_call():
    payload = "data: " + json.dumps({
        "candidates": [{
            "content": {"parts": [{
                "functionCall": {"name": "lookup", "args": {"q": "a"}},
            }]},
            "finishReason": "STOP",
        }],
        "responseId": "resp-2",
    }) + "\n\n"
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, text=payload))
    async with aclosing(_client()) as client:
        events = [event async for event in client.generate_stream_events(_request())]
    kinds = [event.type for event in events]
    assert kinds == [
        StreamEventType.TOOL_USE_START,
        StreamEventType.TOOL_USE_DELTA,
        StreamEventType.TOOL_USE_STOP,
        StreamEventType.COMPLETE,
    ]
    assert events[-1].finish_reason == "tool_calls"
    assert events[-1].content_blocks[0]["functionCall"]["name"] == "lookup"


@pytest.mark.asyncio
@respx.mock
async def test_count_tokens_posts_one_content_per_string():
    route = respx.post(url__regex=r":countTokens$").mock(side_effect=[
        httpx.Response(200, json={"totalTokens": 2}),
        httpx.Response(200, json={"totalTokens": 5}),
    ])
    async with aclosing(_client()) as client:
        assert await client.count_tokens([]) == []
        counts = await client.count_tokens(["aa", "bbbbb"], model="gemini-override")
    assert counts == [2, 5]
    assert [call.request.url.path for call in route.calls] == [
        "/v1beta/models/gemini-override:countTokens",
        "/v1beta/models/gemini-override:countTokens",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_probe_counts_tokens():
    route = respx.post(COUNT_URL).respond(200, json={"totalTokens": 3})
    async with aclosing(_client(label="gemini-probe")) as client:
        result = await check_client(client)
    assert result.ok is True
    assert result.kind == "gemini"
    assert _body(route.calls.last.request)["contents"] == [
        {"role": "user", "parts": [{"text": "connectivity probe"}]},
    ]


def test_catalog_route_defaults_the_endpoint(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    route = {
        "id": "gemini-25-flash",
        "kind": "gemini",
        "model": MODEL,
        "provider": "google",
        "api_key_env": "GEMINI_API_KEY",
    }
    assert missing_credentials(route) == ["GEMINI_API_KEY"]
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = make_client(route)
    assert isinstance(client, GeminiClient)
    assert client._model == MODEL
    assert client._base == "https://generativelanguage.googleapis.com/v1beta"
    assert client._key == "test-key"
    assert client.PROVIDER == "google"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate_stream", "generate_stream_events"])
async def test_parallel_streams_honor_the_concurrency_limit(monkeypatch, method):
    """Two streams at LLM_MAX_CONCURRENT=1 overlap in the caller and not in the HTTP call."""
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "1")
    inflight = 0
    peak = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_iter(_body, *, model):
        nonlocal inflight, peak
        del model
        inflight += 1
        peak = max(peak, inflight)
        entered.set()
        await release.wait()
        inflight -= 1
        yield {
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }],
        }

    client = _client()
    client._iter_payloads = fake_iter  # type: ignore[method-assign]

    async def consume():
        generator = getattr(client, method)(_request())
        return [item async for item in generator]

    first = asyncio.create_task(consume())
    await entered.wait()
    second = asyncio.create_task(consume())
    for _ in range(5):
        await asyncio.sleep(0)
    assert peak == 1
    assert inflight == 1
    release.set()
    done = await asyncio.gather(first, second)
    assert len(done) == 2
    assert all(done)
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_schema_violation_is_rejected_and_can_be_disabled():
    bad = _answer("", function={"name": "build_artifact", "args": {}})
    respx.post(URL).respond(200, json=bad)
    request = _request(schema=SCHEMA, function_name="build_artifact")
    async with aclosing(_client()) as client:
        with pytest.raises(LLMValidationError, match="do not satisfy"):
            await client.generate_structured(request)
    async with aclosing(_client(validate_schema=False)) as client:
        response = await client.generate_structured(request)
    assert response.arguments == {}
    respx.post(URL).respond(200, json=_answer('{"extra": 1}'))
    async with aclosing(_client()) as client:
        with pytest.raises(LLMValidationError, match="do not satisfy"):
            await client.generate_structured(_request(mode="json_schema", schema=SCHEMA))
