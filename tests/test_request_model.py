"""Per-call model override. Mocked transports only."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from llm_mesh.gigachat.batch import GigaChatBatchClient
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL, GigaChatClient
from llm_mesh.openai.client import OpenAIClient, chat_completions_url
from llm_mesh.types import LLMRequest


OPENAI_URL = chat_completions_url("https://openai.test/v1")
GIGA_CHAT = f"{GIGACHAT_BASE_URL}/chat/completions"


def _text(**kwargs) -> LLMRequest:
    return LLMRequest(system="sys", user="user", mode="text", **kwargs)


def _openai() -> OpenAIClient:
    return OpenAIClient(
        "client-model",
        base_url="https://openai.test/v1",
        api_key="test-key",
        label="openai",
    )


def _chat(content: str = "ok", *, finish: str = "stop", model: str = "reported") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _tool(model: str = "reported") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "build_artifact",
                        "arguments": json.dumps({"x": "1"}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_empty_request_model_matches_an_unset_one():
    client = _openai()
    assert client._effective_model(_text(model="")) == "client-model"
    assert client._effective_model(_text()) == "client-model"


def _sse() -> str:
    chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_openai_request_model_overrides_text_and_structured(monkeypatch):
    monkeypatch.delenv("LLM_DISABLE_TOOLS", raising=False)
    monkeypatch.delenv("LLM_RESPONSE_FORMAT", raising=False)
    bodies: list[dict] = []

    def reply(request: httpx.Request) -> httpx.Response:
        body = _body(request)
        bodies.append(body)
        payload = _tool() if body.get("tools") else _chat()
        return httpx.Response(200, json=payload)

    respx.post(OPENAI_URL).mock(side_effect=reply)
    client = _openai()
    try:
        await client.generate_text(_text(model="override-model", length_retry=False))
        await client.generate_text(_text(length_retry=False))
        await client.generate_structured(LLMRequest(
            system="sys",
            user="user",
            schema={"type": "object", "properties": {"x": {"type": "string"}}},
            function_name="build_artifact",
            mode="function_call",
            model="structured-model",
            length_retry=False,
        ))
    finally:
        await client.aclose()
    assert [body["model"] for body in bodies] == [
        "override-model", "client-model", "structured-model",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_openai_stream_sends_the_override():
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200, text=_sse(), headers={"content-type": "text/event-stream"},
        )
    )
    client = _openai()
    try:
        chunks = []
        async for chunk in client.generate_stream(
            _text(model="stream-model", length_retry=False)
        ):
            chunks.append(chunk)
    finally:
        await client.aclose()
    assert chunks
    assert _body(route.calls.last.request)["model"] == "stream-model"


def test_openai_route_id_follows_the_request_without_replacing_the_client():
    client = _openai()
    request = _text(model="other-model")
    assert client._route_id == "openai/client-model"
    assert client._route_id_for(request) == "openai/other-model"
    assert client._route_id == "openai/client-model"


@pytest.mark.asyncio
@respx.mock
async def test_gigachat_override_changes_model_and_token_ceiling():
    route = respx.post(GIGA_CHAT).respond(200, json=_chat(model="GigaChat-2-Max"))
    client = GigaChatClient(token="t", model="GigaChat")
    try:
        await client.generate_text(
            _text(model="GigaChat-2-Max", max_tokens=20000, length_retry=False)
        )
        override = _body(route.calls.last.request)
        await client.generate_text(_text(max_tokens=20000, length_retry=False))
        instance = _body(route.calls.last.request)
    finally:
        await client.aclose()
    assert override["model"] == "GigaChat-2-Max"
    assert override["max_tokens"] == 16384
    assert instance["model"] == "GigaChat"
    assert instance["max_tokens"] == 4096


@pytest.mark.asyncio
@respx.mock
async def test_gigachat_length_retry_without_a_ceiling_does_not_raise(monkeypatch):
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("LLM_LENGTH_RETRIES", "1")
    seen: list[int] = []

    def reply(request: httpx.Request) -> httpx.Response:
        body = _body(request)
        seen.append(body["max_tokens"])
        finish = "length" if len(seen) == 1 else "stop"
        return httpx.Response(200, json=_chat(content="partial answer", finish=finish))

    respx.post(GIGA_CHAT).mock(side_effect=reply)
    client = GigaChatClient(
        token="t", model="GigaChat", use_model_token_limits=False,
    )
    try:
        result = await client.generate_text(_text(max_tokens=100))
    finally:
        await client.aclose()
    assert result.text == "partial answer"
    assert seen == [100, 200]


def test_batch_lines_keep_each_request_model():
    captured: dict[str, str] = {}

    def create(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"id": "b1", "status": "created"})

    results = "\n".join([
        json.dumps({
            "id": "1",
            "result": {
                "choices": [{
                    "message": {"content": "second"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        }),
        json.dumps({
            "id": "0",
            "result": {
                "choices": [{
                    "message": {"content": "first"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        }),
    ])

    async def run():
        with respx.mock:
            respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(side_effect=create)
            respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
                return_value=httpx.Response(
                    200,
                    json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}],
                )
            )
            respx.get(f"{GIGACHAT_BASE_URL}/files/f1/content").mock(
                return_value=httpx.Response(200, text=results)
            )
            client = GigaChatBatchClient(token="tok", poll_interval_s=0.0)
            return await client.run_chat_batch(
                [
                    _text(model="GigaChat-2"),
                    _text(),
                ],
                model="GigaChat-Max",
            )

    out = asyncio.run(run())
    lines = [json.loads(line) for line in captured["body"].splitlines()]
    assert lines[0]["request"]["model"] == "GigaChat-2"
    assert lines[1]["request"]["model"] == "GigaChat-Max"
    assert out[0].text == "first"
    assert out[1].text == "second"


def test_coalesce_copy_keeps_request_model():
    original = _text(model="kept-model")
    copied = original.model_copy(update={"mode": "function_call"})
    assert copied.model == "kept-model"
    assert copied.mode == "function_call"


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_request_model_overrides_the_body():
    anthropic = pytest.importorskip("llm_mesh.anthropic.client")
    url = "https://anthropic.test/v1/messages"
    route = respx.post(url).respond(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-reported",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    client = anthropic.AnthropicClient(
        model="claude-client",
        base_url="https://anthropic.test/v1",
        api_key="test-key",
    )
    try:
        await client.generate_text(_text(model="claude-override", length_retry=False))
    finally:
        await client.aclose()
    assert _body(route.calls.last.request)["model"] == "claude-override"
    assert "thinking" not in _body(route.calls.last.request)
