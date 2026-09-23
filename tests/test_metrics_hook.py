"""Process-wide call metrics. The default hook is a no-op."""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_mesh import LLMRequest, LLMRequestBlocked, OpenAIClient
from llm_mesh.hooks import configure_metrics_hook, configure_request_hook
from llm_mesh.openai.client import OpenAIError
from llm_mesh.types import LLMStreamChunk

URL = "https://host/v1/chat/completions"


def _request() -> LLMRequest:
    return LLMRequest(system="s", user="hello", mode="text")


def _ok() -> dict:
    return {
        "id": "c",
        "model": "m",
        "choices": [{
            "message": {"role": "assistant", "content": "response"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _client() -> OpenAIClient:
    return OpenAIClient(model="m", api_key="k", base_url="https://host/v1")


@respx.mock
@pytest.mark.asyncio
async def test_success_record_fields():
    seen = []
    configure_metrics_hook(on_call=seen.append)
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = _client()
    try:
        await client.generate_text(_request())
    finally:
        await client.aclose()
    assert len(seen) == 1
    record = seen[0]
    assert record.provider == client.PROVIDER
    assert record.model == "m"
    assert record.method == "generate_text"
    assert record.ok is True
    assert record.error_type is None
    assert record.latency_ms >= 0
    assert record.usage is not None
    assert record.usage.total_tokens == 7


@respx.mock
@pytest.mark.asyncio
async def test_provider_error_record():
    seen = []
    configure_metrics_hook(on_call=seen.append)
    respx.post(URL).mock(return_value=httpx.Response(500, text="down"))
    client = _client()
    try:
        with pytest.raises(OpenAIError):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].error_type == "OpenAIError"
    assert seen[0].method == "generate_text"


@respx.mock
@pytest.mark.asyncio
async def test_stream_interruption_record(monkeypatch):
    seen = []
    configure_metrics_hook(on_call=seen.append)

    async def cut(self, request):
        request = self._guarded_request(request)
        yield LLMStreamChunk(delta_text="partial")
        raise OpenAIError("stream cut")

    monkeypatch.setattr(OpenAIClient, "_generate_stream_impl", cut)
    client = _client()
    try:
        with pytest.raises(OpenAIError, match="stream cut"):
            async for _chunk in client.generate_stream(_request()):
                pass
    finally:
        await client.aclose()
    assert len(seen) == 1
    assert seen[0].method == "generate_stream"
    assert seen[0].ok is False
    assert seen[0].error_type == "OpenAIError"


@respx.mock
@pytest.mark.asyncio
async def test_hook_exception_is_swallowed(caplog):
    def explode(_record):
        raise RuntimeError("sink down")

    configure_metrics_hook(on_call=explode)
    respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = _client()
    caplog.set_level("WARNING")
    try:
        response = await client.generate_text(_request())
    finally:
        await client.aclose()
    assert response.text == "response"
    assert "METRICS_HOOK_FAILED" in caplog.text
    assert "hello" not in caplog.text


@respx.mock
@pytest.mark.asyncio
async def test_guard_refusal_emits_a_failed_record():
    seen = []
    configure_metrics_hook(on_call=seen.append)
    configure_request_hook(check_request=lambda _request: None)
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = _client()
    try:
        with pytest.raises(LLMRequestBlocked):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 0
    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].error_type == "LLMRequestBlocked"


@pytest.mark.asyncio
async def test_count_tokens_emits():
    seen = []
    configure_metrics_hook(on_call=seen.append)
    client = OpenAIClient(model="gpt-4", api_key="k", base_url="https://host/v1")
    try:
        counts = await client.count_tokens(["tokenizer"])
    finally:
        await client.aclose()
    assert counts == [1]
    assert len(seen) == 1
    assert seen[0].method == "count_tokens"
    assert seen[0].ok is True
    assert seen[0].model == "gpt-4"
    assert seen[0].usage is None
