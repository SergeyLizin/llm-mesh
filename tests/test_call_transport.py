"""Per-call timeout and retries override the constructor and the env default."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh import LLMRequest, LLMValidationError, OpenAIClient
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL, GigaChatClient

URL = "https://host/v1/chat/completions"


def _ok() -> dict:
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _request(**updates) -> LLMRequest:
    values = {"system": "s", "user": "hi", "mode": "text"}
    values.update(updates)
    return LLMRequest(**values)


@respx.mock
@pytest.mark.asyncio
async def test_per_call_timeout_reaches_httpx_and_beats_the_constructor():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = OpenAIClient(
        model="m", api_key="k", base_url="https://host/v1", http_timeout=1.0,
    )
    try:
        await client.generate_text(_request(timeout_s=12.5))
    finally:
        await client.aclose()
    timeout = route.calls[0].request.extensions["timeout"]
    assert timeout["read"] == 12.5
    assert timeout["connect"] == 12.5


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_per_call_timeout_keeps_connect_split():
    route = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok()),
    )
    client = GigaChatClient(token="dummy", model="GigaChat", timeout_s=1.0)
    try:
        await client.generate_text(_request(timeout_s=12.5))
    finally:
        await client.aclose()
    timeout = route.calls[0].request.extensions["timeout"]
    assert timeout["read"] == 12.5
    assert timeout["connect"] == 30


@respx.mock
@pytest.mark.asyncio
async def test_stream_timeout_reaches_httpx():
    payload = {
        "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    body = "data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n"
    route = respx.post(URL).mock(return_value=httpx.Response(200, text=body))
    client = OpenAIClient(
        model="m", api_key="k", base_url="https://host/v1", http_timeout=1.0,
    )
    try:
        async for _chunk in client.generate_stream(_request(timeout_s=8.0)):
            pass
    finally:
        await client.aclose()
    assert route.calls[0].request.extensions["timeout"]["read"] == 8.0


@respx.mock
@pytest.mark.asyncio
async def test_max_retries_zero_fails_on_one_attempt(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    route = respx.post(URL).mock(return_value=httpx.Response(503, text="down"))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        with pytest.raises(Exception):
            await client.generate_text(_request(max_retries=0))
    finally:
        await client.aclose()
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_negative_transport_knobs_raise_before_http():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        with pytest.raises(LLMValidationError, match="timeout_s"):
            await client.generate_text(_request(timeout_s=-1))
        with pytest.raises(LLMValidationError, match="max_retries"):
            await client.generate_text(_request(max_retries=-1))
    finally:
        await client.aclose()
    assert route.call_count == 0
