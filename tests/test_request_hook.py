"""Request-guard hook: pass, replace, block, and batch isolation.

Transports are mocked. The default hook is the identity, so the rest of the
suite is the proof that an unconfigured client is unchanged.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from llm_mesh import (
    AnthropicClient,
    GeminiClient,
    LLMRequest,
    LLMRequestBlocked,
    LLMResponse,
    LLMValidationError,
)
from llm_mesh.gigachat.batch import BatchingLLMClient, GigaChatBatchClient
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL, GigaChatClient
from llm_mesh.hooks import configure_request_hook, identity_check_request
from llm_mesh.openai.batch import OpenAIBatchClient
from llm_mesh.openai.client import OpenAIClient

OPENAI_URL = "https://host/v1/chat/completions"
_METHODS = (
    "generate_text",
    "generate_stream",
    "generate_structured",
    "generate_stream_events",
)


def _request(user: str = "do-not-log-this-payload") -> LLMRequest:
    return LLMRequest(system="original system", user=user)


def _clients():
    return {
        "openai": OpenAIClient(model="m", api_key="k", base_url="https://host/v1"),
        "anthropic": AnthropicClient(
            model="claude", api_key="k", base_url="https://example.test",
        ),
        "gemini": GeminiClient(
            model="gemini-2.5-flash", api_key="k", base_url="https://gemini.test/v1beta",
        ),
        "gigachat": GigaChatClient(token="pre-baked", model="GigaChat"),
    }


def _ok_openai() -> dict:
    return {
        "id": "c",
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def _invoke(client, method: str, request: LLMRequest):
    call = getattr(client, method)
    if method in ("generate_stream", "generate_stream_events"):
        async for _item in call(request):
            pass
        return None
    return await call(request)


@respx.mock
@pytest.mark.asyncio
async def test_identity_hook_sends_the_same_body():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_ok_openai()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    request = _request("hello")
    try:
        await client.generate_text(request)
        configure_request_hook(check_request=identity_check_request)
        await client.generate_text(request)
    finally:
        await client.aclose()
    first = route.calls[0].request.content
    second = route.calls[1].request.content
    assert first == second
    body = json.loads(second)
    assert body["messages"][0]["content"] == "original system"


@respx.mock
@pytest.mark.asyncio
async def test_replace_hook_changes_the_wire_body():
    def redact(request: LLMRequest):
        return request.model_copy(update={"system": "redacted"})

    configure_request_hook(check_request=redact)
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_ok_openai()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        await client.generate_text(_request("hello"))
    finally:
        await client.aclose()
    body = json.loads(route.calls.last.request.content)
    assert body["messages"][0]["content"] == "redacted"
    assert "original system" not in route.calls.last.request.content.decode()


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini", "gigachat"])
@pytest.mark.parametrize("method", _METHODS)
@respx.mock
@pytest.mark.asyncio
async def test_block_raises_before_http(name, method, caplog):
    configure_request_hook(check_request=lambda _request: None)
    route = respx.route(url__regex=r"https://.*").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    client = _clients()[name]
    caplog.set_level("WARNING")
    try:
        with pytest.raises(LLMRequestBlocked) as caught:
            await _invoke(client, method, _request())
    finally:
        await client.aclose()
    assert caught.value.reason == "request hook returned None"
    assert not isinstance(caught.value, LLMValidationError)
    assert name in str(caught.value) or client.PROVIDER in str(caught.value)
    assert route.call_count == 0
    assert "REQUEST_BLOCKED" in caplog.text
    assert "do-not-log-this-payload" not in caplog.text


@respx.mock
@pytest.mark.asyncio
async def test_hook_exception_propagates_unwrapped():
    class AppError(RuntimeError):
        pass

    def explode(_request: LLMRequest):
        raise AppError("application refused")

    configure_request_hook(check_request=explode)
    route = respx.route(url__regex=r"https://.*").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        with pytest.raises(AppError, match="application refused"):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 0


class _StubBatch:
    def __init__(self) -> None:
        self.users: list[list[str]] = []

    async def run_chat_batch(self, requests, *, model=None, return_exceptions=False, **kwargs):
        self.users.append([req.user for req in requests])
        return [LLMResponse(text=req.user, model=model or "m") for req in requests]


@pytest.mark.asyncio
async def test_coalesced_block_fails_one_future():
    def hook(request: LLMRequest):
        if request.user == "bad":
            return None
        return request

    configure_request_hook(check_request=hook)
    stub = _StubBatch()
    client = BatchingLLMClient(stub, model="m", max_batch_size=8, max_delay_s=0.05)

    async def run():
        bad = asyncio.create_task(client.generate_text(_request("bad")))
        good = asyncio.create_task(client.generate_text(_request("good")))
        return await asyncio.gather(bad, good, return_exceptions=True)

    try:
        blocked, passed = await run()
    finally:
        await client.aclose()
    assert isinstance(blocked, LLMRequestBlocked)
    assert passed.text == "good"
    assert stub.users == [["good"]]


@respx.mock
@pytest.mark.asyncio
async def test_openai_batch_block_raises_before_upload():
    configure_request_hook(check_request=lambda _request: None)
    files = respx.post(url__regex=r".*/files$").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    openai = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    batch = OpenAIBatchClient(client=openai)
    try:
        with pytest.raises(LLMRequestBlocked) as caught:
            await batch.run_chat_batch([_request("a"), _request("b")])
    finally:
        await openai.aclose()
        await batch.aclose()
    assert caught.value.reason == "request hook returned None"
    assert files.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_openai_batch_block_sits_at_its_index():
    configure_request_hook(check_request=lambda _request: None)
    files = respx.post(url__regex=r".*/files$").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    openai = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    batch = OpenAIBatchClient(client=openai)
    try:
        out = await batch.run_chat_batch(
            [_request("a"), _request("b")], return_exceptions=True,
        )
    finally:
        await openai.aclose()
        await batch.aclose()
    assert [type(item) for item in out] == [LLMRequestBlocked, LLMRequestBlocked]
    assert out[1].reason == "request hook returned None"
    assert files.call_count == 0


def _gigachat_text_result(sub_id: str, content: str) -> dict:
    return {
        "id": sub_id,
        "result": {
            "choices": [{
                "message": {"content": content},
                "index": 0,
                "finish_reason": "stop",
            }],
            "model": "GigaChat",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    }


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_batch_block_raises_before_submit():
    configure_request_hook(check_request=lambda _request: None)
    posted = respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    batch = GigaChatBatchClient(token="tok", poll_interval_s=0.0)
    try:
        with pytest.raises(LLMRequestBlocked):
            await batch.run_chat_batch([_request("a")], model="GigaChat")
    finally:
        await batch.aclose()
    assert posted.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_batch_block_keeps_its_index():
    def hook(request: LLMRequest):
        if request.user == "bad":
            return None
        return request

    configure_request_hook(check_request=hook)
    captured: dict[str, str] = {}

    def create(request):
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"id": "b1", "status": "created"})

    respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(side_effect=create)
    respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(
            200, json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}],
        )
    )
    respx.get(f"{GIGACHAT_BASE_URL}/files/f1/content").mock(
        return_value=httpx.Response(
            200, text=json.dumps(_gigachat_text_result("0", "kept")),
        )
    )
    batch = GigaChatBatchClient(token="tok", poll_interval_s=0.0)
    try:
        out = await batch.run_chat_batch(
            [
                LLMRequest(system="original system", user="keep", mode="text"),
                LLMRequest(system="original system", user="bad", mode="text"),
            ],
            model="GigaChat",
            return_exceptions=True,
        )
    finally:
        await batch.aclose()
    assert out[0].text == "kept"
    assert isinstance(out[1], LLMRequestBlocked)
    lines = [json.loads(line) for line in captured["body"].splitlines() if line]
    assert [line["id"] for line in lines] == ["0"]
