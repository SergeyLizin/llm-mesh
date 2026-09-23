"""Tests for GigaChat Batch API and request coalescing. Mock HTTP with respx against the batch
contract; verify concurrent-call aggregation using a stub batch backend.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from llm_mesh.gigachat.client import GIGACHAT_BASE_URL
from llm_mesh.gigachat.batch import (
    BatchingLLMClient,
    GigaChatBatchClient,
    GigaChatBatchError,
)
from llm_mesh.types import LLMError, LLMRequest, LLMResponse, LLMValidationError


def _req(user: str, mode: str = "function_call") -> LLMRequest:
    return LLMRequest(
        system="You are an assistant",
        user=user,
        schema={"type": "object", "properties": {"echo": {"type": "string"}}},
        function_name="build_artifact",
        mode=mode,
    )


def _structured_result(sub_id: str, args: dict) -> dict:
    return {
        "id": sub_id,
        "result": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "build_artifact",
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    },
                    "index": 0,
                    "finish_reason": "function_call",
                }
            ],
            "model": "GigaChat",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
                      "completion_tokens_details": {"reasoning_tokens": 2},
                      "prompt_tokens_details": {"cached_tokens": 4}},
        },
    }


def _text_result(sub_id: str, content: str) -> dict:
    return {
        "id": sub_id,
        "result": {
            "choices": [{"message": {"content": content}, "index": 0, "finish_reason": "stop"}],
            "model": "GigaChat",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    }


def _bc(**kwargs) -> GigaChatBatchClient:
    return GigaChatBatchClient(token="tok", poll_interval_s=0.0, **kwargs)


@pytest.mark.asyncio
async def test_owned_http_client_is_reused_and_closed():
    bc = _bc()
    first = bc._http()
    assert bc._http() is first
    await bc.aclose()
    assert bc._owned_client is None
    again = bc._http()
    assert again is not first
    await bc.aclose()


@pytest.mark.asyncio
async def test_external_http_client_is_not_closed():
    external = httpx.AsyncClient()
    bc = _bc(http_client=external)
    try:
        assert bc._http() is external
        await bc.aclose()
        assert external.is_closed is False
    finally:
        await external.aclose()


# ====================== GigaChatBatchClient: real protocol ==================


@respx.mock
def test_run_chat_batch_real_protocol_happy():
    respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "created"})
    )
    respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(
            200, json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}]
        )
    )
    jsonl = "\n".join(
        json.dumps(r, ensure_ascii=False)
        for r in [_structured_result("0", {"echo": "a"}), _text_result("1", "hi")]
    )
    respx.get(f"{GIGACHAT_BASE_URL}/files/f1/content").mock(
        return_value=httpx.Response(200, text=jsonl)
    )

    async def run():
        bc = _bc()
        out = await bc.run_chat_batch([_req("a", "function_call"), _req("b", "text")], model="GigaChat")
        return out

    out = asyncio.run(run())
    assert out[0].arguments == {"echo": "a"}
    assert out[1].text == "hi"
    assert out[0].usage.total_tokens == 8
    assert out[0].usage.reasoning_tokens == 2
    assert out[0].usage.cache_hit_tokens == 4
    assert out[0].usage.cache_miss_tokens == 1


@respx.mock
def test_run_chat_batch_sends_jsonl_octet_stream_with_method():
    captured = {}

    def create_handler(request):
        captured["content_type"] = request.headers.get("content-type")
        captured["method"] = request.url.params.get("method")
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"id": "b1", "status": "created"})

    respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(side_effect=create_handler)
    respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}])
    )
    respx.get(f"{GIGACHAT_BASE_URL}/files/f1/content").mock(
        return_value=httpx.Response(200, text=json.dumps(_structured_result("0", {"echo": "a"})))
    )

    async def run():
        bc = _bc()
        await bc.run_chat_batch([_req("a")], model="GigaChat-Max")

    asyncio.run(run())
    assert captured["content_type"] == "application/octet-stream"
    assert captured["method"] == "chat_completions"
    line = json.loads(captured["body"].splitlines()[0])
    assert line["id"] == "0"
    assert line["request"]["model"] == "GigaChat-Max"
    assert line["request"]["function_call"] == {"name": "build_artifact"}


@respx.mock
def test_run_chat_batch_error_isolation():
    respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "created"})
    )
    respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}])
    )
    jsonl = "\n".join(
        json.dumps(r)
        for r in [
            _structured_result("0", {"echo": "ok"}),
            {"id": "1", "error": {"status": 400, "message": "No such model"}},
        ]
    )
    respx.get(f"{GIGACHAT_BASE_URL}/files/f1/content").mock(return_value=httpx.Response(200, text=jsonl))

    async def run():
        bc = _bc()
        return await bc.run_chat_batch([_req("ok"), _req("bad")], model="GigaChat", return_exceptions=True)

    out = asyncio.run(run())
    assert out[0].arguments == {"echo": "ok"}
    assert isinstance(out[1], GigaChatBatchError)


@respx.mock
def test_wait_for_completion_polls_until_done():
    respx.get(f"{GIGACHAT_BASE_URL}/batches").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "b1", "status": "in_progress"}]),
            httpx.Response(200, json=[{"id": "b1", "status": "in_progress"}]),
            httpx.Response(200, json=[{"id": "b1", "status": "completed", "output_file_id": "f1"}]),
        ]
    )

    async def run():
        bc = _bc()
        return await bc.wait_for_completion("b1", poll_interval=0.0)

    final = asyncio.run(run())
    assert final["status"] == "completed"
    assert final["output_file_id"] == "f1"


@respx.mock
def test_create_batch_400_raises():
    respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(400, json={"status": 400, "message": "Bad request"})
    )

    async def run():
        bc = _bc()
        await bc.create_batch(b'{"id":"0"}', method="chat_completions")

    with pytest.raises(Exception):
        asyncio.run(run())


# ====================== BatchingLLMClient: coalescing ======================


class _StubBatch:
    """Stub batch backend recording submitted batch sizes."""

    def __init__(self, *, fail_ids: set[int] | None = None) -> None:
        self.batch_sizes: list[int] = []
        self._fail_ids = fail_ids or set()

    async def run_chat_batch(self, requests, *, model=None, return_exceptions=False, **kwargs):
        self.batch_sizes.append(len(requests))
        await asyncio.sleep(0)  # Yield to the event loop.
        out: list = []
        for req in requests:
            idx = int(req.user.removeprefix("u"))
            if idx in self._fail_ids:
                out.append(GigaChatBatchError(f"fail {idx}"))
            else:
                out.append(LLMResponse(arguments={"echo": req.user}, model=model or "m"))
        return out


def test_batching_coalesces_concurrent_calls():
    stub = _StubBatch()

    async def run():
        cli = BatchingLLMClient(stub, model="GigaChat", max_batch_size=16, max_delay_s=0.05)
        reqs = [_req(f"u{i}") for i in range(8)]
        return await asyncio.gather(*[cli.generate_structured(r) for r in reqs])

    outs = asyncio.run(run())
    assert [o.arguments["echo"] for o in outs] == [f"u{i}" for i in range(8)]
    # Eight concurrent calls form a batch rather than eight individual submissions.
    assert sum(stub.batch_sizes) == 8
    assert max(stub.batch_sizes) >= 2


def test_batching_respects_max_batch_size():
    stub = _StubBatch()

    async def run():
        cli = BatchingLLMClient(stub, model="GigaChat", max_batch_size=3, max_delay_s=0.05)
        reqs = [_req(f"u{i}") for i in range(7)]
        return await asyncio.gather(*[cli.generate_structured(r) for r in reqs])

    outs = asyncio.run(run())
    assert len(outs) == 7
    assert sum(stub.batch_sizes) == 7
    assert max(stub.batch_sizes) <= 3


def test_batching_propagates_per_item_errors():
    stub = _StubBatch(fail_ids={1})

    async def run():
        cli = BatchingLLMClient(stub, model="GigaChat", max_batch_size=8, max_delay_s=0.05)
        reqs = [_req(f"u{i}") for i in range(3)]
        return await asyncio.gather(
            *[cli.generate_structured(r) for r in reqs], return_exceptions=True
        )

    outs = asyncio.run(run())
    assert outs[0].arguments == {"echo": "u0"}
    assert isinstance(outs[1], GigaChatBatchError)
    assert outs[2].arguments == {"echo": "u2"}


def test_batching_generate_text_sets_text_mode():
    captured = {}

    class _ModeStub:
        async def run_chat_batch(self, requests, *, model=None, return_exceptions=False, **kwargs):
            captured["modes"] = [r.mode for r in requests]
            return [LLMResponse(text="ok", model="m") for _ in requests]

    async def run():
        cli = BatchingLLMClient(_ModeStub(), model="GigaChat", max_delay_s=0.02)
        return await cli.generate_text(_req("u0", mode="function_call"))

    resp = asyncio.run(run())
    assert resp.text == "ok"
    assert captured["modes"] == ["text"]  # generate_text forces text mode.


@respx.mock
def test_create_batch_refreshes_on_401_token_expired():
    """Refresh an expired token once on 401 and retry, as in the chat client."""

    class _Auth:
        def __init__(self):
            self.refreshes = 0

        async def _ensure_token(self):
            return "expired"

        async def _refresh_token(self):
            self.refreshes += 1
            return "fresh"

    route = respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        side_effect=[
            httpx.Response(401, json={"status": 401, "message": "Token has expired"}),
            httpx.Response(200, json={"id": "b1", "status": "created"}),
        ]
    )

    async def run():
        auth = _Auth()
        bc = GigaChatBatchClient(auth=auth, poll_interval_s=0.0)
        r = await bc.create_batch(b'{"id":"0"}', method="chat_completions")
        return r, auth

    r, auth = asyncio.run(run())
    assert r["id"] == "b1"
    assert auth.refreshes == 1
    assert route.call_count == 2
    assert route.calls[-1].request.headers["Authorization"] == "Bearer fresh"


@pytest.mark.asyncio
async def test_aclose_fails_a_pending_generate_text_instead_of_hanging():
    """A cancelled batch task used to leave the caller's future pending forever.

    CancelledError is a BaseException, so ``except Exception`` in ``_process``
    never resolved it. ``aclose`` has to fail that future itself.
    """
    started = asyncio.Event()

    class _Hang:
        async def run_chat_batch(self, requests, **kwargs):
            started.set()
            await asyncio.Event().wait()

    cli = BatchingLLMClient(_Hang(), model="m", max_delay_s=0.01)
    pending = asyncio.create_task(
        cli.generate_text(LLMRequest(system="s", user="u", mode="text"))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(cli.aclose(), timeout=1)
        with pytest.raises(LLMError, match="batch client closed"):
            await asyncio.wait_for(pending, timeout=1)
    finally:
        pending.cancel()
        await cli.aclose()


@pytest.mark.asyncio
async def test_tools_required_rejected_instead_of_forcing_a_function():
    """Reject unsupported native tool loops instead of silently forcing the fallback function.
    Otherwise a loop could accept an immediate finish call as the model's choice without
    executing any checks. The caller must explicitly choose text emulation.
    """
    client = BatchingLLMClient(batch_client=None, model="GigaChat")  # type: ignore[arg-type]
    req = _req("Analyze")
    object.__setattr__(req, "tools", [{"name": "finish", "parameters": {}}])
    object.__setattr__(req, "tools_required", True)
    object.__setattr__(req, "function_name", "finish")
    with pytest.raises(LLMValidationError) as caught:
        await client.generate_structured(req)
    assert "tools_required" in str(caught.value)
