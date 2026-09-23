"""OpenAI Batch API. Mocked transports only — respx, no network."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
import respx

from llm_mesh.gigachat.batch import BatchingLLMClient
from llm_mesh.models_catalog import _MANAGED_ENV, make_client
from llm_mesh.openai.batch import (
    OpenAIBatchClient,
    OpenAIBatchError,
)
from llm_mesh.openai.client import OpenAIClient
from llm_mesh.types import LLMRequest, LLMTimeoutError

BASE = "https://api.openai.com/v1"


def _openai(**kwargs) -> OpenAIClient:
    return OpenAIClient(
        "client-model",
        base_url=BASE,
        api_key="test-key",
        label="openai",
        **kwargs,
    )


def _batch(client: OpenAIClient | None = None, **kwargs) -> OpenAIBatchClient:
    return OpenAIBatchClient(
        client=client or _openai(),
        poll_interval_s=kwargs.pop("poll_interval_s", 0.0),
        **kwargs,
    )


def _req(user: str, *, mode: str = "text", model: str | None = None) -> LLMRequest:
    return LLMRequest(
        system="sys",
        user=user,
        mode=mode,
        model=model,
        schema={"type": "object", "properties": {"echo": {"type": "string"}}},
        function_name="build_artifact",
    )


def _completion(text: str, *, model: str) -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _tool_completion(args: dict, *, model: str = "client-model") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "build_artifact",
                        "arguments": json.dumps(args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def _line(custom_id: str, *, body: dict | None = None, status_code: int = 200,
          error: dict | None = None) -> dict:
    if error is not None:
        return {"custom_id": custom_id, "error": error}
    return {
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": body or {}},
    }


def _jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(row) for row in rows)


def _install_happy(router, results: list[dict], *, file_responses=None):
    """files upload, batch create, one in_progress poll, then completed + download."""
    router.post(f"{BASE}/files").mock(side_effect=file_responses or [
        httpx.Response(200, json={"id": "file_in"}),
    ])
    router.post(f"{BASE}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    router.get(f"{BASE}/batches/batch_1").mock(side_effect=[
        httpx.Response(200, json={"id": "batch_1", "status": "in_progress"}),
        httpx.Response(200, json={
            "id": "batch_1", "status": "completed", "output_file_id": "file_out",
        }),
    ])
    router.get(f"{BASE}/files/file_out/content").mock(
        return_value=httpx.Response(200, text=_jsonl(results))
    )


def test_chat_lines_match_completion_bodies_and_per_line_model():
    client = _openai()
    bc = _batch(client)
    default = _req("a", mode="function_call")
    override = _req("b", mode="text", model="line-model")
    lines = bc.build_chat_lines([default, override], model="batch-model")
    assert [line["custom_id"] for line in lines] == ["0", "1"]
    assert all(line["method"] == "POST" and line["url"] == "/v1/chat/completions" for line in lines)
    stamped = default.model_copy(update={"model": "batch-model"})
    assert lines[0]["body"] == client.build_completion_body(stamped, structured=True)
    assert lines[0]["body"]["model"] == "batch-model"
    assert lines[0]["body"]["tool_choice"] == {
        "type": "function", "function": {"name": "build_artifact"},
    }
    assert "tools" not in lines[1]["body"]
    assert lines[1]["body"] == client.build_completion_body(override, structured=False)
    assert lines[1]["body"]["model"] == "line-model"


def test_response_from_payload_does_not_scan_the_canary(monkeypatch):
    client = _openai()

    def boom(*_args, **_kwargs):
        raise AssertionError("canary scanned")

    monkeypatch.setattr(client, "_check_response_canary", boom)
    text = client.response_from_payload(
        _completion("hello", model="m"), _req("a"), False,
    )
    structured = client.response_from_payload(
        _tool_completion({"echo": "z"}), _req("a", mode="function_call"), True,
    )
    assert text.text == "hello"
    assert structured.arguments == {"echo": "z"}


@respx.mock
def test_happy_path_maps_results_in_input_order():
    captured = {}

    def on_upload(request: httpx.Request):
        captured["upload"] = request.content
        captured["purpose"] = request.content
        return httpx.Response(200, json={"id": "file_in"})

    def on_create(request: httpx.Request):
        captured["create"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "batch_1", "status": "validating"})

    respx.post(f"{BASE}/files").mock(side_effect=on_upload)
    respx.post(f"{BASE}/batches").mock(side_effect=on_create)
    respx.get(f"{BASE}/batches/batch_1").mock(side_effect=[
        httpx.Response(200, json={"id": "batch_1", "status": "in_progress"}),
        httpx.Response(200, json={
            "id": "batch_1", "status": "completed", "output_file_id": "file_out",
        }),
    ])
    # File order is the reverse of input order. Mapping is by custom_id.
    respx.get(f"{BASE}/files/file_out/content").mock(return_value=httpx.Response(
        200, text=_jsonl([
            _line("1", body=_completion("beta", model="gpt-b")),
            _line("0", body=_completion("alpha", model="gpt-a")),
        ]),
    ))

    async def run():
        bc = _batch()
        try:
            return await bc.run_chat_batch([_req("first"), _req("second")], model="client-model")
        finally:
            await bc.aclose()

    out = asyncio.run(run())
    assert [item.text for item in out] == ["alpha", "beta"]
    assert out[0].model == "gpt-a"
    assert out[1].model == "gpt-b"
    assert out[0].usage.total_tokens == 8
    assert out[0].usage.prompt_tokens == 5
    uploaded = captured["upload"].decode()
    assert b"name=\"purpose\"" in captured["purpose"] or b"purpose" in captured["purpose"]
    assert "batch" in uploaded
    records = [json.loads(line) for line in uploaded.splitlines() if line.strip().startswith("{")]
    # Multipart may wrap the JSONL; fall back to scanning the body for custom_id.
    if len(records) < 2:
        assert '"custom_id": "0"' in uploaded and '"custom_id": "1"' in uploaded
    else:
        assert [row["custom_id"] for row in records] == ["0", "1"]
    assert captured["create"] == {
        "input_file_id": "file_in",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }


def test_mixed_item_errors_raise_first_or_return_in_place():
    rows = [
        _line("0", status_code=400, body={"error": {"message": "status-400-body"}}),
        _line("1", error={"message": "top-level-boom"}),
    ]

    async def run(return_exceptions: bool):
        with respx.mock:
            _install_happy(respx, rows)
            bc = _batch()
            try:
                return await bc.run_chat_batch(
                    [_req("a", mode="function_call"), _req("b", mode="function_call")],
                    return_exceptions=return_exceptions,
                )
            finally:
                await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="status-400-body"):
        asyncio.run(run(False))
    out = asyncio.run(run(True))
    assert isinstance(out[0], OpenAIBatchError)
    assert isinstance(out[1], OpenAIBatchError)
    assert "status-400-body" in str(out[0])
    assert "top-level-boom" in str(out[1])
    assert "status-400-body" not in str(out[1])


@respx.mock
def test_missing_custom_id_is_a_positional_error():
    rows = [
        _line("0", body=_completion("ok", model="m")),
        {"response": {"status_code": 200, "body": _completion("orphan", model="m")}},
    ]

    async def run():
        _install_happy(respx, rows)
        bc = _batch()
        try:
            return await bc.run_chat_batch(
                [_req("a"), _req("b")], return_exceptions=True,
            )
        finally:
            await bc.aclose()

    out = asyncio.run(run())
    assert out[0].text == "ok"
    assert isinstance(out[1], OpenAIBatchError)
    assert "missing custom_id" in str(out[1])


@respx.mock
def test_count_mismatch_raises_even_when_returning_exceptions():
    async def run():
        _install_happy(respx, [_line("0", body=_completion("only", model="m"))])
        bc = _batch()
        try:
            return await bc.run_chat_batch(
                [_req("a"), _req("b")], return_exceptions=True,
            )
        finally:
            await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="count mismatch"):
        asyncio.run(run())


@respx.mock
def test_failed_batch_attaches_errors():
    respx.post(f"{BASE}/files").mock(return_value=httpx.Response(200, json={"id": "file_in"}))
    respx.post(f"{BASE}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    respx.get(f"{BASE}/batches/batch_1").mock(return_value=httpx.Response(200, json={
        "id": "batch_1",
        "status": "failed",
        "errors": {"data": [{"message": "quota exceeded"}]},
    }))

    async def run():
        bc = _batch()
        try:
            await bc.run_chat_batch([_req("a")])
        finally:
            await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="quota exceeded") as caught:
        asyncio.run(run())
    assert "failed" in str(caught.value)


@respx.mock
def test_max_wait_raises_timeout():
    respx.post(f"{BASE}/files").mock(return_value=httpx.Response(200, json={"id": "file_in"}))
    respx.post(f"{BASE}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    respx.get(f"{BASE}/batches/batch_1").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})
    )

    async def run():
        bc = _batch(max_wait_s=0)
        try:
            await bc.run_chat_batch([_req("a")])
        finally:
            await bc.aclose()

    with pytest.raises(LLMTimeoutError, match="did not complete"):
        asyncio.run(run())


@respx.mock
def test_429_on_create_retries_with_backoff(monkeypatch):
    sleeps: list[float] = []

    async def record(delay, *_args, **_kwargs):
        sleeps.append(delay)

    monkeypatch.setattr("llm_mesh.openai.batch.asyncio.sleep", record)
    respx.post(f"{BASE}/files").mock(side_effect=[
        httpx.Response(429, text="slow down"),
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={"id": "file_in"}),
    ])
    respx.post(f"{BASE}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    respx.get(f"{BASE}/batches/batch_1").mock(return_value=httpx.Response(200, json={
        "id": "batch_1", "status": "completed", "output_file_id": "file_out",
    }))
    respx.get(f"{BASE}/files/file_out/content").mock(return_value=httpx.Response(
        200, text=_jsonl([_line("0", body=_completion("ok", model="m"))]),
    ))

    async def run():
        bc = _batch(http_retries=4, backoff_start_s=2.0)
        try:
            return await bc.run_chat_batch([_req("a")])
        finally:
            await bc.aclose()

    out = asyncio.run(run())
    assert out[0].text == "ok"
    assert sleeps[0] == 2.0
    assert sleeps[1] == 3.0


@respx.mock
def test_exhausted_429_raises(monkeypatch):
    async def no_sleep(_delay, *_args, **_kwargs):
        return None

    monkeypatch.setattr("llm_mesh.openai.batch.asyncio.sleep", no_sleep)
    route = respx.post(f"{BASE}/files").mock(return_value=httpx.Response(429, text="slow down"))

    async def run():
        bc = _batch(http_retries=2, backoff_start_s=0)
        try:
            await bc.run_chat_batch([_req("a")])
        finally:
            await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="429"):
        asyncio.run(run())
    assert route.call_count == 2


def test_size_limits_raise(monkeypatch):
    monkeypatch.setattr("llm_mesh.openai.batch.MAX_BATCH_REQUESTS", 1)

    async def too_many():
        bc = _batch()
        try:
            await bc.run_chat_batch([_req("a"), _req("b")])
        finally:
            await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="at most 1"):
        asyncio.run(too_many())

    monkeypatch.setattr("llm_mesh.openai.batch.MAX_BATCH_FILE_BYTES", 8)

    async def too_big():
        bc = _batch()
        try:
            await bc.run_chat_batch([_req("a")])
        finally:
            await bc.aclose()

    with pytest.raises(OpenAIBatchError, match="bytes"):
        asyncio.run(too_big())


def test_owned_http_client_is_reused_and_external_is_not_closed():
    async def run():
        external = httpx.AsyncClient()
        bc = OpenAIBatchClient(
            api_key="k", base_url=BASE, http_client=external, poll_interval_s=0,
        )
        assert bc._http() is external
        await bc.aclose()
        assert not external.is_closed
        await external.aclose()

        owned = OpenAIBatchClient(api_key="k", base_url=BASE, poll_interval_s=0)
        first = owned._http()
        assert owned._http() is first
        await owned.aclose()
        assert first.is_closed

    asyncio.run(run())


@respx.mock
def test_batching_client_coalesces_two_structured_calls():
    uploads: list[str] = []

    def on_upload(request: httpx.Request):
        uploads.append(request.content.decode("utf-8", "replace"))
        return httpx.Response(200, json={"id": "file_in"})

    respx.post(f"{BASE}/files").mock(side_effect=on_upload)
    respx.post(f"{BASE}/batches").mock(
        return_value=httpx.Response(200, json={"id": "batch_1", "status": "validating"})
    )
    respx.get(f"{BASE}/batches/batch_1").mock(return_value=httpx.Response(200, json={
        "id": "batch_1", "status": "completed", "output_file_id": "file_out",
    }))
    respx.get(f"{BASE}/files/file_out/content").mock(return_value=httpx.Response(
        200, text=_jsonl([
            _line("1", body=_tool_completion({"echo": "second"})),
            _line("0", body=_tool_completion({"echo": "first"})),
        ]),
    ))

    async def run():
        cli = BatchingLLMClient(
            _batch(), model="client-model", max_delay_s=0.05,
        )
        try:
            first, second = await asyncio.gather(
                cli.generate_structured(_req("first", mode="function_call")),
                cli.generate_structured(_req("second", mode="function_call")),
            )
            return first, second
        finally:
            await cli.aclose()

    first, second = asyncio.run(run())
    assert first.arguments == {"echo": "first"}
    assert second.arguments == {"echo": "second"}
    assert len(uploads) == 1
    assert '"custom_id": "0"' in uploads[0]
    assert '"custom_id": "1"' in uploads[0]


def test_catalog_openai_batch_mode_wraps_only_when_enabled(monkeypatch):
    for key in _MANAGED_ENV:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.delenv("LLM_BATCH_MODE", raising=False)
    route = {
        "id": "oa-batch",
        "kind": "openai",
        "model": "gpt-test",
        "base_url": BASE,
        "api_key": "sk-test",
        "provider": "openai",
    }
    plain = make_client(route)
    assert type(plain) is OpenAIClient
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    wrapped = make_client(route)
    assert isinstance(wrapped, BatchingLLMClient)
    assert isinstance(wrapped._batch, OpenAIBatchClient)
    assert wrapped._model == "gpt-test"
    assert type(wrapped._batch._openai) is OpenAIClient
