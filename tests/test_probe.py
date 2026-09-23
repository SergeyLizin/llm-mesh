"""Connection checks against mocked transports. No live provider calls."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx
import pytest
import respx

from llm_mesh import (
    ConnectionCheck,
    ProbeKind,
    check_client,
    check_route,
    check_routes,
)
from llm_mesh.anthropic.client import AnthropicClient
from llm_mesh.gigachat.client import GigaChatAsyncClient
from llm_mesh.models_catalog import _MANAGED_ENV, _cli, _format_check_line, make_client
from llm_mesh.openai.client import OpenAIClient, chat_completions_url


OPENAI_URL = chat_completions_url("https://openai.test/v1")
DOWN_URL = chat_completions_url("https://down.test/v1")
UP_URL = chat_completions_url("https://up.test/v1")
GIGA_API = "https://gigachat.test/api"
GIGA_AUTH = "https://auth.test/oauth"
ANTHROPIC_URL = "https://anthropic.test/v1/messages"


def _chat(text: str = "ok") -> dict:
    return {
        "id": "cmpl",
        "model": "probe-model",
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic(text: str = "ok") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-probe",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _openai(**kwargs) -> OpenAIClient:
    return OpenAIClient(
        "probe-model",
        base_url="https://openai.test/v1",
        api_key="test-key",
        label="openai-probe",
        **kwargs,
    )


def _routes() -> list[dict]:
    return [
        {
            "id": "demo",
            "kind": "openai",
            "provider": "down",
            "label": "demo (down)",
            "model": "probe-model",
            "base_url": "https://down.test/v1",
            "api_key": "test-key",
        },
        {
            "id": "demo",
            "kind": "openai",
            "provider": "up",
            "label": "demo (up)",
            "model": "probe-model",
            "base_url": "https://up.test/v1",
            "api_key": "test-key",
        },
    ]


@pytest.fixture
def restored_env():
    """make_client writes managed variables. Put the process back afterwards."""
    saved = {key: os.environ.get(key) for key in _MANAGED_ENV}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def quiet_retries(monkeypatch):
    """One retry with no sleep, so timeout tests observe the retry loop."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


@pytest.mark.asyncio
@respx.mock
async def test_openai_check_succeeds_with_a_tiny_completion():
    route = respx.post(OPENAI_URL).respond(200, json=_chat())
    client = _openai()
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is True
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.model == "probe-model"
    assert result.label == "openai-probe"
    assert result.kind == "openai"
    assert result.error is None
    body = _body(route.calls.last.request)
    assert body["max_tokens"] <= 8
    assert body["messages"][0]["content"] == (
        "You are a connectivity probe. Reply with exactly: ok"
    )


@pytest.mark.asyncio
@respx.mock
async def test_openai_auth_failure_is_typed():
    respx.post(OPENAI_URL).respond(401, text="nope")
    client = _openai()
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is False
    assert result.error_type == "LLMAuthError"
    assert result.error


@pytest.mark.asyncio
@respx.mock
async def test_openai_timeout_includes_retries(quiet_retries):
    route = respx.post(OPENAI_URL).mock(
        side_effect=httpx.ReadTimeout("timed out"),
    )
    client = _openai()
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is False
    assert result.error_type == "LLMTimeoutError"
    assert route.call_count == 2
    assert result.latency_ms is not None and result.latency_ms >= 0


@pytest.mark.asyncio
@respx.mock
async def test_openai_transport_refusal_does_not_escape(quiet_retries):
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    client = _openai()
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is False
    assert result.error_type == "OpenAIError"


@pytest.mark.asyncio
@respx.mock
async def test_openai_error_text_is_capped():
    respx.post(OPENAI_URL).respond(400, text="E" * 2000)
    client = _openai()
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is False
    assert result.error is not None and len(result.error) == 500


@pytest.mark.asyncio
@respx.mock
async def test_check_client_does_not_close_a_caller_owned_client(monkeypatch):
    respx.post(OPENAI_URL).respond(200, json=_chat())
    closed: list[int] = []
    original = OpenAIClient.aclose

    async def spy(self):
        closed.append(1)
        await original(self)

    monkeypatch.setattr(OpenAIClient, "aclose", spy)
    client = _openai()
    result = await check_client(client)
    assert result.ok is True
    assert closed == []
    await client.aclose()
    assert closed == [1]


@pytest.mark.asyncio
@respx.mock
async def test_gigachat_counts_tokens_and_does_not_chat():
    auth = respx.post(GIGA_AUTH).respond(
        200, json={"access_token": "tok-1", "expires_at": 0},
    )
    count = respx.post(f"{GIGA_API}/tokens/count").respond(
        200, json=[{"tokens": 2}],
    )
    chat = respx.post(f"{GIGA_API}/chat/completions").respond(500, text="no")
    client = GigaChatAsyncClient(
        credentials="credentials",
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2",
        label="corp-giga",
        api_url=GIGA_API,
        auth_url=GIGA_AUTH,
        verify=False,
        timeout_s=5,
    )
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is True
    assert result.label == "corp-giga"
    assert result.kind == "gigachat"
    assert result.model == "GigaChat-2"
    assert result.detail == "count_tokens"
    assert auth.called
    assert count.called
    assert json.loads(count.calls.last.request.content)["input"] == [
        "connectivity probe",
    ]
    assert not chat.called


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_probe_counts_tokens():
    count_url = ANTHROPIC_URL + "/count_tokens"
    route = respx.post(count_url).respond(200, json={"input_tokens": 3})
    client = AnthropicClient(
        model="claude-probe",
        base_url="https://anthropic.test/v1",
        api_key="test-key",
        label="anthropic-probe",
    )
    try:
        result = await check_client(client)
    finally:
        await client.aclose()
    assert result.ok is True
    assert result.kind == "anthropic"
    body = _body(route.calls.last.request)
    assert body["model"] == "claude-probe"
    assert body["messages"] == [{"role": "user", "content": "connectivity probe"}]
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_unknown_client_raises_type_error():
    with pytest.raises(TypeError):
        await check_client(object())


@respx.mock
def test_check_routes_keeps_catalog_order(restored_env, monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    respx.post(DOWN_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(UP_URL).respond(200, json=_chat())
    results = check_routes("demo", _routes())
    assert [item.label for item in results] == ["demo (down)", "demo (up)"]
    assert results[0].ok is False
    assert results[0].error_type == "OpenAIError"
    assert results[1].ok is True
    assert results[1].model == "probe-model"


@respx.mock
def test_check_route_closes_the_client_it_created(restored_env, monkeypatch):
    respx.post(UP_URL).respond(200, json=_chat())
    closed: list[int] = []
    original = OpenAIClient.aclose

    async def spy(self):
        closed.append(1)
        await original(self)

    monkeypatch.setattr(OpenAIClient, "aclose", spy)
    route = _routes()[1]
    result = check_route(route)
    assert result.ok is True
    assert closed == [1]


def _write_catalog(tmp_path, routes: list[dict]):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(routes))
    return path


@respx.mock
def test_cli_check_exits_zero_when_any_route_works(
    tmp_path, capsys, restored_env, monkeypatch,
):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    respx.post(DOWN_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(UP_URL).respond(200, json=_chat())
    path = _write_catalog(tmp_path, _routes())
    assert _cli(["--config", str(path), "--check", "demo"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "--check makes real API calls (unlike --list)."
    assert out[1].startswith("demo (down)  [FAIL] OpenAIError: ")
    assert len(out[1].split(": ", 1)[1]) <= 120
    assert re.fullmatch(r"demo \(up\)  \[ok\]  \d+ms", out[2])


@respx.mock
def test_cli_check_exits_one_when_every_route_fails(
    tmp_path, capsys, restored_env, monkeypatch,
):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_S", "0")
    respx.post(DOWN_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(UP_URL).mock(side_effect=httpx.ConnectError("refused"))
    path = _write_catalog(tmp_path, _routes())
    assert _cli(["--config", str(path), "--check", "demo"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("--check makes real API calls")
    assert lines[1].startswith("demo (down)  [FAIL] ")
    assert lines[2].startswith("demo (up)  [FAIL] ")


@respx.mock
def test_cli_check_one_provider_uses_that_route_only(
    tmp_path, capsys, restored_env, monkeypatch,
):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    respx.post(DOWN_URL).mock(side_effect=httpx.ConnectError("refused"))
    respx.post(UP_URL).respond(200, json=_chat())
    path = _write_catalog(tmp_path, _routes())
    assert _cli(["--config", str(path), "--check", "demo@down"]) == 1
    failed = capsys.readouterr().out.splitlines()
    assert len(failed) == 2
    assert failed[1].startswith("demo (down)  [FAIL] ")
    assert _cli(["--config", str(path), "--check", "demo@up"]) == 0
    ok = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r"demo \(up\)  \[ok\]  \d+ms", ok[1])


def test_cli_help_documents_check(capsys):
    assert _cli([]) == 2
    assert "--check" in capsys.readouterr().out


def test_batch_mode_route_is_a_failed_check_not_a_traceback(
    restored_env, monkeypatch, capsys, tmp_path,
):
    """LLM_BATCH_MODE builds BatchingLLMClient. Probing it would enqueue a batch.

    check_client still raises TypeError. check_route and --check report the
    refusal as a failed line and still walk every route.
    """
    from llm_mesh.gigachat.batch import BatchingLLMClient, _BATCH_SINGLETONS

    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    gigachat = {
        "id": "batch-probe",
        "kind": "gigachat",
        "provider": "gigachat",
        "label": "batch-probe (gigachat)",
        "model": "GigaChat-probe",
        "base_url": "https://gigachat.test/v1",
        "api_key": "test-key",
        "auth_url": "https://gigachat.test/oauth",
        "scope": "GIGACHAT_API_PERS",
    }
    openai = {
        "id": "batch-probe",
        "kind": "openai",
        "provider": "openai",
        "label": "batch-probe (openai)",
        "model": "gpt-probe",
        "base_url": "https://openai.test/v1",
        "api_key": "test-key",
    }
    before = set(_BATCH_SINGLETONS)
    try:
        wrapped = make_client(gigachat)
        assert isinstance(wrapped, BatchingLLMClient)
        with pytest.raises(TypeError, match="BatchingLLMClient"):
            asyncio.run(check_client(wrapped))

        results = check_routes("batch-probe", [gigachat, openai])
        assert [item.label for item in results] == [
            "batch-probe (gigachat)", "batch-probe (openai)",
        ]
        assert all(item.ok is False and item.error_type == "TypeError" for item in results)
        assert all(item.latency_ms is None for item in results)
        assert "BatchingLLMClient" in (results[0].error or "")

        path = _write_catalog(tmp_path, [gigachat])
        assert _cli(["--config", str(path), "--check", "batch-probe"]) == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        lines = captured.out.splitlines()
        assert lines[0] == "--check makes real API calls (unlike --list)."
        assert lines[1].startswith("batch-probe (gigachat)  [FAIL] TypeError: ")
        assert "BatchingLLMClient" in lines[1]
    finally:
        for key in list(_BATCH_SINGLETONS):
            if key not in before:
                _BATCH_SINGLETONS.pop(key, None)


def test_probe_request_is_not_shared():
    from llm_mesh.probe import _probe_request

    first = _probe_request()
    first.max_tokens = 1
    assert _probe_request().max_tokens == 8


@pytest.mark.asyncio
async def test_probe_timeout_caps_a_hung_call():
    class _Hang:
        async def generate_text(self, request):
            await asyncio.sleep(30)

    started = time.perf_counter()
    result = await check_client(
        _Hang(), probe=ProbeKind.GENERATE_TEXT, timeout=0.05,
    )
    assert result.ok is False
    assert result.error_type == "LLMTimeoutError"
    assert "0.05s" in (result.error or "")
    assert time.perf_counter() - started < 1


@pytest.mark.asyncio
async def test_probe_timeout_none_uses_the_client_timeout():
    class _Slow:
        async def generate_text(self, request):
            await asyncio.sleep(0.15)

    capped = await check_client(
        _Slow(), probe=ProbeKind.GENERATE_TEXT, timeout=0.05,
    )
    assert capped.ok is False
    assert capped.error_type == "LLMTimeoutError"
    open_timeout = await check_client(
        _Slow(), probe=ProbeKind.GENERATE_TEXT, timeout=None,
    )
    assert open_timeout.ok is True


@pytest.mark.asyncio
async def test_unimplemented_probe_is_a_caller_error():
    client = AnthropicClient(model="m", api_key="k")
    try:
        with pytest.raises(TypeError, match="embed"):
            await check_client(client, probe=ProbeKind.EMBED)
    finally:
        await client.aclose()


def test_fail_line_caps_error_text():
    result = ConnectionCheck(
        ok=False,
        label="demo (down)",
        kind="openai",
        model="probe-model",
        latency_ms=4,
        error="e" * 300,
        error_type="OpenAIError",
    )
    line = _format_check_line(result)
    assert line == f"demo (down)  [FAIL] OpenAIError: {'e' * 120}"
