"""Unit tests for the async GigaChat client using respx HTTP mocks."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from llm_mesh.gigachat.client import (
    GIGACHAT_BASE_URL,
    GIGACHAT_AUTH_URL,
    GigaChatClient,
)
from llm_mesh.types import (
    LLMAuthError,
    LLMError,
    LLMRequest,
    LLMValidationError,
)


CREDS = "dGVzdC1jbGllbnQ6dGVzdC1zZWNyZXQ="  # base64('test-client:test-secret')


@pytest.fixture(autouse=True)
def _clean_gigachat_env(monkeypatch):
    """Isolate credentials from environment files loaded by imported packages. Tests must use only
    explicitly configured client values.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_AUTH_SCOPE", raising=False)
    monkeypatch.delenv("LLM_DISABLE_REASONING", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)


def _request(schema: dict | None = None) -> LLMRequest:
    return LLMRequest(
        system="You are an assistant",
        user="Generate seq()",
        schema=schema or {"type": "object", "properties": {"x": {"type": "string"}}},
        function_name="build_artifact",
    )


def _ok_chat_response(args: dict, model: str = "GigaChat") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "build_artifact",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            }
        ],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# --- Credential resolution --------------------------------------------------


def test_no_credentials_raises():
    with pytest.raises(LLMAuthError):
        GigaChatClient(credentials=None, token=None)


def test_token_short_circuit_skips_oauth(monkeypatch):
    """An explicitly supplied token skips OAuth."""
    client = GigaChatClient(token="pre-baked")

    async def run():
        with respx.mock(base_url=GIGACHAT_BASE_URL) as mock:
            mock.post("/chat/completions").mock(
                return_value=httpx.Response(200, json=_ok_chat_response({"x": "hi"}))
            )
            resp = await client.generate_structured(_request())
            await client.aclose()
        return resp, mock

    resp, _ = asyncio.run(run())
    assert resp.arguments == {"x": "hi"}
    assert resp.usage.total_tokens == 15


# --- OAuth ------------------------------------------------------------------


@respx.mock
def test_oauth_success_then_chat():
    auth = respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_at": 0})
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "ok"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert auth.called
    assert chat.called
    assert resp.arguments == {"x": "ok"}
    # The chat request carries the bearer token.
    assert chat.calls.last.request.headers["Authorization"] == "Bearer tok-1"


@respx.mock
def test_oauth_scope_fallback():
    """Try the next OAuth scope after a 401 until one succeeds."""
    scopes_seen: list[str] = []

    def auth_handler(request):
        # data — `application/x-www-form-urlencoded`, scope=...
        body = request.content.decode()
        scope = next(
            (kv.split("=", 1)[1] for kv in body.split("&") if kv.startswith("scope=")),
            "",
        )
        scopes_seen.append(scope)
        if len(scopes_seen) == 1:
            return httpx.Response(401, text="unauthorized")
        return httpx.Response(200, json={"access_token": "tok-2"})

    auth = respx.post(GIGACHAT_AUTH_URL).mock(side_effect=auth_handler)
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"y": 1}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)  # No explicit scope: use the fallback list.
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert auth.call_count == 2
    assert len(scopes_seen) == 2
    assert scopes_seen[0] != scopes_seen[1]  # The next attempt uses a different scope.
    assert resp.arguments == {"y": 1}


@respx.mock
def test_oauth_all_fail():
    respx.post(GIGACHAT_AUTH_URL).mock(return_value=httpx.Response(403, text="forbidden"))

    async def run():
        client = GigaChatClient(credentials=CREDS)
        with pytest.raises(LLMAuthError, match="last=403"):
            await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())


@respx.mock
def test_oauth_403_dns_blip_retries_within_scope():
    """Retry a transient resolve_no_records 403 within the same scope."""
    responses: list[httpx.Response] = [
        httpx.Response(403, text="Host resolves to a private/reserved IP: resolve_no_records"),
        httpx.Response(403, text="Host resolves to a private/reserved IP: resolve_no_records"),
        httpx.Response(200, json={"access_token": "tok-after-dns-recovered"}),
    ]
    auth = respx.post(GIGACHAT_AUTH_URL).mock(side_effect=responses)
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": 1}))
    )

    async def run():
        # Mock asyncio.sleep to avoid real delays.
        import asyncio as _aio
        original_sleep = _aio.sleep
        _aio.sleep = lambda *a, **kw: original_sleep(0)
        try:
            client = GigaChatClient(credentials=CREDS)
            resp = await client.generate_structured(_request())
            await client.aclose()
            return resp
        finally:
            _aio.sleep = original_sleep

    resp = asyncio.run(run())
    # Make three attempts instead of failing immediately on the first 403.
    assert auth.call_count == 3
    assert resp.arguments == {"x": 1}


@respx.mock
def test_oauth_504_gateway_timeout_retries_within_scope():
    """Retry a transient OAuth 504 with backoff within the same scope. Token acquisition must
    tolerate gateway outages just as chat requests do.
    """
    responses: list[httpx.Response] = [
        httpx.Response(
            504,
            text="<html><head><title>504 Gateway Time-out</title></head></html>",
        ),
        httpx.Response(200, json={"access_token": "tok-after-504-recovered"}),
    ]
    auth = respx.post(GIGACHAT_AUTH_URL).mock(side_effect=responses)
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": 1}))
    )

    async def run():
        import asyncio as _aio
        original_sleep = _aio.sleep
        _aio.sleep = lambda *a, **kw: original_sleep(0)
        try:
            client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
            resp = await client.generate_structured(_request())
            await client.aclose()
            return resp
        finally:
            _aio.sleep = original_sleep

    resp = asyncio.run(run())
    assert auth.call_count == 2  # Retry the initial 504 and receive 200.
    assert resp.arguments == {"x": 1}


@respx.mock
def test_oauth_503_then_502_then_success():
    """Retry both transient failures in a 503, 502, 200 sequence."""
    responses: list[httpx.Response] = [
        httpx.Response(503, text="service unavailable"),
        httpx.Response(502, text="bad gateway"),
        httpx.Response(200, json={"access_token": "tok-recovered"}),
    ]
    auth = respx.post(GIGACHAT_AUTH_URL).mock(side_effect=responses)
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": 1}))
    )

    async def run():
        import asyncio as _aio
        original_sleep = _aio.sleep
        _aio.sleep = lambda *a, **kw: original_sleep(0)
        try:
            client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
            resp = await client.generate_structured(_request())
            await client.aclose()
            return resp
        finally:
            _aio.sleep = original_sleep

    resp = asyncio.run(run())
    assert auth.call_count == 3


# --- Token refresh after chat 401 -------------------------------------------


@respx.mock
def test_chat_401_triggers_refresh_then_retry():
    """Refresh an expired chat token once, then retry successfully."""
    auth = respx.post(GIGACHAT_AUTH_URL).mock(
        side_effect=[
            httpx.Response(200, json={"access_token": "stale"}),
            httpx.Response(200, json={"access_token": "fresh"}),
        ]
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(401, text="token expired"),
            httpx.Response(200, json=_ok_chat_response({"ok": True})),
        ]
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert auth.call_count == 2
    assert chat.call_count == 2
    assert chat.calls[1].request.headers["Authorization"] == "Bearer fresh"
    assert resp.arguments == {"ok": True}


@respx.mock
def test_chat_401_after_refresh_raises():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok"})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(401, text="still bad")
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        with pytest.raises(LLMAuthError, match="after refresh"):
            await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())


# --- Response parsing -------------------------------------------------------


@respx.mock
def test_no_function_call_raises_validation():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "plain text"}}],
                "model": "GigaChat",
            },
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        with pytest.raises(LLMValidationError, match="function_call"):
            await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())


@respx.mock
def test_no_function_call_retries_and_succeeds():
    """Retry a missing function call and return the subsequent successful response."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    _no_fc = httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": ""}}], "model": "GigaChat"},
    )
    _ok = httpx.Response(
        200,
        json={
            "choices": [{"message": {"function_call": {"name": "build_artifact", "arguments": '{"x": "1"}'}}}],
            "model": "GigaChat",
        },
    )
    responses = iter([_no_fc, _ok])
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(side_effect=lambda _r: next(responses))

    async def run():
        client = GigaChatClient(credentials=CREDS)
        result = await client.generate_structured(_request())
        assert result.arguments == {"x": "1"}
        await client.aclose()

    asyncio.run(run())


@respx.mock
def test_arguments_as_dict_supported():
    """Accept arguments supplied as an already parsed dictionary."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "function_call": {
                                "name": "build_artifact",
                                "arguments": {"already": "dict"},
                            }
                        }
                    }
                ],
                "model": "GigaChat",
            },
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert resp.arguments == {"already": "dict"}


@respx.mock
def test_invalid_json_in_arguments_raises():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "function_call": {
                                "name": "build_artifact",
                                "arguments": "not-a-json{{{",
                            }
                        }
                    }
                ],
                "model": "GigaChat",
            },
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        with pytest.raises(LLMValidationError, match="invalid JSON"):
            await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())


@respx.mock
def test_5xx_propagates_as_llmerror():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(503, text="upstream down")
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        # After retries are exhausted, the message reports a generic 5xx retry failure rather
        # than the original 503 status.
        with pytest.raises(LLMError, match="5xx"):
            await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())


# --- reasoning effort / reasoning_content -------------------------------------


@respx.mock
def test_reasoning_effort_sent_in_structured_body():
    """Forward request.reasoning_effort to the request body."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, model="GigaChat-2-Max")
        req = _request()
        req = req.model_copy(update={"reasoning_effort": "high"})
        await client.generate_structured(req)
        await client.aclose()

    asyncio.run(run())
    body = json.loads(chat.calls.last.request.content)
    assert body["reasoning_effort"] == "high"


@respx.mock
def test_reasoning_effort_from_env(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())
    body = json.loads(chat.calls.last.request.content)
    assert body["reasoning_effort"] == "medium"


@respx.mock
def test_reasoning_effort_invalid_value_not_sent(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "ultra")  # Outside the accepted low/medium/high values.
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())
    body = json.loads(chat.calls.last.request.content)
    assert "reasoning_effort" not in body


@respx.mock
def test_reasoning_content_and_request_id_parsed():
    """Parse message reasoning and the x-request-id response header."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    resp_json = _ok_chat_response({"x": "1"})
    resp_json["choices"][0]["message"]["reasoning_content"] = "First I will think..."
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, json=resp_json, headers={"x-request-id": "req-abc-123"}
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert resp.reasoning_content == "First I will think..."
    assert resp.request_id == "req-abc-123"


@respx.mock
def test_request_id_falls_back_to_rquid_when_no_header():
    """Use the generated RqUID when no x-request-id header is returned."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    sent_rquid = chat.calls.last.request.headers["RqUID"]
    assert resp.request_id == sent_rquid


# --- Token streaming --------------------------------------------------------


def _sse(*chunks: dict) -> str:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


@respx.mock
def test_generate_stream_yields_text_deltas():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    sse = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream",
                                    "x-request-id": "rid-stream-1"}
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        chunks = []
        async for c in client.generate_stream(
            LLMRequest(system="s", user="u", mode="text")
        ):
            chunks.append(c)
        await client.aclose()
        return chunks

    chunks = asyncio.run(run())
    text = "".join(c.delta_text for c in chunks)
    assert text == "Hello"
    assert chunks[0].request_id == "rid-stream-1"
    assert chunks[-1].finish_reason == "stop"
    final_usage = [c.usage for c in chunks if c.usage is not None]
    assert final_usage and final_usage[-1].total_tokens == 5


@respx.mock
def test_generate_stream_yields_reasoning_deltas():
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    sse = _sse(
        {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
        {"choices": [{"delta": {"content": "response"}, "finish_reason": "stop"}]},
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse)
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        reasoning, content = "", ""
        async for c in client.generate_stream(
            LLMRequest(system="s", user="u", mode="text", reasoning_effort="high")
        ):
            reasoning += c.delta_reasoning
            content += c.delta_text
        await client.aclose()
        return reasoning, content

    reasoning, content = asyncio.run(run())
    assert reasoning == "thinking"
    assert content == "response"


@respx.mock
def test_generate_stream_401_refreshes_token():
    auth = respx.post(GIGACHAT_AUTH_URL).mock(
        side_effect=[
            httpx.Response(200, json={"access_token": "stale"}),
            httpx.Response(200, json={"access_token": "fresh"}),
        ]
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(401, text="token expired"),
            httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "ok"}}]})),
        ]
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        out = ""
        async for c in client.generate_stream(
            LLMRequest(system="s", user="u", mode="text")
        ):
            out += c.delta_text
        await client.aclose()
        return out

    out = asyncio.run(run())
    assert out == "ok"
    assert auth.call_count == 2
    assert chat.call_count == 2
    assert chat.calls[1].request.headers["Authorization"] == "Bearer fresh"


# --- Proactive refresh using expires_at -------------------------------------


def test_parse_expires_at_normalizes_ms_and_falsy():
    from llm_mesh.gigachat.client import _parse_expires_at
    import time as _t
    assert _parse_expires_at(None) is None
    assert _parse_expires_at(0) is None
    assert _parse_expires_at("") is None
    assert _parse_expires_at("junk") is None
    # Convert epoch milliseconds to seconds.
    assert _parse_expires_at(1_900_000_000_000) == 1_900_000_000.0
    # Preserve absolute epoch seconds.
    assert _parse_expires_at(1_900_000_000) == 1_900_000_000.0
    # Convert relative seconds below 1e9 to now plus the value.
    rel = _parse_expires_at(1800)
    assert rel is not None and abs(rel - (_t.time() + 1800)) < 5


@respx.mock
def test_proactive_refresh_before_expiry():
    """Refresh an expired token before sending the request, without waiting for a 401."""
    auth = respx.post(GIGACHAT_AUTH_URL).mock(
        side_effect=[
            # The first token expired in 2017; its expiry is expressed in milliseconds.
            httpx.Response(200, json={"access_token": "tok-stale",
                                      "expires_at": 1_500_000_000_000}),
            httpx.Response(200, json={"access_token": "tok-fresh",
                                      "expires_at": 9_900_000_000_000}),
        ]
    )
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        await client.generate_structured(_request())  # Fetch the expired tok-stale token explicitly.
        await client.generate_structured(_request())  # Proactive refresh yields tok-fresh.
        await client.aclose()

    asyncio.run(run())
    assert auth.call_count == 2  # The second call proactively refreshed the token.
    # Chat used the fresh bearer token without an intermediate 401.
    assert chat.calls.last.request.headers["Authorization"] == "Bearer tok-fresh"


@respx.mock
def test_no_proactive_refresh_when_token_valid():
    """Reuse a valid token; authenticate only once."""
    auth = respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok",
                                               "expires_at": 9_900_000_000_000})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_ok_chat_response({"x": "1"}))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        await client.generate_structured(_request())
        await client.generate_structured(_request())
        await client.aclose()

    asyncio.run(run())
    assert auth.call_count == 1  # The token remains valid, so no second fetch is needed.


def test_static_token_never_proactively_refreshed():
    """An explicit static token with no expiry is never proactively refreshed."""
    client = GigaChatClient(token="static")
    assert client._token_expired() is False  # An absent expiry does not count as expired.


# --- Retry HTTP 500 and honor Retry-After on 429 -----------------------------


@respx.mock
def test_chat_500_retried_then_success():
    """Retry a generic 500 instead of failing immediately."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    responses = iter([
        httpx.Response(500, text="internal error"),
        httpx.Response(200, json=_ok_chat_response({"x": "1"})),
    ])
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        side_effect=lambda _r: next(responses)
    )

    async def run():
        import asyncio as _aio
        orig = _aio.sleep
        _aio.sleep = lambda *a, **k: orig(0)
        try:
            client = GigaChatClient(credentials=CREDS)
            resp = await client.generate_structured(_request())
            await client.aclose()
            return resp
        finally:
            _aio.sleep = orig

    resp = asyncio.run(run())
    assert resp.arguments == {"x": "1"}


@respx.mock
def test_chat_429_retried_honoring_retry_after():
    """Retry 429 and pass the server's Retry-After delay to sleep."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "7"}, text="rate limited"),
        httpx.Response(200, json=_ok_chat_response({"x": "1"})),
    ])
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        side_effect=lambda _r: next(responses)
    )
    slept: list[float] = []

    async def run():
        import asyncio as _aio
        orig = _aio.sleep
        async def _capture(d, *a, **k):
            slept.append(d)
            return await orig(0)
        _aio.sleep = _capture
        try:
            client = GigaChatClient(credentials=CREDS)
            resp = await client.generate_structured(_request())
            await client.aclose()
            return resp
        finally:
            _aio.sleep = orig

    resp = asyncio.run(run())
    assert resp.arguments == {"x": "1"}
    assert 7.0 in slept  # Honor the server's Retry-After value.


def test_retry_after_delay_parsing():
    from llm_mesh._retry import parse_retry_after
    assert parse_retry_after(httpx.Response(429, headers={"Retry-After": "12"})) == 12.0
    assert parse_retry_after(httpx.Response(429)) is None  # No header is present.
    # Parse HTTP-date as a nonnegative delay.
    d = parse_retry_after(httpx.Response(
        429, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}))
    assert d is not None and d > 0


# --- Canary scans on interrupted streams ------------------------------------


def test_stream_canary_scans_partial_on_error(monkeypatch):
    """Scan accumulated partial output after ReadError, then propagate LLMTimeoutError."""
    from llm_mesh.gigachat import client as _gc

    scanned: list[str] = []
    monkeypatch.setattr(
        _gc, "_check_response_canary",
        lambda text, context="": scanned.append(text),
    )

    class _FakeStreamResp:
        status_code = 200
        headers: dict = {}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "partial"}}]}'
            raise httpx.ReadError("network disconnected")
        async def aread(self): return b""

    class _FakeClient:
        def stream(self, method, url, **kw): return _FakeStreamResp()

    async def run():
        client = GigaChatClient(token="t")
        client._client = _FakeClient()  # type: ignore[assignment]
        got = []
        with pytest.raises(_gc.LLMTimeoutError):
            async for c in client.generate_stream(
                LLMRequest(system="s", user="u", mode="text")
            ):
                got.append(c)
        return got

    got = asyncio.run(run())
    assert "".join(c.delta_text for c in got) == "partial"
    # The canary scan receives partial output despite the interruption.
    assert scanned and "partial" in scanned[-1]


def test_stream_canary_scans_on_early_consumer_break(monkeypatch):
    """Scan partial output in finally when the consumer closes the generator early."""
    from llm_mesh.gigachat import client as _gc

    scanned: list[str] = []
    monkeypatch.setattr(
        _gc, "_check_response_canary",
        lambda text, context="": scanned.append(text),
    )

    class _FakeStreamResp:
        status_code = 200
        headers: dict = {}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "first"}}]}'
            yield 'data: {"choices": [{"delta": {"content": "second"}}]}'
            yield "data: [DONE]"
        async def aread(self): return b""

    class _FakeClient:
        def stream(self, method, url, **kw): return _FakeStreamResp()

    async def run():
        client = GigaChatClient(token="t")
        client._client = _FakeClient()  # type: ignore[assignment]
        async for c in client.generate_stream(
            LLMRequest(system="s", user="u", mode="text")
        ):
            break  # Stop after the first chunk.

    asyncio.run(run())
    # The finally block scans the first delta fragment.
    assert scanned and scanned[-1] == "first"


def test_chunk_from_sse_payload_tolerates_nonstring_reasoning():
    """Treat non-string reasoning content as empty instead of raising ValidationError."""
    from llm_mesh._streaming import chunk_from_sse_payload
    chunk = chunk_from_sse_payload(
        {"choices": [{"delta": {"content": "ok", "reasoning_content": {"oops": 1}}}]},
        request_id="r1", first=True,
    )
    assert chunk.delta_text == "ok"
    assert chunk.delta_reasoning == ""  # Coerce non-string input to an empty string.
    assert chunk.request_id == "r1"


@pytest.mark.parametrize("payload", [
    {"choices": [{"delta": "string-instead-of-object"}]},  # delta is not a dictionary.
    {"choices": ["string-instead-of-choice"]},               # choice is not a dictionary.
    {"choices": []},                                     # Empty choices.
    {"choices": [None]},                                 # None-choice
    {},                                                  # Missing choices.
    {"choices": "not-a-list"},                            # choices is not a list.
])
def test_chunk_from_sse_payload_tolerates_malformed_shapes(payload):
    """Malformed payload shapes produce an empty chunk rather than AttributeError inside the stream
    loop.
    """
    from llm_mesh._streaming import chunk_from_sse_payload
    chunk = chunk_from_sse_payload(payload, request_id="r1", first=True)
    assert chunk.delta_text == ""
    assert chunk.delta_reasoning == ""


# --- Retry length-truncated output ------------------------------------------


@respx.mock
def test_length_retry_doubles_max_tokens_then_succeeds():
    """Double max_tokens after length truncation and return the subsequent stop response."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    truncated = _ok_chat_response({"x": "1"})
    truncated["choices"][0]["finish_reason"] = "length"
    full = _ok_chat_response({"x": "complete"})
    full["choices"][0]["finish_reason"] = "stop"
    responses = iter([httpx.Response(200, json=truncated), httpx.Response(200, json=full)])
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        side_effect=lambda _r: next(responses)
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, model="GigaChat-2-Max")
        resp = await client.generate_structured(
            _request().model_copy(update={"max_tokens": 4096})
        )
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert resp.arguments == {"x": "complete"}
    assert chat.call_count == 2
    # The second request doubles max_tokens.
    body2 = json.loads(chat.calls[1].request.content)
    assert body2["max_tokens"] == 8192


@respx.mock
def test_length_retry_noop_when_at_model_cap():
    """Stop length retries when max_tokens has reached the model ceiling."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    truncated = _ok_chat_response({"x": "1"})
    truncated["choices"][0]["finish_reason"] = "length"
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=truncated)
    )

    async def run():
        # Request the configured Pro ceiling of 8192; no growth is possible.
        client = GigaChatClient(credentials=CREDS, model="GigaChat-2-Pro")
        resp = await client.generate_structured(
            _request().model_copy(update={"max_tokens": 8192})
        )
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert resp.arguments == {"x": "1"}
    assert chat.call_count == 1  # Avoid a useless retry.


@respx.mock
def test_length_retry_disabled_via_env(monkeypatch):
    """LLM_LENGTH_RETRIES=0 disables length retries."""
    monkeypatch.setenv("LLM_LENGTH_RETRIES", "0")
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    truncated = _ok_chat_response({"x": "1"})
    truncated["choices"][0]["finish_reason"] = "length"
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=truncated)
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, model="GigaChat-2-Max")
        await client.generate_structured(_request().model_copy(update={"max_tokens": 4096}))
        await client.aclose()

    asyncio.run(run())
    assert chat.call_count == 1  # Retries are disabled.


@respx.mock
def test_length_retry_skipped_on_degenerate_text(monkeypatch):
    """Do not retry repetitive length-truncated text; a larger budget cannot repair the loop. Send
    one POST.
    """
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t"})
    )
    truncated = {
        "choices": [{
            "message": {"role": "assistant", "content": "}\n" * 3000},
            "finish_reason": "length",
        }],
        "model": "GigaChat-2-Max",
        "usage": {"prompt_tokens": 10, "completion_tokens": 8192, "total_tokens": 8202},
    }
    chat = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=truncated)
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, model="GigaChat-2-Max")
        req = _request().model_copy(update={"max_tokens": 4096, "mode": "text"})
        await client.generate_text(req)
        await client.aclose()

    asyncio.run(run())
    assert chat.call_count == 1  # The repetition guard prevents budget doubling.


# --- Token counting API -----------------------------------------------------


# --- Per-model output limits ------------------------------------------------


def test_max_tokens_clip_to_pro_limit_8k():
    client = GigaChatClient(token="t", model="GigaChat-2-Pro")
    assert client._clip_max_tokens(4096) == 4096  # Within the limit.
    assert client._clip_max_tokens(8192) == 8192  # Exactly at the limit.
    assert client._clip_max_tokens(16384) == 8192  # Clip to the limit.


def test_max_tokens_clip_to_max_limit_16k():
    client = GigaChatClient(token="t", model="GigaChat-2-Max")
    assert client._clip_max_tokens(8192) == 8192
    assert client._clip_max_tokens(16384) == 16384  # Exactly at the limit.
    assert client._clip_max_tokens(32768) == 16384


def test_max_tokens_unknown_model_falls_back_to_default():
    client = GigaChatClient(token="t", model="GigaChat-Mystery")
    # The default output limit is 4096.
    assert client._clip_max_tokens(2048) == 2048
    assert client._clip_max_tokens(16384) == 4096


def test_max_tokens_clip_warns_once_per_model_across_instances(caplog):
    """Deduplicate clipping warnings per model across client instances, including session-per-call
    usage.
    """
    import logging as _logging
    from llm_mesh.gigachat import client as _gc

    _gc._MAX_TOKENS_CLIP_WARNED.discard("GigaChat-2")  # Start with a clean warning cache.
    with caplog.at_level(_logging.WARNING, logger="llm_mesh.gigachat.client"):
        # Three separate instances clip the same model's budget.
        for _ in range(3):
            GigaChatClient(token="t", model="GigaChat-2")._clip_max_tokens(8192)
    clips = [r for r in caplog.records if "exceeds per-model limit" in r.message]
    assert len(clips) == 1  # Emit one warning per model, not one per instance.
    assert "GigaChat-2" in _gc._MAX_TOKENS_CLIP_WARNED
    # Clipping still produces the correct limit.
    assert GigaChatClient(token="t", model="GigaChat-2")._clip_max_tokens(8192) == 4096


def test_max_tokens_warns_only_once(caplog):
    import logging
    from llm_mesh.gigachat import client as _gc
    # Clear this model's process-wide warning cache to keep the test independent of execution
    # order.
    _gc._MAX_TOKENS_CLIP_WARNED.discard("GigaChat-2-Pro")
    client = GigaChatClient(token="t", model="GigaChat-2-Pro")
    with caplog.at_level(logging.WARNING, logger="llm_mesh.gigachat.client"):
        client._clip_max_tokens(16384)
        client._clip_max_tokens(20000)
        client._clip_max_tokens(50000)
    warns = [r for r in caplog.records if "exceeds" in r.message]
    assert len(warns) == 1  # Warn only on the first exceeding request.


# --- Outgoing concurrency limits --------------------------------------------


def test_concurrency_default_no_limit(monkeypatch):
    """Leave concurrency unlimited without an argument or environment setting."""
    monkeypatch.delenv("LLM_MAX_CONCURRENT", raising=False)
    client = GigaChatClient(token="t")
    assert client._max_concurrent is None
    assert client._semaphore is None
    assert client._ensure_semaphore() is None


def test_concurrency_explicit_arg_creates_semaphore_lazily():
    """Store max_concurrent immediately but create its semaphore lazily."""
    client = GigaChatClient(token="t", max_concurrent=3)
    assert client._max_concurrent == 3
    assert client._semaphore is None  # No event loop is required during construction.

    async def run():
        sem = client._ensure_semaphore()
        assert sem is not None
        assert not sem.locked()
        assert client._ensure_semaphore() is sem  # Subsequent access returns the same semaphore.

    asyncio.run(run())


def test_concurrency_env_var_respected(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "5")
    client = GigaChatClient(token="t")
    assert client._max_concurrent == 5


@pytest.mark.parametrize("bad", ["", "0", "-1", "abc", " "])
def test_concurrency_env_var_invalid_falls_back_to_none(monkeypatch, bad):
    monkeypatch.setenv("LLM_MAX_CONCURRENT", bad)
    client = GigaChatClient(token="t")
    assert client._max_concurrent is None


def test_concurrency_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "10")
    client = GigaChatClient(token="t", max_concurrent=2)
    assert client._max_concurrent == 2


def test_concurrency_limits_inflight_requests():
    """With max_concurrent=2, allow at most two requests in flight."""
    client = GigaChatClient(token="t", max_concurrent=2)

    inflight = 0
    peak = 0

    async def run():
        nonlocal inflight, peak
        gate = asyncio.Event()

        async def fake_do(body):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            await gate.wait()
            inflight -= 1
            return {"choices": [{"message": {"function_call": {
                "name": "build_artifact", "arguments": "{}",
            }}}], "model": "GigaChat", "usage": {}}

        client._do_post_chat_with_retry = fake_do  # type: ignore[assignment]
        # Gate five concurrent requests together to measure the peak.
        tasks = [asyncio.create_task(client._post_chat_with_retry({})) for _ in range(5)]
        await asyncio.sleep(0.05)  # Let tasks start and reach the semaphore.
        assert peak <= 2, f"peak={peak} > limit"
        gate.set()
        await asyncio.gather(*tasks)
        assert peak == 2  # The workload actually reaches the configured limit.

    asyncio.run(run())


def test_disable_reasoning_flag_off_by_default():
    """Without LLM_DISABLE_REASONING, preserve the body and requested reasoning effort."""
    client = GigaChatClient(token="t")
    assert client._disable_reasoning is False
    body: dict = {}
    client._apply_reasoning_disable(body)
    assert body == {}  # no-op
    # Forward request reasoning_effort.
    req = _request()
    object.__setattr__(req, "reasoning_effort", "high")
    assert client._resolve_reasoning_effort(req) == "high"


def test_disable_reasoning_flag_on(monkeypatch):
    """Disabling reasoning adds enable_thinking=false and suppresses reasoning_effort."""
    monkeypatch.setenv("LLM_DISABLE_REASONING", "true")
    client = GigaChatClient(token="t")
    assert client._disable_reasoning is True
    body: dict = {}
    client._apply_reasoning_disable(body)
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}
    # Suppress reasoning_effort even when explicitly requested.
    req = _request()
    object.__setattr__(req, "reasoning_effort", "high")
    assert client._resolve_reasoning_effort(req) is None


def test_disable_reasoning_preserves_existing_kwargs(monkeypatch):
    """Merge enable_thinking without erasing other chat_template_kwargs."""
    monkeypatch.setenv("LLM_DISABLE_REASONING", "1")
    client = GigaChatClient(token="t")
    body: dict = {"chat_template_kwargs": {"foo": 1}}
    client._apply_reasoning_disable(body)
    assert body == {"chat_template_kwargs": {"foo": 1, "enable_thinking": False}}


# --- Unsupported native loops in the legacy functions API -------------------


@pytest.mark.asyncio
async def test_tools_required_rejected_honestly():
    """Reject tools_required instead of silently forcing a single function. Legacy function calls
    cannot satisfy the native loop contract; the caller must explicitly choose text emulation.
    """
    client = GigaChatClient(token="t")
    req = _request()
    object.__setattr__(req, "tools", [{"name": "finish", "parameters": {}}])
    object.__setattr__(req, "tools_required", True)
    with pytest.raises(LLMValidationError) as caught:
        await client.generate_structured(req)
    assert "tools_required" in str(caught.value)


def test_tool_turns_dropped_from_legacy_body():
    """Do not send the unsupported tool role through the production legacy wire format."""
    client = GigaChatClient(token="t")
    req = _request()
    object.__setattr__(req, "history", [
        {"role": "user", "content": "OLD-Q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
    ])
    body = client._build_body(req)
    assert [m["role"] for m in body["messages"]] == ["system", "user", "user"]
    assert all("tool_calls" not in m for m in body["messages"])


# --- Response finish reasons ------------------------------------------------


def _chat_response_with_finish(reason: str | None, *, text: bool) -> dict:
    message: dict = {"role": "assistant", "content": "part of the response"}
    if not text:
        message["content"] = ""
        message["function_call"] = {"name": "build_artifact", "arguments": "{\"x\": 1}"}
    choice: dict = {"message": message}
    if reason is not None:
        choice["finish_reason"] = reason
    return {
        "choices": [choice],
        "model": "GigaChat",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


@respx.mock
@pytest.mark.parametrize("reason", ["length", "stop", "function_call"])
def test_finish_reason_propagated_structured(reason):
    """Propagate the provider finish_reason through structured generation."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_at": 0})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response_with_finish(reason, text=False))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    assert asyncio.run(run()).finish_reason == reason


@respx.mock
def test_finish_reason_propagated_text():
    """Propagate finish_reason through text generation as well."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_at": 0})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response_with_finish("length", text=True))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
        resp = await client.generate_text(
            LLMRequest(system="s", user="u", mode="text", length_retry=False)
        )
        await client.aclose()
        return resp

    resp = asyncio.run(run())
    assert resp.finish_reason == "length"
    assert resp.text == "part of the response"  # Return truncated text unchanged.


@respx.mock
def test_finish_reason_absent_is_none():
    """Represent an unreported finish reason as None, distinct from a normal stop."""
    respx.post(GIGACHAT_AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_at": 0})
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response_with_finish(None, text=False))
    )

    async def run():
        client = GigaChatClient(credentials=CREDS, scope="GIGACHAT_API_CORP")
        resp = await client.generate_structured(_request())
        await client.aclose()
        return resp

    assert asyncio.run(run()).finish_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize('debug_enabled', [False, True])
async def test_rejected_structured_response_is_available_only_in_debug(caplog, monkeypatch, debug_enabled):
    import logging
    from unittest.mock import AsyncMock

    caplog.set_level(logging.DEBUG if debug_enabled else logging.WARNING, logger='llm_mesh.gigachat.client')
    payload = _ok_chat_response({})
    raw_arguments = '{"name": "' + 'x' * 500 + 'END_OF_RAW'
    payload['choices'][0]['message']['function_call']['arguments'] = raw_arguments
    client = GigaChatClient(credentials=CREDS)
    monkeypatch.setattr(client, '_post_chat_with_length_retry', AsyncMock(return_value=(payload, 'request-probe')))
    try:
        with pytest.raises(LLMValidationError, match='invalid JSON'):
            await client.generate_structured(_request())
    finally:
        await client.aclose()
    records = [r for r in caplog.records if 'raw_response=' in r.getMessage()]
    assert bool(records) is debug_enabled
    if debug_enabled:
        assert len(records) == 1
        assert raw_arguments in records[0].getMessage()
        assert 'request-probe' in records[0].getMessage()
        assert records[0].exc_info
        assert CREDS not in records[0].getMessage()
