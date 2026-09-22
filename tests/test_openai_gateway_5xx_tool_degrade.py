"""After tool request 5xx retries, try text once to distinguish a gateway outage from rejection of
the tool payload. Use a provider-independent signal, preserve no-degrade behavior, and do not
cache an ambiguous outage as permanent text preference. If text also fails, propagate the error.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_mesh.openai.client import OpenAIClient, OpenAIError, _is_gateway_5xx
from llm_mesh.types import LLMRequest, LLMValidationError


BASE = "https://host/v1"
URL = "https://host/v1/chat/completions"

_BAD_GATEWAY = {
    "error": {
        "code": "BAD_GATEWAY",
        "message": "Service temporarily unavailable. Try again later.",
        "metadata": {"raw": "Provider returned an invalid tool call"},
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LLM_BASE_URL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_API_KEY",
        "OPENROUTER_API_KEY", "LLM_DISABLE_TOOLS", "LLM_TOOL_CHOICE_PREF",
        "LLM_RESPONSE_FORMAT", "LLM_NO_DEGRADE", "LLM_OPEN_OBJECT_SCHEMAS",
    ):
        monkeypatch.delenv(var, raising=False)
    # Disable transport delays here to test the fallback ladder rather than retry timing.
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {"kind": {"type": "string"}},
        "required": ["kind"],
    }


def _request(**over) -> LLMRequest:
    kw = dict(
        system="You dispatch fragments",
        user="Write a handler for ON_OPEN",
        tools=[{"name": "emit", "description": "emit", "parameters": _schema()}],
        schema=_schema(),
        function_name="emit",
        mode="function_call",
    )
    kw.update(over)
    return LLMRequest(**kw)


def _client(monkeypatch) -> OpenAIClient:
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    return OpenAIClient(model="google/gemma-4-26b-a4b-it", label="polza")


def _text_answer(payload: str) -> dict:
    return {
        "model": "google/gemma-4-26b-a4b-it",
        "choices": [{"message": {"role": "assistant", "content": payload}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _route_by_tools(text_payload: str, *, status: int = 502):
    """Mock a gateway that rejects tool payloads but serves text requests, distinguishing payload
    rejection from a full outage.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        if body.get("tools") or body.get("functions"):
            return httpx.Response(status, json=_BAD_GATEWAY)
        return httpx.Response(200, json=_text_answer(text_payload))

    return _handler


# --- Predicate --------------------------------------------------------------


def test_predicate_takes_5xx_and_ignores_429_and_4xx():
    """Match 5xx but exclude rate limits and client errors; removing tools cannot resolve quota
    exhaustion.
    """
    for code in (500, 502, 503, 504):
        assert _is_gateway_5xx(OpenAIError("x", status_code=code)), code
    for code in (400, 404, 422, 429):
        assert not _is_gateway_5xx(OpenAIError("x", status_code=code)), code


# --- Reactive fallback ------------------------------------------------------


@respx.mock
def test_tool_5xx_is_served_by_text_tier(monkeypatch):
    """Recover a tool request 5xx through the text tier instead of returning the gateway error."""
    respx.post(URL).mock(side_effect=_route_by_tools('{"kind": "handler"}'))
    client = _client(monkeypatch)

    import asyncio

    resp = asyncio.run(client.generate_structured(_request()))
    assert resp.arguments == {"kind": "handler"}
    assert client._last_served_tier == "text"


@respx.mock
def test_forced_tier_degrades_too(monkeypatch):
    """Apply the same recovery on the forced-function path, independently of the multi-tool path."""
    respx.post(URL).mock(side_effect=_route_by_tools('{"kind": "handler"}'))
    client = _client(monkeypatch)

    import asyncio

    resp = asyncio.run(client.generate_structured(_request(tools=None)))
    assert resp.arguments == {"kind": "handler"}
    assert client._last_served_tier == "text"


@respx.mock
def test_gateway_really_down_still_fails_honestly(monkeypatch):
    """If the gateway also fails without tools, propagate the failure after one distinguishing text
    attempt.
    """
    respx.post(URL).mock(return_value=httpx.Response(502, json=_BAD_GATEWAY))
    client = _client(monkeypatch)

    import asyncio

    with pytest.raises(OpenAIError) as ei:
        asyncio.run(client.generate_structured(_request()))
    assert ei.value.status_code == 502


@respx.mock
def test_no_degrade_keeps_the_original_5xx(monkeypatch):
    """Under no-degrade, preserve the original 5xx and status instead of replacing it with a
    text tier refusal.
    """
    monkeypatch.setenv("LLM_NO_DEGRADE", "1")
    respx.post(URL).mock(side_effect=_route_by_tools('{"kind": "handler"}'))
    client = _client(monkeypatch)

    import asyncio

    with pytest.raises(OpenAIError) as ei:
        asyncio.run(client.generate_structured(_request()))
    assert ei.value.status_code == 502


@respx.mock
def test_429_is_not_degraded(monkeypatch):
    """Do not use request-form fallback for rate limits."""
    respx.post(URL).mock(side_effect=_route_by_tools('{"kind": "x"}', status=429))
    client = _client(monkeypatch)

    import asyncio

    with pytest.raises(OpenAIError) as ei:
        asyncio.run(client.generate_structured(_request()))
    assert ei.value.status_code == 429


@respx.mock
def test_native_tool_loop_is_not_substituted_by_text(monkeypatch):
    """Never substitute text fallback for a native tool-loop turn; the caller could mistake it for
    the model's own decision.
    """
    respx.post(URL).mock(side_effect=_route_by_tools('{"kind": "x"}'))
    client = _client(monkeypatch)

    import asyncio

    with pytest.raises((OpenAIError, LLMValidationError)):
        asyncio.run(client.generate_structured(_request(tools_required=True)))
