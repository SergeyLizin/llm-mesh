"""Corrupt auto tool arguments should advance the fallback ladder. Repeating a deterministic
request reproduces the same corruption, while forced forms may succeed. Keep recovery
provider-independent and expose corruption under no-degrade.
"""

from __future__ import annotations

from llm_mesh.models_catalog import load_catalog_data

import json
from pathlib import Path

import httpx
import pytest
import respx

from llm_mesh.openai.client import OpenAIClient, ToolArgsCorruptedError
from llm_mesh.types import LLMRequest, LLMValidationError


BASE = "https://host/v1"
URL = "https://host/v1/chat/completions"

# Prefix of a captured corrupt response, shortened while preserving quote-token artifacts and
# the malformed trtrue token.
CORRUPTED_ARGS = (
    '{"form_id": "Event_14f0t7n", "full_schema": {"name": "<|\\"|Form<|\\"|", '
    '"xsdContent": [{"options": {"displayName": "<|\\"|Full name<|", '
    '"required": "trtrue}, "type": "<|i'
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LLM_BASE_URL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_API_KEY",
        "OPENROUTER_API_KEY", "LLM_DISABLE_TOOLS", "LLM_TOOL_CHOICE_PREF",
        "LLM_RESPONSE_FORMAT", "LLM_MAX_RETRIES", "LLM_NO_DEGRADE",
    ):
        monkeypatch.delenv(var, raising=False)


def _tools() -> list[dict]:
    return [{
        "name": "emit_form",
        "description": "Declare a form.",
        "parameters": {
            "type": "object",
            "properties": {"form_id": {"type": "string"}},
            "required": ["form_id"],
        },
    }]


def _request() -> LLMRequest:
    return LLMRequest(
        system="You generate forms",
        user="Create an application form",
        tools=_tools(),
        schema={"type": "object", "properties": {"form_id": {"type": "string"}}},
        function_name="emit_form",
        function_description="Declare a form.",
        mode="function_call",
    )


def _tool_call_response(arguments: str) -> dict:
    return {
        "model": "google/gemma-4-31b-it",
        "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "emit_form", "arguments": arguments},
            }],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _client(monkeypatch) -> OpenAIClient:
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    return OpenAIClient(model="google/gemma-4-31b-it", label="silicon")


@pytest.mark.asyncio
@respx.mock
async def test_corrupted_multitool_args_degrade_to_forced(monkeypatch):
    """Recover corrupt auto arguments through a forced-function tier."""
    bodies: list[dict] = []

    def _route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("tool_choice") == "auto":
            return httpx.Response(200, json=_tool_call_response(CORRUPTED_ARGS))
        return httpx.Response(
            200, json=_tool_call_response(json.dumps({"form_id": "Event_14f0t7n"})),
        )

    respx.post(URL).mock(side_effect=_route)
    client = _client(monkeypatch)

    resp = await client.generate_structured(_request())

    assert resp.arguments == {"form_id": "Event_14f0t7n"}
    # Exactly two requests: corrupt multi-tool and forced fallback.
    assert len(bodies) == 2
    assert bodies[0]["tool_choice"] == "auto"
    assert bodies[1]["tool_choice"] == {
        "type": "function", "function": {"name": "emit_form"},
    }


@pytest.mark.asyncio
@respx.mock
async def test_corruption_on_every_tier_fails_honestly(monkeypatch):
    """When every tier returns corrupt output, traverse the complete ladder and fail. Count
    requests so the test proves the actual auto, strict, required, text sequence rather than
    passing accidentally because every mock response is invalid.
    """
    bodies: list[dict] = []

    def _route(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_call_response(CORRUPTED_ARGS))

    respx.post(URL).mock(side_effect=_route)
    client = _client(monkeypatch)

    with pytest.raises(LLMValidationError):
        await client.generate_structured(_request())

    # Traverse auto, forced, required, then text without tools.
    assert [b.get("tool_choice") for b in bodies] == [
        "auto", {"type": "function", "function": {"name": "emit_form"}},
        "required", None,
    ]
    assert "tools" not in bodies[-1]


@pytest.mark.asyncio
@respx.mock
async def test_no_degrade_keeps_corruption_visible(monkeypatch):
    """Under no-degrade, keep argument corruption visible instead of turning a failed native
    measurement into success through another tier.
    """
    bodies: list[dict] = []

    def _route(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if json.loads(request.content).get("tool_choice") == "auto":
            return httpx.Response(200, json=_tool_call_response(CORRUPTED_ARGS))
        return httpx.Response(
            200, json=_tool_call_response(json.dumps({"form_id": "Event_14f0t7n"})),
        )

    respx.post(URL).mock(side_effect=_route)
    monkeypatch.setenv("LLM_NO_DEGRADE", "1")
    client = _client(monkeypatch)

    with pytest.raises(ToolArgsCorruptedError):
        await client.generate_structured(_request())

    # Exactly one request; no forced recovery under the flag.
    assert len(bodies) == 1


def _forced_request() -> LLMRequest:
    """Build a request without tools so the ladder starts at strict forcing."""
    return LLMRequest(
        system="You generate forms",
        user="Create an application form",
        schema={"type": "object", "properties": {"form_id": {"type": "string"}}},
        function_name="emit_form",
        function_description="Declare a form.",
        mode="function_call",
    )


@pytest.mark.asyncio
@respx.mock
async def test_no_degrade_raises_corruption_on_forced_tier(monkeypatch):
    """Under no-degrade, propagate corruption on forced tiers as well as auto. Schema negotiation
    may still change forms, but recovering damaged arguments would conceal the measured failure.
    """
    bodies: list[dict] = []

    def _route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if isinstance(body.get("tool_choice"), dict):
            return httpx.Response(200, json=_tool_call_response(CORRUPTED_ARGS))
        return httpx.Response(
            200, json=_tool_call_response(json.dumps({"form_id": "Event_14f0t7n"})),
        )

    respx.post(URL).mock(side_effect=_route)
    monkeypatch.setenv("LLM_NO_DEGRADE", "1")
    client = _client(monkeypatch)

    with pytest.raises(ToolArgsCorruptedError):
        await client.generate_structured(_forced_request())

    # Exactly one strict request; no required recovery under the flag.
    assert [b.get("tool_choice") for b in bodies] == [
        {"type": "function", "function": {"name": "emit_form"}},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_forced_tier_corruption_degrades_without_flag(monkeypatch):
    """Without no-degrade, preserve strict-to-required recovery from corruption."""
    bodies: list[dict] = []

    def _route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if isinstance(body.get("tool_choice"), dict):
            return httpx.Response(200, json=_tool_call_response(CORRUPTED_ARGS))
        return httpx.Response(
            200, json=_tool_call_response(json.dumps({"form_id": "Event_14f0t7n"})),
        )

    respx.post(URL).mock(side_effect=_route)
    client = _client(monkeypatch)

    resp = await client.generate_structured(_forced_request())

    assert resp.arguments == {"form_id": "Event_14f0t7n"}
    assert [b.get("tool_choice") for b in bodies] == [
        {"type": "function", "function": {"name": "emit_form"}}, "required",
    ]


def test_corrupted_error_stays_llm_validation_error():
    """ToolArgsCorruptedError must remain a LLMValidationError so existing bounded retries and
    exception handlers keep working.
    """
    assert issubclass(ToolArgsCorruptedError, LLMValidationError)


def test_silicon_routes_declare_measured_response_format():
    """Declare SiliconFlow's measured json_object dialect, not unsupported json_schema. This adds a
    native fallback between forced tools and text without changing the happy path.
    """
    catalog = load_catalog_data()
    missing = [
        f"{model.get('id')}@{route.get('name')}"
        for model in catalog
        for route in model.get("providers", [])
        if route.get("base_url_env") == "SILICON_BASE_URL"
        and route.get("response_format") != "json_object"
    ]
    assert not missing, f"silicon-routes without a declared dialect: {missing}"
