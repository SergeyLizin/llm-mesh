"""Validate structured arguments against the function schema before returning success.
Syntactically valid but schema-invalid JSON must follow strict, required, then text fallback;
schema-valid native calls must still succeed immediately.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh.openai.client import OpenAIClient, _args_satisfy_schema
from llm_mesh.types import LLMRequest

BASE = "https://api.example.test/v1"
URL = f"{BASE}/chat/completions"

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def _req() -> LLMRequest:
    return LLMRequest(
        system="You design forms",
        user="Create a report form",
        schema=_SCHEMA,
        function_name="design_entity_form",
        mode="function_call",
    )


def _tool_call(args: dict) -> dict:
    return {
        "model": "gemma",
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "design_entity_form",
                             "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        }}],
        "usage": {},
    }


def _text(args: dict) -> dict:
    return {
        "model": "gemma",
        "choices": [{"message": {
            "role": "assistant",
            "content": json.dumps(args, ensure_ascii=False),
        }}],
        "usage": {},
    }


# ── unit: _args_satisfy_schema ──────────────────────────────────────────────

def test_args_satisfy_schema_valid():
    assert _args_satisfy_schema({"name": "Report"}, _SCHEMA) is True


def test_args_satisfy_schema_violation():
    # Valid JSON missing the required name field triggers fallback.
    assert _args_satisfy_schema({"type": "date"}, _SCHEMA) is False


def test_args_satisfy_schema_conservative_noops():
    # An empty or absent schema, or non-dictionary arguments, does not block this check.
    assert _args_satisfy_schema({"anything": 1}, None) is True
    assert _args_satisfy_schema({"anything": 1}, {}) is True
    assert _args_satisfy_schema("not dict", _SCHEMA) is True


# --- Structured fallback integration ----------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_schema_invalid_toolcall_degrades_to_text(monkeypatch):
    """Fall back through schema-invalid strict and required calls to valid text output rather than
    returning invalid JSON as success.
    """
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(200, json=_tool_call({"type": "date"})),   # Strict output violates the schema.
        httpx.Response(200, json=_tool_call({"foo": "bar"})),     # Required output also violates the schema.
        httpx.Response(200, json=_text({"name": "Inspection report"})),  # Text output is valid.
    ])
    client = OpenAIClient(model="google/gemma-4-31b-it")
    try:
        resp = await client.generate_structured(_req())
    finally:
        await client.aclose()
    # Return the schema-valid text emulation result.
    assert resp.arguments == {"name": "Inspection report"}
    assert route.call_count == 3
    assert client._tool_choice_pref == "text"
    # The final text request contains no tools.
    assert "tools" not in json.loads(route.calls[2].request.content)


@respx.mock
@pytest.mark.asyncio
async def test_schema_valid_toolcall_no_degradation(monkeypatch):
    """A schema-valid strict call succeeds in one request without fallback."""
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=_tool_call({"name": "Inspection report"}))
    )
    client = OpenAIClient(model="qwen/qwen3.6-27b")
    try:
        resp = await client.generate_structured(_req())
    finally:
        await client.aclose()
    assert resp.arguments == {"name": "Inspection report"}
    assert route.call_count == 1  # No extra round trips.
    # Object-form strict tool_choice was accepted.
    assert json.loads(route.calls[0].request.content)["tool_choice"]["type"] == "function"
