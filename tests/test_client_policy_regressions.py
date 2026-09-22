"""Wire-level checks for explicit selection and response preservation policies."""
import json

import httpx
import pytest

from llm_mesh import LLMRequest
from llm_mesh.openai import OpenAIClient
from llm_mesh.gigachat import GigaChatAsyncClient

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["ok"]}},
          "required": ["answer"]}


def response(answer):
    return {"choices": [{"message": {"tool_calls": [{"id": "call", "function": {
        "name": "f", "arguments": json.dumps({"answer": answer})}}]}, "finish_reason": "stop"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("dialect", ["tools", "json_schema"])
async def test_schema_policy_preserves_or_retries_invalid_enum(monkeypatch, validate, dialect):
    monkeypatch.setenv("LLM_OPTIONS", json.dumps({"response_format": "json_schema"} if dialect != "tools" else {}))
    client = OpenAIClient(model="m", base_url="https://example.invalid/v1", api_key="k",
                          validate_schema=validate, no_degrade=False)
    bodies = []

    async def post(body):
        bodies.append(body)
        value = "bad" if len(bodies) == 1 else "ok"
        if dialect == "json_schema":
            return {"choices": [{"message": {"content": json.dumps({"answer": value})}, "finish_reason": "stop"}]}
        return response(value)

    client._post = post
    if dialect == "json_schema":
        client._tool_choice_pref = "response_format"
    result = await client.generate_structured(LLMRequest(system="s", user="u", schema=SCHEMA, function_name="f"))
    assert result.arguments == {"answer": "ok" if validate else "bad"}
    assert len(bodies) == (2 if validate else 1)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["auto", "required"])
@pytest.mark.parametrize("multiple", [False, True])
async def test_catalog_tool_choice_reaches_wire(monkeypatch, preference, multiple):
    monkeypatch.setenv("LLM_OPTIONS", json.dumps({"tool_choice_pref": preference}))
    client = OpenAIClient(model="m", base_url="https://example.invalid/v1", api_key="k")
    bodies = []

    async def post(body):
        bodies.append(body)
        return response("ok")

    client._post = post
    await client.generate_structured(LLMRequest(
        system="s", user="u", function_name="f", schema=SCHEMA,
        tools=[{"name": "f", "parameters": SCHEMA}] if multiple else None,
    ))
    assert len(bodies) == 1
    assert bodies[0]["tool_choice"] == preference
    await client.aclose()


@pytest.mark.asyncio
async def test_gigachat_stream_uses_configured_reasoning_field(monkeypatch):
    monkeypatch.setenv("LLM_OPTIONS", '{"reasoning_field":"custom_thought"}')
    client = GigaChatAsyncClient(token="dummy", model="m")
    payload = {"choices": [{"delta": {"custom_thought": "thought", "reasoning": "wrong"}, "finish_reason": "stop"}]}
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n")))
    try:
        chunks = [chunk async for chunk in client.generate_stream(LLMRequest(system="s", user="u"))]
        assert "".join(chunk.delta_reasoning for chunk in chunks) == "thought"
    finally:
        await client.aclose()
