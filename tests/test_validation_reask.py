"""One corrective retry after a confirmed schema violation.

Measurement modes keep the first response. GigaChat does not validate
arguments, so it never re-asks.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh import (
    AnthropicClient,
    Budget,
    GeminiClient,
    GigaChatClient,
    LLMRequest,
    LLMUsage,
    LLMValidationError,
    OpenAIClient,
)
from llm_mesh.hooks import configure_metrics_hook
from llm_mesh._common import merge_usage
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["ok"]}},
    "required": ["answer"],
}


def test_merge_usage_adds_tokens_and_prefers_the_second_cache():
    merged = merge_usage(
        LLMUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11,
                 reasoning_tokens=2, cache_hit_tokens=7, cache_miss_tokens=3,
                 cost_usd=0.1),
        LLMUsage(prompt_tokens=20, completion_tokens=4, total_tokens=24,
                 reasoning_tokens=1, cache_hit_tokens=4, cost_usd=0.2),
    )
    assert merged.prompt_tokens == 30
    assert merged.completion_tokens == 5
    assert merged.total_tokens == 35
    assert merged.reasoning_tokens == 3
    assert merged.cache_hit_tokens == 4
    assert merged.cache_miss_tokens == 3
    assert merged.cost_usd == pytest.approx(0.3)


def _openai_tool(answer: str, *, prompt: int, cache: int | None = None) -> dict:
    usage: dict = {
        "prompt_tokens": prompt,
        "completion_tokens": 2,
        "total_tokens": prompt + 2,
    }
    if cache is not None:
        usage["prompt_cache_hit_tokens"] = cache
    return {
        "model": "m",
        "choices": [{"message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": json.dumps({"answer": answer})},
            }],
        }, "finish_reason": "stop"}],
        "usage": usage,
    }


def _request() -> LLMRequest:
    return LLMRequest(system="s", user="ask", schema=SCHEMA, function_name="f")


@respx.mock
@pytest.mark.asyncio
async def test_openai_reask_returns_corrected_args_and_summed_usage():
    route = respx.post("https://host/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, json=_openai_tool("WRONG", prompt=10, cache=7)),
        httpx.Response(200, json=_openai_tool("ok", prompt=20, cache=4)),
    ])
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        response = await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert route.call_count == 2
    body = json.loads(route.calls[1].request.content)
    blob = json.dumps(body["messages"])
    assert "WRONG" in blob
    assert "failed schema validation" in blob
    assert response.arguments == {"answer": "ok"}
    assert response.validation_reasks == 1
    assert response.usage.prompt_tokens == 30
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 34
    assert response.usage.cache_hit_tokens == 4
    assert client._last_served_tier == "strict"


@respx.mock
@pytest.mark.asyncio
async def test_openai_second_failure_uses_the_existing_fallback():
    route = respx.post("https://host/v1/chat/completions").mock(side_effect=[
        httpx.Response(200, json=_openai_tool("WRONG", prompt=1)),
        httpx.Response(200, json=_openai_tool("ALSO_BAD", prompt=1)),
        httpx.Response(200, json=_openai_tool("STILL", prompt=1)),
        httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": json.dumps({"answer": "ok"})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }),
    ])
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        response = await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert response.arguments == {"answer": "ok"}
    assert route.call_count == 4
    reask = json.dumps(json.loads(route.calls[1].request.content)["messages"])
    assert "WRONG" in reask and "failed schema validation" in reask
    required = json.loads(route.calls[2].request.content)
    assert required["tool_choice"] == "required"
    assert "failed schema validation" not in json.dumps(required["messages"])


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["no_degrade", "preserve"])
async def test_openai_measurement_modes_do_not_reask(policy: str):
    route = respx.post("https://host/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_tool("WRONG", prompt=1)),
    )
    kwargs = {"no_degrade": True} if policy == "no_degrade" else {"fallback_policy": "preserve"}
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1", **kwargs)
    try:
        with pytest.raises(LLMValidationError):
            await client.generate_structured(_request())
    finally:
        await client.aclose()
    bodies = [json.dumps(json.loads(call.request.content)) for call in route.calls]
    assert all("failed schema validation" not in body for body in bodies)
    if policy == "preserve":
        assert route.call_count == 1


def _anthropic(answer: str, *, input_tokens: int) -> dict:
    return {
        "id": "msg",
        "model": "claude",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": input_tokens, "output_tokens": 2},
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "f",
            "input": {"answer": answer},
        }],
    }


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_reask_corrects_and_sums_usage():
    route = respx.post("https://anthropic.test/v1/messages").mock(side_effect=[
        httpx.Response(200, json=_anthropic("WRONG", input_tokens=10)),
        httpx.Response(200, json=_anthropic("ok", input_tokens=6)),
    ])
    client = AnthropicClient(model="claude", api_key="k", base_url="https://anthropic.test")
    try:
        response = await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert route.call_count == 2
    blob = route.calls[1].request.content.decode()
    assert "WRONG" in blob and "failed schema validation" in blob
    assert response.arguments == {"answer": "ok"}
    assert response.validation_reasks == 1
    assert response.usage.prompt_tokens == 16
    assert response.usage.completion_tokens == 4
    assert client._last_served_tier == "tool"


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_preserve_does_not_reask():
    route = respx.post("https://anthropic.test/v1/messages").mock(
        return_value=httpx.Response(200, json=_anthropic("WRONG", input_tokens=1)),
    )
    client = AnthropicClient(
        model="claude", api_key="k", base_url="https://anthropic.test",
        fallback_policy="preserve",
    )
    try:
        with pytest.raises(LLMValidationError):
            await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert route.call_count == 1


def _gemini(answer: str, *, prompt: int) -> dict:
    return {
        "responseId": "r",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{
            "content": {"role": "model", "parts": [{
                "functionCall": {"name": "f", "args": {"answer": answer}},
            }]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": prompt,
            "candidatesTokenCount": 2,
            "totalTokenCount": prompt + 2,
        },
    }


@respx.mock
@pytest.mark.asyncio
async def test_gemini_reask_corrects_and_sums_usage():
    model = "gemini-2.5-flash"
    url = f"https://gemini.test/v1beta/models/{model}:generateContent"
    route = respx.post(url).mock(side_effect=[
        httpx.Response(200, json=_gemini("WRONG", prompt=8)),
        httpx.Response(200, json=_gemini("ok", prompt=5)),
    ])
    client = GeminiClient(model=model, api_key="k", base_url="https://gemini.test")
    try:
        response = await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert route.call_count == 2
    blob = route.calls[1].request.content.decode()
    assert "WRONG" in blob and "failed schema validation" in blob
    assert response.arguments == {"answer": "ok"}
    assert response.validation_reasks == 1
    assert response.usage.prompt_tokens == 13
    assert response.usage.completion_tokens == 4


def _gemini_text(text: str, *, prompt: int) -> dict:
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": prompt,
            "candidatesTokenCount": 2,
            "totalTokenCount": prompt + 2,
        },
    }


@respx.mock
@pytest.mark.asyncio
async def test_gemini_unparseable_reask_records_both_attempts():
    seen = []
    configure_metrics_hook(on_call=seen.append)
    model = "gemini-2.5-flash"
    url = f"https://gemini.test/v1beta/models/{model}:generateContent"
    route = respx.post(url).mock(side_effect=[
        httpx.Response(200, json=_gemini_text('{"answer":"WRONG"}', prompt=8)),
        httpx.Response(200, json=_gemini_text("not-json", prompt=5)),
    ])
    client = GeminiClient(
        model=model, api_key="k", base_url="https://gemini.test",
        budget=Budget(max_total_tokens=1000),
    )
    try:
        with pytest.raises(LLMValidationError, match="unparseable JSON"):
            await client.generate_structured(LLMRequest(
                system="s", user="ask", schema=SCHEMA, function_name="f",
                mode="json_schema",
            ))
    finally:
        await client.aclose()
    assert route.call_count == 2
    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].usage is not None
    assert seen[0].usage.prompt_tokens == 13
    assert seen[0].usage.total_tokens == 17
    assert client.budget_state().total_tokens == 17


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_does_not_reask_invalid_arguments():
    route = respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {
                "function_call": {"name": "f", "arguments": json.dumps({"answer": "WRONG"})},
            }}],
            "model": "GigaChat",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }),
    )
    client = GigaChatClient(token="dummy", model="GigaChat")
    try:
        response = await client.generate_structured(_request())
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert response.arguments == {"answer": "WRONG"}
    assert response.validation_reasks == 0
    assert b"failed schema validation" not in route.calls[0].request.content
