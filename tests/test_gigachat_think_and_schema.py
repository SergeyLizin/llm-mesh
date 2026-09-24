"""GigaChat think-tag fallback and schema typing. The provider reasoning field stays primary.
Think tags are removed from visible text even when that field is present.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import respx

from llm_mesh.gigachat._common import (
    ReasoningContentParser,
    simplify_schema_for_gigachat,
    split_reasoning_content,
)
from llm_mesh.gigachat.client import GIGACHAT_AUTH_URL, GIGACHAT_BASE_URL, GigaChatClient
from llm_mesh.stream_events import Complete, ContentDelta, ReasoningDelta
from llm_mesh.types import LLMRequest


CREDS = "dGVzdC1jbGllbnQ6dGVzdC1zZWNyZXQ="


def test_schema_infers_property_leaf_and_object_shape():
    """A description-only property becomes a string. A node with properties becomes an object."""
    src = {
        "properties": {
            "name": {"description": "user name"},
            "score": {"enum": [1, 2, 2]},
            "flag": {"enum": [True, False]},
            "label": {"enum": ["a", "a", 1]},
            "tags": {"type": "array"},
        }
    }
    out = simplify_schema_for_gigachat(src)
    assert out["type"] == "object"
    assert out["properties"]["name"]["type"] == "string"
    assert "enum" not in out["properties"]["score"]
    assert out["properties"]["score"]["type"] == "integer"
    assert out["properties"]["flag"]["type"] == "boolean"
    assert "enum" not in out["properties"]["flag"]
    assert out["properties"]["label"] == {"type": "string", "enum": ["a"]}
    assert out["properties"]["tags"]["items"] == {"type": "string"}
    assert src["properties"]["score"]["enum"] == [1, 2, 2]


def test_schema_keeps_ref_resolution_and_union_collapse():
    """Local refs still inline, and oneOf still keeps the first non-null variant."""
    out = simplify_schema_for_gigachat(
        {
            "type": "object",
            "$defs": {"Step": {"type": "object", "properties": {"n": {"type": "integer"}}}},
            "properties": {
                "step": {"$ref": "#/$defs/Step"},
                "kind": {
                    "description": "kind",
                    "oneOf": [{"type": "null"}, {"type": "string", "enum": ["ok", "ok"]}],
                },
            },
        }
    )
    assert out["properties"]["step"]["properties"]["n"]["type"] == "integer"
    assert "$ref" not in out["properties"]["step"]
    kind = out["properties"]["kind"]
    assert "oneOf" not in kind
    assert kind["type"] == "string"
    assert kind["enum"] == ["ok"]
    assert kind["description"] == "kind"


def test_think_blocks_split_from_visible_text():
    visible, reasoning = split_reasoning_content("<think>plan</think>answer")
    assert visible == "answer"
    assert reasoning == "plan"
    visible, reasoning = split_reasoning_content(
        "<THINK>hidden</THINK>shown",
        "from-field",
    )
    assert visible == "shown"
    assert reasoning == "from-field"
    visible, reasoning = split_reasoning_content("plain")
    assert visible == "plain"
    assert reasoning is None


def test_think_parser_holds_a_tag_split_across_deltas():
    parser = ReasoningContentParser()
    first = parser.feed("hello <thi")
    assert first.content == "hello "
    assert first.reasoning_content == ""
    second = parser.feed("nk>secret</thi")
    assert second.content == ""
    assert second.reasoning_content == "secret"
    third = parser.feed("nk>ans")
    assert third.content == "ans"
    assert third.reasoning_content == ""
    tail = ReasoningContentParser()
    assert tail.feed("<thi").content == ""
    flushed = tail.flush()
    assert flushed.content == "<thi"
    assert flushed.reasoning_content == ""


def _sse(*chunks: dict) -> str:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def _auth() -> None:
    respx.post(GIGACHAT_AUTH_URL).mock(return_value=httpx.Response(200, json={"access_token": "t"}))


@respx.mock
def test_generate_text_strips_think_and_keeps_field():
    _auth()
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "content": "<think>secret</think>Hello",
                        "reasoning_content": "from-field",
                    },
                    "finish_reason": "stop",
                }],
                "model": "GigaChat",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        response = await client.generate_text(LLMRequest(system="s", user="u", mode="text"))
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.text == "Hello"
    assert response.reasoning_content == "from-field"


@respx.mock
def test_generate_text_uses_think_when_field_is_absent():
    _auth()
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"content": "<think>plan</think>{\"x\": 1}"},
                    "finish_reason": "stop",
                }],
                "model": "GigaChat",
            },
        )
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        response = await client.generate_text(
            LLMRequest(system="s", user="u", mode="json_schema", schema={"type": "object"})
        )
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.arguments == {"x": 1}
    assert response.reasoning_content == "plan"
    assert "<think>" not in (response.text or "")


@respx.mock
def test_stream_reassembles_a_split_think_tag():
    _auth()
    sse = _sse(
        {"choices": [{"delta": {"content": "Hi <thi"}}]},
        {"choices": [{"delta": {"content": "nk>because</think>!"}, "finish_reason": "stop"}]},
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        chunks = [
            chunk
            async for chunk in client.generate_stream(LLMRequest(system="s", user="u", mode="text"))
        ]
        await client.aclose()
        return chunks

    chunks = asyncio.run(run())
    assert "".join(chunk.delta_text for chunk in chunks) == "Hi !"
    assert "".join(chunk.delta_reasoning for chunk in chunks) == "because"


@respx.mock
def test_stream_events_emit_reasoning_before_the_answer():
    _auth()
    sse = _sse(
        {"choices": [{"delta": {"content": "<think>why</think>ok"}, "finish_reason": "stop"}]},
    )
    respx.post(f"{GIGACHAT_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )

    async def run():
        client = GigaChatClient(credentials=CREDS)
        events = [
            event
            async for event in client.generate_stream_events(
                LLMRequest(system="s", user="u", mode="text")
            )
        ]
        await client.aclose()
        return events

    events = asyncio.run(run())
    assert isinstance(events[0], ReasoningDelta)
    assert events[0].delta_reasoning == "why"
    assert isinstance(events[1], ContentDelta)
    assert events[1].delta_text == "ok"
    assert isinstance(events[-1], Complete)
