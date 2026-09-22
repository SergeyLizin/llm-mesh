"""Tests for public capability protocols and provider-neutral helpers."""

from __future__ import annotations

import logging

import pytest

from llm_mesh._common import build_text_messages, post_with_length_retry
from llm_mesh.gigachat.client import GigaChatAsyncClient
from llm_mesh.gigachat.batch import BatchingLLMClient
from llm_mesh.openai.client import OpenAIClient
from llm_mesh.protocol import BatchLLMClient, LLMClient, TextGenerator
from llm_mesh.types import LLMRequest


def _request(**updates: object) -> LLMRequest:
    values = {
        "system": "SYS",
        "user": "NOW",
        "schema": {"type": "object"},
        "function_name": "emit",
        "history": [
            {"role": "user", "content": "OLD-Q"},
            {"role": "assistant", "content": "OLD-A"},
            {"role": "developer", "content": "DROP"},
        ],
    }
    values.update(updates)
    return LLMRequest(**values)


def test_protocols_are_capability_split_for_batch_clients() -> None:
    """The batch protocol must not promise unsupported streaming methods."""
    assert "generate_text" in TextGenerator.__dict__
    assert "generate_stream" in LLMClient.__dict__ or any(
        "generate_stream" in base.__dict__ for base in LLMClient.__mro__
    )
    assert "generate_stream" not in BatchLLMClient.__dict__
    assert not hasattr(BatchingLLMClient, "generate_stream")


def test_common_text_messages_filter_history_roles() -> None:
    messages = build_text_messages(_request())
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in messages] == [
        "SYS",
        "OLD-Q",
        "OLD-A",
        "NOW",
    ]


def test_common_text_messages_carry_tool_turns() -> None:
    """Forward assistant tool calls and linked tool results together so strict providers receive
    consistent history.
    """
    tool_calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "validate_record", "arguments": "{}"},
    }]
    messages = build_text_messages(_request(history=[
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
    ]))
    assert [m["role"] for m in messages] == ["system", "assistant", "tool", "user"]
    assert messages[1]["tool_calls"] == tool_calls
    assert messages[2]["tool_call_id"] == "c1"
    assert messages[2]["content"] == "OBS"


def test_common_text_messages_drop_tool_turns_when_unsupported() -> None:
    """When tool turns are unsupported, drop both the assistant call and its result to avoid
    orphaned history.
    """
    messages = build_text_messages(
        _request(history=[
            {"role": "user", "content": "OLD-Q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
        ]),
        tool_turns=False,
    )
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert all("tool_calls" not in m for m in messages)


def test_structured_history_included_both_providers() -> None:
    """Include history in structured requests for both OpenAI and GigaChat legacy clients, matching
    their text-generation paths.
    """
    request = _request()
    openai = OpenAIClient(model="m", base_url="https://example.test/v1", api_key="k")
    openai_body = openai._fc_body(
        request,
        request.function_name,
        request.schema_ or {},
        {"type": "function", "function": {"name": request.function_name}},
    )
    gigachat = GigaChatAsyncClient(token="t")
    gigachat_body = gigachat._build_body(request)

    expected_roles = ["system", "user", "assistant", "user"]
    assert [message["role"] for message in openai_body["messages"]] == expected_roles
    assert [message["role"] for message in gigachat_body["messages"]] == expected_roles


@pytest.mark.asyncio
async def test_length_retry_does_not_translate_provider_exception() -> None:
    sentinel = RuntimeError("provider-specific")

    async def post(_: dict[str, object]) -> dict[str, object]:
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        await post_with_length_retry(
            {"max_tokens": 10},
            post=post,
            payload_of=lambda result: result,
            retries=2,
            next_max_tokens=lambda current: current * 2,
            provider="test",
            logger=logging.getLogger(__name__),
        )
    assert caught.value is sentinel
