"""OpenAI count_tokens uses tiktoken and does not call the gateway."""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_mesh import OpenAIClient
from llm_mesh.gigachat.batch import BatchingLLMClient
from llm_mesh.models_catalog import make_client
from llm_mesh.openai.batch import OpenAIBatchClient
from llm_mesh.openai.tokenize import encoding_name
from llm_mesh.probe import ProbeKind, _probe_for
from llm_mesh.types import LLMValidationError


def _client(**kwargs) -> OpenAIClient:
    defaults = {
        "model": "gpt-4o",
        "base_url": "https://host/v1",
        "api_key": "k",
    }
    defaults.update(kwargs)
    return OpenAIClient(**defaults)


def test_known_models_select_their_encoding() -> None:
    assert encoding_name("gpt-4o", None) == "o200k_base"
    assert encoding_name("gpt-4", None) == "cl100k_base"
    assert encoding_name("Qwen/Qwen3", "cl100k_base") == "cl100k_base"


def test_unknown_model_is_not_counted_with_a_stand_in() -> None:
    with pytest.raises(LLMValidationError, match="Qwen"):
        encoding_name("Qwen/Qwen3", None)
    with pytest.raises(LLMValidationError, match="unknown tiktoken encoding"):
        encoding_name("gpt-4o", "not-an-encoding")


@pytest.mark.asyncio
async def test_count_tokens_matches_the_model_encoding() -> None:
    import tiktoken

    text = "tokenizer"
    client = _client()
    try:
        assert await client.count_tokens([]) == []
        counts = await client.count_tokens([text, ""], model="gpt-4")
        encoding = tiktoken.get_encoding("cl100k_base")
        assert counts == [
            len(encoding.encode(text, disallowed_special=())),
            0,
        ]
        # cl100k packs this word into one token; o200k uses two.
        assert counts[0] == 1
        assert await client.count_tokens([text]) == [2]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_explicit_encoding_wins_over_the_model_name() -> None:
    import tiktoken

    text = "hello tokenizer"
    client = _client(model="gpt-4o", tiktoken_encoding="cl100k_base")
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        assert await client.count_tokens([text], model="gpt-4o") == [
            len(encoding.encode(text, disallowed_special=())),
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_special_token_text_is_counted() -> None:
    import tiktoken

    text = "<|endoftext|>"
    client = _client()
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        assert await client.count_tokens([text]) == [
            len(encoding.encode(text, disallowed_special=())),
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_count_tokens_does_not_call_the_gateway() -> None:
    route = respx.post("https://host/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={}),
    )
    client = _client()
    try:
        await client.count_tokens(["budget check"])
        assert route.call_count == 0
    finally:
        await client.aclose()


def test_chat_probe_stays_a_generation() -> None:
    client = _client()
    assert _probe_for(client) is ProbeKind.GENERATE_TEXT


def test_catalog_route_sets_the_encoding(monkeypatch) -> None:
    import os

    from llm_mesh.models_catalog import _MANAGED_ENV

    for key in _MANAGED_ENV:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    client = make_client({
        "id": "embedder",
        "kind": "openai",
        "model": "Qwen/Qwen3",
        "base_url": "https://example.test/v1",
        "api_key": "k",
        "tiktoken_encoding": "o200k_base",
    })
    assert isinstance(client, OpenAIClient)
    assert client._tiktoken_encoding == "o200k_base"


@pytest.mark.asyncio
async def test_batch_wrapper_counts_on_the_inner_client() -> None:
    import tiktoken

    openai = _client(tiktoken_encoding="o200k_base")
    wrapped = BatchingLLMClient(OpenAIBatchClient(client=openai), model="gpt-4o")
    text = "hello tokenizer"
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        assert await wrapped.count_tokens([text]) == [
            len(encoding.encode(text, disallowed_special=())),
        ]
    finally:
        await wrapped.aclose()


def test_count_is_stable_for_a_known_phrase() -> None:
    """Lock one count so a tiktoken upgrade that changes o200k is visible."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    assert len(encoding.encode("hello", disallowed_special=())) == 1
