"""Gateway /score, llama.cpp /v1/rerank, and a local cross-encoder."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh import AnthropicClient, Capability, LocalCrossEncoder, OpenAIClient
from llm_mesh.models_catalog import ModelCatalogError, make_client
from llm_mesh.probe import ProbeKind, _probe_for
from llm_mesh.rerank import llama_rerank_url, score_url
from llm_mesh.types import LLMAuthError, LLMValidationError


def _client(**kwargs) -> OpenAIClient:
    defaults = {
        "model": "bge-reranker",
        "base_url": "https://host/v1",
        "api_key": "k",
    }
    defaults.update(kwargs)
    return OpenAIClient(**defaults)


def test_score_url_drops_v1() -> None:
    assert score_url("https://host/v1") == "https://host/score"
    assert score_url("https://host") == "https://host/score"
    assert score_url("https://host/score") == "https://host/score"
    assert llama_rerank_url("https://host/v1") == "https://host/v1/rerank"
    assert llama_rerank_url("https://host") == "https://host/v1/rerank"


@pytest.mark.asyncio
@respx.mock
async def test_score_orders_by_score_and_sends_text_fields() -> None:
    seen: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": [
            {"index": 0, "score": 0.1},
            {"index": 1, "score": 0.9},
        ]})

    respx.post("https://host/score").mock(side_effect=_capture)
    client = _client()
    try:
        hits = await client.rerank("q", ["first", "second"])
    finally:
        await client.aclose()
    assert seen["url"] == "https://host/score"
    assert seen["body"] == {
        "model": "bge-reranker",
        "encoding_format": "float",
        "text_1": "q",
        "text_2": ["first", "second"],
    }
    assert [(hit.index, hit.text, hit.score) for hit in hits] == [
        (1, "second", 0.9),
        (0, "first", 0.1),
    ]


@pytest.mark.asyncio
@respx.mock
async def test_llama_protocol_posts_query_and_documents() -> None:
    seen: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.2},
        ]})

    respx.post("https://host/v1/rerank").mock(side_effect=_capture)
    client = _client(rerank_protocol="llama")
    try:
        hits = await client.rerank("q", ["first", "second"])
    finally:
        await client.aclose()
    assert seen["body"]["query"] == "q"
    assert seen["body"]["documents"] == ["first", "second"]
    assert hits[0].text == "second"
    assert hits[0].score == 0.8


@pytest.mark.asyncio
@respx.mock
async def test_short_score_list_raises() -> None:
    respx.post("https://host/score").mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "score": 0.5}]}),
    )
    client = _client()
    try:
        with pytest.raises(LLMValidationError, match="2 documents"):
            await client.rerank("q", ["first", "second"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_auth_failure_does_not_return_the_input_order() -> None:
    respx.post("https://host/score").mock(return_value=httpx.Response(401, text="no"))
    client = _client()
    try:
        with pytest.raises(LLMAuthError):
            await client.rerank("q", ["first", "second"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_min_score_and_top_k() -> None:
    respx.post("https://host/score").mock(
        return_value=httpx.Response(200, json={"data": [
            {"index": 0, "score": 0.2},
            {"index": 1, "score": 0.4},
            {"index": 2, "score": 0.9},
        ]}),
    )
    client = _client()
    try:
        kept = await client.rerank("q", ["a", "b", "c"], min_score=0.3, top_k=1)
        assert [(hit.text, hit.score) for hit in kept] == [("c", 0.9)]
        below = await client.rerank("q", ["a", "b", "c"], min_score=1.5)
        assert [hit.text for hit in below] == ["c", "b", "a"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_local_scorer_orders_pairs() -> None:
    encoder = LocalCrossEncoder(
        "test-model",
        scorer=lambda query, documents: [0.1 if text == "a" else 0.9 for text in documents],
    )
    hits = await encoder.rerank("q", ["a", "b"])
    assert [hit.text for hit in hits] == ["b", "a"]
    with pytest.raises(LLMValidationError, match="test-model"):
        await encoder.rerank("q", ["a"], model="other")


@pytest.mark.asyncio
async def test_local_model_requires_sentence_transformers(monkeypatch) -> None:
    real_import = __import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _blocked)
    encoder = LocalCrossEncoder("BAAI/bge-reranker-v2-m3")
    with pytest.raises(LLMValidationError, match="sentence-transformers"):
        await encoder.rerank("q", ["a"])


@pytest.mark.asyncio
async def test_anthropic_does_not_rerank() -> None:
    client = AnthropicClient(model="m", api_key="k")
    assert not client.supports(Capability.RERANK)
    with pytest.raises(NotImplementedError):
        await client.rerank("q", ["a"])
    await client.aclose()


def test_rerank_route_is_not_a_chat_batch(monkeypatch) -> None:
    import os

    from llm_mesh.models_catalog import _MANAGED_ENV

    for key in _MANAGED_ENV:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    client = make_client({
        "id": "reranker",
        "kind": "openai",
        "model": "BAAI/bge-reranker-v2-m3",
        "base_url": "https://example.test/v1",
        "api_key": "k",
        "task": "rerank",
        "rerank_protocol": "llama",
        "rerank_top_k": 4,
        "rerank_min_score": 0.2,
    })
    assert type(client) is OpenAIClient
    assert client._catalog_task == "rerank"
    assert client._rerank_protocol == "llama"
    assert client._rerank_top_k == 4
    assert client._rerank_min_score == 0.2
    assert _probe_for(client) is ProbeKind.RERANK
    with pytest.raises(ModelCatalogError, match="rerank"):
        make_client({
            "id": "nope",
            "kind": "gigachat",
            "model": "GigaChat-2",
            "api_key": "k",
            "scope": "GIGACHAT_API_PERS",
            "task": "rerank",
        })
