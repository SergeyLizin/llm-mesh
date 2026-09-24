"""Embeddings: shared parsing, provider wire format, catalog task, and batch lines."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_mesh import (
    AnthropicClient,
    Capability,
    GigaChatClient,
    GeminiClient,
    OpenAIClient,
    QueryInstruction,
)
from llm_mesh.embeddings import (
    checked_dimensions,
    embeddings_url,
    parse_embedding_response,
)
from llm_mesh.gigachat.batch import BatchingLLMClient
from llm_mesh.models_catalog import ModelCatalogError, make_client, route_task
from llm_mesh.openai.batch import (
    EMBEDDINGS_BATCH_ENDPOINT,
    OpenAIBatchClient,
)
from llm_mesh.probe import ProbeKind, _probe_for
from llm_mesh.types import LLMValidationError


def _pin(monkeypatch) -> None:
    import os

    from llm_mesh.models_catalog import _MANAGED_ENV

    for key in _MANAGED_ENV:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.delenv("LLM_BATCH_MODE", raising=False)


def test_query_instruction_empty_is_identity() -> None:
    instruction = QueryInstruction.parse("  ")
    assert instruction.is_empty()
    assert instruction.apply("hello") == "hello"


def test_query_instruction_wraps_the_placeholder() -> None:
    instruction = QueryInstruction.parse("search: {query}")
    assert instruction.apply("tax") == "search: tax"


def test_query_instruction_requires_the_placeholder() -> None:
    with pytest.raises(ValueError, match="query"):
        QueryInstruction.parse("search documents")


def test_embeddings_url_inserts_v1_once() -> None:
    assert embeddings_url("https://host") == "https://host/v1/embeddings"
    assert embeddings_url("https://host/v1") == "https://host/v1/embeddings"
    assert embeddings_url("https://host/v1/embeddings") == "https://host/v1/embeddings"


def test_parse_sorts_by_index_and_rejects_a_short_list() -> None:
    parsed = parse_embedding_response(
        {"data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]},
        2,
    )
    assert [item.dense for item in parsed] == [[1.0], [2.0]]
    with pytest.raises(LLMValidationError, match="2 inputs"):
        parse_embedding_response({"data": [{"index": 0, "embedding": [1.0]}]}, 2)


def test_parse_sparse_shapes_drop_non_positive_weights() -> None:
    parsed = parse_embedding_response(
        {"data": [{
            "index": 0,
            "embedding": [1.0],
            "sparse_embedding": {"indices": [3, 4], "values": [0.5, 0.0]},
        }]},
        1,
        sparse=True,
    )
    assert parsed[0].sparse == {"3": 0.5}
    mapped = parse_embedding_response(
        {"data": [{
            "embedding": [1.0],
            "sparse_embedding": {"tax": 0.2, "noise": -1.0},
        }]},
        1,
        sparse=True,
    )
    assert mapped[0].sparse == {"tax": 0.2}


def test_parse_rejects_a_missing_dense_vector() -> None:
    with pytest.raises(LLMValidationError, match="dense"):
        parse_embedding_response({"data": [{"index": 0}]}, 1)


def test_checked_dimensions_rejects_bool_and_negative() -> None:
    assert checked_dimensions(0) is None
    assert checked_dimensions(None) is None
    with pytest.raises(ValueError):
        checked_dimensions(True)
    with pytest.raises(ValueError, match="negative"):
        checked_dimensions(-1)


@pytest.mark.asyncio
@respx.mock
async def test_openai_embed_orders_results_and_wraps_queries() -> None:
    client = OpenAIClient(model="embed", base_url="https://host/v1", api_key="k")
    client._query_instruction = QueryInstruction.parse("q: {query}")
    seen: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [2.0], "sparse_embedding": {"a": 0.4}},
            {"index": 0, "embedding": [1.0], "sparse_embedding": {"b": 0.1}},
        ]})

    respx.post("https://host/v1/embeddings").mock(side_effect=_capture)
    result = await client.embed(
        ["one", "two"], task="query", dimensions=8, sparse=True,
    )
    assert seen["body"] == {
        "model": "embed",
        "input": ["q: one", "q: two"],
        "encoding_format": "float",
        "return_sparse": True,
        "dimensions": 8,
    }
    assert [item.dense for item in result] == [[1.0], [2.0]]
    assert result[0].sparse == {"b": 0.1}
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_embed_empty_input_does_not_call() -> None:
    client = OpenAIClient(model="embed", base_url="https://host/v1", api_key="k")
    assert await client.embed([]) == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gigachat_embed_refreshes_on_401_and_refuses_sparse() -> None:
    client = GigaChatClient(token="t", api_url="https://giga.test/api/v1", model="Embeddings")
    calls = {"n": 0}

    async def _refresh() -> str:
        client._token = "t2"
        return "t2"

    client._refresh_token = _refresh  # type: ignore[method-assign]

    def _capture(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, text="expired")
        assert request.headers["Authorization"] == "Bearer t2"
        body = json.loads(request.content)
        assert body == {"model": "Embeddings", "input": ["doc"]}
        assert "dimensions" not in body
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5]}]})

    respx.post("https://giga.test/api/v1/embeddings").mock(side_effect=_capture)
    result = await client.embed(["doc"])
    assert result[0].dense == [0.5]
    with pytest.raises(LLMValidationError, match="sparse"):
        await client.embed(["doc"], sparse=True)
    with pytest.raises(LLMValidationError, match="dimensions"):
        await client.embed(["doc"], dimensions=32)
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_embed_sends_task_type_and_keeps_order() -> None:
    client = GeminiClient(
        model="text-embedding-004",
        api_key="k",
        base_url="https://gemini.test/v1beta",
    )
    client._query_instruction = QueryInstruction.parse("q: {query}")
    seen: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [
            {"values": [1.0, 0.0]},
            {"values": [0.0, 1.0]},
        ]})

    respx.post(
        "https://gemini.test/v1beta/models/text-embedding-004:batchEmbedContents"
    ).mock(side_effect=_capture)
    result = await client.embed(["tax", "fee"], task="query", dimensions=2)
    request = seen["body"]["requests"][0]
    assert request["model"] == "models/text-embedding-004"
    assert request["taskType"] == "RETRIEVAL_QUERY"
    assert request["content"]["parts"][0]["text"] == "q: tax"
    assert seen["body"]["requests"][1]["content"]["parts"][0]["text"] == "q: fee"
    assert request["outputDimensionality"] == 2
    assert [item.dense for item in result] == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(LLMValidationError, match="sparse"):
        await client.embed(["tax"], sparse=True)
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_does_not_embed() -> None:
    client = AnthropicClient(model="m", api_key="k")
    assert not client.supports(Capability.EMBEDDINGS)
    with pytest.raises(NotImplementedError):
        await client.embed(["x"])
    await client.aclose()


def test_route_task_defaults_to_chat_and_rejects_unknown() -> None:
    assert route_task({"id": "m"}) == "chat"
    with pytest.raises(ModelCatalogError, match="embeddings"):
        route_task({"id": "m", "task": "compress"})


def test_embeddings_route_is_not_a_chat_batch(monkeypatch) -> None:
    _pin(monkeypatch)
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    client = make_client({
        "id": "embedder",
        "kind": "openai",
        "model": "text-embedding-3-large",
        "base_url": "https://example.test/v1",
        "api_key": "k",
        "task": "embeddings",
        "query_instruction": "q: {query}",
        "sparse": True,
        "dimensions": 16,
    })
    assert type(client) is OpenAIClient
    assert not isinstance(client, BatchingLLMClient)
    assert client._catalog_task == "embeddings"
    assert client._query_instruction.apply("tax") == "q: tax"
    assert client._embedding_sparse is True
    assert client._embedding_dimensions == 16


def test_unknown_catalog_task_is_a_catalog_error(monkeypatch) -> None:
    _pin(monkeypatch)
    with pytest.raises(ModelCatalogError, match="task"):
        make_client({
            "id": "bad",
            "kind": "openai",
            "model": "m",
            "base_url": "https://example.test/v1",
            "api_key": "k",
            "task": "compress",
        })


def test_probe_of_an_embedding_client_calls_embed() -> None:
    client = OpenAIClient(model="embed", base_url="https://host/v1", api_key="k")
    client._catalog_task = "embeddings"
    assert _probe_for(client) is ProbeKind.EMBED
    anthropic = AnthropicClient(model="m", api_key="k")
    anthropic._catalog_task = "embeddings"
    assert _probe_for(anthropic) is ProbeKind.EMBED


def test_embedding_batch_lines_use_the_embeddings_endpoint() -> None:
    openai = OpenAIClient(model="embed", base_url="https://host/v1", api_key="k")
    openai._query_instruction = QueryInstruction.parse("q: {query}")
    batch = OpenAIBatchClient(client=openai)
    lines = batch.build_embedding_lines(["tax"], task="query", dimensions=4, sparse=True)
    assert lines[0]["url"] == EMBEDDINGS_BATCH_ENDPOINT
    assert lines[0]["body"]["input"] == ["q: tax"]
    assert lines[0]["body"]["return_sparse"] is True
    assert lines[0]["body"]["dimensions"] == 4


@pytest.mark.asyncio
async def test_create_batch_sends_the_embeddings_endpoint() -> None:
    openai = OpenAIClient(model="embed", base_url="https://host/v1", api_key="k")
    batch = OpenAIBatchClient(client=openai)
    captured: dict = {}

    async def _request(method: str, path: str, *, context: str, **kwargs):
        captured["json"] = kwargs["json"]
        return httpx.Response(200, json={"id": "batch_1"})

    batch._request = _request  # type: ignore[method-assign]
    created = await batch.create_batch("file_1", endpoint=EMBEDDINGS_BATCH_ENDPOINT)
    assert created["id"] == "batch_1"
    assert captured["json"]["endpoint"] == "/v1/embeddings"
    assert captured["json"]["input_file_id"] == "file_1"
    await openai.aclose()
