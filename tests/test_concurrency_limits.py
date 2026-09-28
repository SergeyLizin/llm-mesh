"""Concurrency limits shared per (base URL, credential) across clients."""

from __future__ import annotations

import asyncio
import logging

import pytest

from llm_mesh.anthropic import AnthropicClient
from llm_mesh.concurrency import limit_scope, scope_limit
from llm_mesh.gemini import GeminiClient
from llm_mesh.gigachat import GigaChatAsyncClient
from llm_mesh.openai import OpenAIClient

BASE = "https://gw.example/v1"


@pytest.fixture(autouse=True)
def _no_env_limit(monkeypatch):
    monkeypatch.delenv("LLM_MAX_CONCURRENT", raising=False)


def _openai(model="m", key="k1", base=BASE, **kw):
    return OpenAIClient(model=model, base_url=base, api_key=key, **kw)


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: OpenAIClient(model="m", base_url=BASE, api_key="k", **kw),
        lambda **kw: AnthropicClient(model="m", api_key="k", **kw),
        lambda **kw: GeminiClient(model="m", api_key="k", **kw),
        lambda **kw: GigaChatAsyncClient(token="t", **kw),
    ],
    ids=["openai", "anthropic", "gemini", "gigachat"],
)
def test_constructor_argument_and_precedence(monkeypatch, factory):
    assert factory(max_concurrent=4)._max_concurrent == 4
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "7")
    assert factory()._max_concurrent == 7  # env when no argument
    assert factory(max_concurrent=2)._max_concurrent == 2  # argument wins over env


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_invalid_argument_rejected(bad):
    with pytest.raises(ValueError, match="max_concurrent"):
        _openai(max_concurrent=bad)


def test_same_endpoint_and_key_share_one_semaphore():
    chat = _openai(model="chat", max_concurrent=3)
    embed = _openai(model="embed", max_concurrent=3)
    other_key = _openai(model="chat", key="k2", max_concurrent=3)
    other_host = _openai(model="chat", base="https://other.example/v1", max_concurrent=3)

    async def run():
        sem = chat._ensure_semaphore()
        assert embed._ensure_semaphore() is sem
        assert other_key._ensure_semaphore() is not sem
        assert other_host._ensure_semaphore() is not sem

    asyncio.run(run())


def test_trailing_slash_and_case_are_same_endpoint():
    a = _openai(base="https://GW.example/v1/", max_concurrent=2)
    b = _openai(base="https://gw.example/v1", max_concurrent=2)
    assert a._limit_scope == b._limit_scope


def test_smallest_limit_wins_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_mesh.concurrency"):
        _openai(max_concurrent=8)
        _openai(max_concurrent=3)
    assert scope_limit(limit_scope(BASE, "k1")) == 3
    assert "different max_concurrent" in caplog.text

    async def run():
        return _openai(max_concurrent=8)._ensure_semaphore()

    sem = asyncio.run(run())
    assert sem._value == 3


def test_client_without_own_limit_joins_endpoint_limit():
    limited = _openai(max_concurrent=2)
    unlimited = _openai(model="other")
    assert unlimited._max_concurrent is None

    async def run():
        assert unlimited._ensure_semaphore() is limited._ensure_semaphore()

    asyncio.run(run())


def test_no_limit_anywhere_means_no_semaphore():
    client = _openai()
    assert client._ensure_semaphore() is None  # no loop required without a limit


def test_credential_is_not_kept_in_scope():
    scope = limit_scope(BASE, "super-secret-key")
    assert "super-secret" not in scope and len(scope) == 32


def test_semaphore_is_per_event_loop():
    client = _openai(max_concurrent=2)

    async def get():
        return client._ensure_semaphore()

    first = asyncio.run(get())
    second = asyncio.run(get())
    assert first is not second
    assert first._value == second._value == 2


def test_limit_holds_across_clients_of_one_endpoint(monkeypatch):
    """Two clients (chat + embeddings) of one key never exceed the shared limit together."""
    active = 0
    peak = 0

    async def fake_post(self, url, body, *, what):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"data": [{"index": i, "embedding": [0.0]} for i in range(len(body["input"]))]}

    monkeypatch.setattr(OpenAIClient, "_post_json", fake_post)
    a = _openai(model="emb-a", max_concurrent=2)
    b = _openai(model="emb-b", max_concurrent=2)

    async def run():
        await asyncio.gather(*(c.embed(["x"]) for c in (a, b) for _ in range(5)))

    asyncio.run(run())
    assert peak == 2
