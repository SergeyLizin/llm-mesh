"""Per-client spend ceilings. Currencies are not converted."""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_mesh import (
    Budget,
    LLMBudgetExceeded,
    LLMRequest,
    LLMValidationError,
    OpenAIClient,
)
from llm_mesh.openai.batch import OpenAIBatchClient
from llm_mesh.types import LLMStreamChunk, LLMUsage

URL = "https://host/v1/chat/completions"


def _request() -> LLMRequest:
    return LLMRequest(system="s", user="hello", mode="text")


def _ok(*, cost: float | None = None, cost_rub: float | None = None, total: int = 4) -> dict:
    usage: dict = {"prompt_tokens": 1, "completion_tokens": total - 1, "total_tokens": total}
    if cost is not None:
        usage["cost"] = cost
    if cost_rub is not None:
        usage["cost_rub"] = cost_rub
    return {
        "id": "c",
        "model": "m",
        "choices": [{
            "message": {"role": "assistant", "content": "response"},
            "finish_reason": "stop",
        }],
        "usage": usage,
    }


def _client(budget: Budget) -> OpenAIClient:
    return OpenAIClient(model="m", api_key="k", base_url="https://host/v1", budget=budget)


@respx.mock
@pytest.mark.asyncio
async def test_ceiling_raises_before_http():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok()))
    client = _client(Budget(max_total_tokens=0))
    try:
        with pytest.raises(LLMBudgetExceeded, match="tokens"):
            await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 0
    assert not issubclass(LLMBudgetExceeded, LLMValidationError)


@respx.mock
@pytest.mark.asyncio
async def test_spend_accumulates_and_reset_clears_it():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_ok(cost=0.25, total=4)))
    client = _client(Budget(max_cost_usd=1.0, max_total_tokens=100))
    try:
        await client.generate_text(_request())
        await client.generate_text(_request())
        state = client.budget_state()
        assert state.cost_usd == pytest.approx(0.5)
        assert state.total_tokens == 8
        assert state.cost_rub == 0.0
        with pytest.raises(LLMBudgetExceeded):
            client._budget_ledger.cost_usd = 1.0
            await client.generate_text(_request())
        assert route.call_count == 2
        client.reset_budget()
        assert client.budget_state().cost_usd == 0.0
        assert client.budget_state().total_tokens == 0
        await client.generate_text(_request())
    finally:
        await client.aclose()
    assert route.call_count == 3


@respx.mock
@pytest.mark.asyncio
async def test_currencies_are_independent():
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=_ok(cost_rub=9.0, total=2)),
    )
    client = _client(Budget(max_cost_usd=0.01))
    try:
        await client.generate_text(_request())
        state = client.budget_state()
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert state.cost_rub == pytest.approx(9.0)
    assert state.cost_usd == 0.0


@pytest.mark.asyncio
async def test_stream_usage_is_accumulated(monkeypatch):
    async def stream(self, request):
        request = self._guarded_request(request)
        yield LLMStreamChunk(
            delta_text="hi",
            usage=LLMUsage(total_tokens=5, cost_usd=0.2),
        )

    monkeypatch.setattr(OpenAIClient, "_generate_stream_impl", stream)
    client = _client(Budget(max_total_tokens=100))
    try:
        async for _chunk in client.generate_stream(_request()):
            pass
    finally:
        await client.aclose()
    state = client.budget_state()
    assert state.total_tokens == 5
    assert state.cost_usd == pytest.approx(0.2)


@respx.mock
@pytest.mark.asyncio
async def test_batch_ceiling_raises_before_upload():
    files = respx.post(url__regex=r".*/files$").mock(
        return_value=httpx.Response(500, text="should not be called"),
    )
    openai = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    batch = OpenAIBatchClient(client=openai, budget=Budget(max_total_tokens=0))
    try:
        with pytest.raises(LLMBudgetExceeded, match="tokens"):
            await batch.run_chat_batch([_request()])
    finally:
        await openai.aclose()
        await batch.aclose()
    assert files.call_count == 0
