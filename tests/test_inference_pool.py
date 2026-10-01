"""One queue across inference servers, each with its own concurrency cap."""

from __future__ import annotations

import asyncio
import json
import os
import threading

import httpx
import pytest
import respx

from llm_mesh.base import Capability
from llm_mesh.concurrency import STREAMS, limit_scope, scope_limit, scope_semaphore
from llm_mesh.gigachat import GigaChatClient
from llm_mesh.models_catalog import (
    ModelCatalogError,
    _MANAGED_ENV,
    _cli,
    _patch_model,
    expand_catalog,
    make_client,
    missing_credentials,
    resolve_route,
    route_env,
)
from llm_mesh.openai import OpenAIClient
from llm_mesh.pool import InferencePool
from llm_mesh.probe import check_client, check_routes
from llm_mesh.types import LLMRequest, LLMResponse


def _client(base: str, limit: int, **kwargs: object) -> OpenAIClient:
    return OpenAIClient(
        model="qwen", base_url=base, api_key="k", max_concurrent=limit, **kwargs,
    )


def _request() -> LLMRequest:
    return LLMRequest(system="s", user="u", mode="text")


@pytest.fixture
def restored_env():
    """make_client writes the route into the process environment."""
    saved = {key: os.environ.get(key) for key in _MANAGED_ENV}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_pool_requires_two_limited_distinct_servers():
    only = _client("http://gpu1:8000/v1", 1)
    with pytest.raises(ValueError, match="at least two"):
        InferencePool([only])
    unlimited = OpenAIClient(model="qwen", base_url="http://gpu2:8000/v1", api_key="k")
    with pytest.raises(ValueError, match="no max_concurrent"):
        InferencePool([only, unlimited])
    again = _client("http://gpu1:8000/v1", 4)
    with pytest.raises(ValueError, match="already in the pool"):
        InferencePool([only, again])


def test_calls_fill_free_slots_and_never_pass_a_server_limit():
    """Limits 2 and 1. The first three calls occupy both servers; a freed slot runs the older waiter."""
    first = _client("http://gpu1:8000/v1", 2)
    second = _client("http://gpu2:8000/v1", 1)
    pool = InferencePool([first, second])
    inflight = {first._base: 0, second._base: 0}
    peak = {first._base: 0, second._base: 0}
    order: list[str] = []
    blocked = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    def install(client: OpenAIClient) -> None:
        async def generate_text(request: LLMRequest) -> str:
            assert client._ensure_semaphore() is None  # the pool already holds the slot
            order.append(client._base)
            inflight[client._base] += 1
            peak[client._base] = max(peak[client._base], inflight[client._base])
            try:
                slot = len(order)
                if slot <= 3:
                    await blocked[slot - 1].wait()
            finally:
                inflight[client._base] -= 1
            return "ok"

        client.generate_text = generate_text  # type: ignore[method-assign]

    install(first)
    install(second)

    async def run() -> None:
        tasks = [asyncio.create_task(pool.generate_text(_request())) for _ in range(4)]
        for _ in range(50):
            if (
                len(order) == 3
                and inflight[first._base] == 2
                and inflight[second._base] == 1
            ):
                break
            await asyncio.sleep(0)
        assert order == [first._base, first._base, second._base]
        blocked[0].set()
        for _ in range(50):
            if len(order) == 4:
                break
            await asyncio.sleep(0)
        assert order[3] == first._base
        assert peak == {first._base: 2, second._base: 1}
        blocked[1].set()
        blocked[2].set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert scope_semaphore(first._limit_scope).held == 0
        assert scope_semaphore(second._limit_scope).held == 0

    asyncio.run(run())


def test_a_cancelled_waiter_does_not_keep_a_slot():
    first = _client("http://gpu1:8000/v1", 1)
    second = _client("http://gpu2:8000/v1", 1)
    pool = InferencePool([first, second])
    gates: list[asyncio.Event] = []

    def install(client: OpenAIClient) -> None:
        async def generate_text(request: LLMRequest) -> str:
            gate = asyncio.Event()
            gates.append(gate)
            await gate.wait()
            return "ok"

        client.generate_text = generate_text  # type: ignore[method-assign]

    install(first)
    install(second)

    async def run() -> None:
        running = [asyncio.create_task(pool.generate_text(_request())) for _ in range(2)]
        for _ in range(50):
            if len(gates) == 2:
                break
            await asyncio.sleep(0)
        cancelled = asyncio.create_task(pool.generate_text(_request()))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert scope_semaphore(first._limit_scope).held == 1
        assert scope_semaphore(second._limit_scope).held == 1
        for gate in gates:
            gate.set()
        await asyncio.gather(*running)
        assert scope_semaphore(first._limit_scope).held == 0
        assert scope_semaphore(second._limit_scope).held == 0

    asyncio.run(run())


def test_stream_uses_the_server_that_still_has_a_stream_slot():
    """A full stream cap on the larger server sends the next stream to the other one."""
    larger = _client("http://gpu1:8000/v1", 2, max_concurrent_streams=1)
    smaller = _client("http://gpu2:8000/v1", 1)
    pool = InferencePool([larger, smaller])
    started: list[str] = []
    release = asyncio.Event()

    def install(client: OpenAIClient) -> None:
        async def generate_stream(request: LLMRequest):
            async with client._stream_slot():
                started.append(client._base)
                yield "chunk"
                await release.wait()

        client.generate_stream = generate_stream  # type: ignore[method-assign]

    install(larger)
    install(smaller)

    async def consume() -> None:
        async for _chunk in pool.generate_stream(_request()):
            pass

    async def run() -> None:
        first = asyncio.create_task(consume())
        for _ in range(50):
            if started:
                break
            await asyncio.sleep(0)
        assert started == [larger._base]
        second = asyncio.create_task(consume())
        for _ in range(50):
            if len(started) == 2:
                break
            await asyncio.sleep(0)
        assert started == [larger._base, smaller._base]
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)

    asyncio.run(run())


def test_a_stream_waits_when_every_request_slot_is_taken():
    """A stream that cannot take a request slot waits. It must not retry on this stack.

    Releasing the speculative stream slot used to wake the pool, which tried
    the same grant again without awaiting. The event loop never ran, so a
    timeout could not interrupt it. The join is the regression lock.
    """
    first = _client("http://gpu1:8000/v1", 1, max_concurrent_streams=1)
    second = _client("http://gpu2:8000/v1", 1, max_concurrent_streams=1)
    pool = InferencePool([first, second])
    release = asyncio.Event()
    entered: list[str] = []

    async def blocking(request: LLMRequest) -> str:
        entered.append("b")
        await release.wait()
        return "ok"

    async def gen(request: LLMRequest):
        entered.append("s")
        yield "x"

    first.generate_text = blocking  # type: ignore[method-assign]
    second.generate_text = blocking  # type: ignore[method-assign]
    first.generate_stream = gen  # type: ignore[method-assign]
    second.generate_stream = gen  # type: ignore[method-assign]

    async def scenario() -> None:
        tasks = [asyncio.create_task(pool.generate_text(_request())) for _ in range(2)]
        for _ in range(50):
            if entered.count("b") == 2:
                break
            await asyncio.sleep(0)
        assert entered == ["b", "b"]
        stream = pool.generate_stream(_request())
        nxt = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.05)
        assert "s" not in entered
        assert scope_semaphore(first._limit_scope, STREAMS).held == 0
        assert scope_semaphore(second._limit_scope, STREAMS).held == 0
        release.set()
        assert await nxt == "x"
        assert entered.count("s") == 1
        await asyncio.gather(*tasks)
        await stream.aclose()

    box: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            asyncio.run(scenario())
        except BaseException as exc:  # noqa: BLE001 - the join reports it
            box["exc"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(2)
    if thread.is_alive():
        pytest.fail("event loop stuck while a stream waited for a request slot")
    if "exc" in box:
        raise box["exc"]


@respx.mock
def test_real_post_does_not_deadlock_on_the_pool_slot():
    """The member must not take a second slot: with max_concurrent=1 that deadlocks."""
    first = _client("http://gpu1:8000/v1", 1)
    second = _client("http://gpu2:8000/v1", 1)
    pool = InferencePool([first, second])
    body = {
        "model": "qwen",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    seen: list[str] = []
    both = asyncio.Event()
    hold = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if len(seen) == 2:
            both.set()
        await hold.wait()
        return httpx.Response(200, json=body)

    respx.post("http://gpu1:8000/v1/chat/completions").mock(side_effect=respond)
    respx.post("http://gpu2:8000/v1/chat/completions").mock(side_effect=respond)

    async def run() -> None:
        tasks = [
            asyncio.create_task(pool.generate_text(_request())),
            asyncio.create_task(pool.generate_text(_request())),
        ]
        await asyncio.wait_for(both.wait(), 2)
        assert seen == [
            "http://gpu1:8000/v1/chat/completions",
            "http://gpu2:8000/v1/chat/completions",
        ]
        hold.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        await pool.aclose()

    asyncio.run(run())


def _pool_route() -> dict:
    return {
        "id": "qwen-local",
        "kind": "openai",
        "model": "qwen",
        "provider": "local",
        "label": "qwen-local (local)",
        "api_key": "k",
        "endpoints": [
            {"base_url": "http://gpu1:8000/v1", "max_concurrent": 4},
            {"base_url": "http://gpu2:8000/v1", "max_concurrent": 2, "max_concurrent_streams": 1},
        ],
    }


def test_catalog_endpoints_build_a_pool(restored_env):
    pool = make_client(_pool_route())
    try:
        assert isinstance(pool, InferencePool)
        assert [client._base for client in pool.clients] == [
            "http://gpu1:8000/v1",
            "http://gpu2:8000/v1",
        ]
        assert [client._max_concurrent for client in pool.clients] == [4, 2]
        assert [client._max_concurrent_streams for client in pool.clients] == [None, 1]
    finally:
        asyncio.run(pool.aclose())


def test_catalog_route_limit_fills_an_endpoint_that_omits_one(restored_env):
    route = _pool_route()
    route["max_concurrent"] = 5
    route["endpoints"] = [
        {"base_url": "http://gpu1:8000/v1"},
        {"base_url": "http://gpu2:8000/v1", "max_concurrent": 2},
    ]
    pool = make_client(route)
    try:
        assert [client._max_concurrent for client in pool.clients] == [5, 2]
    finally:
        asyncio.run(pool.aclose())


def test_catalog_without_endpoints_stays_one_client(restored_env):
    client = make_client({
        "id": "one",
        "kind": "openai",
        "model": "qwen",
        "provider": "local",
        "label": "one (local)",
        "api_key": "k",
        "base_url": "http://gpu1:8000/v1",
        "max_concurrent": 3,
    })
    try:
        assert isinstance(client, OpenAIClient)
        assert client._max_concurrent == 3
    finally:
        asyncio.run(client.aclose())


def test_catalog_rejects_a_pool_that_is_not_two_servers(restored_env):
    route = _pool_route()
    route["base_url"] = "http://gpu1:8000/v1"
    with pytest.raises(ModelCatalogError, match="not both"):
        make_client(route)
    route = _pool_route()
    route["endpoints"] = [
        {"base_url": "http://gpu1:8000/v1", "max_concurrent": 1},
        {"base_url": "http://gpu1:8000/v1", "max_concurrent": 2},
    ]
    with pytest.raises(ModelCatalogError, match="repeats"):
        make_client(route)
    assert missing_credentials({
        "id": "qwen-local",
        "kind": "openai",
        "model": "qwen",
        "endpoints": [
            {"base_url_env": "GPU1_URL"},
            {"base_url": "http://gpu2:8000/v1"},
        ],
    }) == ["GPU1_URL", "endpoints[0].max_concurrent", "endpoints[1].max_concurrent", "LLM_API_KEY"]


def test_env_export_refuses_a_pool(tmp_path, capsys):
    with pytest.raises(ModelCatalogError, match="make_client"):
        route_env(_pool_route())
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps([{
        "id": "qwen-local",
        "kind": "openai",
        "model": "qwen",
        "providers": [{
            "name": "local",
            "api_key": "k",
            "endpoints": _pool_route()["endpoints"],
        }],
    }]), encoding="utf-8")
    assert _cli(["--config", str(catalog), "--env", "qwen-local"]) == 1
    assert "make_client" in capsys.readouterr().err
    assert _cli(["--config", str(catalog), "--json", "qwen-local"]) == 0


def test_batch_mode_refuses_endpoints(monkeypatch, restored_env):
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    with pytest.raises(ModelCatalogError, match="LLM_BATCH_MODE"):
        make_client(_pool_route())


def test_pool_does_not_clear_an_existing_base_url(restored_env):
    os.environ["LLM_BASE_URL"] = "http://keep.example/v1"
    pool = make_client(_pool_route())
    try:
        assert os.environ["LLM_BASE_URL"] == "http://keep.example/v1"
        assert os.environ["LLM_MODEL"] == "qwen"
    finally:
        asyncio.run(pool.aclose())


def test_endpoint_models_build_without_a_parent_model(restored_env):
    os.environ["LLM_BASE_URL"] = "http://keep.example/v1"
    os.environ["LLM_MODEL"] = "keep-model"
    route = {
        "id": "qwen-local",
        "kind": "openai",
        "api_key": "k",
        "endpoints": [
            {"base_url": "http://gpu1:8000/v1", "model": "model-a", "max_concurrent": 1},
            {"base_url": "http://gpu2:8000/v1", "model": "model-b", "max_concurrent": 1},
        ],
    }
    assert missing_credentials(route) == []
    pool = make_client(route)
    try:
        assert [client._model for client in pool.clients] == ["model-a", "model-b"]
        assert pool.model == "model-a"
        assert os.environ["LLM_BASE_URL"] == "http://keep.example/v1"
        assert os.environ["LLM_MODEL"] == "keep-model"
    finally:
        asyncio.run(pool.aclose())
    bare = {
        "id": "qwen-local",
        "kind": "openai",
        "api_key": "k",
        "endpoints": [
            {"base_url": "http://gpu1:8000/v1", "max_concurrent": 1},
            {"base_url": "http://gpu2:8000/v1", "max_concurrent": 1},
        ],
    }
    assert missing_credentials(bare) == ["endpoints[0].model", "endpoints[1].model"]
    with pytest.raises(ModelCatalogError, match="model"):
        make_client(bare)


def test_budget_state_sums_members():
    first = _client("http://gpu1:8000/v1", 1)
    second = _client("http://gpu2:8000/v1", 1)
    pool = InferencePool([first, second])
    first._budget_ledger.cost_usd = 1.5
    first._budget_ledger.total_tokens = 10
    second._budget_ledger.cost_rub = 2.0
    second._budget_ledger.total_tokens = 3
    state = pool.budget_state()
    assert state.cost_usd == 1.5
    assert state.cost_rub == 2.0
    assert state.total_tokens == 13
    pool.reset_budget()
    assert pool.budget_state().total_tokens == 0


@pytest.mark.asyncio
@respx.mock
async def test_check_client_probes_the_pool(restored_env):
    body = {
        "model": "qwen",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    gpu1 = respx.post("http://gpu1:8000/v1/chat/completions").respond(200, json=body)
    gpu2 = respx.post("http://gpu2:8000/v1/chat/completions").respond(200, json=body)
    pool = make_client(_pool_route())
    try:
        result = await check_client(pool, timeout=5)
    finally:
        await pool.aclose()
    assert result.ok is True
    assert result.kind == "openai"
    assert result.model == "qwen"
    assert result.detail == "each member"
    assert gpu1.called
    assert gpu2.called


def _provider_cluster_catalog() -> list[dict]:
    """spare is the first ready provider and is not in the cluster."""
    return [{
        "id": "mix",
        "cluster": ["local", "giga"],
        "providers": [
            {
                "name": "spare",
                "kind": "openai",
                "model": "spare-model",
                "base_url": "http://spare:8000/v1",
                "api_key": "k3",
                "max_concurrent": 1,
            },
            {
                "name": "local",
                "kind": "openai",
                "model": "qwen-a",
                "base_url": "http://gpu1:8000/v1",
                "api_key": "k1",
                "max_concurrent": 2,
                "force_temperature": 0,
                "extra_body": {"top_p": 0.2},
            },
            {
                "name": "giga",
                "kind": "gigachat",
                "model": "GigaChat-2",
                "api_key": "tok",
                "max_concurrent": 1,
                "reasoning_effort": "low",
            },
        ],
    }]


def test_provider_cluster_keeps_each_members_call_parameters(restored_env):
    os.environ["LLM_BASE_URL"] = "http://keep.example/v1"
    os.environ["LLM_MODEL"] = "keep-model"
    routes = expand_catalog(_provider_cluster_catalog())
    route = resolve_route("mix", routes)
    assert route["provider"] == "cluster"
    assert [member["provider"] for member in route["cluster"]] == ["local", "giga"]
    pool = make_client(route)
    try:
        local, giga = pool.clients
        assert isinstance(local, OpenAIClient)
        assert isinstance(giga, GigaChatClient)
        assert local._model == "qwen-a"
        assert local._force_temperature == 0.0
        assert local._extra_body.get("top_p") == 0.2
        assert local._max_concurrent == 2
        assert giga.model == "GigaChat-2"
        assert giga._reasoning_effort == "low"
        assert giga._max_concurrent == 1
        assert os.environ["LLM_BASE_URL"] == "http://keep.example/v1"
        assert os.environ["LLM_MODEL"] == "keep-model"
    finally:
        asyncio.run(pool.aclose())
    # Without cluster, the first ready provider is still the one that is selected.
    plain = _provider_cluster_catalog()
    plain[0].pop("cluster")
    assert resolve_route("mix", expand_catalog(plain))["provider"] == "spare"


def test_named_provider_stays_one_client(restored_env):
    routes = expand_catalog(_provider_cluster_catalog())
    route = resolve_route("mix@local", routes)
    assert route["provider"] == "local"
    client = make_client(route)
    try:
        assert isinstance(client, OpenAIClient)
        assert client._model == "qwen-a"
        assert client._force_temperature == 0.0
    finally:
        asyncio.run(client.aclose())


def test_provider_cluster_flattens_a_members_endpoints(restored_env):
    catalog = _provider_cluster_catalog()
    local = catalog[0]["providers"][1]
    local.pop("base_url")
    local["endpoints"] = [
        {"base_url": "http://gpu1:8000/v1", "max_concurrent": 4},
        {"base_url": "http://gpu2:8000/v1", "max_concurrent": 3},
    ]
    pool = make_client(resolve_route("mix", expand_catalog(catalog)))
    try:
        assert [type(client) for client in pool.clients] == [
            OpenAIClient, OpenAIClient, GigaChatClient,
        ]
        assert [client._max_concurrent for client in pool.clients] == [4, 3, 1]
        assert pool.clients[0]._extra_body.get("top_p") == 0.2
        assert pool.clients[1]._extra_body.get("top_p") == 0.2
        assert pool.clients[2]._reasoning_effort == "low"
    finally:
        asyncio.run(pool.aclose())


def test_provider_cluster_requires_limits_and_known_names(restored_env):
    catalog = _provider_cluster_catalog()
    catalog[0]["providers"][1].pop("max_concurrent")
    routes = expand_catalog(catalog)
    with pytest.raises(ModelCatalogError, match="max_concurrent"):
        make_client(resolve_route("mix", routes))
    with pytest.raises(ModelCatalogError, match="at least two"):
        expand_catalog([{
            "id": "mix",
            "cluster": ["only"],
            "providers": [{"name": "only", "kind": "openai"}],
        }])
    with pytest.raises(ModelCatalogError, match="unknown provider"):
        expand_catalog([{
            "id": "mix",
            "cluster": ["local", "missing"],
            "providers": [
                {"name": "local", "kind": "openai"},
                {"name": "giga", "kind": "gigachat"},
            ],
        }])
    with pytest.raises(ModelCatalogError, match="repeats"):
        expand_catalog([{
            "id": "mix",
            "cluster": ["local", "local"],
            "providers": [
                {"name": "local", "kind": "openai"},
                {"name": "giga", "kind": "gigachat"},
            ],
        }])


def test_provider_cluster_is_not_ready_when_a_member_lacks_credentials():
    catalog = _provider_cluster_catalog()
    catalog[0]["providers"][2].pop("api_key")
    catalog[0]["providers"][2]["api_key_env"] = "GIGA_KEY_UNSET"
    with pytest.raises(ModelCatalogError, match="GIGA_KEY_UNSET"):
        resolve_route("mix", expand_catalog(catalog))


def test_env_export_refuses_a_provider_cluster(tmp_path, capsys):
    routes = expand_catalog(_provider_cluster_catalog())
    with pytest.raises(ModelCatalogError, match="make_client"):
        route_env(resolve_route("mix", routes))
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps(_provider_cluster_catalog()), encoding="utf-8")
    assert _cli(["--config", str(catalog), "--env", "mix"]) == 1
    assert "make_client" in capsys.readouterr().err


def test_batch_mode_refuses_a_provider_cluster(monkeypatch, restored_env):
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    routes = expand_catalog(_provider_cluster_catalog())
    with pytest.raises(ModelCatalogError, match="LLM_BATCH_MODE"):
        make_client(resolve_route("mix", routes))


def _reply(model: str) -> LLMResponse:
    return LLMResponse(model=model, text="ok")


def _giga(url: str, limit: int) -> GigaChatClient:
    return GigaChatClient(
        credentials="tok",
        model="GigaChat-2",
        api_url=url,
        max_concurrent=limit,
        timeout_s=1,
        verify=False,
    )


def test_a_call_runs_only_on_a_member_that_implements_it():
    """GigaChat has more free slots. rerank and tools_required still go to OpenAI."""
    local = _client("http://gpu1:8000/v1", 1)
    giga = _giga("http://giga.example/api", 4)
    pool = InferencePool([local, giga])
    assert pool.supports(Capability.RERANK)
    assert pool.supports(Capability.TOOLS_REQUIRED)
    seen: list[str] = []

    async def rerank(*_args: object, **_kwargs: object) -> list[object]:
        seen.append("rerank")
        return []

    async def structured(request: LLMRequest) -> LLMResponse:
        seen.append("structured")
        assert request.tools_required
        return _reply("qwen")

    local.rerank = rerank  # type: ignore[method-assign]
    local.generate_structured = structured  # type: ignore[method-assign]

    async def run() -> None:
        await pool.rerank("q", ["d"])
        request = _request()
        request.tools_required = True
        await pool.generate_structured(request)

    asyncio.run(run())
    assert seen == ["rerank", "structured"]


def test_an_unimplemented_call_fails_without_waiting():
    pool = InferencePool([
        _giga("http://giga-a.example/api", 1),
        _giga("http://giga-b.example/api", 1),
    ])
    assert not pool.supports(Capability.RERANK)

    async def run() -> None:
        with pytest.raises(NotImplementedError, match="rerank"):
            await asyncio.wait_for(pool.rerank("q", ["d"]), 0.2)

    asyncio.run(run())


def test_a_narrow_call_does_not_block_a_server_that_can_run_another():
    local = _client("http://gpu1:8000/v1", 1)
    giga = _giga("http://giga.example/api", 1)
    pool = InferencePool([local, giga])
    started: list[str] = []
    release_local = asyncio.Event()

    async def local_text(_request: LLMRequest) -> LLMResponse:
        started.append("local-text")
        await release_local.wait()
        return _reply("qwen")

    async def giga_text(_request: LLMRequest) -> LLMResponse:
        started.append("giga-text")
        return _reply("GigaChat-2")

    async def local_rerank(*_args: object, **_kwargs: object) -> list[object]:
        started.append("rerank")
        return []

    local.generate_text = local_text  # type: ignore[method-assign]
    giga.generate_text = giga_text  # type: ignore[method-assign]
    local.rerank = local_rerank  # type: ignore[method-assign]

    async def run() -> None:
        first = asyncio.create_task(pool.generate_text(_request()))
        for _ in range(50):
            if started == ["local-text"]:
                break
            await asyncio.sleep(0)
        assert started == ["local-text"]
        rerank_task = asyncio.create_task(pool.rerank("q", ["d"]))
        await asyncio.sleep(0)
        assert "rerank" not in started
        second = asyncio.create_task(pool.generate_text(_request()))
        for _ in range(50):
            if "giga-text" in started:
                break
            await asyncio.sleep(0)
        assert started == ["local-text", "giga-text"]
        await second
        release_local.set()
        await first
        await rerank_task
        assert started[-1] == "rerank"

    asyncio.run(run())


@pytest.mark.asyncio
async def test_check_client_uses_each_members_own_probe(restored_env):
    """GigaChat is first and has fewer slots. Its probe must still reach GigaChat."""
    catalog = _provider_cluster_catalog()
    catalog[0]["cluster"] = ["giga", "local"]
    pool = make_client(resolve_route("mix", expand_catalog(catalog)))
    seen: list[str] = []

    async def generate_text(_request: LLMRequest) -> LLMResponse:
        seen.append("text")
        return _reply("qwen-a")

    async def count_tokens(_texts: list[str], model: str | None = None) -> list[int]:
        seen.append("count")
        return [1]

    pool.clients[1].generate_text = generate_text  # type: ignore[method-assign]
    pool.clients[0].count_tokens = count_tokens  # type: ignore[method-assign]
    try:
        result = await check_client(pool, timeout=5)
    finally:
        await pool.aclose()
    assert result.ok is True
    assert seen == ["count", "text"]


def test_check_routes_probes_the_cluster_and_keeps_the_environment(restored_env, monkeypatch):
    os.environ["LLM_BASE_URL"] = "http://keep.example/v1"
    os.environ["LLM_MODEL"] = "keep-model"
    seen: list[str] = []

    async def generate_text(self: OpenAIClient, _request: LLMRequest) -> LLMResponse:
        seen.append(self._base)
        return _reply(self._model)

    async def count_tokens(
        self: GigaChatClient, _texts: list[str], model: str | None = None,
    ) -> list[int]:
        seen.append("giga")
        return [1]

    monkeypatch.setattr(OpenAIClient, "generate_text", generate_text)
    monkeypatch.setattr(GigaChatClient, "count_tokens", count_tokens)
    routes = expand_catalog(_provider_cluster_catalog())
    results = check_routes("mix", routes)
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].label == "mix (cluster)"
    assert "http://gpu1:8000/v1" in seen
    assert "http://spare:8000/v1" not in seen
    assert "giga" in seen
    assert os.environ["LLM_BASE_URL"] == "http://keep.example/v1"
    assert os.environ["LLM_MODEL"] == "keep-model"
    seen.clear()
    again = check_routes("mix@cluster", routes)
    assert len(again) == 1 and again[0].ok is True
    assert "http://spare:8000/v1" not in seen
    assert resolve_route("mix@cluster", routes)["provider"] == "cluster"


def test_exclude_providers_updates_cluster():
    original = {
        "id": "mix",
        "cluster": ["a", "b", "c"],
        "providers": [
            {"name": "a", "kind": "openai"},
            {"name": "b", "kind": "openai"},
            {"name": "c", "kind": "openai"},
            {"name": "spare", "kind": "openai"},
        ],
    }
    kept = _patch_model(original, {"id": "mix", "exclude_providers": ["c"]})
    assert kept["cluster"] == ["a", "b"]
    expand_catalog([kept])
    spare = _patch_model(original, {"id": "mix", "exclude_providers": ["spare"]})
    assert spare["cluster"] == ["a", "b", "c"]
    with pytest.raises(ModelCatalogError, match="exclude_providers"):
        _patch_model(original, {"id": "mix", "exclude_providers": ["b", "c"]})
    with pytest.raises(ModelCatalogError, match="which this patch removes"):
        _patch_model(original, {
            "id": "mix",
            "cluster": ["a", "c"],
            "exclude_providers": ["c"],
        })


def test_expand_rejects_duplicate_and_reserved_provider_names():
    with pytest.raises(ModelCatalogError, match="duplicate provider"):
        expand_catalog([{
            "id": "m",
            "providers": [
                {"name": "a", "kind": "openai"},
                {"name": "a", "kind": "openai"},
            ],
        }])
    with pytest.raises(ModelCatalogError, match="cannot be named"):
        expand_catalog([{
            "id": "mix",
            "cluster": ["local", "cluster"],
            "providers": [
                {"name": "local", "kind": "openai"},
                {"name": "cluster", "kind": "gigachat"},
            ],
        }])


def test_a_failed_cluster_build_drops_the_members_limits(restored_env):
    keeper = OpenAIClient(
        model="qwen", base_url="http://gpu1:8000/v1", api_key="k1", max_concurrent=4,
    )
    scope = limit_scope("http://gpu1:8000/v1", "k1")
    assert scope_limit(scope) == 4
    catalog = _provider_cluster_catalog()
    catalog[0]["providers"][1]["max_concurrent"] = 1
    catalog[0]["providers"][2].pop("max_concurrent")
    routes = expand_catalog(catalog)
    with pytest.raises(ModelCatalogError, match="max_concurrent"):
        make_client(resolve_route("mix", routes))
    assert scope_limit(scope) == 4
    asyncio.run(keeper.aclose())
