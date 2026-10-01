"""One queue, several inference servers, each with its own concurrency cap.

A single client already shares a limiter with every other client of the same
endpoint and credential. That does not help when the process has several
servers and each server allows a different number of in-flight calls. Sending
the next call to a fixed server queues it there while another server sits
idle. ``InferencePool`` keeps one FIFO of calls and gives a call to a server
only when that server has a free slot, so no server is asked for more than
its ``max_concurrent``.

A call that cannot run yet stays in the queue. The oldest call that fits a
free server runs first. A call that does not fit (a stream, when that
server's stream slots are full, or a call the server does not implement)
stays queued and does not block a later call that does fit another server.
The slot is held for the whole call, including retries and the rest of a
stream.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Sequence
from typing import Any

from llm_mesh.base import Capability
from llm_mesh.concurrency import (
    REQUESTS,
    STREAMS,
    EndpointLimiter,
    holding_dispatched_slot,
    scope_limit,
    scope_semaphore,
)
from llm_mesh.types import BudgetState, LLMRequest, LLMResponse

__all__ = ["InferencePool"]


# Which capability a pool method requires. A member that does not declare it
# is not given the call, even when it has a free slot.
_METHOD_CAPABILITY: dict[str, Capability] = {
    "generate_text": Capability.TEXT,
    "generate_structured": Capability.STRUCTURED,
    "generate_stream": Capability.STREAM,
    "generate_stream_events": Capability.STREAM_EVENTS,
    "embed": Capability.EMBEDDINGS,
    "rerank": Capability.RERANK,
    "count_tokens": Capability.COUNT_TOKENS,
}


class _Waiter:
    def __init__(
        self,
        stream: bool,
        future: asyncio.Future[None],
        servers: frozenset[int],
    ) -> None:
        self.stream = stream
        self.future = future
        self.servers = servers
        self.grant: _Grant | None = None


class _Grant:
    """One server's slot, already taken. ``release`` returns it."""

    def __init__(
        self,
        client: Any,
        requests: EndpointLimiter,
        streams: EndpointLimiter | None,
    ) -> None:
        self.client = client
        self._requests = requests
        self._streams = streams
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # Reverse of acquire: requests, then streams.
        self._requests.release()
        if self._streams is not None:
            self._streams.release()


class _Server:
    def __init__(self, client: Any, index: int) -> None:
        self.client = client
        self.index = index
        self.scope = client._limit_scope

    def request_limiter(self) -> EndpointLimiter:
        limiter = scope_semaphore(self.scope, REQUESTS)
        if limiter is None:
            raise RuntimeError(
                "llm-mesh: inference server lost its max_concurrent "
                f"(scope {self.scope})"
            )
        return limiter

    def stream_limiter(self) -> EndpointLimiter | None:
        return scope_semaphore(self.scope, STREAMS)

    def spare_requests(self) -> int:
        limiter = self.request_limiter()
        limit = limiter.limit
        if limit is None:
            return 0
        return max(0, limit - limiter.held)


def _endpoint_of(client: Any) -> str:
    for attr in ("_base", "_api_url", "URL"):
        value = getattr(client, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(getattr(client, "_limit_scope", type(client).__name__))


class InferencePool:
    """Dispatch calls from one queue onto inference servers.

    Each member is an llm-mesh client constructed with its own positive
    ``max_concurrent``. Two members must not share an endpoint and credential:
    that is one server, and it already has one limiter. A member with no
    limit is rejected, because the pool would not know when that server is
    full.

    The member is not called except through the pool. A direct call on the
    member uses the same limiter, so it still cannot exceed the server, but
    it skips this queue.

    A call is given only to a member that implements it. ``rerank`` does not
    land on a chat client that has no rerank, and ``tools_required`` does not
    land on a client whose tool loop cannot run it. ``supports`` is true when
    at least one member implements the capability. A call that nobody
    implements fails immediately and does not wait for a slot.
    """

    def __init__(self, clients: Sequence[Any]) -> None:
        if isinstance(clients, (str, bytes)) or not isinstance(clients, Sequence):
            raise TypeError("InferencePool expected a sequence of clients")
        members = list(clients)
        if len(members) < 2:
            raise ValueError("InferencePool needs at least two inference servers")
        seen: list[str] = []
        for client in members:
            scope = getattr(client, "_limit_scope", None)
            if not isinstance(scope, str):
                raise ValueError(
                    "InferencePool member must be an llm-mesh client "
                    "constructed with max_concurrent "
                    f"(got {type(client).__name__})"
                )
            if scope_limit(scope, REQUESTS) is None:
                raise ValueError(
                    "InferencePool member has no max_concurrent; "
                    f"{_endpoint_of(client)} needs its own positive limit"
                )
            if scope in seen:
                raise ValueError(
                    "InferencePool members must be different servers; "
                    f"{_endpoint_of(client)} is already in the pool"
                )
            seen.append(scope)
        self._clients = tuple(members)
        self._servers = tuple(_Server(client, index) for index, client in enumerate(members))
        self._waiters: list[_Waiter] = []
        self._pumping = False
        self._pump_again = False
        self._listened: list[weakref.ReferenceType[EndpointLimiter]] = []
        self.PROVIDER = getattr(members[0], "PROVIDER", "pool")

    @property
    def clients(self) -> tuple[Any, ...]:
        """Member clients, in the order they were given."""
        return self._clients

    def __repr__(self) -> str:
        parts = []
        for server in self._servers:
            limit = scope_limit(server.scope, REQUESTS)
            parts.append(f"{_endpoint_of(server.client)} max_concurrent={limit}")
        return "InferencePool(" + ", ".join(parts) + ")"

    @property
    def model(self) -> str:
        """The first member's model. A call uses the model of the member it runs on."""
        client = self._clients[0]
        value = getattr(client, "model", None)
        if isinstance(value, str) and value:
            return value
        value = getattr(client, "_model", None)
        return value if isinstance(value, str) else ""

    def budget_state(self) -> BudgetState:
        """Sum of each member's accumulated spend."""
        usd = 0.0
        rub = 0.0
        tokens = 0
        for client in self._clients:
            state = client.budget_state()
            usd += state.cost_usd
            rub += state.cost_rub
            tokens += state.total_tokens
        return BudgetState(cost_usd=usd, cost_rub=rub, total_tokens=tokens)

    def supports(self, capability: Any) -> bool:
        """True when a member can serve ``capability``.

        The call is then sent only to such members. A capability nobody
        implements is refused before the call waits for a slot.
        """
        return any(_client_supports(client, (capability,)) for client in self._clients)

    def bind_catalog_route(self, route: dict) -> None:
        """Apply catalog embedding and rerank fields to every member."""
        for client in self._clients:
            bind = getattr(client, "bind_catalog_route", None)
            if bind is not None:
                bind(route)
        task = getattr(self._clients[0], "_catalog_task", None)
        if task is not None:
            self._catalog_task = task

    def reset_budget(self) -> None:
        """Zero the accumulated spend on every member."""
        for client in self._clients:
            reset = getattr(client, "reset_budget", None)
            if reset is not None:
                reset()

    async def aclose(self) -> None:
        """Close every member. The first failure is raised after the rest close."""
        errors: list[BaseException] = []
        for client in self._clients:
            close = getattr(client, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        return await self._call("generate_text", request)

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        return await self._call("generate_structured", request)

    async def embed(self, texts: list[str], **kwargs: Any) -> list[Any]:
        return await self._call("embed", texts, **kwargs)

    async def rerank(self, query: str, documents: list[str], **kwargs: Any) -> list[Any]:
        return await self._call("rerank", query, documents, **kwargs)

    async def count_tokens(self, texts: list[str], **kwargs: Any) -> list[int]:
        """Count on one server. The call takes a request slot, like any other."""
        return await self._call("count_tokens", texts, **kwargs)

    def generate_stream(self, request: LLMRequest) -> AsyncIterator[Any]:
        return self._stream("generate_stream", request)

    def generate_stream_events(self, request: LLMRequest) -> AsyncIterator[Any]:
        return self._stream("generate_stream_events", request)

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        grant = await self._acquire(
            stream=False, servers=self._servers_for(name, args, kwargs),
        )
        try:
            with holding_dispatched_slot():
                method = getattr(grant.client, name)
                return await method(*args, **kwargs)
        finally:
            grant.release()

    async def _stream(self, name: str, request: LLMRequest) -> AsyncIterator[Any]:
        grant = await self._acquire(
            stream=True, servers=self._servers_for(name, (request,), {}),
        )
        agen = None
        try:
            agen = getattr(grant.client, name)(request)
            while True:
                try:
                    # The flag covers the member's own acquire, then drops
                    # before the chunk is handed to the caller. Leaving it
                    # set would let unrelated work on this task skip limits.
                    with holding_dispatched_slot():
                        item = await anext(agen)
                except StopAsyncIteration:
                    break
                yield item
        finally:
            try:
                if agen is not None:
                    with holding_dispatched_slot():
                        await agen.aclose()
            finally:
                grant.release()

    def _ensure_listeners(self) -> None:
        self._listened = [ref for ref in self._listened if ref() is not None]
        for server in self._servers:
            self._listen(server.request_limiter())
            streams = server.stream_limiter()
            if streams is not None:
                self._listen(streams)

    def _listen(self, limiter: EndpointLimiter) -> None:
        if any(ref() is limiter for ref in self._listened):
            return
        self._listened.append(weakref.ref(limiter))
        pool = weakref.ref(self)

        def on_release() -> None:
            owner = pool()
            if owner is not None:
                owner._on_release()

        limiter.add_release_listener(on_release)

    def _on_release(self) -> None:
        self._pump()

    def _servers_for(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> frozenset[int]:
        """Indexes of members that implement this call. Empty raises."""
        required = [_METHOD_CAPABILITY[name]]
        if name == "generate_structured":
            request = args[0] if args else kwargs.get("request")
            if getattr(request, "tools_required", False):
                required.append(Capability.TOOLS_REQUIRED)
        servers = frozenset(
            server.index
            for server in self._servers
            if _client_supports(server.client, required)
        )
        if not servers:
            names = ", ".join(capability.value for capability in required)
            raise NotImplementedError(f"InferencePool: no member implements {names}")
        return servers

    async def _acquire(self, *, stream: bool, servers: frozenset[int]) -> _Grant:
        self._ensure_listeners()
        if not self._waiters:
            grant = self._try_grant(stream, servers)
            if grant is not None:
                return grant
        waiter = _Waiter(stream, asyncio.get_running_loop().create_future(), servers)
        self._waiters.append(waiter)
        self._pump()
        try:
            await waiter.future
        except asyncio.CancelledError:
            self._waiters = [item for item in self._waiters if item is not waiter]
            if waiter.grant is not None:
                waiter.grant.release()
            raise
        assert waiter.grant is not None
        return waiter.grant

    def _pump(self) -> None:
        if self._pumping:
            self._pump_again = True
            return
        self._pumping = True
        try:
            while True:
                self._pump_again = False
                self._grant_waiters()
                if not self._pump_again:
                    return
        finally:
            self._pumping = False

    def _grant_waiters(self) -> None:
        pending: list[_Waiter] = []
        for waiter in self._waiters:
            if waiter.future.done():
                continue
            grant = self._try_grant(waiter.stream, waiter.servers)
            if grant is None:
                pending.append(waiter)
                continue
            waiter.grant = grant
            waiter.future.set_result(None)
        self._waiters = pending

    def _try_grant(self, stream: bool, servers: frozenset[int]) -> _Grant | None:
        ranked = sorted(self._servers, key=lambda server: (-server.spare_requests(), server.index))
        for server in ranked:
            if server.index not in servers:
                continue
            streams: EndpointLimiter | None = None
            if stream:
                streams = server.stream_limiter()
                if streams is not None and not streams.try_acquire():
                    continue
            requests = server.request_limiter()
            if not requests.try_acquire():
                if streams is not None:
                    # Speculative. Waking listeners would retry this grant on
                    # the same stack: the request slots are still full, so the
                    # attempt fails again and the event loop never runs.
                    streams.release(notify=False)
                continue
            return _Grant(server.client, requests, streams)
        return None


def _client_supports(client: Any, required: tuple[Any, ...] | list[Any]) -> bool:
    check = getattr(client, "supports", None)
    if check is None:
        return False
    return all(check(capability) for capability in required)
