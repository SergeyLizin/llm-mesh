"""Concurrency limits shared per endpoint and credential.

A provider enforces its concurrency limit per account, not per client object.
Clients that call the same endpoint with the same credential therefore share
one limiter here, whatever their model, label or task (chat, embeddings,
rerank). The scope key is ``(base URL, credential)``; the credential is only
stored as a SHA-256 digest.

Limits:

- A client registers its ``max_concurrent`` for its scope. When clients of one
  scope ask for different limits, the smallest wins and a warning is logged:
  exceeding the lower limit would fail with HTTP 429 anyway.
- A client without its own limit still waits on the scope's limiter when
  another client registered one; the limit belongs to the endpoint.
- The limiter is created lazily in the running event loop. Each loop gets
  its own limiter for a scope: the limit holds per event loop. Applications
  normally drive all clients from a single loop.
- A lowered limit applies to the next acquire, including a limiter that
  already exists. Calls that already hold a slot finish; they are not
  cancelled.

Streams: a scope may also have a smaller ``streams`` limit
(``max_concurrent_streams``). A stream takes a streams slot, then a regular
slot, and holds both until it ends; blocking calls take only the regular
slot. With ``max_concurrent=8`` and ``max_concurrent_streams=5`` at most five
long streams run at once and at least three slots stay free for short calls
(retrieval, embeddings, rerank), while the endpoint still never sees more than
eight requests. The fixed order (streams, then regular) cannot deadlock.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import weakref

logger = logging.getLogger(__name__)

__all__ = [
    "limit_scope",
    "register_limit",
    "reset_limits",
    "scope_limit",
    "scope_semaphore",
    "validate_max_concurrent",
    "EndpointLimiter",
    "REQUESTS",
    "STREAMS",
]

REQUESTS = "requests"
STREAMS = "streams"

_lock = threading.Lock()
# (kind, scope) -> limit
_limits: dict[tuple[str, str], int] = {}
# loop -> {(kind, scope): limiter}; loops are held weakly so closed loops drop out.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, str], EndpointLimiter]]" = (
    weakref.WeakKeyDictionary()
)
# Warnings already logged, so each is logged once.
_reserve_warned: set[tuple[str, int, int]] = set()
_streams_alone_warned: set[str] = set()


def validate_max_concurrent(value: int | None, *, name: str = "max_concurrent") -> int | None:
    """Return a positive int or None. Zero, negatives and non-integers are rejected."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return value


def limit_scope(base_url: str | None, credential: str | None) -> str:
    """Scope key for an endpoint and credential. The credential is hashed, never kept."""
    url = (base_url or "").strip().rstrip("/").lower()
    material = f"{url}\n{credential or ''}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def register_limit(scope: str, max_concurrent: int | None, *, label: str = "", kind: str = REQUESTS) -> None:
    """Record a client's limit of ``kind`` for its scope. The smallest registered limit wins."""
    if max_concurrent is None:
        return
    key = (kind, scope)
    with _lock:
        current = _limits.get(key)
        if current is None:
            _limits[key] = max_concurrent
        elif max_concurrent != current:
            effective = min(current, max_concurrent)
            logger.warning(
                "llm-mesh: clients of one endpoint ask for different %s (%s vs %s)%s; the endpoint limit is %s",
                "max_concurrent_streams" if kind == STREAMS else "max_concurrent",
                current,
                max_concurrent,
                f" [{label}]" if label else "",
                effective,
            )
            _limits[key] = effective
        _check_stream_reserve(scope)


def _check_stream_reserve(scope: str) -> None:
    """Warn once when the streams limit does not reserve a blocking slot. Holds ``_lock``."""
    streams = _limits.get((STREAMS, scope))
    requests = _limits.get((REQUESTS, scope))
    if streams is None:
        return
    if requests is None:
        if scope in _streams_alone_warned:
            return
        _streams_alone_warned.add(scope)
        logger.warning(
            "llm-mesh: max_concurrent_streams=%s is set without max_concurrent; "
            "streams are capped, blocking calls are not",
            streams,
        )
        return
    if streams < requests:
        return
    if (scope, streams, requests) in _reserve_warned:
        return
    _reserve_warned.add((scope, streams, requests))
    logger.warning(
        "llm-mesh: max_concurrent_streams=%s is not below max_concurrent=%s; "
        "streams can still take every slot of the endpoint",
        streams,
        requests,
    )


class EndpointLimiter:
    """Async limiter whose capacity follows the registered limit.

    ``asyncio.Semaphore`` cannot shrink. Each acquire reads the current limit,
    so a limit lowered after this object was created applies to the next
    acquire. Slots already held are not revoked.

    The limiter does not store the event loop. ``asyncio.Semaphore`` keeps
    ``_loop`` after the first wait, and a registry keyed weakly by that loop
    would then retain every closed loop (registry → semaphore → loop).
    Waiters are futures dropped as soon as they are resumed.

    A released slot is handed to the oldest waiter and stays counted in
    ``_held``. A later acquire cannot take that slot ahead of a request that
    is already waiting; waking every waiter and racing them would let a tight
    loop of new calls starve the queue.
    """

    def __init__(self, kind: str, scope: str) -> None:
        self._kind = kind
        self._scope = scope
        self._held = 0
        self._waiters: list[asyncio.Future[None]] = []
        # Futures that already own a handed-off slot. They must not acquire again.
        self._granted: set[asyncio.Future[None]] = set()

    @property
    def limit(self) -> int | None:
        return scope_limit(self._scope, self._kind)

    def locked(self) -> bool:
        """True when a new acquire would wait."""
        limit = self.limit
        return limit is not None and self._held >= limit

    def _release(self) -> None:
        """Free one slot, or hand it to the oldest waiter."""
        if self.limit is not None:
            while self._waiters:
                fut = self._waiters.pop(0)
                if fut.done():
                    continue
                try:
                    fut.set_result(None)
                except asyncio.InvalidStateError:
                    continue
                # The waiter resumes only after this task yields, so the mark
                # is visible before it returns from ``await``.
                self._granted.add(fut)
                return
        self._held -= 1
        if self.limit is None and self._waiters:
            waiters = self._waiters
            self._waiters = []
            for fut in waiters:
                if not fut.done():
                    fut.set_result(None)

    async def __aenter__(self) -> EndpointLimiter:
        while True:
            limit = self.limit
            if limit is None or self._held < limit:
                self._held += 1
                return self
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await fut
            except asyncio.CancelledError:
                self._waiters = [waiter for waiter in self._waiters if waiter is not fut]
                if fut in self._granted:
                    # The slot was already handed over, then this wait was cancelled.
                    self._granted.discard(fut)
                    self._release()
                raise
            if fut in self._granted:
                self._granted.discard(fut)
                return self

    async def __aexit__(self, *_exc: object) -> None:
        self._release()


def scope_limit(scope: str, kind: str = REQUESTS) -> int | None:
    """The registered limit of a scope, or None when no client set one."""
    with _lock:
        return _limits.get((kind, scope))


def scope_semaphore(scope: str, kind: str = REQUESTS) -> EndpointLimiter | None:
    """The scope's limiter in the running loop, or None without a limit.

    With a limit it must be called from a coroutine: waiters are bound to the
    running loop. The limiter reads the registered limit on each acquire.
    """
    key = (kind, scope)
    with _lock:
        if _limits.get(key) is None:
            return None
    loop = asyncio.get_running_loop()
    with _lock:
        if _limits.get(key) is None:
            return None
        per_loop = _semaphores.get(loop)
        if per_loop is None:
            per_loop = {}
            _semaphores[loop] = per_loop
        limiter = per_loop.get(key)
        if limiter is None:
            limiter = EndpointLimiter(kind, scope)
            per_loop[key] = limiter
        return limiter


def reset_limits() -> None:
    """Forget all scopes and semaphores (tests, reconfiguration)."""
    with _lock:
        _limits.clear()
        _semaphores.clear()
        _reserve_warned.clear()
        _streams_alone_warned.clear()
