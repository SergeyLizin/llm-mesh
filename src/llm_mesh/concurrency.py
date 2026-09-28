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
- The semaphore is created lazily in the running event loop. ``asyncio``
  primitives are bound to one loop, so each loop gets its own semaphore for a
  scope: the limit holds per event loop. Applications normally drive all
  clients from a single loop.
- A lowered limit applies to semaphores created afterwards. A semaphore that
  already exists keeps its size; a warning names the scope.

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
    "REQUESTS",
    "STREAMS",
]

REQUESTS = "requests"
STREAMS = "streams"

_lock = threading.Lock()
# (kind, scope) -> limit
_limits: dict[tuple[str, str], int] = {}
# loop -> {(kind, scope): (semaphore, size)}; loops are held weakly so closed loops drop out.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, str], tuple[asyncio.Semaphore, int]]]" = (
    weakref.WeakKeyDictionary()
)
# Warnings already logged, so each is logged once.
_resize_warned: set[tuple[str, str, int, int]] = set()
_reserve_warned: set[tuple[str, int, int]] = set()


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
    """Warn once when the streams limit leaves no slot for blocking calls. Holds ``_lock``."""
    streams = _limits.get((STREAMS, scope))
    requests = _limits.get((REQUESTS, scope))
    if streams is None or requests is None or streams < requests:
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


def scope_limit(scope: str, kind: str = REQUESTS) -> int | None:
    """The registered limit of a scope, or None when no client set one."""
    with _lock:
        return _limits.get((kind, scope))


def scope_semaphore(scope: str, kind: str = REQUESTS) -> asyncio.Semaphore | None:
    """The scope's semaphore in the running loop, or None without a limit.

    With a limit it must be called from a coroutine: the semaphore is bound to
    the running loop.
    """
    key = (kind, scope)
    with _lock:
        if _limits.get(key) is None:
            return None
    loop = asyncio.get_running_loop()
    with _lock:
        limit = _limits.get(key)
        if limit is None:
            return None
        per_loop = _semaphores.get(loop)
        if per_loop is None:
            per_loop = {}
            _semaphores[loop] = per_loop
        entry = per_loop.get(key)
        if entry is None:
            entry = (asyncio.Semaphore(limit), limit)
            per_loop[key] = entry
        elif entry[1] != limit and (kind, scope, entry[1], limit) not in _resize_warned:
            _resize_warned.add((kind, scope, entry[1], limit))
            logger.warning(
                "llm-mesh: endpoint limit lowered to %s after its semaphore of size %s was created; "
                "the existing semaphore keeps its size in this event loop",
                limit,
                entry[1],
            )
        return entry[0]


def reset_limits() -> None:
    """Forget all scopes and semaphores (tests, reconfiguration)."""
    with _lock:
        _limits.clear()
        _semaphores.clear()
        _resize_warned.clear()
        _reserve_warned.clear()
