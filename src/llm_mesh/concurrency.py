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
]

_lock = threading.Lock()
_limits: dict[str, int] = {}
# loop -> {scope: (semaphore, size)}; loops are held weakly so closed loops drop out.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, tuple[asyncio.Semaphore, int]]]" = (
    weakref.WeakKeyDictionary()
)
# (scope, stale size, new limit) already reported, so the warning is logged once.
_resize_warned: set[tuple[str, int, int]] = set()


def validate_max_concurrent(value: int | None) -> int | None:
    """Return a positive int or None. Zero, negatives and non-integers are rejected."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_concurrent must be a positive integer or None, got {value!r}")
    return value


def limit_scope(base_url: str | None, credential: str | None) -> str:
    """Scope key for an endpoint and credential. The credential is hashed, never kept."""
    url = (base_url or "").strip().rstrip("/").lower()
    material = f"{url}\n{credential or ''}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


def register_limit(scope: str, max_concurrent: int | None, *, label: str = "") -> None:
    """Record a client's limit for its scope. The smallest registered limit wins."""
    if max_concurrent is None:
        return
    with _lock:
        current = _limits.get(scope)
        if current is None:
            _limits[scope] = max_concurrent
            return
        if max_concurrent == current:
            return
        effective = min(current, max_concurrent)
        logger.warning(
            "llm-mesh: clients of one endpoint ask for different max_concurrent (%s vs %s)%s; "
            "the endpoint limit is %s",
            current,
            max_concurrent,
            f" [{label}]" if label else "",
            effective,
        )
        _limits[scope] = effective


def scope_limit(scope: str) -> int | None:
    """The registered limit of a scope, or None when no client set one."""
    with _lock:
        return _limits.get(scope)


def scope_semaphore(scope: str) -> asyncio.Semaphore | None:
    """The scope's semaphore in the running loop, or None without a limit.

    With a limit it must be called from a coroutine: the semaphore is bound to
    the running loop.
    """
    with _lock:
        if _limits.get(scope) is None:
            return None
    loop = asyncio.get_running_loop()
    with _lock:
        limit = _limits.get(scope)
        if limit is None:
            return None
        per_loop = _semaphores.get(loop)
        if per_loop is None:
            per_loop = {}
            _semaphores[loop] = per_loop
        entry = per_loop.get(scope)
        if entry is None:
            entry = (asyncio.Semaphore(limit), limit)
            per_loop[scope] = entry
        elif entry[1] != limit and (scope, entry[1], limit) not in _resize_warned:
            _resize_warned.add((scope, entry[1], limit))
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
