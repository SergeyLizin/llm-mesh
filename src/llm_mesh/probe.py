"""Live connection checks for catalog routes and constructed clients.

A check proves that the endpoint is reachable, credentials are accepted, TLS
is accepted, and the named model can be called. It does not judge reply
text, and it does not install a canary. Success is the absence of an
exception from the cheapest call that still exercises that path.

``check_client`` does not close the client. The caller created it and still
owns ``aclose()``. ``check_route`` builds the client with ``make_client``,
which writes the route into the process environment, then closes that
client exactly once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_mesh.anthropic.client import AnthropicClient
from llm_mesh.gemini.client import GeminiClient
from llm_mesh.gigachat.client import GigaChatAsyncClient
from llm_mesh.models_catalog import make_client, route_model, routes_for
from llm_mesh.openai.client import OpenAIClient
from llm_mesh.types import LLMRequest, LLMTimeoutError

logger = logging.getLogger(__name__)

_ERROR_LIMIT = 500

# OpenAI's client timeout is 600s per attempt, and retries multiply it.
# A diagnostic must not wait that long for a gateway that never answers.
# timeout=None on the check functions leaves the client timeout in place.
DEFAULT_PROBE_TIMEOUT_S = 60.0


class ProbeKind(str, Enum):
    """Which call proves the path.

    ``generate_text`` is a chat completion. ``count_tokens`` is an
    authenticated tokenizer call that does not spend generation tokens.
    """

    GENERATE_TEXT = "generate_text"
    COUNT_TOKENS = "count_tokens"


class ConnectionCheck(BaseModel):
    """Outcome of one route or one client.

    ``error_type`` is the exception class name (``LLMAuthError``,
    ``LLMTimeoutError``, ``OpenAIError``, ...) so callers can classify a
    failure without parsing ``error``.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    label: str
    kind: str
    model: str
    latency_ms: int | None
    error: str | None = None
    error_type: str | None = None
    detail: str = ""


def _identity(client: Any) -> tuple[str, str, str]:
    if isinstance(client, OpenAIClient):
        return client.PROVIDER, "openai", client._model
    if isinstance(client, AnthropicClient):
        return client.PROVIDER, "anthropic", client._model
    if isinstance(client, GeminiClient):
        return client.PROVIDER, "gemini", client._model
    if isinstance(client, GigaChatAsyncClient):
        return client.PROVIDER, "gigachat", client.model
    return type(client).__name__, "", ""


def _probe_for(client: Any) -> ProbeKind:
    """Pick the cheapest call that still proves this client's path.

    GigaChat's tokenizer, Anthropic's messages count endpoint, and
    Gemini's countTokens authenticate without a chat completion.
    OpenAI-compatible endpoints have no such call, so they send one
    tiny generation.
    """
    if isinstance(client, (GigaChatAsyncClient, AnthropicClient, GeminiClient)):
        return ProbeKind.COUNT_TOKENS
    if isinstance(client, OpenAIClient):
        return ProbeKind.GENERATE_TEXT
    raise TypeError(
        f"no connection probe for {type(client).__name__}; "
        "pass an OpenAI, Anthropic, Gemini, or GigaChat client"
    )


def _probe_request() -> LLMRequest:
    """Return a fresh probe request.

    ``LLMRequest`` is mutable. A module-level instance would let one caller
    change ``max_tokens`` for every later probe. Short enough that a gateway
    which bills by output tokens spends almost nothing, and long enough that
    ``max_tokens=1`` rejections are avoided.
    """
    return LLMRequest(
        system="You are a connectivity probe. Reply with exactly: ok",
        user="ok",
        max_tokens=8,
        temperature=0,
        length_retry=False,
    )


async def _invoke(client: Any, probe: ProbeKind) -> None:
    if probe is ProbeKind.COUNT_TOKENS:
        await client.count_tokens(["connectivity probe"])
        return
    if probe is ProbeKind.GENERATE_TEXT:
        # reasoning_effort stays unset so Anthropic does not enable thinking.
        await client.generate_text(_probe_request())
        return
    raise TypeError(f"unknown probe {probe!r}")


async def _invoke_bounded(
    client: Any, probe: ProbeKind, timeout: float | None,
) -> None:
    """Run the probe, optionally under a wall-clock cap.

    ``timeout=None`` uses only the client's own timeout. That is the
    production budget, including its retries. A number is a ceiling around
    the whole call, so a silent gateway cannot hold ``--check`` for the
    client's 600s timeout several times over.
    """
    if timeout is None:
        await _invoke(client, probe)
        return
    try:
        await asyncio.wait_for(_invoke(client, probe), timeout)
    except TimeoutError as exc:
        raise LLMTimeoutError(
            f"connection probe exceeded {timeout:g}s; "
            "pass timeout=None to use the client timeout"
        ) from exc


def _failure(
    *,
    label: str,
    kind: str,
    model: str,
    exc: BaseException,
    latency_ms: int | None,
    detail: str,
) -> ConnectionCheck:
    return ConnectionCheck(
        ok=False,
        label=label,
        kind=kind,
        model=model,
        latency_ms=latency_ms,
        error=str(exc)[:_ERROR_LIMIT],
        error_type=type(exc).__name__,
        detail=detail,
    )


async def check_client(
    client: Any,
    *,
    probe: ProbeKind | None = None,
    timeout: float | None = DEFAULT_PROBE_TIMEOUT_S,
) -> ConnectionCheck:
    """Probe ``client`` and return the outcome.

    The caller owns ``client``. This function does not call ``aclose()``.
    An unrecognized client type raises ``TypeError`` instead of reporting
    a failed connection. A probe the client does not implement
    (``count_tokens`` on OpenAI, for example) also raises ``TypeError``:
    that is a caller error, not a dead route. ``CancelledError`` and
    ``KeyboardInterrupt`` propagate. Every other exception becomes
    ``ok=False``.

    ``timeout`` defaults to 60 seconds around the whole probe, including
    the client's retries. ``timeout=None`` leaves only the client timeout.
    """
    if probe is not None and not isinstance(probe, ProbeKind):
        raise TypeError(
            f"probe must be ProbeKind or None, got {type(probe).__name__}"
        )
    selected = probe if probe is not None else _probe_for(client)
    label, kind, model = _identity(client)
    started = time.perf_counter()
    try:
        await _invoke_bounded(client, selected, timeout)
    except NotImplementedError as exc:
        raise TypeError(
            f"{type(client).__name__} does not implement probe {selected.value}"
        ) from exc
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        return _failure(
            label=label,
            kind=kind,
            model=model,
            exc=exc,
            latency_ms=_elapsed_ms(started),
            detail=selected.value,
        )
    return ConnectionCheck(
        ok=True,
        label=label,
        kind=kind,
        model=model,
        latency_ms=_elapsed_ms(started),
        detail=selected.value,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _overlay(route: dict, result: ConnectionCheck) -> ConnectionCheck:
    return result.model_copy(update={
        "label": str(route.get("label") or result.label),
        "kind": str(route.get("kind") or result.kind),
        "model": route_model(route) or result.model,
    })


async def _probe_owned(
    client: Any, *, timeout: float | None,
) -> ConnectionCheck:
    """Probe a client this module constructed, then close it once.

    ``TypeError`` means the caller passed a client this module cannot probe.
    Closing still happens, and the ``TypeError`` is raised afterwards.
    A close error after a successful probe is the result: ``ok=True`` would
    hide a transport that could not be shut down. A close error after a
    failed probe is logged and does not replace the probe's error.
    """
    result: ConnectionCheck | None = None
    pending: BaseException | None = None
    try:
        result = await check_client(client, timeout=timeout)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except TypeError as exc:
        pending = exc
    except Exception as exc:
        pending = exc
    try:
        await client.aclose()
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        logger.warning(
            "connection check could not close %s",
            type(client).__name__,
        )
        if pending is None and result is not None and result.ok:
            return _failure(
                label=result.label,
                kind=result.kind,
                model=result.model,
                exc=exc,
                latency_ms=result.latency_ms,
                detail=result.detail,
            )
    if pending is not None:
        raise pending
    if result is None:
        raise TypeError("connection probe produced no result")
    return result


def check_route(
    route: dict, *, timeout: float | None = DEFAULT_PROBE_TIMEOUT_S,
) -> ConnectionCheck:
    """Build ``route``'s client, probe it, and close it.

    Construction goes through ``make_client``, so route options such as
    ``verify_ssl`` apply, and the process environment is updated exactly
    as ``make_client`` updates it. Call this when no event loop is running.
    ``timeout`` is the probe cap; see ``check_client``.
    """
    try:
        client = make_client(route)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        return _failure(
            label=str(route.get("label") or ""),
            kind=str(route.get("kind") or ""),
            model=route_model(route),
            exc=exc,
            latency_ms=None,
            detail="",
        )
    try:
        result = asyncio.run(_probe_owned(client, timeout=timeout))
    except TypeError as exc:
        # Batch mode returns BatchingLLMClient. Probing it would enqueue a real
        # batch and wait. check_client still raises TypeError for an unknown
        # object; a catalog route is a diagnostic result, so --check prints a
        # line and keeps walking the other routes.
        return _overlay(route, _failure(
            label=str(route.get("label") or ""),
            kind=str(route.get("kind") or ""),
            model=route_model(route),
            exc=exc,
            latency_ms=None,
            detail="",
        ))
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        return _failure(
            label=str(route.get("label") or ""),
            kind=str(route.get("kind") or ""),
            model=route_model(route),
            exc=exc,
            latency_ms=None,
            detail="",
        )
    return _overlay(route, result)


def check_routes(
    selector: str,
    routes: list[dict] | None = None,
    *,
    timeout: float | None = DEFAULT_PROBE_TIMEOUT_S,
) -> list[ConnectionCheck]:
    """Probe every provider route for ``selector``, in catalog order.

    ``selector`` is a model id or ``id@provider``. Each route is built with
    ``make_client``, which replaces the process environment. The last route
    leaves its environment in place. ``timeout`` is the per-route probe cap;
    see ``check_client``.
    """
    return [
        check_route(route, timeout=timeout)
        for route in routes_for(selector, routes)
    ]
