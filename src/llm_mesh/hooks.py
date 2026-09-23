"""Optional application hooks for prompt-injection checks.

The built-in canary in llm_mesh.canary is active whenever a context token is
set. configure_canary_hooks replaces the token getter, prompt builder, and
response check. A custom token getter may use ContextVar to isolate concurrent
sessions.

configure_request_hook replaces the process-wide request guard. One hook
serves every client and thread in the process, the same way the canary hooks
do. A hook that needs per-session isolation has to do that itself (for
example with a ContextVar); the library does not scope the hook per client.

The library ships no detection patterns and no redaction. The hook body
belongs to the application. count_tokens is not guarded: it takes raw strings
and is a diagnostic, not a prompt sent to a model. A probe's generate_text
request is guarded like any other call, so a hook that blocks it surfaces as
the route check's error_type.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

from llm_mesh.types import LLMRequest, LLMRequestBlocked

from .canary import (
    build_canary_prompt as _default_build_prompt,
    check_and_warn as _default_check,
    get_canary_context_token as _default_get_token,
)

logger = logging.getLogger(__name__)

_get_token: Callable[[], str | None] = _default_get_token
_build_prompt: Callable[[str], str] = _default_build_prompt
_check: Callable[..., Any] = _default_check


def configure_canary_hooks(
    *, get_token: Callable[[], str | None],
    build_prompt: Callable[[str], str], check: Callable[..., Any],
) -> None:
    global _get_token, _build_prompt, _check
    _get_token, _build_prompt, _check = get_token, build_prompt, check


def get_canary_context_token() -> str | None:
    return _get_token()


def build_canary_prompt(token: str) -> str:
    return _build_prompt(token)


def check_and_warn(*args: Any, **kwargs: Any) -> Any:
    return _check(*args, **kwargs)


def identity_check_request(request: LLMRequest) -> LLMRequest | None:
    """Return the request unchanged. This is the default guard."""
    return request


_check_request: Callable[[LLMRequest], LLMRequest | None] = identity_check_request


def configure_request_hook(
    *, check_request: Callable[[LLMRequest], LLMRequest | None],
) -> None:
    """Replace the process-wide request guard.

    Return the same object to pass the request through. Return a different
    LLMRequest to replace it: the library never mutates the payload itself.
    Return None to block. An exception propagates unchanged.

    count_tokens is not a call site. It takes raw strings and does not build
    an LLMRequest. The connectivity probe's generate_text request is guarded,
    so a refusal shows up as that route check's error_type.
    """
    global _check_request
    _check_request = check_request


def check_request(request: LLMRequest) -> LLMRequest | None:
    """Run the configured request hook. The default returns the same object."""
    return _check_request(request)


def apply_request_guard(request: LLMRequest, *, provider: str) -> LLMRequest:
    """Translate the hook result into a request or LLMRequestBlocked.

    None is a refusal. The warning names the provider and carries the
    REQUEST_BLOCKED marker. The payload is not logged. A hook exception is
    not caught here.
    """
    result = check_request(request)
    if result is None:
        logger.warning("REQUEST_BLOCKED provider=%s", provider)
        raise LLMRequestBlocked(
            f"{provider}: request blocked by the request hook",
            reason="request hook returned None",
        )
    return result


def guard_batch_requests(
    requests: Sequence[LLMRequest],
    *,
    provider: str,
    return_exceptions: bool,
) -> tuple[list[tuple[int, LLMRequest]], dict[int, LLMRequestBlocked]]:
    """Guard each request before a file or JSONL body is built.

    A refusal raises unless return_exceptions is set, in which case it
    occupies its input index and that request is left out of the upload.
    A partial file would renumber custom ids, so blocked rows are omitted
    and the surviving rows keep their original indexes. Other exceptions
    from the hook still propagate.
    """
    accepted: list[tuple[int, LLMRequest]] = []
    blocked: dict[int, LLMRequestBlocked] = {}
    for index, request in enumerate(requests):
        try:
            accepted.append((index, apply_request_guard(request, provider=provider)))
        except LLMRequestBlocked as exc:
            if not return_exceptions:
                raise
            blocked[index] = exc
    return accepted, blocked
