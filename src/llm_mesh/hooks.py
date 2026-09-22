"""Optional application hooks for prompt-injection checks.

The built-in canary in llm_mesh.canary is active whenever a context token is
set. configure_canary_hooks replaces the token getter, prompt builder, and
response check. A custom token getter may use ContextVar to isolate concurrent
sessions.
"""

from collections.abc import Callable
from typing import Any

from .canary import (
    build_canary_prompt as _default_build_prompt,
    check_and_warn as _default_check,
    get_canary_context_token as _default_get_token,
)

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
