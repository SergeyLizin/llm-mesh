"""Provider-neutral helpers shared by LLM clients."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from llm_mesh.text_parsing import looks_degenerate_repetition
from llm_mesh.types import LLMRequest
from .hooks import (
    build_canary_prompt,
    check_and_warn,
    get_canary_context_token,
)

_ResultT = TypeVar("_ResultT")


def apply_canary(system: str) -> str:
    """Append the canary block to the system prompt when a token is active."""
    token = get_canary_context_token()
    return system + build_canary_prompt(token) if token else system


def check_response_canary(response_text: str, *, context: str) -> None:
    """Scan a response for the canary without changing exception semantics."""
    token = get_canary_context_token()
    if not token or not response_text:
        return
    check_and_warn(response_text, token, session_id="", context=context)


def build_text_messages(
    request: LLMRequest, *, tool_turns: bool = True,
) -> list[dict[str, Any]]:
    """Build [system, *history, user] messages. Preserve user/assistant text, assistant tool_calls,
    and tool results linked by tool_call_id; discard other roles. Strict providers require the
    parent assistant tool-call turn before its results. With tool_turns=False, discard both tool
    results and assistant tool-call turns entirely: legacy GigaChat functions do not support the
    OpenAI tool role.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": apply_canary(request.system)}
    ]
    for turn in request.history or []:
        role = turn.get("role")
        is_tool_turn = role == "tool" or (
            role == "assistant" and bool(turn.get("tool_calls"))
        )
        if is_tool_turn and not tool_turns:
            continue
        if role == "assistant" and turn.get("tool_calls"):
            messages.append({
                "role": "assistant",
                "content": turn.get("content") or "",
                "tool_calls": turn["tool_calls"],
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": turn.get("tool_call_id") or "",
                "content": str(turn.get("content", "")),
            })
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": request.user})
    return messages


def finish_reason(payload: dict[str, Any]) -> str:
    """Extract the finish reason from an OpenAI-compatible payload."""
    try:
        choice = payload["choices"][0] or {}
    except (KeyError, IndexError, TypeError):
        return ""
    return str(choice.get("finish_reason") or choice.get("finishReason") or "")


def finish_reason_opt(payload: dict[str, Any]) -> str | None:
    """Return the response finish reason, or None when absent. Missing metadata and an explicit
    stop are distinct outcomes for evaluation and truncation diagnostics.
    """
    return finish_reason(payload) or None


def response_content(payload: dict[str, Any]) -> str:
    """Extract message.content from an OpenAI-compatible payload."""
    try:
        choice = payload["choices"][0] or {}
    except (KeyError, IndexError, TypeError):
        return ""
    message = choice.get("message") or {}
    return str(message.get("content") or "")


def warn_if_truncated(
    choice: dict[str, Any],
    request: LLMRequest,
    content: str | None,
    *,
    provider: str,
    logger: logging.Logger,
    sent_max_tokens: int | None = None,
) -> None:
    """Log output truncation using the token budget actually sent after clipping. The requested
    budget may exceed the model limit; clipping itself is logged only once per model. Length
    retries send a copied body with a larger budget, reported by the adjacent retry log. When
    escalation does not occur, sent_max_tokens is the exact wire budget.
    """
    reason = choice.get("finish_reason") or choice.get("finishReason") or ""
    if reason != "length":
        return
    sent = request.max_tokens if sent_max_tokens is None else sent_max_tokens
    # Keep the original log format when the budget was not clipped.
    clipped = (
        ""
        if sent == request.max_tokens
        else f" (requested {request.max_tokens}, clipped to model limit)"
    )
    logger.warning(
        "%s: response truncated by max_tokens=%d%s (finish_reason=length, "
        "completion_len=%d chars). Output likely structurally invalid.",
        provider,
        sent,
        clipped,
        len(content or ""),
    )


async def post_with_length_retry(
    body: dict[str, Any],
    *,
    post: Callable[[dict[str, Any]], Awaitable[_ResultT]],
    payload_of: Callable[[_ResultT], dict[str, Any]],
    retries: int,
    next_max_tokens: Callable[[int], int],
    provider: str,
    logger: logging.Logger,
) -> _ResultT:
    """Retry a provider POST after genuine length truncation, changing only max_tokens. The
    supplied post function owns transport, authentication, retries, and exceptions. An empty
    length-limited response may have spent its budget on hidden reasoning. Allow one budget
    increase for that case; if it remains empty, stop rather than repeatedly paying for a
    response that never produces visible output.
    """
    result = await post(body)
    current = int(body.get("max_tokens") or 0)
    empty_length_retries = 0
    for _ in range(retries):
        payload = payload_of(result)
        if finish_reason(payload) != "length":
            return result
        if not response_content(payload).strip():
            if empty_length_retries >= 1:
                logger.warning(
                    "%s: empty content with finish_reason=length after doubling the budget "
                    "— the budget may not be the cause, stopping budget escalation",
                    provider,
                )
                return result
            empty_length_retries += 1
        if looks_degenerate_repetition(response_content(payload)):
            logger.warning(
                "%s: finish_reason=length, but output is degenerate "
                "(repetition) — skipping length-retry",
                provider,
            )
            return result
        target = next_max_tokens(current)
        if target <= current:
            break
        logger.warning(
            "%s: finish_reason=length → retry with max_tokens %d→%d",
            provider,
            current,
            target,
        )
        current = target
        result = await post({**body, "max_tokens": target})
    return result


__all__ = [
    "apply_canary",
    "build_text_messages",
    "check_response_canary",
    "finish_reason",
    "finish_reason_opt",
    "post_with_length_retry",
    "response_content",
    "warn_if_truncated",
]
