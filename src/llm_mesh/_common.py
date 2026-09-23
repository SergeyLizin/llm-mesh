"""Provider-neutral helpers shared by LLM clients."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from llm_mesh.config import get_env
from llm_mesh.text_parsing import looks_degenerate_repetition
from llm_mesh.types import LLMRequest, LLMResponse, LLMUsage
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


def schema_violation_detail(arguments: Any, schema: dict[str, Any] | None) -> str | None:
    """First confirmed jsonschema error, or None when the arguments may pass.

    A missing schema, non-object arguments, a missing jsonschema package, or
    an unsupported schema are not confirmed violations. Only
    ``jsonschema.ValidationError`` is. The text is the error message plus
    the path, which a corrective re-ask feeds back to the model.
    """
    if not schema or not isinstance(arguments, dict):
        return None
    try:
        import jsonschema  # noqa: PLC0415 -- import the validator lazily
    except Exception:
        return None
    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return None
    except jsonschema.ValidationError as exc:
        path = "/".join(str(part) for part in exc.absolute_path)
        where = f"/{path}" if path else "/"
        return f"{exc.message} at {where}"
    except Exception:
        return None


def _args_satisfy_schema(arguments: Any, schema: dict[str, Any] | None) -> bool:
    """Check parsed arguments against a JSON Schema.

    An explicit validation failure is False, so the caller can reject the
    response. A missing schema, non-object arguments, a missing jsonschema
    package, or an unsupported schema do not block the response: only a
    confirmed ``jsonschema.ValidationError`` returns False.
    """
    return schema_violation_detail(arguments, schema) is None


def corrective_request(
    request: LLMRequest,
    *,
    function_name: str,
    failed_answer: str,
    detail: str,
) -> LLMRequest:
    """Copy the request with one corrective turn appended to history.

    Body builders already emit history and then the current user turn, so
    the failed answer and the error text ride along without a special body.
    The original user text stays on the request.
    """
    history = list(request.history or [])
    history.append({"role": "assistant", "content": failed_answer})
    history.append({
        "role": "user",
        "content": (
            f"Your previous answer for function {function_name} failed "
            f"schema validation: {detail}. Respond again with corrected output only."
        ),
    })
    return request.model_copy(update={"history": history})


def merge_usage(first: LLMUsage | None, second: LLMUsage | None) -> LLMUsage:
    """Add two attempts' token counts.

    Prompt, completion, total, and reasoning add. Cache fields keep the
    attempt that reported them, and the second attempt wins when both did:
    -1 means unreported, not an empty cache. Money adds when either side
    reported it. None plus None stays None, so an unreported cost is not
    treated as zero.
    """
    left = first or LLMUsage()
    right = second or LLMUsage()

    def _cache(left_value: int, right_value: int) -> int:
        if right_value != -1:
            return right_value
        return left_value

    def _money(left_value: float | None, right_value: float | None) -> float | None:
        if left_value is None and right_value is None:
            return None
        return (left_value or 0.0) + (right_value or 0.0)

    return LLMUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        cache_hit_tokens=_cache(left.cache_hit_tokens, right.cache_hit_tokens),
        cache_miss_tokens=_cache(left.cache_miss_tokens, right.cache_miss_tokens),
        cost_rub=_money(left.cost_rub, right.cost_rub),
        cost_usd=_money(left.cost_usd, right.cost_usd),
    )


def stamp_validation_reask(prior: LLMUsage, response: LLMResponse) -> LLMResponse:
    """Return the second attempt with both attempts' usage and reask=1."""
    return response.model_copy(update={
        "usage": merge_usage(prior, response.usage),
        "validation_reasks": 1,
    })


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
    # Images ride on the current user turn only. An empty list keeps the
    # string content, so a text-only body stays byte-for-byte the same.
    if request.images:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": request.user},
            *[
                {"type": "image_url", "image_url": {"url": image.openai_url()}}
                for image in request.images
            ],
        ]
        messages.append({"role": "user", "content": parts})
    else:
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


# Positive flags accept only these tokens. Call sites that historically did not
# strip keep that behavior through the strip parameter: a padded " 1 " must not
# start meaning true on a site that never stripped.
_ENV_TRUE = frozenset({"1", "true", "yes"})
# LLM_VERIFY_SSL is an exclusion list, not a positive flag. The two polarities
# and the provider-specific defaults are part of the client contract.
_ENV_FALSE = frozenset({"0", "false", "no"})


def _env_flag(name: str, *, default: str = "", strip: bool = True) -> bool:
    """Return whether the setting is 1, true, or yes.

    Most sites strip. OpenAI's LLM_STREAM_TRANSPORT, LLM_DISABLE_TOOLS, and
    LLM_NO_DEGRADE do not; pass strip=False so a padded value keeps its old
    meaning. Values always come from get_env, which honors LLM_OPTIONS.
    """
    raw = get_env(name, default)
    if strip:
        raw = raw.strip()
    return raw.lower() in _ENV_TRUE


def _env_is_disabled(name: str, *, default: str, strip: bool = False) -> bool:
    """Return whether the setting is 0, false, or no.

    LLM_VERIFY_SSL uses this list. OpenAI and Anthropic default to verified
    ("1"); GigaChat defaults to unverified ("false") because those
    installations often lack the CA roots. Do not rewrite those sites as
    _env_flag: the polarity and the defaults differ by provider.
    """
    raw = get_env(name, default)
    if strip:
        raw = raw.strip()
    return raw.lower() in _ENV_FALSE


def _env_positive_int(name: str) -> int | None:
    """Read a positive integer, or None when missing, zero, or not digits."""
    raw = get_env(name, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def _env_float_default(
    name: str, default: float, *, logger: logging.Logger,
) -> float:
    """Read a float. Empty or non-numeric values keep ``default`` and warn.

    A bad ``LLM_HTTP_TIMEOUT`` used to raise from the constructor. Neighboring
    numeric options already ignore garbage; this matches them.
    """
    raw = get_env(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s is not a number (%r) — ignoring", name, raw)
        return default


def _env_int_default(
    name: str, default: int, *, logger: logging.Logger,
) -> int:
    """Read an int. Empty or non-numeric values keep ``default`` and warn."""
    raw = get_env(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s is not an integer (%r) — ignoring", name, raw)
        return default


def _env_nonneg_int(name: str, default: int) -> int:
    """Read a nonnegative integer. Zero is valid; anything else uses default.

    GigaChat's length-retry budget defaults to 1. OpenAI and Anthropic
    default to 2. The default stays at the call site.
    """
    raw = get_env(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def _parse_json_dict_env(
    name: str,
    *,
    logger: logging.Logger,
    include_error: bool = True,
) -> dict[str, Any] | None:
    """Parse an environment variable as a JSON object.

    Missing, invalid, and non-object values warn and return None so client
    construction does not fail. include_error keeps the OpenAI warning, which
    names the parser exception. Anthropic's warning does not. The logger is
    the caller's so the warning stays on that module.
    """
    raw = get_env(name, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        if include_error:
            logger.warning("%s invalid JSON (%s) — ignoring", name, exc)
        else:
            logger.warning("%s is not valid JSON — ignoring", name)
        return None
    if not isinstance(parsed, dict):
        logger.warning("%s is not a JSON object — ignoring", name)
        return None
    return parsed


__all__ = [
    "apply_canary",
    "build_text_messages",
    "check_response_canary",
    "finish_reason",
    "finish_reason_opt",
    "post_with_length_retry",
    "response_content",
    "warn_if_truncated",
    "_env_flag",
    "_env_float_default",
    "_env_int_default",
    "_env_is_disabled",
    "_env_nonneg_int",
    "_env_positive_int",
    "_parse_json_dict_env",
]
