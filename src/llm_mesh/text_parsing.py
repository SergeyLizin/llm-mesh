"""Helpers for parsing artifacts returned in chat/completions content instead of function calls."""

from __future__ import annotations

import json
import re
import zlib
from typing import Any, cast


def looks_degenerate_repetition(content: str | None) -> bool:
    """Detect pathological repetition in length-truncated output using tail compression. Repeating
    one line or token compresses far more than ordinary JSON or code; a ratio below 0.05
    distinguishes the observed cases. Skip short output, where a larger budget may still help.
    Increasing max_tokens cannot repair a repetitive generation loop.
    """
    if not content:
        return False
    tail = content[-4000:].encode("utf-8", "replace")
    if len(tail) < 1000:
        return False
    return len(zlib.compress(tail, 6)) / len(tail) < 0.05


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.+?)```", re.DOTALL | re.IGNORECASE)


# Strip function-call control markers emitted in content by servers without tool parsers. Apply
# stripping independently in generate_text and extract_json_from_text because they cover
# distinct caller paths. Keep the regex shared and stripping idempotent.
_CONTROL_MARKER_RE = re.compile(r"<\|function_call\|>")


# JSON allows only specific backslash escapes. Double invalid backslashes in model prose so
# json.loads can parse them as literals; preserve already valid escapes.
_INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})')


def extract_json_fence(text: str) -> str:
    """Extract a closed `````json`` fence, otherwise return stripped text.

    This is the conservative tier used by consumers that must not silently
    accept a response truncated before the closing fence.
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_json_fence_tolerant(text: str) -> str:
    """Extract JSON fence content, accepting a missing closing fence."""
    payload = extract_json_fence(text)
    if payload != text.strip():
        return payload
    match = re.search(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else payload


def extract_json_fence_lenient(text: str) -> str:
    """Tolerant fence extraction plus repair of invalid JSON escapes."""
    return _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", extract_json_fence_tolerant(text))


def _try_raw_decode(candidate: str) -> dict[str, Any] | None:
    """Decode the first JSON object and ignore trailing text, explanations, repeated objects, or
    fences that would otherwise cause Extra data errors.
    """
    try:
        obj, _end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _loads_lenient(candidate: str) -> dict[str, Any]:
    """Parse model output tolerantly: try strict JSON, decode the first object after Extra data,
    repair invalid backslash escapes, then use json_repair. Preserve valid JSON and require a
    dictionary result.
    """
    try:
        return cast("dict[str, Any]", json.loads(candidate))
    except json.JSONDecodeError as exc:
        # Decode the valid object before trailing text.
        if "Extra data" in str(exc):
            obj = _try_raw_decode(candidate)
            if obj is not None:
                return obj
        # Repair invalid escapes without changing valid ones.
        if "Invalid \\escape" in str(exc):
            cleaned = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", candidate)
            try:
                return cast("dict[str, Any]", json.loads(cleaned))
            except json.JSONDecodeError as exc2:
                # Escape repair may reveal the same trailing-data case.
                if "Extra data" in str(exc2):
                    obj = _try_raw_decode(cleaned)
                    if obj is not None:
                        return obj
        # Use json_repair as the final fallback.
        try:
            from json_repair import loads as repair_loads
        except ImportError:
            raise exc
        result = repair_loads(candidate)
        if not isinstance(result, dict):
            raise exc
        return result


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract an object from a JSON Markdown fence or the first balanced brace block. Parse it
    into a dictionary for caller-side schema validation; raise RuntimeError when no object can
    be recovered.
    """
    text = text.strip()
    if not text:
        raise RuntimeError("LLM returned an empty text response")

    # Remove any function-call control marker before searching fences or balanced braces.
    text = _CONTROL_MARKER_RE.sub("", text).strip()

    # 1. markdown fence
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            return _loads_lenient(candidate)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"JSON in Markdown fence cannot be parsed: {exc}; payload: {candidate[:300]}"
            ) from exc

    # Find the first balanced object.
    start = text.find("{")
    if start == -1:
        raise RuntimeError(f"Text response is missing JSON: {text[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return _loads_lenient(candidate)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"JSON object cannot be parsed: {exc}; payload: {candidate[:300]}"
                    ) from exc
    raise RuntimeError(f"Unclosed JSON object in text response: {text[:200]!r}")


# ============== Suggestions parsing + markdown block ==============


# Share the suggestions heading between rendering and duplicate detection so nested application
# flows do not append a second next-steps block.
NEXT_STEPS_HEADING = "**Next steps:**"


def parse_suggestions_json(raw: str) -> list[str]:
    """Leniently parse a suggestions object from plain JSON or a JSON fence. Return the first four
    nonempty strings; return an empty list for invalid JSON, missing keys, or parsing failures.
    """
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    payload = m.group(1).strip() if m else raw.strip()
    try:
        data = json.loads(payload)
        items = data.get("suggestions") or []
        return [str(s).strip() for s in items if isinstance(s, str) and s.strip()][:4]
    except Exception:
        return []


def format_suggestions_block(suggestions: list[str]) -> str:
    """Render the suggestions heading and Markdown bullets without leading newlines."""
    if not suggestions:
        return ""
    return f"\n\n---\n\n{NEXT_STEPS_HEADING}\n" + "\n".join(
        f"* {s}" for s in suggestions
    )


# ============== Code-fence stripping (structured-output code fields) ==============


_CODE_FENCE_WRAP_RE = re.compile(
    r"\A\s*```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)\n?```[ \t]*\Z", re.DOTALL,
)


def strip_code_fence(code: str) -> str:
    """Remove one Markdown code fence only when it wraps the entire string. This prevents
    double-fenced previews and fence markers being persisted as executable code. Preserve
    embedded fences in legitimate source code and return unwrapped input unchanged.
    """
    m = _CODE_FENCE_WRAP_RE.match(code)
    return m.group(1) if m else code

def _loads_lenient_list(candidate: str) -> list[Any]:
    """Parse a top-level JSON array with the same repair strategy as _loads_lenient. Keep this
    separate so object extraction retains its dictionary return contract.
    """
    try:
        return cast("list[Any]", json.loads(candidate))
    except json.JSONDecodeError as exc:
        if "Extra data" in str(exc):
            try:
                obj, _end = json.JSONDecoder().raw_decode(candidate)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, list):
                return obj
        if "Invalid \\escape" in str(exc):
            cleaned = _INVALID_JSON_ESCAPE_RE.sub(r"\\\\", candidate)
            try:
                return cast("list[Any]", json.loads(cleaned))
            except json.JSONDecodeError:
                pass
        try:
            from json_repair import loads as repair_loads
        except ImportError:
            raise exc
        result = repair_loads(candidate)
        if not isinstance(result, list):
            raise exc
        return result


def extract_json_array_from_text(text: str) -> list[Any]:
    """Extract a top-level JSON array from text by balancing brackets. An object-only parser would
    incorrectly return just the first object inside the array and discard the remaining items.
    """
    text = text.strip()
    if not text:
        raise RuntimeError("LLM returned an empty text response")
    text = _CONTROL_MARKER_RE.sub("", text).strip()

    match = _JSON_FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            return _loads_lenient_list(candidate)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"JSON array in Markdown fence cannot be parsed: {exc}; payload: {candidate[:300]}"
            ) from exc

    start = text.find("[")
    if start == -1:
        raise RuntimeError(f"Text response is missing JSON array: {text[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return _loads_lenient_list(candidate)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"JSON array cannot be parsed: {exc}; payload: {candidate[:300]}"
                    ) from exc
    raise RuntimeError(f"No balanced JSON array found in text response: {text[:200]!r}")
