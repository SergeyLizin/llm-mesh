"""Stateless GigaChat helpers: environment parsing, OAuth expiry normalization, function-schema
simplification, and ``<think>`` extraction. Transport and request policies are implemented by
the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from llm_mesh._common import _env_nonneg_int
from llm_mesh.config import get_env
import time
from typing import Any


_JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"



def _parse_env_float(name: str) -> float | None:
    """Read a sampling override as a float, or None if missing or invalid. Zero is a valid value."""
    raw = get_env(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_expires_at(raw: Any) -> float | None:
    """Normalize OAuth expires_at to absolute epoch seconds for proactive refresh. GigaChat returns
    epoch milliseconds. Missing, zero, or invalid values mean unknown expiry and reactive
    refresh on 401. Positive values below 1e9 are interpreted as relative seconds for
    nonstandard gateways.
    """
    if not raw:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    if val > 1e12:          # Convert epoch milliseconds to seconds.
        return val / 1000.0
    if val < 1e9:           # Relative seconds rather than an absolute epoch timestamp.
        return time.time() + val
    return val              # Already expressed as absolute epoch seconds.


def _normalize_schema_type(value: Any) -> Any:
    """Lowercase a known JSON Schema type. Leave unknown type names unchanged."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized if normalized in _JSON_SCHEMA_TYPES else value
    if isinstance(value, list):
        return [_normalize_schema_type(item) for item in value]
    return value


def _infer_missing_type(schema: dict[str, Any], *, default: str = "string") -> str:
    """Guess a scalar type from the keys GigaChat needs. Booleans are checked before integers
    because bool is a subclass of int.
    """
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    enum = schema.get("enum")
    if isinstance(enum, list):
        for item in enum:
            if isinstance(item, str):
                return "string"
            if isinstance(item, bool):
                return "boolean"
            if isinstance(item, int):
                return "integer"
            if isinstance(item, float):
                return "number"
    return default


def _normalize_schema_enum(schema: dict[str, Any]) -> None:
    """Keep unique string enum values. Drop the keyword when none remain: GigaChat rejects
    non-string enum entries.
    """
    enum = schema.get("enum")
    if not isinstance(enum, list):
        return
    string_enum: list[str] = []
    seen: set[str] = set()
    for item in enum:
        if isinstance(item, str) and item not in seen:
            seen.add(item)
            string_enum.append(item)
    if string_enum:
        schema["enum"] = string_enum
    else:
        schema.pop("enum", None)


def _adapt_schema_node(schema: dict[str, Any], *, default: str | None = None) -> dict[str, Any]:
    """Give one schema node a concrete type. Property nodes pass default='string' so a leaf
    without properties, items, or enum becomes a string. Other nodes are typed only when a
    structural signal is present.
    """
    node = dict(schema)
    props = node.get("properties")
    if isinstance(props, dict):
        node["properties"] = {
            name: _adapt_schema_node(prop, default="string") if isinstance(prop, dict) else prop
            for name, prop in props.items()
        }
    if "type" in node:
        node["type"] = _normalize_schema_type(node["type"])
    if isinstance(node.get("type"), list):
        non_null = [item for item in node["type"] if item != "null"]
        node["type"] = non_null[0] if non_null else "string"
    if "type" not in node and (
        default is not None
        or any(key in node for key in ("properties", "items", "additionalProperties", "enum"))
    ):
        node["type"] = _infer_missing_type(node, default=default or "string")
    _normalize_schema_enum(node)
    if node.get("type") == "object" and "properties" not in node:
        node["properties"] = {}
        node.setdefault("additionalProperties", True)
    if node.get("type") == "array":
        items = node.get("items")
        if items is None:
            node["items"] = {"type": "string"}
        elif isinstance(items, dict) and "type" not in items:
            node["items"] = _adapt_schema_node(items, default="string")
    return node


def simplify_schema_for_gigachat(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt JSON Schema to legacy GigaChat functions. Resolve local $defs/$ref references, reduce
    oneOf/anyOf/allOf to the first non-null variant while retaining wrapper metadata, infer a
    missing type, keep only string enum values, and ensure objects have properties. GigaChat
    rejects these unsupported schema constructs with 422. The caller must still validate output
    against the full schema, for example with Pydantic.
    """
    defs: dict[str, Any] = schema.get("$defs", {}) or schema.get("definitions", {}) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            # Replace a local $ref with its target.
            if "$ref" in node and isinstance(node["$ref"], str):
                ref_path = node["$ref"]
                if ref_path.startswith("#/$defs/") or ref_path.startswith("#/definitions/"):
                    name = ref_path.rsplit("/", 1)[-1]
                    target = defs.get(name)
                    if target is not None:
                        return _resolve(target)
                # Leave unresolved references unchanged.
                return {"type": "object", "properties": {}, "additionalProperties": True}

            # Choose the first non-null union variant and retain wrapper fields other than the
            # combinator.
            for combinator in ("oneOf", "anyOf", "allOf"):
                if combinator in node:
                    variants = node[combinator]
                    if not (isinstance(variants, list) and variants):
                        return {"type": "object", "properties": {}, "additionalProperties": True}
                    chosen = next(
                        (v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")),
                        variants[0],
                    )
                    resolved_variant = _resolve(chosen)
                    if not isinstance(resolved_variant, dict):
                        resolved_variant = {"type": "string"}
                    merged = dict(resolved_variant)
                    # Overlay wrapper metadata such as description, default, and title on the
                    # chosen variant. Wrapper values take precedence over metadata inherited
                    # from a referenced schema.
                    for key, value in node.items():
                        if key == combinator:
                            continue
                        merged[key] = value
                    return _adapt_schema_node(merged)

            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key in ("$defs", "definitions", "discriminator"):
                    continue
                cleaned[key] = _resolve(value)
            return _adapt_schema_node(cleaned)

        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    result = _resolve(schema)
    if not isinstance(result, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    # Function parameters are objects. A root with no type is not a scalar leaf.
    if "type" not in result:
        result = dict(result)
        result["type"] = "object"
    if result.get("type") == "object" and "properties" not in result:
        result = dict(result)
        result["properties"] = {}
        result.setdefault("additionalProperties", True)
    return result


def _longest_tag_prefix_suffix(text: str, tag: str) -> int:
    """Return how many trailing characters of text can still grow into tag."""
    text_lower = text.lower()
    tag_lower = tag.lower()
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if tag_lower.startswith(text_lower[-length:]):
            return length
    return 0


@dataclass
class ReasoningContent:
    """Visible text and the reasoning extracted from it."""

    content: str
    reasoning_content: str


class ReasoningContentParser:
    """Incrementally extract ``<think>...</think>`` from streamed content. A tag split across
    deltas stays buffered until it is complete or the stream ends.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False

    def feed(self, text: str | None) -> ReasoningContent:
        """Parse one fragment. Hold an incomplete tag prefix instead of emitting it."""
        if not text:
            return ReasoningContent(content="", reasoning_content="")
        self._buffer += text
        return self._consume(final=False)

    def flush(self) -> ReasoningContent:
        """Emit buffered text at the end of the stream. An unclosed think block is reasoning."""
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> ReasoningContent:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        while self._buffer:
            if self._in_reasoning:
                close_index = self._buffer.lower().find(_THINK_CLOSE_TAG)
                if close_index >= 0:
                    reasoning_parts.append(self._buffer[:close_index])
                    self._buffer = self._buffer[close_index + len(_THINK_CLOSE_TAG):]
                    self._in_reasoning = False
                    continue
                hold_len = 0 if final else _longest_tag_prefix_suffix(self._buffer, _THINK_CLOSE_TAG)
                emit_len = len(self._buffer) - hold_len
                reasoning_parts.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
                break
            open_index = self._buffer.lower().find(_THINK_OPEN_TAG)
            if open_index >= 0:
                content_parts.append(self._buffer[:open_index])
                self._buffer = self._buffer[open_index + len(_THINK_OPEN_TAG):]
                self._in_reasoning = True
                continue
            hold_len = 0 if final else _longest_tag_prefix_suffix(self._buffer, _THINK_OPEN_TAG)
            emit_len = len(self._buffer) - hold_len
            content_parts.append(self._buffer[:emit_len])
            self._buffer = self._buffer[emit_len:]
            break
        return ReasoningContent(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
        )


def extract_reasoning_from_content(text: str | None) -> ReasoningContent:
    """Extract every ``<think>`` block from a complete message."""
    parser = ReasoningContentParser()
    parsed = parser.feed(text)
    flushed = parser.flush()
    return ReasoningContent(
        content=f"{parsed.content}{flushed.content}",
        reasoning_content=f"{parsed.reasoning_content}{flushed.reasoning_content}",
    )


def split_reasoning_content(
    content: str | None,
    field_reasoning: str | None = None,
) -> tuple[str, str | None]:
    """Separate visible text from reasoning. A non-empty provider field wins; think tags are
    still removed from the visible text so they are not returned as the answer.
    """
    parsed = extract_reasoning_from_content(content if isinstance(content, str) else None)
    if isinstance(field_reasoning, str) and field_reasoning:
        return parsed.content, field_reasoning
    return parsed.content, parsed.reasoning_content or None


def _resolve_scope(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return get_env("LLM_AUTH_SCOPE")
