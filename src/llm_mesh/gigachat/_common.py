"""Stateless GigaChat helpers: environment parsing, OAuth expiry normalization, and function-schema
simplification. Transport and request policies are implemented by the client.
"""

from __future__ import annotations

from llm_mesh.config import get_env
import time
from typing import Any



def _env_nonneg_int(name: str, default: int) -> int:
    """Read a nonnegative integer from the environment; use default for missing, invalid, or
    negative values.
    """
    raw = get_env(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


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


def simplify_schema_for_gigachat(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt JSON Schema to legacy GigaChat functions. Resolve local $defs/$ref references, reduce
    oneOf/anyOf/allOf to the first non-null variant while retaining wrapper metadata, and ensure
    objects have properties. GigaChat rejects these unsupported schema constructs with 422. The
    caller must still validate output against the full schema, for example with Pydantic.
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
                    return merged

            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key in ("$defs", "definitions", "discriminator"):
                    continue
                cleaned[key] = _resolve(value)

            # GigaChat expects a scalar type: select the first non-null type. Optionality is
            # represented by omission from required.
            if isinstance(cleaned.get("type"), list):
                non_null = [t for t in cleaned["type"] if t != "null"]
                cleaned["type"] = non_null[0] if non_null else "string"

            # Add an empty properties map to objects that lack one.
            if cleaned.get("type") == "object" and "properties" not in cleaned:
                cleaned["properties"] = {}
                cleaned.setdefault("additionalProperties", True)
            return cleaned

        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    result = _resolve(schema)
    if not isinstance(result, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    return result


def _resolve_scope(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return get_env("LLM_AUTH_SCOPE")
