"""Local token counts for OpenAI-compatible clients.

OpenAI has no tokenizer endpoint. ``count_tokens`` uses tiktoken, so the
call does not contact the gateway and does not prove that the route is up.
The count is the string itself. Chat-message overhead is not added: that
overhead depends on the model and would be a guess.

A model name tiktoken knows (``gpt-4o``, ``gpt-4``) selects its encoding.
Any other name needs ``tiktoken_encoding`` on the client or the catalog
route. Counting an unknown model with a stand-in encoding would report a
budget the model's own tokenizer does not use.
"""

from __future__ import annotations

from llm_mesh.types import LLMValidationError

_ENCODINGS: dict[str, object] = {}


def encoding_name(model: str, explicit: str | None) -> str:
    """Resolve the tiktoken encoding name. An unknown model is an error."""
    if explicit:
        name = explicit.strip()
        if not name:
            raise LLMValidationError("tiktoken_encoding is empty")
        _load(name)
        return name
    chosen = (model or "").strip()
    if not chosen:
        raise LLMValidationError(
            "count_tokens needs a model name or tiktoken_encoding"
        )
    try:
        import tiktoken
    except ImportError as exc:
        raise LLMValidationError(
            "count_tokens requires the tiktoken package"
        ) from exc
    try:
        return tiktoken.encoding_for_model(chosen).name
    except KeyError as exc:
        raise LLMValidationError(
            f"tiktoken has no encoding for model {chosen!r}. "
            "Set tiktoken_encoding on the client or the catalog route "
            "(for example 'o200k_base' or 'cl100k_base')."
        ) from exc


def count_text_tokens(texts: list[str], encoding_name_value: str) -> list[int]:
    """Count each string with that encoding. Special-token text is ordinary BPE."""
    encoding = _load(encoding_name_value)
    return [
        len(encoding.encode(text, disallowed_special=()))
        for text in texts
    ]


def _load(name: str):
    cached = _ENCODINGS.get(name)
    if cached is not None:
        return cached
    try:
        import tiktoken
    except ImportError as exc:
        raise LLMValidationError(
            "count_tokens requires the tiktoken package"
        ) from exc
    try:
        encoding = tiktoken.get_encoding(name)
    except (KeyError, ValueError) as exc:
        raise LLMValidationError(
            f"unknown tiktoken encoding {name!r}"
        ) from exc
    _ENCODINGS[name] = encoding
    return encoding
