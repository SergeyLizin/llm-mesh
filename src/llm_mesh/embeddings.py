"""Shared embedding request, response, and response parsing.

One call returns a dense vector and, when the provider sends it, a sparse
vector. Document text is sent as-is. Query text is wrapped with the model's
``query_instruction`` when that template is set. A symmetric model leaves the
template empty, so a query and a document are the same string.

OpenAI-compatible gateways and GigaChat both answer with ``data[].embedding``
and ``data[].index``. The index is the order, not the array position.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_mesh.types import LLMValidationError

TASK_CHAT = "chat"
TASK_EMBEDDINGS = "embeddings"
TASK_RERANK = "rerank"
TASKS = frozenset({TASK_CHAT, TASK_EMBEDDINGS, TASK_RERANK})

_MISSING_INDEX = 2**31 - 1


class Embedding(BaseModel):
    """One input's vectors. ``sparse`` is absent when the provider did not return one."""

    model_config = ConfigDict(extra="forbid")

    dense: list[float]
    sparse: dict[str, float] | None = None


class QueryInstruction:
    """Template applied to query text. ``{query}`` is the only placeholder.

    An empty template leaves the query unchanged. A non-empty template without
    the placeholder would turn every query into the same string, so it is
    rejected.
    """

    PLACEHOLDER = "{query}"

    def __init__(self, template: str) -> None:
        self.template = template

    @classmethod
    def parse(cls, template: str | None) -> QueryInstruction:
        if template is None or not str(template).strip():
            return cls("")
        text = str(template)
        if cls.PLACEHOLDER not in text:
            raise ValueError(
                "query_instruction must contain {query}, "
                f"got {template!r}"
            )
        return cls(text)

    def is_empty(self) -> bool:
        return self.template == ""

    def apply(self, query: str) -> str:
        if not self.template:
            return query
        return self.template.replace(self.PLACEHOLDER, query)


def normalize_task(value: object) -> str:
    """Catalog task. Missing means chat. Anything else is an error."""
    if value is None or str(value).strip() == "":
        return TASK_CHAT
    task = str(value).strip().lower()
    if task not in TASKS:
        raise ValueError(
            f"task must be 'chat', 'embeddings', or 'rerank', got {value!r}"
        )
    return task


def normalize_embedding_side(task: str) -> str:
    """``document`` or ``query``. These are not catalog tasks."""
    if task not in ("document", "query"):
        raise ValueError("embedding task must be 'document' or 'query'")
    return task


def checked_dimensions(value: object) -> int | None:
    """Positive output size, or None when the model default should be used.

    Zero and None omit the field. A negative value is rejected: an indexing
    run would otherwise store vectors of the wrong width.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"dimensions must be a positive integer, got {value!r}")
    if value < 0:
        raise ValueError(
            f"dimensions={value} cannot be negative; "
            "0 or omitting the field keeps the model default"
        )
    if value == 0:
        return None
    return value


def embedding_options(
    client: Any,
    *,
    dimensions: int | None,
    sparse: bool | None,
    allow_sparse: bool,
) -> tuple[QueryInstruction, bool, int | None]:
    """Defaults from the client, overridden by the call. Sparse is refused when the API has none."""
    use_sparse = client._embedding_sparse if sparse is None else bool(sparse)
    if use_sparse and not allow_sparse:
        provider = getattr(client, "PROVIDER", type(client).__name__)
        raise LLMValidationError(f"{provider}: sparse embeddings are not supported")
    if dimensions is None:
        use_dimensions = client._embedding_dimensions
    else:
        use_dimensions = checked_dimensions(dimensions)
    instruction = getattr(client, "_query_instruction", None)
    if not isinstance(instruction, QueryInstruction):
        instruction = QueryInstruction.parse(None)
    return instruction, use_sparse, use_dimensions


def prepare_inputs(
    texts: list[str],
    task: str,
    instruction: QueryInstruction,
) -> list[str]:
    """Wrap queries when the model has an instruction. Documents stay as given."""
    side = normalize_embedding_side(task)
    if side == "query" and not instruction.is_empty():
        return [instruction.apply(text) for text in texts]
    return list(texts)


def embeddings_url(base_url: str) -> str:
    """OpenAI-compatible embeddings URL.

    Bases are written with ``/v1`` and without it. A base that already ends in
    ``/embeddings`` is kept. Otherwise ``/v1`` is inserted once.
    """
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return base + "/embeddings"
    return base + "/v1/embeddings"


def embedding_body(
    model: str,
    texts: list[str],
    *,
    sparse: bool = False,
    dimensions: int | None = None,
) -> dict[str, Any]:
    """OpenAI-compatible ``/embeddings`` body. GigaChat accepts the same shape
    without ``return_sparse`` and ``dimensions``.
    """
    body: dict[str, Any] = {"model": model, "input": texts}
    if sparse:
        body["encoding_format"] = "float"
        body["return_sparse"] = True
    width = checked_dimensions(dimensions)
    if width is not None:
        body["dimensions"] = width
    return body


def parse_embedding_response(
    payload: dict[str, Any],
    expected: int,
    *,
    sparse: bool = False,
) -> list[Embedding]:
    """Map an OpenAI-shaped payload onto one embedding per input, in input order."""
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise LLMValidationError("embeddings: response data is empty")
    ordered = sorted(
        (item for item in data if isinstance(item, dict)),
        key=lambda item: _item_index(item),
    )
    results: list[Embedding] = []
    for item in ordered:
        dense = _dense(item.get("embedding"))
        parsed_sparse = _sparse(item.get("sparse_embedding")) if sparse else None
        results.append(Embedding(dense=dense, sparse=parsed_sparse))
    if len(results) != expected:
        raise LLMValidationError(
            f"embeddings: got {len(results)} vectors for {expected} inputs"
        )
    return results


def _item_index(item: dict[str, Any]) -> int:
    index = item.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return _MISSING_INDEX
    return index


def _dense(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise LLMValidationError("embeddings: dense vector is missing")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise LLMValidationError("embeddings: dense vector is not numeric") from exc


def _sparse(value: object) -> dict[str, float] | None:
    """Accept ``{indices, values}`` or a token-to-weight map. Non-positive weights are dropped."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LLMValidationError("embeddings: sparse_embedding is not an object")
    if "indices" in value and "values" in value:
        indices = value.get("indices")
        weights = value.get("values")
        if not isinstance(indices, list) or not isinstance(weights, list):
            raise LLMValidationError("embeddings: sparse indices and values must be arrays")
        if len(indices) != len(weights):
            raise LLMValidationError(
                "embeddings: sparse indices and values have different lengths"
            )
        parsed: dict[str, float] = {}
        for index, weight in zip(indices, weights):
            number = float(weight)
            if number > 0:
                parsed[str(int(index))] = number
        return parsed
    parsed = {}
    for key, weight in value.items():
        number = float(weight)
        if number > 0:
            parsed[str(key)] = number
    return parsed


__all__ = [
    "TASK_CHAT",
    "TASK_EMBEDDINGS",
    "TASK_RERANK",
    "Embedding",
    "QueryInstruction",
    "checked_dimensions",
    "embedding_body",
    "embedding_options",
    "embeddings_url",
    "normalize_embedding_side",
    "normalize_task",
    "parse_embedding_response",
    "prepare_inputs",
]
