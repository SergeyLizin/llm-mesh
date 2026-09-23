"""Rerank documents against a query.

Two gateway wires, both measured against live servers:

* ``score`` — ``POST {base}/score`` with ``text_1`` / ``text_2``. A base that
  ends in ``/v1`` drops that suffix first: ``/v1/score`` is not the endpoint.
* ``llama`` — ``POST {base}/v1/rerank`` with ``query`` / ``documents``.

A local cross-encoder scores the same pairs in-process. It is optional:
``sentence-transformers`` is imported only when a local model is loaded.

A short or malformed response raises. Returning the input order would look
like a ranking that never ran.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from llm_mesh.types import LLMValidationError

logger = logging.getLogger(__name__)

PROTOCOL_SCORE = "score"
PROTOCOL_LLAMA = "llama"
_PROTOCOLS = frozenset({PROTOCOL_SCORE, PROTOCOL_LLAMA})


class RerankHit(BaseModel):
    """One document after scoring. ``index`` is its position in the input."""

    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    score: float


def normalize_rerank_protocol(value: object) -> str:
    """``score`` when the route omits the protocol. Anything else is an error."""
    if value is None or str(value).strip() == "":
        return PROTOCOL_SCORE
    protocol = str(value).strip().lower()
    if protocol not in _PROTOCOLS:
        raise ValueError(
            f"rerank_protocol must be 'score' or 'llama', got {value!r}"
        )
    return protocol


def checked_top_k(value: object) -> int | None:
    """Positive cut, or None when every scored document should be returned."""
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"rerank_top_k must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"rerank_top_k={value} must be positive")
    return value


def checked_min_score(value: object) -> float:
    """Zero disables the floor. A negative floor is rejected."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"rerank_min_score must be a number, got {value!r}")
    if value < 0:
        raise ValueError(f"rerank_min_score={value} cannot be negative")
    return float(value)


def score_url(base_url: str) -> str:
    """Gateway ``/score`` URL. ``/v1`` is removed; ``/v1/score`` is a 404."""
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/score"):
        return base
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/score"


def llama_rerank_url(base_url: str) -> str:
    """llama.cpp ``/v1/rerank``. A base that already ends in ``/v1`` is kept."""
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/rerank"):
        return base
    if base.endswith("/v1"):
        return base + "/rerank"
    return base + "/v1/rerank"


def score_body(model: str, query: str, documents: list[str]) -> dict[str, Any]:
    return {
        "model": model,
        "encoding_format": "float",
        "text_1": query,
        "text_2": documents,
    }


def llama_body(model: str, query: str, documents: list[str]) -> dict[str, Any]:
    return {"model": model, "query": query, "documents": documents}


def parse_score_payload(payload: dict[str, Any], expected: int) -> list[tuple[int, float]]:
    """``data[{index, score}]``. Every input index is required exactly once."""
    data = payload.get("data")
    if not isinstance(data, list):
        raise LLMValidationError("rerank: response data is not an array")
    return _pairs(data, expected, score_key="score", label="data")


def parse_llama_payload(payload: dict[str, Any], expected: int) -> list[tuple[int, float]]:
    """``results[{index, relevance_score}]``. Every input index is required once."""
    data = payload.get("results")
    if not isinstance(data, list):
        raise LLMValidationError("rerank: response results is not an array")
    return _pairs(data, expected, score_key="relevance_score", label="results")


def rank_hits(
    documents: list[str],
    pairs: list[tuple[int, float]],
    *,
    top_k: int | None,
    min_score: float,
) -> list[RerankHit]:
    """Highest score first. Equal scores keep the earlier input.

    A positive ``min_score`` drops documents under the floor. When every
    document is under it, the floor is not applied: an empty list would hide
    the scores the caller needs to see.
    """
    width = checked_top_k(top_k)
    floor = checked_min_score(min_score)
    ordered = sorted(pairs, key=lambda item: (-item[1], item[0]))
    if floor > 0.0:
        above = [item for item in ordered if item[1] >= floor]
        if above:
            ordered = above
        else:
            logger.warning(
                "rerank: every score is below min_score=%s; returning the ranking without the floor",
                floor,
            )
    if width is not None:
        ordered = ordered[:width]
    return [
        RerankHit(index=index, text=documents[index], score=score)
        for index, score in ordered
    ]


class LocalCrossEncoder:
    """In-process cross-encoder. The model is loaded on the first ``rerank`` call.

    Pass ``scorer`` to supply scores without ``sentence-transformers``. A
    production instance leaves ``scorer`` unset and names a Hugging Face model.
    """

    def __init__(
        self,
        model: str,
        *,
        scorer: Callable[[str, list[str]], list[float]] | None = None,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        if not (model or "").strip() and scorer is None:
            raise ValueError("LocalCrossEncoder requires a model name")
        self.model = (model or "").strip()
        self._scorer = scorer
        self._encoder: Any = None
        self._top_k = checked_top_k(top_k)
        self._min_score = checked_min_score(min_score)

    async def aclose(self) -> None:
        self._encoder = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankHit]:
        if model and self.model and model != self.model:
            raise LLMValidationError(
                f"LocalCrossEncoder is loaded as {self.model!r}, not {model!r}"
            )
        if not documents:
            return []
        if not isinstance(query, str) or not query.strip():
            raise LLMValidationError("rerank: query is empty")
        if self._scorer is not None:
            scores = [float(score) for score in self._scorer(query, documents)]
        else:
            scores = await asyncio.to_thread(self._predict, query, documents)
        if len(scores) != len(documents):
            raise LLMValidationError(
                f"rerank: got {len(scores)} scores for {len(documents)} documents"
            )
        pairs = [(index, float(score)) for index, score in enumerate(scores)]
        return rank_hits(
            documents,
            pairs,
            top_k=self._top_k if top_k is None else top_k,
            min_score=self._min_score if min_score is None else min_score,
        )

    def _predict(self, query: str, documents: list[str]) -> list[float]:
        encoder = self._load()
        raw = encoder.predict([(query, text) for text in documents])
        return [float(score) for score in raw]

    def _load(self) -> Any:
        if self._encoder is None:
            cross_encoder = _import_cross_encoder()
            self._encoder = cross_encoder(self.model)
        return self._encoder


def _import_cross_encoder() -> Callable[..., Any]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise LLMValidationError(
            "local rerank requires the sentence-transformers package"
        ) from exc
    return CrossEncoder


def _pairs(
    rows: list[Any],
    expected: int,
    *,
    score_key: str,
    label: str,
) -> list[tuple[int, float]]:
    found: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise LLMValidationError(f"rerank: {label} entry is not an object")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise LLMValidationError(f"rerank: {label} entry has no integer index")
        if index < 0 or index >= expected:
            raise LLMValidationError(
                f"rerank: index {index} is outside 0..{expected - 1}"
            )
        if index in found:
            raise LLMValidationError(f"rerank: index {index} is repeated")
        raw = row.get(score_key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise LLMValidationError(f"rerank: index {index} has no numeric {score_key}")
        found[index] = float(raw)
    if len(found) != expected:
        raise LLMValidationError(
            f"rerank: got {len(found)} scores for {expected} documents"
        )
    return [(index, found[index]) for index in range(expected)]


__all__ = [
    "PROTOCOL_LLAMA",
    "PROTOCOL_SCORE",
    "LocalCrossEncoder",
    "RerankHit",
    "checked_min_score",
    "checked_top_k",
    "llama_body",
    "llama_rerank_url",
    "normalize_rerank_protocol",
    "parse_llama_payload",
    "parse_score_payload",
    "rank_hits",
    "score_body",
    "score_url",
]
