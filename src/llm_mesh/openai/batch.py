"""OpenAI-compatible Batch API. Upload a JSONL file of chat completions, create a
batch, poll until a terminal status, and map results back in input order.

The wire shape is the official one (and the one compatible gateways copy): each
line is ``{custom_id, method, url, body}``, files are uploaded with
``purpose=batch``, and the batch itself asks for ``/v1/chat/completions`` with a
24h completion window. This repo's base URL already includes ``/v1``; batch paths
are ``{base}/files`` and ``{base}/batches``, the same tolerant join as
``chat_completions_url``.

Bodies and response parsing stay on ``OpenAIClient``. This module only owns the
files/batches transport, so gateway adaptations cannot drift from the live path.
Canary scanning is not applied here, matching GigaChat batch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence, cast

import httpx

from llm_mesh._common import _env_float_default, _env_is_disabled
from llm_mesh.config import get_env
from llm_mesh.base import BudgetLedger
from llm_mesh.types import (
    Budget,
    BudgetState,
    LLMAuthError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)

from .client import OpenAIClient, OpenAIError

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(get_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(get_env(name, str(default)))
    except (TypeError, ValueError):
        return default

# Official OpenAI limits for one input file. v1 does not chunk past them.
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

# Fixed by the Batch API. Callers cannot ask for a shorter window.
COMPLETION_WINDOW = "24h"
BATCH_ENDPOINT = "/v1/chat/completions"
EMBEDDINGS_BATCH_ENDPOINT = "/v1/embeddings"

# 429 and 529 are overload; 500-504 are transient gateway failures. Same backoff
# curve as GigaChat batch: start at LLM_BATCH_429_BACKOFF_START, grow ×1.5, cap 45s.
_RETRYABLE_STATUS = frozenset({429, 500, 501, 502, 503, 504, 529})
_TERMINAL_STATUS = frozenset({"completed", "failed", "expired", "cancelled"})
_BACKOFF_CAP_S = 45.0


class OpenAIBatchError(LLMError):
    """Batch upload, submission, polling, or per-item result failure."""


def _batch_base_url(base_url: str | None, client: OpenAIClient | None) -> str:
    """Return ``<base>/`` where ``<base>`` already includes ``/v1`` when the chat URL does.

    ``OpenAIClient.URL`` is the chat-completions URL. Strip that suffix so ``files``
    and ``batches`` sit next to ``chat/completions``, not under it. A trailing slash
    is required: httpx joins a relative path against the base, and without the slash
    ``urljoin`` drops the last segment.
    """
    if base_url:
        raw = base_url.strip()
    elif client is not None:
        url = client.URL
        suffix = "/chat/completions"
        raw = url[: -len(suffix)] if url.endswith(suffix) else url
    else:
        raw = get_env("LLM_BASE_URL", "")
    raw = raw.strip().rstrip("/")
    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")]
    if not raw:
        raise OpenAIBatchError(
            "OpenAIBatchClient: missing base_url — pass base_url or an OpenAIClient"
        )
    return raw + "/"


class OpenAIBatchClient:
    """Async OpenAI batch client. An injected ``http_client`` stays caller-owned and
    is never closed here. The owned client is created on the first request and lives
    until ``aclose``.
    """

    def __init__(
        self,
        *,
        client: OpenAIClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        verify: bool | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        max_wait_s: float | None = None,
        http_retries: int | None = None,
        backoff_start_s: float | None = None,
        budget: Budget | None = None,
    ) -> None:
        if client is None and not api_key:
            raise LLMAuthError(
                "OpenAIBatchClient: requires `client` (OpenAIClient) or `api_key`"
            )
        self._openai = client
        # Explicit key wins. Otherwise reuse the completion client's key.
        self._api_key = api_key or (client._key if client is not None else None)
        self._base_url = _batch_base_url(base_url, client)
        self._external_client = http_client
        self._owned_client: httpx.AsyncClient | None = None
        # GigaChat batch defaults verify off because its corporate CA is not in the
        # trust store. OpenAI endpoints are public TLS: verify unless the caller or
        # the wrapped client already turned it off.
        if verify is not None:
            self._verify = verify
        elif client is not None:
            self._verify = client._verify
        else:
            self._verify = not _env_is_disabled("LLM_VERIFY_SSL", default="1")
        self._timeout = httpx.Timeout(
            timeout_s if timeout_s is not None
            else _env_float_default("LLM_HTTP_TIMEOUT", 120.0, logger=logger)
        )
        self._poll_interval = (
            poll_interval_s
            if poll_interval_s is not None
            else _env_float("LLM_BATCH_POLL_INTERVAL_S", 5.0)
        )
        self._max_wait = (
            max_wait_s if max_wait_s is not None else _env_float("LLM_BATCH_MAX_WAIT_S", 3600.0)
        )
        self._http_retries = max(
            1, http_retries if http_retries is not None else _env_int("LLM_BATCH_HTTP_RETRIES", 12)
        )
        self._backoff_start = (
            backoff_start_s
            if backoff_start_s is not None
            else _env_float("LLM_BATCH_429_BACKOFF_START", 1.0)
        )
        self._budget_ledger = BudgetLedger(budget)

    def reset_budget(self) -> None:
        self._budget_ledger.reset()

    def budget_state(self) -> BudgetState:
        return self._budget_ledger.state()

    def _http(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._owned_client is None:
            self._owned_client = httpx.AsyncClient(
                base_url=self._base_url, verify=self._verify, timeout=self._timeout,
            )
        return self._owned_client

    async def aclose(self) -> None:
        """Close the client this instance created. A caller-owned client is left open."""
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

    def _auth_headers(self) -> dict[str, str]:
        key = self._api_key
        if not key and self._openai is not None:
            key = self._openai._key
        if not key:
            raise LLMAuthError("OpenAIBatchClient: no API key")
        headers = {"Authorization": f"Bearer {key}"}
        if self._openai is not None:
            headers.update(self._openai._extra_headers)
        return headers

    async def _request(
        self, method: str, path: str, *, context: str, **kwargs: Any
    ) -> httpx.Response:
        """One authenticated call. Retry 429/529/500-504 inside the shared budget."""
        delay = self._backoff_start
        extra_headers = kwargs.pop("headers", None) or {}
        headers = {**self._auth_headers(), **extra_headers}
        last_status: int | None = None
        for attempt in range(self._http_retries):
            resp = await self._http().request(method, path, headers=headers, **kwargs)
            if resp.status_code == 200:
                return resp
            last_status = resp.status_code
            if resp.status_code in (401, 403):
                raise LLMAuthError(
                    f"OpenAI Batch API ({context}): authentication error — {resp.text[:300]}"
                )
            if resp.status_code in _RETRYABLE_STATUS and attempt + 1 < self._http_retries:
                logger.warning(
                    "OpenAI Batch: %s (%s), waiting %.1fs (%d/%d)",
                    resp.status_code, context, delay, attempt + 1, self._http_retries,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, _BACKOFF_CAP_S)
                continue
            raise OpenAIBatchError(
                f"OpenAI Batch API ({context}): error {resp.status_code} — {resp.text[:300]}"
            )
        raise OpenAIBatchError(
            f"OpenAI Batch: attempts exhausted for {context} (status {last_status})"
        )

    def build_chat_lines(
        self, requests: Sequence[LLMRequest], *, model: str | None = None,
        ids: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        """JSONL records in input order. Each line's model is ``request.model`` or the
        batch default. The body is exactly ``OpenAIClient.build_completion_body``.
        """
        if self._openai is None:
            raise OpenAIBatchError(
                "OpenAIBatchClient: a completion client is required to build request bodies"
            )
        resolved = model or self._openai._model
        if ids is not None and len(ids) != len(requests):
            raise OpenAIBatchError(
                "OpenAIBatchClient: line ids must match the request list"
            )
        lines: list[dict[str, Any]] = []
        for n, req in enumerate(requests):
            i = n if ids is None else ids[n]
            line_model = req.model or resolved
            # Stamp the batch default onto the request so the body builder does not
            # fall back to a different model configured on the client.
            prepared = req if req.model else req.model_copy(update={"model": line_model or None})
            structured = req.mode != "text"
            lines.append({
                "custom_id": str(i),
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": self._openai.build_completion_body(prepared, structured=structured),
            })
        return lines

    def build_embedding_lines(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        task: str = "document",
        dimensions: int | None = None,
        sparse: bool | None = None,
    ) -> list[dict[str, Any]]:
        """One ``/v1/embeddings`` line per input. Query instructions are applied first."""
        from llm_mesh.embeddings import (
            embedding_body,
            embedding_options,
            prepare_inputs,
        )

        if self._openai is None:
            raise OpenAIBatchError(
                "OpenAIBatchClient: a completion client is required to build embedding bodies"
            )
        instruction, use_sparse, use_dimensions = embedding_options(
            self._openai, dimensions=dimensions, sparse=sparse, allow_sparse=True,
        )
        prepared = prepare_inputs(list(texts), task, instruction)
        chosen = model or self._openai._model
        lines: list[dict[str, Any]] = []
        for i, text in enumerate(prepared):
            lines.append({
                "custom_id": str(i),
                "method": "POST",
                "url": EMBEDDINGS_BATCH_ENDPOINT,
                "body": embedding_body(
                    chosen, [text], sparse=use_sparse, dimensions=use_dimensions,
                ),
            })
        return lines

    @staticmethod
    def build_jsonl(lines: Sequence[dict[str, Any]]) -> bytes:
        return "\n".join(json.dumps(line, ensure_ascii=False) for line in lines).encode("utf-8")

    @staticmethod
    def _check_request_count(requests: Sequence[LLMRequest]) -> None:
        # Reject before building bodies. Past 50 000 lines the file is illegal anyway,
        # and constructing it would only allocate a payload we are about to refuse.
        if len(requests) > MAX_BATCH_REQUESTS:
            raise OpenAIBatchError(
                f"OpenAI Batch API accepts at most {MAX_BATCH_REQUESTS} requests "
                f"per input file (got {len(requests)})"
            )

    @staticmethod
    def _check_payload_size(payload: bytes) -> None:
        if len(payload) > MAX_BATCH_FILE_BYTES:
            raise OpenAIBatchError(
                f"OpenAI Batch API accepts at most {MAX_BATCH_FILE_BYTES} bytes "
                f"per input file (got {len(payload)})"
            )

    async def upload_file(self, jsonl_data: bytes) -> str:
        resp = await self._request(
            "POST", "files", context="upload_file",
            files={"file": ("batch.jsonl", jsonl_data, "application/jsonl")},
            data={"purpose": "batch"},
        )
        payload = resp.json()
        file_id = payload.get("id") if isinstance(payload, dict) else None
        if not file_id:
            raise OpenAIBatchError("OpenAI Batch API: file upload did not return an id")
        return str(file_id)

    async def create_batch(
        self, input_file_id: str, *, endpoint: str = BATCH_ENDPOINT,
    ) -> dict[str, Any]:
        resp = await self._request(
            "POST", "batches", context="create_batch",
            json={
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": COMPLETION_WINDOW,
            },
        )
        data = resp.json()
        if not isinstance(data, dict) or not data.get("id"):
            raise OpenAIBatchError("OpenAI Batch API: batch create did not return an id")
        return data

    async def get_batch(self, batch_id: str) -> dict[str, Any]:
        resp = await self._request("GET", f"batches/{batch_id}", context="get_batch")
        data = resp.json()
        if not isinstance(data, dict):
            raise OpenAIBatchError(
                f"OpenAI Batch API: unexpected status payload ({type(data).__name__})"
            )
        return data

    async def wait_for_completion(
        self,
        batch_id: str,
        *,
        poll_interval: float | None = None,
        max_wait: float | None = None,
    ) -> dict[str, Any]:
        """Poll until a terminal status. Non-completed terminals raise with the batch
        ``errors`` object attached. Anything else keeps polling until ``max_wait``.
        """
        interval = poll_interval if poll_interval is not None else self._poll_interval
        deadline = max_wait if max_wait is not None else self._max_wait
        loop = asyncio.get_running_loop()
        start = loop.time()
        while True:
            data = await self.get_batch(batch_id)
            status = str(data.get("status", "unknown"))
            if status in _TERMINAL_STATUS:
                if status != "completed":
                    errors = data.get("errors")
                    detail = f"Batch {batch_id} finished with status: {status}"
                    if errors:
                        detail += f" errors={errors}"
                    raise OpenAIBatchError(detail)
                return data
            if loop.time() - start >= deadline:
                raise LLMTimeoutError(
                    f"Batch {batch_id} did not complete within {deadline}s (status: {status})"
                )
            await asyncio.sleep(interval)

    async def download_results(self, output_file_id: str) -> list[dict[str, Any]]:
        resp = await self._request(
            "GET", f"files/{output_file_id}/content", context="download_results",
        )
        results: list[dict[str, Any]] = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("OpenAI Batch result: invalid JSONL line: %s", line[:100])
                results.append({})
                continue
            if isinstance(parsed, dict):
                results.append(parsed)
            else:
                results.append({})
        return results

    async def run_chat_batch(
        self,
        requests: Sequence[LLMRequest],
        *,
        model: str | None = None,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        return_exceptions: bool = False,
    ) -> list[LLMResponse | BaseException]:
        """Submit, poll, download, and map a chat batch. Responses follow input order.

        Per-item failures (top-level ``error``, non-200 ``response.status_code``, a
        missing ``custom_id``, or a parse error) become ``OpenAIBatchError``. The
        first one is raised unless ``return_exceptions`` is set, in which case each
        sits at its input position. A sent/received count mismatch raises always:
        the file cannot be mapped, so an empty success would be a fabrication.
        """
        if not requests:
            return []
        if self._openai is None:
            raise OpenAIBatchError(
                "OpenAIBatchClient: a completion client is required to build request bodies"
            )
        self._budget_ledger.check(self._openai.PROVIDER)
        from llm_mesh.hooks import guard_batch_requests

        provider = self._openai.PROVIDER
        accepted, blocked = guard_batch_requests(
            requests, provider=provider, return_exceptions=return_exceptions,
        )
        if not accepted:
            return [blocked[i] for i in range(len(requests))]
        passing = [req for _, req in accepted]
        self._check_request_count(passing)
        lines = self.build_chat_lines(
            passing, model=model, ids=[i for i, _ in accepted],
        )
        payload = self.build_jsonl(lines)
        self._check_payload_size(payload)

        file_id = await self.upload_file(payload)
        batch = await self.create_batch(file_id)
        status = await self.wait_for_completion(
            str(batch["id"]), poll_interval=poll_interval, max_wait=max_wait,
        )
        output_file_id = status.get("output_file_id")
        if not output_file_id:
            raise OpenAIBatchError(
                f"OpenAI Batch API: missing output_file_id (batch_id={batch.get('id')!r}, "
                f"status={status.get('status')!r})"
            )
        raw_results = await self.download_results(str(output_file_id))
        if len(raw_results) != len(accepted):
            raise OpenAIBatchError(
                f"OpenAI Batch: result count mismatch "
                f"(sent {len(accepted)} ids, received {len(raw_results)})"
            )

        by_id: dict[str, dict[str, Any]] = {}
        for entry in raw_results:
            custom_id = entry.get("custom_id")
            if custom_id is None or custom_id == "":
                continue
            by_id[str(custom_id)] = entry

        resolved = model or self._openai._model
        guarded = dict(accepted)
        out: list[LLMResponse | BaseException] = []
        for i, _req in enumerate(requests):
            if i in blocked:
                out.append(blocked[i])
                continue
            req = guarded[i]
            entry = by_id.get(str(i))
            err: BaseException | None = None
            response: LLMResponse | None = None
            if entry is None:
                err = OpenAIBatchError(f"Batch: missing custom_id for id={i}")
            else:
                err, response = self._map_entry(entry, req, i, resolved)
            if err is not None:
                if not return_exceptions:
                    raise err
                out.append(err)
            else:
                assert response is not None
                self._budget_ledger.add(response.usage)
                out.append(response)
        return out

    def _map_entry(
        self,
        entry: dict[str, Any],
        req: LLMRequest,
        index: int,
        resolved_model: str,
    ) -> tuple[BaseException | None, LLMResponse | None]:
        """Turn one result line into a response or a per-item error. Never an empty success."""
        top_error = entry.get("error")
        if top_error:
            return OpenAIBatchError(f"Batch subtask id={index}: {top_error}"), None
        response_obj = entry.get("response")
        if not isinstance(response_obj, dict):
            return OpenAIBatchError(
                f"Batch subtask id={index}: missing response object"
            ), None
        status_code = response_obj.get("status_code")
        if status_code != 200:
            body_preview = response_obj.get("body")
            return OpenAIBatchError(
                f"Batch subtask id={index}: response status {status_code} — {body_preview}"
            ), None
        body = response_obj.get("body")
        if not isinstance(body, dict):
            return OpenAIBatchError(
                f"Batch subtask id={index}: response body is not an object"
            ), None
        line_model = req.model or resolved_model
        prepared = req if req.model else req.model_copy(update={"model": line_model or None})
        try:
            parsed = self._openai.response_from_payload(  # type: ignore[union-attr]
                body, prepared, req.mode != "text",
            )
        except (LLMError, OpenAIError, KeyError, TypeError, ValueError) as exc:
            return OpenAIBatchError(f"Batch subtask id={index}: {exc}"), None
        return None, cast("LLMResponse", parsed)

    async def run_embedding_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        task: str = "document",
        dimensions: int | None = None,
        sparse: bool | None = None,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Submit an embeddings batch and return one vector per input, in input order.

        The file endpoint is ``/v1/embeddings``. A failed line raises, or is
        returned in place when ``return_exceptions`` is set. A count mismatch
        raises for the whole batch.
        """
        from llm_mesh.embeddings import Embedding, parse_embedding_response

        items = list(texts)
        if not items:
            return []
        if self._openai is None:
            raise OpenAIBatchError(
                "OpenAIBatchClient: a completion client is required to build embedding bodies"
            )
        if len(items) > MAX_BATCH_REQUESTS:
            raise OpenAIBatchError(
                f"OpenAI Batch API accepts at most {MAX_BATCH_REQUESTS} requests "
                f"per input file (got {len(items)})"
            )
        lines = self.build_embedding_lines(
            items, model=model, task=task, dimensions=dimensions, sparse=sparse,
        )
        payload = self.build_jsonl(lines)
        self._check_payload_size(payload)
        file_id = await self.upload_file(payload)
        batch = await self.create_batch(file_id, endpoint=EMBEDDINGS_BATCH_ENDPOINT)
        status = await self.wait_for_completion(
            str(batch["id"]), poll_interval=poll_interval, max_wait=max_wait,
        )
        output_file_id = status.get("output_file_id")
        if not output_file_id:
            raise OpenAIBatchError(
                f"OpenAI Batch API: missing output_file_id (batch_id={batch.get('id')!r}, "
                f"status={status.get('status')!r})"
            )
        raw_results = await self.download_results(str(output_file_id))
        if len(raw_results) != len(items):
            raise OpenAIBatchError(
                f"OpenAI Batch: result count mismatch "
                f"(sent {len(items)} ids, received {len(raw_results)})"
            )
        by_id = {
            str(entry.get("custom_id")): entry
            for entry in raw_results
            if entry.get("custom_id") not in (None, "")
        }
        use_sparse = lines[0]["body"].get("return_sparse") is True
        out: list[Embedding | BaseException] = []
        for i in range(len(items)):
            entry = by_id.get(str(i))
            if entry is None:
                err: BaseException = OpenAIBatchError(f"Batch: missing custom_id for id={i}")
                if not return_exceptions:
                    raise err
                out.append(err)
                continue
            parsed, err = self._map_embedding_entry(entry, i, sparse=use_sparse)
            if err is not None:
                if not return_exceptions:
                    raise err
                out.append(err)
            else:
                assert parsed is not None
                out.append(parsed)
        return out

    @staticmethod
    def _map_embedding_entry(
        entry: dict[str, Any], index: int, *, sparse: bool,
    ) -> tuple[Any, BaseException | None]:
        from llm_mesh.embeddings import parse_embedding_response

        top_error = entry.get("error")
        if top_error:
            return None, OpenAIBatchError(f"Batch subtask id={index}: {top_error}")
        response_obj = entry.get("response")
        if not isinstance(response_obj, dict):
            return None, OpenAIBatchError(
                f"Batch subtask id={index}: missing response object"
            )
        status_code = response_obj.get("status_code")
        if status_code != 200:
            return None, OpenAIBatchError(
                f"Batch subtask id={index}: response status {status_code} — "
                f"{response_obj.get('body')}"
            )
        body = response_obj.get("body")
        if not isinstance(body, dict):
            return None, OpenAIBatchError(
                f"Batch subtask id={index}: response body is not an object"
            )
        try:
            parsed = parse_embedding_response(body, 1, sparse=sparse)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            return None, OpenAIBatchError(f"Batch subtask id={index}: {exc}")
        return parsed[0], None
