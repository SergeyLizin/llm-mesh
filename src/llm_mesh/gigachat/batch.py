"""Async GigaChat Batch API for workloads that can wait for completion. Submit JSONL records shaped
as {id, request} to POST /batches?method=chat_completions or embedder, poll GET
/batches?batch_id=..., then download GET /files/{output_file_id}/content. Result records contain
the matching id and either result or error. The client submits real batches rather than issuing
one chat request per item.
"""

from __future__ import annotations

import asyncio
import json
import logging
from llm_mesh.config import CONNECTION_ENV, get_env, read_options
import threading
from typing import Any, Sequence, cast

import httpx

from .client import (
    GIGACHAT_BASE_URL,
    GigaChatAsyncClient,
)
from ._common import simplify_schema_for_gigachat
from llm_mesh._common import finish_reason_opt
from llm_mesh.types import (
    LLMAuthError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMUsage,
    LLMValidationError,
)

logger = logging.getLogger(__name__)


class GigaChatBatchError(LLMError):
    """Batch API submission, polling, or result-download failure."""


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


class GigaChatBatchClient:
    """Async GigaChat batch client. Reuse OAuth from GigaChatAsyncClient or supply a static token.
    An injected http_client, including an ASGI test transport, remains owned by the caller and
    is not closed by this client.
    """

    def __init__(
        self,
        *,
        auth: GigaChatAsyncClient | None = None,
        token: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        verify: bool | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        max_wait_s: float | None = None,
        http_retries: int | None = None,
        backoff_start_s: float | None = None,
    ) -> None:
        if auth is None and token is None:
            raise LLMAuthError(
                "GigaChatBatchClient: requires either `auth` (GigaChatAsyncClient), or `token`"
            )
        self._auth = auth
        self._token = token
        self._base_url = (base_url or getattr(auth, "_api_url", None) or get_env("LLM_BASE_URL", GIGACHAT_BASE_URL)).rstrip("/") + "/"
        self._external_client = http_client
        self._verify = verify if verify is not None else get_env("LLM_VERIFY_SSL", "false").lower() not in ("0", "false", "no")
        self._timeout = httpx.Timeout(timeout_s if timeout_s is not None else float(get_env("LLM_HTTP_TIMEOUT", "120")))
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

    # --- auth / transport ----------------------------------------------

    async def _get_token(self) -> str:
        if self._token:
            return self._token
        if self._auth is not None:
            return await self._auth._ensure_token()
        raise LLMAuthError("GigaChatBatchClient: no token source")

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if self._external_client is not None:
            return await self._external_client.request(method, path, **kwargs)
        async with httpx.AsyncClient(
            base_url=self._base_url, verify=self._verify, timeout=self._timeout
        ) as client:
            return await client.request(method, path, **kwargs)

    @staticmethod
    def _raise_for_status(response: httpx.Response, context: str) -> None:
        status = response.status_code
        try:
            message = response.json().get("message", response.text)
        except Exception:
            message = response.text
        prefix = f"GigaChat Batch API ({context})"
        if status == 401:
            raise LLMAuthError(f"{prefix}: authentication error — {message}")
        if status == 400:
            raise LLMValidationError(f"{prefix}: invalid request — {message}")
        raise GigaChatBatchError(f"{prefix}: error {status} — {message}")

    # Low-level Batch API operations.

    @staticmethod
    def build_jsonl(requests: Sequence[dict[str, Any]]) -> bytes:
        """Serialize {id, request} records into JSONL bytes for an octet-stream upload."""
        return "\n".join(json.dumps(req, ensure_ascii=False) for req in requests).encode("utf-8")

    async def _refresh_token(self) -> str:
        """Refresh an expired token after 401 when an auth client is configured. A static token
        cannot be refreshed, so its error propagates.
        """
        if self._auth is not None:
            return await self._auth._refresh_token()
        raise LLMAuthError("GigaChatBatchClient: token expired, but no `auth` for refresh")

    async def _request_authed(
        self, method: str, path: str, *, context: str, **kwargs: Any
    ) -> httpx.Response:
        """Send a bearer-authenticated request with rate-limit backoff and one refresh on 401.
        Long-running batches may outlive the token issued before submission.
        """
        token = await self._get_token()
        refreshed = False
        delay = self._backoff_start
        extra_headers = kwargs.pop("headers", None) or {}
        for attempt in range(self._http_retries):
            headers = {"Authorization": f"Bearer {token}", **extra_headers}
            resp = await self._send(method, path, headers=headers, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 401 and self._auth is not None and not refreshed:
                logger.info("GigaChat Batch: 401 (%s) → refresh token", context)
                token = await self._refresh_token()
                refreshed = True
                continue
            if resp.status_code == 429 and attempt + 1 < self._http_retries:
                logger.warning(
                    "GigaChat Batch: 429 (%s), waiting %.1fs (%d/%d)",
                    context, delay, attempt + 1, self._http_retries,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 45.0)
                continue
            self._raise_for_status(resp, context)
        raise GigaChatBatchError(f"GigaChat Batch: attempts exhausted for {context}")

    async def create_batch(self, jsonl_data: bytes, method: str) -> dict[str, Any]:
        """Submit a batch using POST /batches?method=...."""
        resp = await self._request_authed(
            "POST", "batches", context="create_batch",
            params={"method": method}, content=jsonl_data,
            headers={"Content-Type": "application/octet-stream"},
        )
        return cast("dict[str, Any]", resp.json())

    @staticmethod
    def _unwrap_batch_list(payload: Any) -> dict[str, Any]:
        """Unwrap the single-item status list returned by GET /batches."""
        if isinstance(payload, list):
            if not payload:
                raise GigaChatBatchError("GigaChat Batch API: empty list in response to GET /batches")
            return cast("dict[str, Any]", payload[0])
        if isinstance(payload, dict):
            return payload
        raise GigaChatBatchError(
            f"GigaChat Batch API: unexpected response format for GET /batches: {type(payload).__name__}"
        )

    async def get_batch(self, batch_id: str) -> dict[str, Any]:
        """Fetch batch status using GET /batches?batch_id=...."""
        resp = await self._request_authed(
            "GET", "batches", context="get_batch", params={"batch_id": batch_id}
        )
        return self._unwrap_batch_list(resp.json())

    async def wait_for_completion(
        self,
        batch_id: str,
        *,
        poll_interval: float | None = None,
        max_wait: float | None = None,
    ) -> dict[str, Any]:
        """Poll until a terminal status and return the final status object."""
        interval = poll_interval if poll_interval is not None else self._poll_interval
        deadline = max_wait if max_wait is not None else self._max_wait
        terminal = {"completed", "failed", "expired", "cancelled"}
        start = asyncio.get_event_loop().time()
        while True:
            data = await self.get_batch(batch_id)
            status = data.get("status", "unknown")
            if status in terminal:
                if status != "completed":
                    raise GigaChatBatchError(f"Batch {batch_id} finished with status: {status}")
                return data
            if asyncio.get_event_loop().time() - start >= deadline:
                raise LLMTimeoutError(
                    f"Batch {batch_id} did not complete within {deadline}s (status: {status})"
                )
            await asyncio.sleep(interval)

    async def get_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Wait for output_file_id and download newline-delimited result records."""
        status = await self.get_batch(batch_id)
        output_file_id = status.get("output_file_id")
        if not output_file_id:
            raise GigaChatBatchError(
                f"GigaChat Batch API: missing output_file_id (batch_id={batch_id!r}, "
                f"status={status.get('status')!r})"
            )

        resp = await self._request_authed(
            "GET", f"files/{output_file_id}/content", context="get_results"
        )
        text = resp.text.strip()
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type and text.startswith("["):
            return cast("list[dict[str, Any]]", json.loads(text))
        results: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Batch result: invalid line of JSONL: %s", line[:100])
        return results

    # High-level chat batching.

    def _build_chat_line(self, sub_id: str, request: LLMRequest, model: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.mode != "text" and request.schema_ is not None:
            body["functions"] = [
                {
                    "name": request.function_name,
                    "description": request.function_description
                    or f"Build a {request.function_name} object.",
                    "parameters": simplify_schema_for_gigachat(request.schema_),
                }
            ]
            body["function_call"] = {"name": request.function_name}
        return {"id": sub_id, "request": body}

    @staticmethod
    def _response_from_result(
        result: dict[str, Any], request: LLMRequest, fallback_model: str
    ) -> LLMResponse:
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        # Centralize usage parsing in LLMUsage.from_raw so new provider fields remain consistent
        # across all clients.
        usage = LLMUsage.from_raw(result.get("usage"))
        model = str(result.get("model", fallback_model))

        if request.mode == "text":
            return LLMResponse(
                arguments={},
                text=message.get("content", "") or "",
                finish_reason=finish_reason_opt(result),
                model=model,
                usage=usage,
                raw=result,
            )

        fc = message.get("function_call")
        if not fc:
            raise LLMValidationError(
                f"GigaChat Batch: missing function_call in result (content="
                f"{(message.get('content') or '')[:200]!r})"
            )
        raw_args = fc.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LLMValidationError(f"GigaChat Batch: invalid JSON in arguments: {exc}") from exc
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            raise LLMValidationError(
                f"GigaChat Batch: arguments have unexpected type {type(raw_args).__name__}"
            )
        return LLMResponse(
            arguments=arguments,
            finish_reason=finish_reason_opt(result),
            model=model,
            usage=usage,
            raw=result,
        )

    async def run_chat_batch(
        self,
        requests: Sequence[LLMRequest],
        *,
        model: str | None = None,
        poll_interval: float | None = None,
        max_wait: float | None = None,
        return_exceptions: bool = False,
    ) -> list[LLMResponse | BaseException]:
        """Submit, poll, download, and map a complete chat batch. Return responses in input order.
        Invalid results and per-item errors become LLMError: raise the first by default, or
        return exceptions in their original positions when return_exceptions=True.
        """
        if not requests:
            return []

        resolved_model = model or (self._auth.model if self._auth is not None else "GigaChat")
        lines = [
            self._build_chat_line(str(i), req, resolved_model)
            for i, req in enumerate(requests)
        ]
        jsonl = self.build_jsonl(lines)

        batch = await self.create_batch(jsonl, method="chat_completions")
        await self.wait_for_completion(
            batch["id"], poll_interval=poll_interval, max_wait=max_wait
        )
        raw_results = await self.get_results(batch["id"])

        # Match results by id, using the stringified input-request index.
        by_id: dict[str, dict[str, Any]] = {str(r.get("id")): r for r in raw_results}
        out: list[LLMResponse | BaseException] = []
        for i, req in enumerate(requests):
            entry = by_id.get(str(i))
            err: BaseException | None = None
            response: LLMResponse | None = None
            if entry is None:
                err = GigaChatBatchError(f"Batch: no result for id={i}")
            elif entry.get("error"):
                e = entry["error"]
                err = GigaChatBatchError(
                    f"Batch subtask id={i}: {e.get('status')} {e.get('message')}"
                )
            else:
                try:
                    response = self._response_from_result(
                        entry.get("result", {}), req, resolved_model
                    )
                except LLMError as parse_exc:
                    err = parse_exc

            if err is not None:
                if not return_exceptions:
                    raise err
                out.append(err)
            else:
                assert response is not None
                out.append(response)
        return out


# Coalescing adapter for generate_text and generate_structured.


class BatchingLLMClient:
    """Coalesce concurrent text or structured calls into real GigaChat batches. Intended for
    offline workloads that tolerate the collection window and batch polling latency. Preserve
    the individual client interface and deliver each result to its waiting caller.
    """

    def __init__(
        self,
        batch_client: GigaChatBatchClient,
        *,
        model: str,
        max_batch_size: int | None = None,
        max_delay_s: float | None = None,
        idle_timeout_s: float = 30.0,
    ) -> None:
        self._batch = batch_client
        self._model = model
        self._max_batch_size = max(1, max_batch_size or _env_int("LLM_BATCH_COALESCE_MAX", 16))
        self._max_delay = (
            max_delay_s if max_delay_s is not None else _env_float("LLM_BATCH_COALESCE_DELAY_S", 0.25)
        )
        self._idle_timeout = idle_timeout_s
        self._queue: asyncio.Queue[tuple[LLMRequest, asyncio.Future]] | None = None
        self._worker: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    # Duck-typed client interface used by callers.

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        return await self._submit(request.model_copy(update={"mode": "text"}))

    async def generate_structured(self, request: LLMRequest) -> LLMResponse:
        # Reject tools_required explicitly. The legacy batch functions API cannot implement a
        # native tool loop; silently forcing function_name could be mistaken for a
        # model-selected final answer without any tool execution.
        if request.tools_required:
            raise LLMValidationError(
                "GigaChat Batch: native tool-loop (tools_required) not "
                "supported — legacy functions API without tool_calls/tool role"
            )
        return await self._submit(request.model_copy(update={"mode": "function_call"}))

    async def aclose(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        for task in list(self._inflight):
            task.cancel()

    # Internal request coalescing.

    def _ensure_running(self) -> None:
        # Initialize lazily inside the active event loop. This synchronous section contains no
        # await and cannot race within a single-threaded loop.
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def _submit(self, request: LLMRequest) -> LLMResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._ensure_running()
        assert self._queue is not None
        await self._queue.put((request, fut))
        return cast("LLMResponse", await fut)

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=self._idle_timeout)
            except asyncio.TimeoutError:
                return  # Exit while idle; the next _submit restarts collection.
            batch = [first]
            deadline = loop.time() + self._max_delay
            while len(batch) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            # Process the batch in the background while collecting the next window.
            task = asyncio.create_task(self._process(batch))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _process(self, batch: list[tuple[LLMRequest, asyncio.Future]]) -> None:
        requests = [req for req, _ in batch]
        try:
            results = await self._batch.run_chat_batch(
                requests, model=self._model, return_exceptions=True
            )
        except Exception as exc:  # noqa: BLE001 -- propagate the failure to every waiting
                                  # caller
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        for (_, fut), res in zip(batch, results):
            if fut.done():
                continue
            if isinstance(res, BaseException):
                fut.set_exception(res)
            else:
                fut.set_result(res)


# Environment-gated process-wide clients coalesce requests across concurrent sessions.

_BATCH_SINGLETONS: dict[tuple, BatchingLLMClient] = {}
_BATCH_SINGLETONS_LOCK = threading.Lock()


def batch_mode_enabled() -> bool:
    """Return whether LLM_BATCH_MODE enables batch generation. Disabled by default; enable it
    for offline workloads that prioritize batching over interactive latency.
    """
    return get_env("LLM_BATCH_MODE", "").strip().lower() in ("1", "true", "yes")


def get_batching_client(
    model: str,
    *,
    credentials: str | None = None,
    scope: str | None = None,
) -> BatchingLLMClient:
    """Return the process-wide BatchingLLMClient for this model. Sharing enables coalescing across
    sessions; the nested GigaChatAsyncClient supplies OAuth.
    """
    key = (model, credentials, scope,
           tuple((name, get_env(name)) for name in CONNECTION_ENV),
           json.dumps(read_options(), sort_keys=True))
    cli = _BATCH_SINGLETONS.get(key)
    if cli is not None:
        return cli
    with _BATCH_SINGLETONS_LOCK:
        cli = _BATCH_SINGLETONS.get(key)
        if cli is None:
            auth = GigaChatAsyncClient(model=model, credentials=credentials, scope=scope)
            cli = BatchingLLMClient(GigaChatBatchClient(auth=auth), model=model)
            _BATCH_SINGLETONS[key] = cli
        return cli
