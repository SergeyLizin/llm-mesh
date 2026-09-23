# llm-mesh

Async Python clients for OpenAI-compatible APIs, the Anthropic Messages API, the Gemini generateContent API, and GigaChat. One request type, one response type, streaming, structured output, retries, and a model catalog.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/SergeyLizin/llm-mesh/actions/workflows/tests.yml/badge.svg)](https://github.com/SergeyLizin/llm-mesh/actions/workflows/tests.yml)

Requires Python 3.11 or newer. The package is not published to PyPI yet.

## Installation

From a checkout:

```sh
python -m pip install -e /path/to/llm-mesh
```

From a reviewed Git revision:

```sh
python -m pip install "llm-mesh @ git+https://github.com/SergeyLizin/llm-mesh.git@<commit>"
```

The library does not read `.env` files. Export variables before constructing a client, or pass them as constructor arguments.

## Quick start

```python
import asyncio
import os

from llm_mesh import LLMRequest, OpenAIClient

async def main():
    client = OpenAIClient(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    try:
        response = await client.generate_text(
            LLMRequest(system="Be concise.", user="Explain HTTP caching.", max_tokens=256)
        )
        print(response.text)
    finally:
        await client.aclose()

asyncio.run(main())
```

The same `LLMRequest` works with `AnthropicClient`, `GeminiClient`, and `GigaChatAsyncClient`. Close the client with `aclose()` when you are done.

To pick a model from the bundled catalog instead of constructing a client by hand:

```python
from llm_mesh.models_catalog import load_catalog, make_client, resolve_route

route = resolve_route("your-model-id@your-provider", load_catalog())
client = make_client(route)
```

`make_client` writes the route into the process environment. Build catalog clients one at a time, before issuing requests.

## What you can call

| Method | Result |
| --- | --- |
| `generate_text` | `LLMResponse.text` |
| `generate_structured` | `LLMResponse.arguments` and `function_name` |
| `generate_stream` | text chunks |
| `generate_stream_events` | typed content, reasoning, tool, and completion events |
| `count_tokens` | token counts, where the provider implements it |
| `embed` | dense vectors, and a sparse vector when the provider returns one |
| `rerank` | documents scored against a query, best first |

`LLMRequest.model` overrides the client's model for that call. An empty string leaves the client's model in place. Limits the client has already learned (which tool form the gateway accepted, an open-object rejection, a forced temperature) stay on the instance and apply to every per-call model. A model that needs different limits needs its own client.

`count_tokens` returns one integer per string. Anthropic posts each string to `/v1/messages/count_tokens`, Gemini to `:countTokens`, and GigaChat posts the whole list to `/tokens/count`. OpenAI-compatible clients count locally with tiktoken and do not call the gateway. A model tiktoken knows (`gpt-4o`, `gpt-4`) selects that encoding. Any other model name needs `tiktoken_encoding` on the client or the catalog route (`o200k_base`, `cl100k_base`). The count is the string itself, without chat-message overhead. `supports(Capability.COUNT_TOKENS)` tells you whether the call can succeed.

`embed` sends a list of strings and returns one `Embedding` per input, in that order. `task="document"` is the default. `task="query"` applies the model's `query_instruction` when the template contains `{query}`. An empty template leaves queries and documents identical. OpenAI-compatible gateways can return a sparse vector in the same response (`sparse=True`, wire field `return_sparse`). GigaChat and Gemini refuse `sparse=True`. GigaChat also refuses `dimensions`; the vector width is the model's. Gemini sends `taskType` `RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY` and, when set, `outputDimensionality`. A catalog route with both `query_instruction` and Gemini's task type wraps the query and still sends `taskType`, so a route that should rely on `taskType` alone leaves `query_instruction` empty.

Anthropic has no embeddings API. `embed` raises `NotImplementedError`, and `supports(Capability.EMBEDDINGS)` is false.

Embedding calls share the client's `LLM_MAX_CONCURRENT` semaphore with chat calls on that same instance. There is no separate process-wide limiter.

OpenAI's file Batch API accepts embeddings as well as chat. `OpenAIBatchClient.build_embedding_lines` writes one `/v1/embeddings` line per input, and `run_embedding_batch` submits that file. `LLM_BATCH_MODE` still coalesces chat completions only. A catalog route with `"task": "embeddings"` or `"task": "rerank"` is not wrapped in the chat coalescer.

`rerank` scores each document against the query and returns `RerankHit` values, highest score first. OpenAI-compatible gateways use `POST /score` with `text_1` (the query) and `text_2` (the documents). A base URL that ends in `/v1` is not sent to `/v1/score`; that path is a 404, and the client calls `/score` on the host. `rerank_protocol="llama"` uses llama.cpp's `POST /v1/rerank` (`query` / `documents`, `relevance_score`). A catalog route sets the same field. `rerank_top_k` and `rerank_min_score` are the defaults for later calls. A positive floor drops documents under it; if every document is under the floor, the ranking is returned without the floor so the scores stay visible. A response that does not score every document raises. The input order is not returned in place of a failed call.

`LocalCrossEncoder` scores the same pairs in-process. Install it with `llm-mesh[rerank]` (`sentence-transformers`). The model loads on the first call. Anthropic, Gemini, and GigaChat have no rerank method.

## Providers

| | OpenAI-compatible | Anthropic | Gemini | GigaChat |
| --- | --- | --- | --- | --- |
| Class | `OpenAIClient` | `AnthropicClient` | `GeminiClient` | `GigaChatAsyncClient` |
| Default endpoint | `LLM_BASE_URL` (required) | `https://api.anthropic.com` | `https://generativelanguage.googleapis.com/v1beta` | `https://gigachat.devices.sberbank.ru/api/v1` |
| API key | `LLM_API_KEY` | `LLM_API_KEY` | `LLM_API_KEY` | OAuth client credentials in `LLM_API_KEY`, or a `token` |
| Structured output | tools, then declared `response_format` | native tool use and `output_config.format` | `responseSchema` and function calling | legacy functions and native JSON Schema |
| `count_tokens` | local tiktoken | yes | yes | yes |
| `embed` | yes | no | yes | yes |
| `rerank` | `/score` or `/v1/rerank` | no | no | no |
| Batch | chat and embeddings | no | no | chat |

```python
from llm_mesh import AnthropicClient, GeminiClient, GigaChatAsyncClient

anthropic = AnthropicClient(model="claude-sonnet-5")
gemini = GeminiClient(model="gemini-2.5-flash")
gigachat = GigaChatAsyncClient(model="GigaChat-2")
```

`GeminiClient` calls the native generateContent API. Catalog routes whose kind is `openai` stay on `OpenAIClient`, including OpenAI-compatible Gemini proxies. The bundled native Gemini route is `gemini-25-flash`. The bundled native Anthropic routes are `claude-opus-5`, `claude-sonnet-4-6`, and `claude-haiku-4-5`.

GigaChat also needs `LLM_AUTH_SCOPE` (`GIGACHAT_API_PERS`, `GIGACHAT_API_B2B`, or `GIGACHAT_API_CORP`). TLS verification defaults to on for OpenAI, Anthropic, and Gemini, and off for GigaChat. Set `LLM_VERIFY_SSL` or pass `verify`.

GigaChat speaks the legacy functions API. `tools_required` raises `LLMValidationError`. `tool_choice="auto"` lets the model pick among `LLMRequest.tools`; the default `"single"` forces `function_name`.

## Structured output

```python
response = await client.generate_structured(
    LLMRequest(
        system="Answer with the schema.",
        user="Name the HTTP method used to read a resource.",
        schema={
            "type": "object",
            "properties": {"method": {"type": "string"}},
            "required": ["method"],
            "additionalProperties": False,
        },
        mode="json_schema",
    )
)
print(response.arguments)
```

`mode="function_call"` (the default) asks for a tool call. `mode="json_schema"` asks for a JSON object. `mode="text"` returns ordinary text.

Native structured output is tried first. If the gateway rejects the schema, the client retries once as text, then parses the text. That retry does not run when `tools_required` is set, when `LLM_NO_DEGRADE` is set, or when `fallback_policy="preserve"`. `validate_schema=True` (the default) rejects arguments that do not match the schema. `validate_schema=False` returns the parsed object unchanged.

GigaChat simplifies schemas for the legacy functions API: local `$ref` are inlined, and non-string enums are dropped. When the reasoning field is empty, `<think>` tags are removed from the visible answer, including tags split across stream chunks. A non-empty reasoning field stays the reasoning text.

## Streaming

```python
async for event in client.generate_stream_events(request):
    print(type(event).__name__, event)
```

Events live in `llm_mesh.stream_events`. `Complete` carries usage and `finish_reason` when the provider sent them. A transport failure yields `Error` and then raises. Streams share the client's `LLM_MAX_CONCURRENT` slot with blocking calls.

`generate_stream` yields text chunks. OpenAI non-streaming calls can also be sent as SSE internally with `LLM_STREAM_TRANSPORT=true`; the public stream methods do not need that flag.

## Model catalog

The installed package ships `models.json`. Credentials stay in the environment variables named by each route. Loading a catalog makes no API calls.

```sh
llm-mesh --list                  # routes and which credentials are missing
llm-mesh --env your-model-id     # shell exports for one route
llm-mesh --json your-model-id    # one route as JSON
llm-mesh --check your-model-id   # real API call; see below
```

A project catalog can extend the bundled one. Copy [`examples/models.json`](examples/models.json):

```json
{
  "extends": "llm-mesh",
  "models": [
    {
      "id": "custom-model",
      "kind": "openai",
      "providers": [{
        "name": "custom",
        "model_env": "CUSTOM_LLM_MODEL",
        "base_url_env": "CUSTOM_LLM_BASE_URL",
        "api_key_env": "CUSTOM_LLM_API_KEY"
      }]
    }
  ]
}
```

```python
routes = load_catalog("/path/to/your/project/models.json")
```

Or set `LLM_MODELS_CONFIG` and call `load_catalog()` with no arguments. An explicit path wins over the variable. With neither, only the bundled catalog is loaded. A file named `models.json` in the working directory is not picked up on its own.

New ids are appended. An id that already exists replaces that model. `exclude` drops ids. `order` lists ids to place first. `patches` changes individual fields and keeps inheriting the rest of the public entry. A JSON array, instead of an object with `extends`, replaces the bundled catalog entirely.

Catalog kinds are `openai`, `anthropic`, `gemini`, and `gigachat`. `anthropic` and `gemini` routes may omit a base URL and use the provider default.

### Connection checks

`llm-mesh --check` and `check_route` send one cheap request: `embed` for a catalog route whose `task` is `embeddings`, `rerank` for `task` `rerank`, `generate_text` for other OpenAI-compatible clients, and `count_tokens` for Anthropic, Gemini, and GigaChat chat routes. OpenAI's token count stays off this path: tiktoken does not contact the gateway, so a chat route is still proved with one generation. The reply text is not judged. The probe is capped at 60 seconds. A route marked `embeddings` on a client that cannot embed fails the check instead of sending a chat completion.

```sh
llm-mesh --check your-model-id
llm-mesh --check your-model-id@your-provider
```

Exit 0 means at least one checked route answered. Exit 1 means every checked route failed, or the selector is not in the catalog. `--list`, `--env`, and `--json` do not call a provider.

From Python, `check_client` probes a client you already built and does not close it. `check_route` and `check_routes` are synchronous, call `asyncio.run`, and close the client they create. Call them when no event loop is running. With `LLM_BATCH_MODE` set, the batch adapter is reported as a failure: a check would sit in the batch queue.

## Configuration

Constructor arguments override the environment, except `length_retry_cap`, where `LLM_LENGTH_RETRY_CAP` wins.

`LLM_OPTIONS` is a JSON object of the same settings, with the `LLM_` prefix removed (`LLM_MAX_OUTPUT_TOKENS` becomes `max_output_tokens`). A key in that object overrides the standalone variable. When `LLM_OPTIONS` is set, omitted route options (output, reasoning, schema) use client defaults and ignore leftover standalone variables. Omitted runtime options (timeouts, retries, batch) still fall back to standalone variables. Catalog selection replaces `LLM_OPTIONS`; it does not merge with an object you already exported.

### Credentials

| Variable | Used by |
| --- | --- |
| `LLM_API_KEY` | all four providers |
| `LLM_BASE_URL` | OpenAI (required). Anthropic, Gemini, and GigaChat use it when set, and otherwise their public endpoint |
| `LLM_AUTH_URL`, `LLM_AUTH_SCOPE` | GigaChat OAuth. Default auth URL is `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`. No default scope |
| `LLM_VERIFY_SSL` | `0`, `false`, or `no` disables TLS verification |

### Output, reasoning, and schema

| Variable | Default | Effect |
| --- | --- | --- |
| `LLM_MAX_OUTPUT_TOKENS` | none; GigaChat uses a per-family ceiling | cap on the output budget |
| `LLM_MIN_OUTPUT_TOKENS` | unset | OpenAI floor, for reasoning models that spend the budget before visible text |
| `LLM_MAX_CONCURRENT` | unlimited | concurrent requests per client |
| `LLM_DISABLE_REASONING` | false | turn reasoning off with the provider's declared dialect |
| `LLM_REASONING_EFFORT` | unset | `low`, `medium`, or `high` |
| `LLM_REASONING_FIELD` | `reasoning_content` | response field that holds reasoning text |
| `LLM_NO_DEGRADE` | false | raise instead of silently parsing a text reply as structured output |
| `LLM_DISABLE_TOOLS` | false | OpenAI and Anthropic: structured output through text emulation |
| `LLM_RESPONSE_FORMAT` | unset | OpenAI: `json_schema` or `json_object` as an extra structured tier |
| `LLM_EXTRA_BODY`, `LLM_EXTRA_HEADERS` | `{}` | extra JSON fields and headers. Generated fields keep their values. GigaChat does not let these replace `Authorization` or `RqUID` |

`fallback_policy="preserve"` on the constructor is the same idea as `LLM_NO_DEGRADE`: return or raise the provider's own result, and do not invent a replacement. The default policy is `"recover"`.

### Timeouts and retries

| Variable | Default |
| --- | --- |
| `LLM_HTTP_TIMEOUT` | 600s for generation, 120s for batch |
| `LLM_MAX_RETRIES` | 3 extra OpenAI attempts on retryable HTTP statuses |
| `LLM_LENGTH_RETRIES` | 2 for OpenAI, 1 for GigaChat. `0` disables them |
| `LLM_LENGTH_RETRY_CAP` | 32768 for OpenAI. GigaChat uses its output ceiling |

A length retry runs when the provider stops for length and the visible output is empty. The next attempt uses a larger `max_tokens`.

### Batch

`LLM_BATCH_MODE=true` makes catalog routes of kind `openai` and `gigachat` return a `BatchingLLMClient`. Concurrent `generate_text` and `generate_structured` calls are grouped and submitted as one provider batch. A client you construct yourself is unchanged. Anthropic and Gemini have no batch client.

OpenAI batch lines follow the official Batch API (`purpose=batch`, `completion_window` of 24h, poll until a terminal status). One file is limited to 50 000 requests and 200 MB. A failed line raises, or is returned in place when `return_exceptions=True`. A short line is never filled in with an empty success. GigaChat posts `{id, request}` JSONL to `POST /batches`.

| Variable | Default |
| --- | --- |
| `LLM_BATCH_POLL_INTERVAL_S` | 5 |
| `LLM_BATCH_MAX_WAIT_S` | 3600 |
| `LLM_BATCH_COALESCE_MAX` | 16 |
| `LLM_BATCH_COALESCE_DELAY_S` | 0.25 |

## Canary

`set_canary_context_token` appends a marker to the system prompt and scans reply text for it. Image bytes are not scanned. Reset the token in a `finally` block.

```python
from llm_mesh.canary import reset_canary_context_token, set_canary_context_token

token = set_canary_context_token("session-marker")
try:
    ...
finally:
    reset_canary_context_token(token)
```

Batch results are not scanned. A custom prompt or detector can replace the defaults through `llm_mesh.hooks.configure_canary_hooks()`.

### Request guard hook

`configure_request_hook` is the seam in front of the wire. The library ships no detection patterns and no redaction. The hook body belongs to the application: a prompt-injection guard, a PII filter, or a quota check.

| Hook returns | Effect |
| --- | --- |
| the same `LLMRequest` | the call proceeds unchanged |
| a different `LLMRequest` | that object is what the provider receives |
| `None` | `LLMRequestBlocked` |
| an exception | that exception propagates, unwrapped |

`LLMRequestBlocked` is not a validation error. Tier ladders, length retry, and degradation catch `LLMValidationError` and provider errors and may try another path. A guard refusal is not one of those paths. `no_degrade` and `preserve` do not apply, and the refusal does not count as degradation.

`count_tokens` is not guarded. It takes raw strings and is a diagnostic. A connectivity probe that calls `generate_text` is guarded like any other request, so a refusal shows up as that route check's `error_type`.

The hook is process-wide: one function for every client and thread. A hook that needs per-session isolation has to do that itself. A blocked request logs `REQUEST_BLOCKED` and the provider name. The payload is not logged, and neither are image bytes.

## Metrics

`configure_metrics_hook` receives one `CallRecord` per interactive call: `generate_text`, `generate_structured`, `generate_stream`, `generate_stream_events`, and `count_tokens`. The record is emitted on success, on a provider error, on a guard refusal (`error_type="LLMRequestBlocked"`), and when a stream ends or is interrupted. `latency_ms` covers the whole call, including retries and a validation re-ask. The default hook is a no-op.

The callable is synchronous. It must not block or await. An exception it raises is logged at WARNING with the marker `METRICS_HOOK_FAILED` and is swallowed, so telemetry cannot fail the call.

Batch clients are not instrumented. A batch submission does not emit a `CallRecord`.

## Budgets

Pass `budget=Budget(...)` to a client. `max_cost_usd`, `max_cost_rub`, and `max_total_tokens` are independent ceilings. A client that is already at a ceiling raises `LLMBudgetExceeded` before the next request is sent. Spend is added after each response, including the terminal chunk of a stream, from the same usage the metrics record carries. `reset_budget()` zeroes the counters. `budget_state()` returns a copy of what has been accumulated.

Currencies are not converted. A ceiling in dollars does not account ruble spend, and the reverse is also true. A route that reports both needs both fields. Unreported cost is not added. `total_tokens` is.

`LLMBudgetExceeded` is not a validation error. Tier ladders do not retry it.

Batch clients take the same constructor argument. The check runs before upload, and each returned response is accumulated. The coalescing adapter checks before it submits and adds the response it gets back.

## Validation re-ask

When structured output fails jsonschema and `validate_schema` is on, the client retries once. The retry is the same request plus two history turns: the failed answer as the assistant, then a user turn that names the function and the first schema error. Body builders are unchanged. If the second answer still fails, the existing path runs: raise, or fall through to the next tier, using that second result.

The re-ask does not run when `no_degrade` is set or `fallback_policy="preserve"`. Those modes are for measurement, and they have to see the raw first response. GigaChat does not validate arguments, so it does not re-ask.

Usage of the two attempts is added. Cache fields keep the attempt that reported them, preferring the second. `LLMResponse.validation_reasks` is `1` when a re-ask happened. The metrics record carries the summed usage and the tier that served the call.

## Per-call timeout and retries

`LLMRequest.timeout_s` and `LLMRequest.max_retries` override the constructor, which overrides the environment. A negative value raises `LLMValidationError` before any HTTP. `timeout_s` is passed to httpx as that request's timeout. GigaChat keeps its connect timeout of 30 seconds and applies `timeout_s` to the other phases. `max_retries=0` is one attempt.

Batch clients and `count_tokens` stay on the constructor timeout. They honor a per-call timeout only where the underlying method already accepts one.

## Images

`LLMRequest.images` attaches images to the current user turn. History turns stay text. An empty list is the text-only body, byte for byte.

| Client | `data` | `url` |
| --- | --- | --- |
| OpenAI | `data:` URL in an `image_url` part | `image_url` part |
| Anthropic | base64 `source` (`anthropic-version` `2023-06-01` accepts it) | `source` type `url` on that same version |
| Gemini | `inlineData` part (`mimeType`, camelCase like the rest of the body) | rejected: generateContent has no public-URL image input |
| GigaChat | rejected: legacy functions chat has no image input | rejected |

`media_type` defaults to `image/png` for bytes and must be png, jpeg, gif, or webp. The canary scans text only. The request guard sees the attachments and may refuse them. Metrics records are unchanged.

## Adding a provider

Subclass `BaseLLMClient`, implement `generate_text`, `generate_structured`, `generate_stream`, and `generate_stream_events`, and set `CAPABILITIES`. `supports()` reads that class set. It does not look at instance flags, so a capability in the set can still be off for one instance (for example `LLM_DISABLE_TOOLS`).

Public code should depend on the `LLMClient` and `BatchLLMClient` protocols. Register a catalog kind in `models_catalog._ROUTE_CLIENT_BUILDERS`.

## Development

Documentation, comments, catalog notes, built-in prompts, log messages, errors, and test fixtures are in English.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Tests mock HTTP and do not need provider credentials.

```sh
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

## Releases

`.github/workflows/release.yml` runs on tags such as `v2.0.0`. The tag must match `project.version` in `pyproject.toml`. The workflow tests, builds the sdist and wheel, publishes to PyPI, and opens a GitHub Release.

Publishing uses GitHub OIDC through the `pypi` environment. No PyPI token is stored in the repository. A manual workflow run builds artifacts and does not publish.

```sh
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin main
git push origin v2.0.0
```

## License

[MIT](LICENSE)
