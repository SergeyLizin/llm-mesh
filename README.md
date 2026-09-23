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

`LLMRequest.model` overrides the client's model for that call. An empty string leaves the client's model in place. Limits the client has already learned (which tool form the gateway accepted, an open-object rejection, a forced temperature) stay on the instance and apply to every per-call model. A model that needs different limits needs its own client.

OpenAI has no `count_tokens`. Anthropic posts each string to `/v1/messages/count_tokens`, Gemini to `:countTokens`, and GigaChat posts the whole list to `/tokens/count`. `supports(Capability.COUNT_TOKENS)` tells you whether the call can succeed.

## Providers

| | OpenAI-compatible | Anthropic | Gemini | GigaChat |
| --- | --- | --- | --- | --- |
| Class | `OpenAIClient` | `AnthropicClient` | `GeminiClient` | `GigaChatAsyncClient` |
| Default endpoint | `LLM_BASE_URL` (required) | `https://api.anthropic.com` | `https://generativelanguage.googleapis.com/v1beta` | `https://gigachat.devices.sberbank.ru/api/v1` |
| API key | `LLM_API_KEY` | `ANTHROPIC_API_KEY`, then `LLM_API_KEY` | `GEMINI_API_KEY`, then `LLM_API_KEY` | OAuth client credentials in `LLM_API_KEY`, or a `token` |
| Structured output | tools, then declared `response_format` | native tool use and `output_config.format` | `responseSchema` and function calling | legacy functions and native JSON Schema |
| `count_tokens` | no | yes | yes | yes |
| Batch | yes | no | no | yes |

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

`llm-mesh --check` and `check_route` send one cheap request: `generate_text` for OpenAI-compatible clients, `count_tokens` for Anthropic, Gemini, and GigaChat. The reply text is not judged. The probe is capped at 60 seconds.

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
| `LLM_API_KEY` | all four; Anthropic and Gemini check their own variable first |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` | Anthropic, before the `LLM_` names |
| `GEMINI_API_KEY`, `GEMINI_BASE_URL` | Gemini, before the `LLM_` names |
| `LLM_BASE_URL` | OpenAI (required), and the others when their own base URL is unset |
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

`set_canary_context_token` appends a marker to the system prompt and scans replies for it. Reset the token in a `finally` block.

```python
from llm_mesh.canary import reset_canary_context_token, set_canary_context_token

token = set_canary_context_token("session-marker")
try:
    ...
finally:
    reset_canary_context_token(token)
```

Batch results are not scanned. A custom prompt or detector can replace the defaults through `llm_mesh.hooks.configure_canary_hooks()`.

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
