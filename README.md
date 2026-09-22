# llm-mesh

Async Python clients for OpenAI-compatible APIs and GigaChat, with shared
request/response types, streaming, structured output, retries, and a model catalog.

Requires Python 3.11 or newer. Licensed under MIT.

## Installation

Install from a checked-out repository:

```sh
python -m pip install -e /path/to/llm-mesh
```

Or install a reviewed Git revision:

```sh
python -m pip install "llm-mesh @ git+https://github.com/SergeyLizin/llm-mesh.git@<commit>"
```

The package is not yet published to PyPI.

## OpenAI-compatible APIs

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

`GigaChatAsyncClient` offers the same text, structured-output and streaming
interfaces, with GigaChat authentication. Batch helpers are available in
`llm_mesh.gigachat.batch`. `LLMUsage` normalizes provider token usage, including
cache and reasoning tokens.

## Environment configuration

Set environment variables before constructing a client. The library does not
load `.env` files automatically. In this section, **OpenAI** means any endpoint
used through `OpenAIClient`; **GigaChat** means `GigaChatAsyncClient`.
Provider-specific environment aliases are not inferred. A catalog route may
explicitly reference any secret variable through `api_key_env` (and similarly
`base_url_env`, `model_env`, `scope_env`, or `auth_url_env`).

### Catalog selection and connection

- `LLM_MODELS_CONFIG`: catalog JSON path. An explicit catalog path argument
  takes precedence; when neither is set, the bundled `models.json` is used.
- `LLM_MODEL_ID`: catalog entry ID consumed by `make_selected_client()`.
  Unset or empty means no selection and the function returns `None`.
- `LLM_PROVIDER`: route kind, `openai` or `gigachat`, exported by catalog
  selection for callers that dispatch by provider. Setting it does not change
  the class of a directly constructed client.
- `LLM_MODEL`: provider model name exported by catalog selection. The catalog
  factory passes it to the client. Direct constructors use their `model`
  argument; pass this variable explicitly, as in the example above.
- `LLM_PROVIDER_LABEL`: diagnostic provider label for OpenAI; defaults to
  `openai` unless a constructor label is supplied. Catalog selection exports
  the route's `provider` label, falling back to its kind.
- `LLM_BASE_URL`: API endpoint. Required for OpenAI unless `base_url` is passed.
  GigaChat defaults to `https://gigachat.devices.sberbank.ru/api/v1`.
  A Batch client also inherits the endpoint of its supplied authentication client.
- `LLM_API_KEY`: OpenAI API key, or GigaChat encoded OAuth client credentials.
  No default. Explicit credentials take precedence. A GigaChat client may
  instead receive an existing access token through the `token` argument.
- `LLM_AUTH_URL`: GigaChat OAuth token endpoint; defaults to
  `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`. Unused by OpenAI.
- `LLM_AUTH_SCOPE`: GigaChat OAuth scope, such as `GIGACHAT_API_PERS`,
  `GIGACHAT_API_B2B`, or `GIGACHAT_API_CORP`; no default scope is configured.
  Unused by OpenAI. These scope strings are protocol values, not environment aliases.
- `LLM_VERIFY_SSL`: TLS certificate verification. Defaults to `true` for
  OpenAI and `false` for GigaChat and Batch. Case-insensitive `0`, `false`,
  and `no` disable verification; other nonempty values enable it.
  An explicit `verify` argument takes precedence.

### Options and precedence

`LLM_OPTIONS` is a JSON object containing request and runtime settings:

```sh
export LLM_OPTIONS='{"max_output_tokens":8192,"max_concurrent":2,"extra_body":{"top_p":0.9}}'
```

For every setting below, the JSON key is the lowercase environment name
without `LLM_`: for example, `LLM_HTTP_TIMEOUT` becomes `http_timeout`.
Use JSON booleans, numbers, and objects directly, without encoding objects as
strings. The connection and catalog selection variables above are read separately
and cannot be configured inside `LLM_OPTIONS`.

- Unset or empty `LLM_OPTIONS` lets clients read standalone variables below.
- A present JSON key overrides its standalone variable. JSON `null` selects
  the client default. Invalid JSON or a non-object value raises `ValueError`.
- When `LLM_OPTIONS` is present, including `{}`, omitted **route options** use
  client defaults and ignore old standalone values. Route options are all
  settings in the next three subsections (output, reasoning, and schema/usage).
- Omitted **runtime options** in the transport/retry and Batch subsections
  still fall back to standalone variables. Thus `{}` does not reset those
  process-wide defaults.

Catalog selection manages only nine variables: the eight connection variables
from `LLM_PROVIDER` through `LLM_VERIFY_SSL`, plus `LLM_OPTIONS`.
`route_env()` builds the options object from route fields, mapping `max_tokens`
to `max_output_tokens`. It also copies `http_timeout`, `no_degrade`,
`length_retries`, and `length_retry_cap` when declared. For GigaChat,
`reasoning_on.reasoning_effort` supplies the default `reasoning_effort` option.
Other runtime controls can be set through standalone variables or a manually
constructed `LLM_OPTIONS` object.

`apply_route_env()` replaces all nine managed values, including empty values;
the CLI's `--env` output does the same with shell exports. Missing OAuth fields
are cleared, so declare `auth_url` or `auth_url_env` on a route that requires a
custom token endpoint. Catalog selection replaces an existing `LLM_OPTIONS`
object rather than merging it.

Explicit constructor overrides such as `verify`, `http_timeout`/`timeout_s`,
`reasoning_field`, and `no_degrade` take precedence where supported.
`length_retry_cap` is an exception: it supplies a default that the environment
can override. Boolean enable flags below accept case-insensitive `1`, `true`,
and `yes`; other values disable them. All durations are in seconds.

### Output and concurrency (route options)

- `LLM_MAX_OUTPUT_TOKENS`: positive integer output-budget ceiling for both
  clients. OpenAI has no default ceiling. GigaChat uses model-family limits
  by default: 4096 for base models, 8192 for Pro, 16384 for Max, and 32768 for
  Ultra. `use_model_token_limits=False` disables these inferred limits, but
  an explicit setting still applies.
- `LLM_MIN_OUTPUT_TOKENS`: positive integer floor for the requested output
  budget, useful for reasoning models. OpenAI only; unset means no floor.
- `LLM_MAX_CONCURRENT`: positive integer limit on concurrent outbound
  generation requests per client instance. Both clients; unset means no limit.
  GigaChat's explicit `max_concurrent` argument takes precedence.
- `LLM_FORCE_TEMPERATURE`: numeric temperature override for every request.
  Both clients; unset preserves the request's value.
- `LLM_FORCE_TOP_P`: numeric `top_p` override. GigaChat only; unset means no
  override. OpenAI endpoints can receive `top_p` through `extra_body`.

### Reasoning (route options)

- `LLM_DISABLE_REASONING`: boolean, default `false`. OpenAI applies the declared
  `reasoning_off` object and removes declared `reasoning_on` fields; without
  a valid off object, the configured baseline is preserved. GigaChat suppresses
  `reasoning_effort` and sends `chat_template_kwargs.enable_thinking=false`.
- `LLM_REASONING_ON`: JSON object declaring OpenAI request fields that enable
  reasoning; default `{}`. Merged into the extra request body.
- `LLM_REASONING_OFF`: JSON object declaring OpenAI request fields that disable
  reasoning; unset means no off dialect. An explicit `{}` removes declared on
  fields without adding replacement fields.
- `LLM_DISABLE_THINKING_FOR_TOOLS`: boolean, default `false`. OpenAI only;
  applies the declared off dialect to native function requests, preserving
  reasoning for text generation and text fallback. Requires a valid off object.
- `LLM_REASONING_EFFORT`: GigaChat default reasoning effort: `low`, `medium`,
  or `high`. Unset or invalid values add no effort setting. A request's
  `reasoning_effort` takes precedence; disabling reasoning suppresses it.
- `LLM_REASONING_FIELD`: response field containing reasoning text. OpenAI
  defaults to checking `reasoning_content` and `reasoning`; GigaChat defaults
  to `reasoning_content`. An explicit `reasoning_field` argument takes precedence.

### Schema, request extensions, and usage (route options)

These settings apply to OpenAI clients only.

- `LLM_DISABLE_TOOLS`: boolean, default `false`. Selects text-based emulation
  for structured output instead of native tool calling.
- `LLM_TOOL_CHOICE_PREF`: initial structured-output preference: `auto`, `required`, or
  `text`. Unset uses automatic negotiation starting with strict named-function
  selection. `auto` starts with automatic function selection before trying
  forced forms. For multi-tool requests, `required` forces a tool call and the
  default is `auto`. An explicit `tool_choice_pref` argument takes precedence.
- `LLM_RESPONSE_FORMAT`: declared native structured-output dialect:
  `json_schema` or `json_object`. Unset disables this additional fallback tier.
  `json_object` guarantees JSON syntax, while `json_schema` includes the schema
  when the request provides one.
- `LLM_OPEN_OBJECT_SCHEMAS`: set to `unsupported` to bypass schema-bearing
  tiers for schemas containing open objects. Unset leaves the client to learn
  this restriction from matching provider errors.
- `LLM_SANITIZE_ENUMS`: boolean, default `false`. Normalizes enum literals to
  strings for Google-compatible function schemas.
- `LLM_EXTRA_BODY`: JSON object of additional provider request fields; default
  `{}`. Existing generated body fields generally take precedence; reasoning
  on/off settings apply their declared overrides.
- `LLM_EXTRA_HEADERS`: JSON object of extra HTTP headers; default `{}`. Header
  values are converted to strings, and matching constructor header keys are
  overwritten by these settings.
- `LLM_CACHE_HIT_FIELD`: flat usage field for cache-hit tokens; defaults to
  `prompt_cache_hit_tokens`.
- `LLM_CACHE_MISS_FIELD`: flat usage field for cache-miss tokens; defaults to
  `prompt_cache_miss_tokens`.
- `LLM_CACHE_NESTED_FIELD`: dotted usage path for cache-hit tokens when flat
  hits are absent; defaults to `prompt_tokens_details.cached_tokens`. Misses
  are derived from prompt tokens when possible. Empty cache-field settings
  select the defaults; unavailable cache statistics are reported as `-1`.

### Transport and retries (runtime options)

- `LLM_HTTP_TIMEOUT`: HTTP timeout, default `600` for OpenAI and GigaChat
  generation, `120` for Batch HTTP calls. GigaChat generation uses a separate
  30-second connect timeout. Explicit timeout arguments take precedence.
- `LLM_MAX_RETRIES`: additional OpenAI retries for retryable failures; default
  `3` (up to four attempts). GigaChat transient retries use constructor settings.
- `LLM_RETRY_BACKOFF_S`: OpenAI exponential-backoff base delay; default `1.0`.
  Retries add jitter and may honor a server's retry delay.
- `LLM_LENGTH_RETRIES`: nonnegative number of additional attempts after output
  truncation, increasing the token budget. Defaults to `2` for OpenAI and `1`
  for GigaChat; `0` disables these retries.
- `LLM_LENGTH_RETRY_CAP`: positive integer OpenAI token-budget ceiling for
  length retries; default `32768`, or the constructor's `length_retry_cap`.
  If the minimum budget reaches an implicit cap, the cap grows to twice that
  minimum; an explicit environment cap is preserved. GigaChat instead uses
  its configured model/output ceiling.
- `LLM_STREAM_TRANSPORT`: boolean, default `false`. OpenAI only; uses SSE
  internally for non-streaming generation calls and reconstructs the final
  response. The normal `generate_stream()` API does not require this flag.
- `LLM_NO_DEGRADE`: boolean, default `false`. OpenAI only; raises instead of
  implicitly falling back from native structured output to text emulation.
  Explicit text mode and negotiation between native tool forms remain allowed.
- `LLM_FREQUENCY_PENALTY`: optional numeric OpenAI `frequency_penalty` value.
  Unset adds no value; an existing `extra_body` value takes precedence.
- `LLM_REPETITION_PENALTY`: optional numeric OpenAI-compatible
  `repetition_penalty` value, with the same precedence. The endpoint must
  support this field; unset adds no value.

### GigaChat Batch API (runtime options)

- `LLM_BATCH_MODE`: boolean, default `false`. Enables Batch generation in the
  catalog factory for GigaChat routes. It does not change directly constructed
  generation clients or enable Batch for OpenAI routes.
- `LLM_BATCH_POLL_INTERVAL_S`: interval between batch status polls; default `5.0`.
- `LLM_BATCH_MAX_WAIT_S`: maximum wait for batch completion; default `3600.0`.
- `LLM_BATCH_HTTP_RETRIES`: maximum number of HTTP attempts, including the
  initial attempt; default `12`, minimum `1`. The budget covers rate-limit
  retries and an authentication refresh attempt.
- `LLM_BATCH_429_BACKOFF_START`: initial delay after HTTP 429; default `1.0`.
  Subsequent delays grow by a factor of 1.5, capped at 45 seconds.
- `LLM_BATCH_COALESCE_MAX`: maximum number of pending generation requests
  grouped into one batch by `BatchingLLMClient`; default `16`, minimum `1`.
- `LLM_BATCH_COALESCE_DELAY_S`: coalescing delay before submitting a partial
  group; default `0.25`.

Batch uses the connection/authentication settings above and `LLM_HTTP_TIMEOUT`.
Explicit Batch constructor settings override their environment defaults.

## Request policies and streaming events

Both clients use the same transport implementation for every caller. Configure
behavior explicitly instead of choosing a separate client implementation:

- `OpenAIClient(fallback_policy="preserve")` exposes corrupt arguments, schema
  validation failures, open-object errors, and exhausted gateway failures without
  generating a replacement response. Declared open-object restrictions do not
  bypass the native request in this mode; the actual provider response is used.
  A multi-tool text turn is returned as text rather than forced into another call.
  Native `tool_choice` negotiation after an explicit unsupported-form error and
  normal transport/length retries remain enabled. Implicit text fallback is
  rejected; explicitly selected text emulation still works. The default policy
  is `"recover"`. This constructor policy is independent of `LLM_NO_DEGRADE`.
- `OpenAIClient(validate_schema=False)` returns parsed structured arguments
  without retrying schema violations, so callers can score the original model
  response. The default is `True`; this policy covers tool and response-format
  tiers. JSON parsing and transport error handling still apply.
- `OpenAIClient(no_degrade=True)` raises when native structured output would
  otherwise fall back to text emulation. Explicit text mode remains available.
- `length_retry_cap` sets the default OpenAI output budget ceiling for length
  retries; `LLM_LENGTH_RETRY_CAP` overrides it. `reasoning_field` selects the
  response field to read and takes precedence over `LLM_REASONING_FIELD`.
- `tool_choice_pref="required"` selects the initial OpenAI tool-choice form.
  `disable_thinking_for_tools=True` applies the declared reasoning-off dialect
  only to native function requests.
- `GigaChatAsyncClient(tool_choice="auto")` sends the tools from `LLMRequest.tools`
  using legacy automatic function selection. The default `"single"` forces
  `function_name` with `schema_`. Native modern tool loops (`tools_required`)
  are unsupported by the GigaChat client and raise a validation error.
- `GigaChatAsyncClient(use_model_token_limits=False)` leaves the output limit
  to the caller unless `LLM_MAX_OUTPUT_TOKENS` supplies an explicit ceiling.
  `LLMRequest(mode="json_schema", schema=...)` selects native JSON Schema output.

`LLM_REASONING_ON` and `LLM_REASONING_OFF` contain JSON objects describing an
OpenAI-compatible provider's request fields. Disabling reasoning uses only the declared off
object; an empty object removes declared on fields. If the off dialect is
missing or invalid, the client preserves the configured baseline and does not
invent vendor fields. Catalog entries expose these objects as `reasoning_on`
and `reasoning_off`.

Alongside `generate_stream()`, both clients expose `generate_stream_events()`.
It yields typed content, reasoning, tool, completion, and error events from
`llm_mesh.stream_events`; `Complete` includes available usage and finish reason.
A transport failure yields `Error` and then raises the corresponding exception.
`EventStreamGenerator` describes this interface for type checking.

For synchronous model discovery before constructing a client, use
`llm_mesh.list_models(base_url, api_key)`. It returns the endpoint's model IDs
in their advertised order; selecting a model belongs to the caller.

## Package layout

Provider code is grouped in `llm_mesh.openai` and `llm_mesh.gigachat`.
Each contains `client.py` and private `_common.py` helpers. OpenAI-compatible
model discovery lives in `openai/discovery.py`; the GigaChat Batch API lives
in `gigachat/batch.py`. Request and response types, retries, SSE parsing,
the built-in canary, optional hooks, and the public model catalog remain
shared at the package root.

Import clients from `llm_mesh`, `llm_mesh.openai`, or `llm_mesh.gigachat`.
Provider subpackages also export `list_models` and the GigaChat batch clients,
respectively.

## Model catalog

The library ships a shared public `models.json` with models and ordered provider
routes. A project can keep its additional models in its own `models.json`, without
copying or modifying the public catalog. Credentials are supplied through the
environment variables named by the routes. Loading a catalog makes no API calls.

```python
from llm_mesh.models_catalog import load_catalog, resolve_route, make_client

routes = load_catalog()
# Choose an id@provider present in the catalog and configure its credentials:
route = resolve_route("your-model-id@your-provider", routes)
client = make_client(route)
# Use the client and await client.aclose() when finished.
```

To add project models, copy [examples/models.json](examples/models.json) into your
project and edit its entries. `extends` includes all models from the public catalog:

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

Load both catalogs together by specifying the project file:

```python
routes = load_catalog("/path/to/your/project/models.json")
# Public model ids and custom-model@custom are now available in routes.
```

Alternatively, set `LLM_MODELS_CONFIG=/path/to/your/project/models.json` and call
`load_catalog()` without arguments. An explicit path takes precedence over this
variable. With neither, only the bundled catalog is loaded. Files in the working
directory are never picked up implicitly.

New ids are appended to the public catalog. If an id already exists, the project
entry replaces that complete model, including its provider list. Public catalog
updates remain available unless the project overrides the same id. Optional
`exclude` removes ids. `order` lists unique known ids to place first; unlisted
ids follow in catalog order, so newly added public models remain available.
Use `patches` to change individual fields without freezing the entire model:

```json
{
  "extends": "llm-mesh",
  "patches": [
    {
      "id": "existing-model-id",
      "providers": [{"name": "existing-provider", "max_concurrent": 2}],
      "exclude_providers": ["unwanted-provider"]
    }
  ]
}
```

Patches apply after `models` and before `exclude`/`order`. Each patch must name
an existing model id. Provider changes match by `name`; new names append routes,
while unmentioned routes and fields keep inheriting public updates. Model and
provider fields are shallow overrides: an object such as `extra_body` replaces
that whole field, and `null` removes a field. Use `exclude_providers` to remove
routes explicitly. Duplicate patch ids/names are rejected. Full replacements
through `models` remain available when intentional isolation is required.

Provider order within a model is preserved. For a fully independent catalog,
a plain JSON array is also supported and replaces the public catalog. Use
`load_catalog_data()` for nested entries, `load_catalog()` for flattened routes,
and `catalog_digest()` for an effective catalog fingerprint.

`make_client()` uses the legacy environment-based configuration and updates
process-wide variables. Construct catalog clients serially, before issuing
requests. For independent configuration use explicit client constructors.

## Canary checks

Set a session marker with `llm_mesh.canary.set_canary_context_token()`. Clients
append it to the system prompt and scan responses. Reset the token in a
`finally` block so it cannot leak into later calls.

```python
from llm_mesh.canary import reset_canary_context_token, set_canary_context_token

token = set_canary_context_token("session-marker")
try:
    ...
finally:
    reset_canary_context_token(token)
```

A different prompt or detector can replace the defaults through
`llm_mesh.hooks.configure_canary_hooks()`.

## Development

Write all documentation, comments, catalog notes, built-in prompts, log messages,
errors, and test fixtures in English. Caller-provided text and provider responses
retain their original language; the library does not translate them.
Keep descriptions and examples self-contained, without references to private
repositories, internal issue trackers, or the library's extraction history.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Tests use mocked HTTP transports and do not require provider credentials.

## Releases

`.github/workflows/release.yml` runs on tags such as `v1.0.0`. It runs the test
matrix, checks that the tag matches `project.version` in `pyproject.toml`, builds
the wheel and source distribution, and checks an installed wheel outside the
checkout. The same distribution artifacts are then published to PyPI and
attached to a GitHub Release with generated release notes.

Before the first release, create the GitHub environment `pypi` and configure a
[PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
(or a pending publisher for a new project) with these exact settings:

- Project: `llm-mesh`.
- GitHub owner: `SergeyLizin`.
- Repository: `llm-mesh`.
- Workflow filename: `release.yml`.
- Environment: `pypi`.

Publishing uses GitHub OIDC; no PyPI API token secret is needed. Restrict the
`pypi` environment to release tags (`v*`). The workflow files do not create the
PyPI publisher or configure the GitHub environment automatically.

For a release, update `project.version`, commit the final contents, then push
the matching tag. For example, once version `1.0.0` is ready:

```sh
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin main
git push origin v1.0.0
```

A manual **Run workflow** invocation only tests and builds downloadable
artifacts, even when a tag is selected; it never publishes. Use it to rehearse
the pipeline. Once a version is published to PyPI, release changes under a new
version. If PyPI succeeds but GitHub Release creation fails, rerun only the
failed job to reuse the already-built artifacts without republishing to PyPI.
