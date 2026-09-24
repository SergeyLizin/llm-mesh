"""Tests for public capability protocols and provider-neutral helpers."""

from __future__ import annotations

import inspect
import logging

import pytest

from llm_mesh._common import (
    _env_flag,
    _env_is_disabled,
    _env_nonneg_int,
    _env_positive_int,
    _parse_json_dict_env,
    build_text_messages,
    post_with_length_retry,
)
from llm_mesh.anthropic import AnthropicClient
from llm_mesh.base import BaseLLMClient, Capability
from llm_mesh.gemini import GeminiClient
from llm_mesh.gigachat.client import GigaChatClient
from llm_mesh.gigachat.batch import BatchingLLMClient
from llm_mesh.models_catalog import (
    VALID_KINDS,
    _ROUTE_CLIENT_BUILDERS,
    ModelCatalogError,
    make_client,
)
from llm_mesh.openai.client import OpenAIClient
from llm_mesh.protocol import (
    AsyncClosable,
    BatchLLMClient,
    EventStreamGenerator,
    LLMClient,
    StreamGenerator,
    StructuredGenerator,
    TextGenerator,
)
from llm_mesh.types import LLMRequest


def _request(**updates: object) -> LLMRequest:
    values = {
        "system": "SYS",
        "user": "NOW",
        "schema": {"type": "object"},
        "function_name": "emit",
        "history": [
            {"role": "user", "content": "OLD-Q"},
            {"role": "assistant", "content": "OLD-A"},
            {"role": "developer", "content": "DROP"},
        ],
    }
    values.update(updates)
    return LLMRequest(**values)


def test_protocols_are_capability_split_for_batch_clients() -> None:
    """The batch protocol must not promise unsupported streaming methods."""
    assert "generate_text" in TextGenerator.__dict__
    assert "generate_stream" in LLMClient.__dict__ or any(
        "generate_stream" in base.__dict__ for base in LLMClient.__mro__
    )
    assert "generate_stream" not in BatchLLMClient.__dict__
    assert not hasattr(BatchingLLMClient, "generate_stream")


def test_common_text_messages_filter_history_roles() -> None:
    messages = build_text_messages(_request())
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in messages] == [
        "SYS",
        "OLD-Q",
        "OLD-A",
        "NOW",
    ]


def test_common_text_messages_carry_tool_turns() -> None:
    """Forward assistant tool calls and linked tool results together so strict providers receive
    consistent history.
    """
    tool_calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "validate_record", "arguments": "{}"},
    }]
    messages = build_text_messages(_request(history=[
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
    ]))
    assert [m["role"] for m in messages] == ["system", "assistant", "tool", "user"]
    assert messages[1]["tool_calls"] == tool_calls
    assert messages[2]["tool_call_id"] == "c1"
    assert messages[2]["content"] == "OBS"


def test_common_text_messages_drop_tool_turns_when_unsupported() -> None:
    """When tool turns are unsupported, drop both the assistant call and its result to avoid
    orphaned history.
    """
    messages = build_text_messages(
        _request(history=[
            {"role": "user", "content": "OLD-Q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "OBS"},
        ]),
        tool_turns=False,
    )
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert all("tool_calls" not in m for m in messages)


def test_structured_history_included_both_providers() -> None:
    """Include history in structured requests for both OpenAI and GigaChat legacy clients, matching
    their text-generation paths.
    """
    request = _request()
    openai = OpenAIClient(model="m", base_url="https://example.test/v1", api_key="k")
    openai_body = openai._fc_body(
        request,
        request.function_name,
        request.schema_ or {},
        {"type": "function", "function": {"name": request.function_name}},
    )
    gigachat = GigaChatClient(token="t")
    gigachat_body = gigachat._build_body(request)

    expected_roles = ["system", "user", "assistant", "user"]
    assert [message["role"] for message in openai_body["messages"]] == expected_roles
    assert [message["role"] for message in gigachat_body["messages"]] == expected_roles


@pytest.mark.asyncio
async def test_length_retry_does_not_translate_provider_exception() -> None:
    sentinel = RuntimeError("provider-specific")

    async def post(_: dict[str, object]) -> dict[str, object]:
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        await post_with_length_retry(
            {"max_tokens": 10},
            post=post,
            payload_of=lambda result: result,
            retries=2,
            next_max_tokens=lambda current: current * 2,
            provider="test",
            logger=logging.getLogger(__name__),
        )
    assert caught.value is sentinel


# Capabilities that share a method still need that method's real signature.
# BATCH is intentionally absent: no interactive client declares it, and a
# declaration without a method must fail this map rather than pass quietly.
_CAPABILITY_METHODS = {
    Capability.TEXT: "generate_text",
    Capability.STREAM: "generate_stream",
    Capability.STREAM_EVENTS: "generate_stream_events",
    Capability.STRUCTURED: "generate_structured",
    Capability.TOOLS: "generate_structured",
    Capability.MULTI_TOOL: "generate_structured",
    Capability.TOOLS_REQUIRED: "generate_structured",
    Capability.JSON_SCHEMA_MODE: "generate_structured",
    Capability.LENGTH_RETRY: "generate_text",
    Capability.COUNT_TOKENS: "count_tokens",
    Capability.EMBEDDINGS: "embed",
    Capability.RERANK: "rerank",
}

_INTERACTIVE = frozenset({
    Capability.TEXT,
    Capability.STREAM,
    Capability.STREAM_EVENTS,
    Capability.STRUCTURED,
    Capability.TOOLS,
    Capability.MULTI_TOOL,
    Capability.JSON_SCHEMA_MODE,
    Capability.LENGTH_RETRY,
})

_EXPECTED_CAPABILITIES = {
    OpenAIClient: _INTERACTIVE | frozenset({
        Capability.TOOLS_REQUIRED,
        Capability.EMBEDDINGS,
        Capability.COUNT_TOKENS,
        Capability.RERANK,
    }),
    AnthropicClient: _INTERACTIVE | frozenset({
        Capability.TOOLS_REQUIRED,
        Capability.COUNT_TOKENS,
    }),
    GeminiClient: _INTERACTIVE | frozenset({
        Capability.TOOLS_REQUIRED,
        Capability.COUNT_TOKENS,
        Capability.EMBEDDINGS,
    }),
    # Legacy functions API: one selected function, no native tool loop.
    GigaChatClient: _INTERACTIVE | frozenset({
        Capability.COUNT_TOKENS,
        Capability.EMBEDDINGS,
    }),
}

_PROTOCOL_BASES = (
    TextGenerator,
    StreamGenerator,
    EventStreamGenerator,
    StructuredGenerator,
    AsyncClosable,
    LLMClient,
)


def _pin_managed_env(monkeypatch) -> None:
    """Record catalog keys so make_client's os.environ.update is undone."""
    import os

    from llm_mesh.models_catalog import _MANAGED_ENV

    for key in _MANAGED_ENV:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    # Batch mode changes the openai and gigachat builders. These tests assert
    # the interactive client, so an ambient flag must not leak in.
    monkeypatch.delenv("LLM_BATCH_MODE", raising=False)


def _route(kind: str) -> dict[str, str]:
    return {
        "id": "custom",
        "kind": kind,
        "model": "test-model",
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
    }


def _openai() -> OpenAIClient:
    return OpenAIClient(model="m", base_url="https://example.test/v1", api_key="k")


def _anthropic() -> AnthropicClient:
    return AnthropicClient(model="m", api_key="k")


def _gemini() -> GeminiClient:
    return GeminiClient(model="m", api_key="k")


def _gigachat() -> GigaChatClient:
    return GigaChatClient(token="t")


def test_valid_kinds_match_the_registry() -> None:
    assert VALID_KINDS == tuple(sorted(_ROUTE_CLIENT_BUILDERS))
    assert set(VALID_KINDS) == {"anthropic", "gemini", "gigachat", "openai"}


def test_unknown_kind_keeps_the_catalog_error(monkeypatch) -> None:
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ModelCatalogError) as caught:
        make_client({"id": "nope", "kind": "nope", "model": "m"})
    assert str(caught.value) == (
        f"'nope': kind='nope', expected one of {VALID_KINDS}"
    )


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("openai", OpenAIClient),
        ("anthropic", AnthropicClient),
        ("gemini", GeminiClient),
        ("gigachat", GigaChatClient),
    ],
)
def test_registry_client_satisfies_llm_client(monkeypatch, kind: str, cls: type) -> None:
    """Every interactive builder returns a BaseLLMClient that has the protocol methods."""
    _pin_managed_env(monkeypatch)
    client = make_client(_route(kind))
    assert type(client) is cls
    assert isinstance(client, BaseLLMClient)
    for protocol in _PROTOCOL_BASES:
        for name, value in protocol.__dict__.items():
            if name.startswith("_") or not callable(value):
                continue
            method = getattr(client, name)
            params = [
                param for param in inspect.signature(method).parameters
                if param != "self"
            ]
            if name == "aclose":
                assert params == []
            else:
                assert params[0] == "request"
    for capability in cls.CAPABILITIES:
        method_name = _CAPABILITY_METHODS[capability]
        method = getattr(client, method_name)
        assert callable(method)
        params = [
            param for param in inspect.signature(method).parameters
            if param != "self"
        ]
        if method_name in ("count_tokens", "embed"):
            assert params[0] == "texts"
        elif method_name == "rerank":
            assert params[0] == "query"
        else:
            assert params[0] == "request"
        assert client.supports(capability)
    assert not client.supports(Capability.BATCH)


def test_capability_matrix_is_the_provider_difference() -> None:
    """Changing a capability means editing this table on purpose."""
    for cls, expected in _EXPECTED_CAPABILITIES.items():
        assert cls.CAPABILITIES == expected
        assert Capability.BATCH not in cls.CAPABILITIES


def test_gigachat_batch_builder_stays_outside_the_base(monkeypatch) -> None:
    _pin_managed_env(monkeypatch)
    monkeypatch.setenv("LLM_BATCH_MODE", "1")
    client = make_client(_route("gigachat"))
    assert isinstance(client, BatchingLLMClient)
    assert not isinstance(client, BaseLLMClient)


@pytest.mark.parametrize(
    ("raw", "stripped", "unstripped"),
    [
        ("", False, False),
        ("0", False, False),
        ("false", False, False),
        ("no", False, False),
        ("TRUE", True, True),
        ("Yes", True, True),
        ("1", True, True),
        (" 1 ", True, False),
        (" true", True, False),
    ],
)
def test_env_flag_keeps_strip_and_nostrip(
    monkeypatch, raw: str, stripped: bool, unstripped: bool,
) -> None:
    monkeypatch.setenv("LLM_PROBE_FLAG", raw)
    assert _env_flag("LLM_PROBE_FLAG") is stripped
    assert _env_flag("LLM_PROBE_FLAG", strip=False) is unstripped


@pytest.mark.parametrize(
    ("raw", "verified"),
    [
        (None, None),
        ("", None),
        ("0", False),
        ("false", False),
        ("no", False),
        ("TRUE", True),
        ("Yes", True),
        ("1", True),
        (" false", True),
    ],
)
def test_verify_ssl_stays_inverted(
    monkeypatch, raw: str | None, verified: bool | None,
) -> None:
    """Unset OpenAI/Anthropic verify; unset GigaChat does not. Padded values are not off."""
    if raw is None:
        monkeypatch.delenv("LLM_VERIFY_SSL", raising=False)
    else:
        monkeypatch.setenv("LLM_VERIFY_SSL", raw)
    openai_default = verified if verified is not None else True
    gigachat_default = verified if verified is not None else False
    assert _openai()._verify is openai_default
    assert _anthropic()._verify is openai_default
    assert _gigachat()._verify is gigachat_default
    if raw in (None, ""):
        assert _env_is_disabled("LLM_VERIFY_SSL", default="1") is False
        assert _env_is_disabled("LLM_VERIFY_SSL", default="false") is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "TRUE", "Yes", "1"])
def test_positive_and_nonneg_int_edges(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("LLM_PROBE_INT", raw)
    positive = _env_positive_int("LLM_PROBE_INT")
    nonneg = _env_nonneg_int("LLM_PROBE_INT", 7)
    if raw == "1":
        assert positive == 1
        assert nonneg == 1
    elif raw == "0":
        assert positive is None
        assert nonneg == 0
    else:
        assert positive is None
        assert nonneg == 7


def test_parse_json_dict_keeps_each_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LLM_EXTRA_BODY", "{not json")
    openai_log = logging.getLogger("llm_mesh.openai.client")
    anthropic_log = logging.getLogger("llm_mesh.anthropic.client")
    with caplog.at_level(logging.WARNING, logger="llm_mesh.openai.client"):
        assert _parse_json_dict_env("LLM_EXTRA_BODY", logger=openai_log) is None
    assert "LLM_EXTRA_BODY invalid JSON" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="llm_mesh.anthropic.client"):
        assert _parse_json_dict_env(
            "LLM_EXTRA_BODY", logger=anthropic_log, include_error=False,
        ) is None
    assert "LLM_EXTRA_BODY is not valid JSON — ignoring" in caplog.text
    caplog.clear()
    monkeypatch.setenv("LLM_EXTRA_BODY", "[]")
    with caplog.at_level(logging.WARNING, logger="llm_mesh.openai.client"):
        assert _parse_json_dict_env("LLM_EXTRA_BODY", logger=openai_log) is None
    assert "LLM_EXTRA_BODY is not a JSON object — ignoring" in caplog.text
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"top_p": 0.8}')
    assert _parse_json_dict_env("LLM_EXTRA_BODY", logger=openai_log) == {"top_p": 0.8}


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_openai, _anthropic, _gemini, _gigachat])
async def test_aclose_is_idempotent_and_http_is_lazy(factory) -> None:
    client = factory()
    assert client._client is None
    await client.aclose()
    await client.aclose()
    assert client._client is None
    client._ensure_http()
    assert client._client is not None
    await client.aclose()
    await client.aclose()
    assert client._client is None


@pytest.mark.asyncio
async def test_ensure_http_keeps_each_providers_timeout() -> None:
    """GigaChat's connect budget is 30s. The other two use one float for every phase."""
    import httpx

    gigachat = _gigachat()
    try:
        transport = gigachat._ensure_http()
        assert transport.timeout.connect == 30.0
        assert transport.timeout.read == gigachat._timeout.read
    finally:
        await gigachat.aclose()

    for factory in (_openai, _anthropic, _gemini):
        client = factory()
        try:
            transport = client._ensure_http()
            assert isinstance(transport.timeout, httpx.Timeout)
            assert transport.timeout.connect == client._http_timeout
            assert transport.timeout.read == client._http_timeout
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_gigachat_context_reuses_and_closes_the_same_client() -> None:
    client = _gigachat()
    first = client._ensure_http()
    async with client as entered:
        assert entered is client
        assert client._client is first
    assert client._client is None
    async with client:
        assert client._client is not None
        assert client._client.timeout.connect == 30.0
    assert client._client is None


def test_supports_ignores_instance_switches(monkeypatch) -> None:
    """supports() is the class set. A disabled instance still reports the path."""
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "1")
    openai = _openai()
    assert openai._tools_enabled is False
    assert openai.supports(Capability.TOOLS)
    gigachat = GigaChatClient(token="t", tool_choice="single")
    assert gigachat.supports(Capability.MULTI_TOOL)
    assert openai.supports(Capability.COUNT_TOKENS)
    assert gigachat.supports(Capability.COUNT_TOKENS)
    assert _anthropic().supports(Capability.COUNT_TOKENS)
    assert _gemini().supports(Capability.COUNT_TOKENS)


@pytest.mark.asyncio
async def test_count_tokens_names_the_missing_capability() -> None:
    """The base implementation still refuses. OpenAI overrides it with tiktoken."""
    from llm_mesh.base import BaseLLMClient

    client = _openai()
    with pytest.raises(NotImplementedError, match="does not support count_tokens"):
        await BaseLLMClient.count_tokens(client, ["hello"])
