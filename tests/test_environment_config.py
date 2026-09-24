"""Neutral configuration, typed options, and isolation between provider routes."""
import json
import os

import pytest

from llm_mesh import OpenAIClient, GigaChatClient, LLMRequest
from llm_mesh.config import get_env, read_options
from llm_mesh.gigachat.batch import GigaChatBatchClient, get_batching_client
from llm_mesh.models_catalog import _MANAGED_ENV, apply_route_env, make_client, route_env


@pytest.fixture(autouse=True)
def clean_connections(monkeypatch):
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("LLM_BATCH_MODE", raising=False)
    yield
    for key in _MANAGED_ENV:
        os.environ.pop(key, None)


def route(kind="openai", **options):
    return {"id": "custom", "kind": kind, "model": "test-model",
            "base_url": "https://example.test/v1", "api_key": "test-key", **options}


def test_managed_environment_is_compact_and_neutral():
    env = route_env(route())
    assert len(_MANAGED_ENV) == 9
    assert set(env) == set(_MANAGED_ENV)
    assert all(name.startswith("LLM_") for name in env)
    assert json.loads(env["LLM_OPTIONS"]) == {}


def test_options_preserve_false_zero_empty_dialect_and_nested_values():
    env = route_env(route(disable_reasoning=False, force_temperature=0,
                          reasoning_off={}, extra_body={"reasoning": {"enabled": False}}))
    options = json.loads(env["LLM_OPTIONS"])
    assert options["disable_reasoning"] is False
    assert options["force_temperature"] == 0
    assert options["reasoning_off"] == {}
    assert options["extra_body"]["reasoning"]["enabled"] is False
    assert "api_key" not in options and "base_url" not in options


def test_route_options_ignore_stale_standalone_settings(monkeypatch):
    monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "16")
    c = make_client(route())
    assert c._tools_enabled is True
    assert c._clip_max_tokens(1000) == 1000


def test_switching_provider_clears_options_and_authentication(monkeypatch):
    first = make_client(route("gigachat", scope="scope-one", auth_url="https://auth.test/token",
                              reasoning_on={"reasoning_effort": "high"}, max_tokens=128))
    second = make_client(route(api_key="second-key"))
    assert first._clip_max_tokens(1000) == 128
    assert first._resolve_reasoning_effort(LLMRequest(system="s", user="u")) == "high"
    assert second._clip_max_tokens(1000) == 1000
    assert os.environ["LLM_AUTH_SCOPE"] == ""
    assert os.environ["LLM_AUTH_URL"] == ""
    assert os.environ["LLM_API_KEY"] == "second-key"
    assert json.loads(os.environ["LLM_OPTIONS"]) == {}


def test_connection_settings_are_read_when_client_is_constructed(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("LLM_AUTH_URL", "https://proxy.test/oauth")
    monkeypatch.setenv("LLM_AUTH_SCOPE", "scope-one")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_VERIFY_SSL", "true")
    monkeypatch.setenv("LLM_OPTIONS", '{"http_timeout": 17}')
    c = GigaChatClient()
    assert c._api_url == "https://proxy.test/v1"
    assert c._auth_url == "https://proxy.test/oauth"
    assert c._scope == "scope-one"
    assert c._timeout.read == 17
    assert c._verify is True
    batch = GigaChatBatchClient(auth=c)
    assert batch._base_url == "https://proxy.test/v1/"
    assert batch._verify is True
    assert batch._timeout.read == 17


def test_explicit_connection_arguments_take_precedence(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("LLM_VERIFY_SSL", "false")
    c = GigaChatClient(token="token", api_url="https://explicit.test/v1", verify=True)
    assert c._api_url == "https://explicit.test/v1" and c._verify is True


def test_batch_cache_does_not_reuse_another_routes_credentials():
    apply_route_env(route("gigachat", scope="scope-one"))
    first = get_batching_client("test-model")
    apply_route_env(route("gigachat", scope="scope-two", api_key="other-key"))
    second = get_batching_client("test-model")
    assert first is not second
    assert second is get_batching_client("test-model")


def test_catalog_resolves_explicitly_named_credentials(monkeypatch):
    monkeypatch.setenv("CUSTOM_SERVICE_TOKEN", "test-key")
    env = route_env({"kind": "openai", "model": "m", "api_key_env": "CUSTOM_SERVICE_TOKEN"})
    assert env["LLM_API_KEY"] == "test-key"


@pytest.mark.parametrize("raw", ["invalid", "[]", "null", '"text"'])
def test_invalid_options_fail_loudly(raw, monkeypatch):
    monkeypatch.setenv("LLM_OPTIONS", raw)
    with pytest.raises(ValueError, match="LLM_OPTIONS"):
        OpenAIClient(base_url="https://example.test", api_key="test-key")


def test_standalone_options_and_runtime_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "7")
    monkeypatch.setenv("LLM_OPTIONS", '{"max_output_tokens": 200, "reasoning_off": {}}')
    assert get_env("LLM_MAX_RETRIES") == "7"
    assert get_env("LLM_MAX_OUTPUT_TOKENS") == "200"
    assert read_options()["reasoning_off"] == {}


def test_garbage_timeout_and_retries_keep_defaults(monkeypatch, caplog):
    monkeypatch.setenv("LLM_HTTP_TIMEOUT", "soon")
    monkeypatch.setenv("LLM_MAX_RETRIES", "lots")
    with caplog.at_level("WARNING"):
        client = OpenAIClient(base_url="https://example.test", api_key="test-key")
        giga = GigaChatClient(token="tok")
    assert client._http_timeout == 600.0
    assert client._max_retries == 3
    assert giga._timeout.read == 600.0
    assert "LLM_HTTP_TIMEOUT" in caplog.text
    assert "LLM_MAX_RETRIES" in caplog.text
