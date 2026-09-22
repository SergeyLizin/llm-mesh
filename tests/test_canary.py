"""Built-in canary policy and its use by shared clients."""
import logging

import pytest

from llm_mesh import LLMRequest, _common, canary, hooks
from llm_mesh.gigachat import GigaChatAsyncClient
from llm_mesh.openai import OpenAIClient


def test_builtin_canary_injects_and_scans(caplog):
    request = LLMRequest(system="system", user="user")
    assert _common.build_text_messages(request)[0]["content"] == "system"
    token = canary.set_canary_context_token("test-canary")
    try:
        assert "test-canary" in _common.build_text_messages(request)[0]["content"]
        with caplog.at_level(logging.CRITICAL, logger="llm_mesh.canary"):
            assert canary.check_and_warn("leaked test-canary", "test-canary") is True
        assert "CANARY_DETECTED" in caplog.text
    finally:
        canary.reset_canary_context_token(token)
    assert _common.build_text_messages(request)[0]["content"] == "system"


@pytest.mark.parametrize("provider", ["openai", "gigachat"])
def test_canary_context_reaches_shared_clients(provider, caplog):
    if provider == "openai":
        client = OpenAIClient(base_url="https://example.test/v1", api_key="test-key")
        messages = client._messages
    else:
        client = GigaChatAsyncClient(token="test-token")
        messages = lambda request: client._build_body(request)["messages"]
    request = LLMRequest(system="system", user="user")
    token = canary.set_canary_context_token("test-canary")
    try:
        assert "test-canary" in messages(request)[0]["content"]
        with caplog.at_level(logging.CRITICAL, logger="llm_mesh.canary"):
            assert hooks.check_and_warn("leaked test-canary", "test-canary") is True
        assert "CANARY_DETECTED" in caplog.text
    finally:
        canary.reset_canary_context_token(token)
    assert messages(request)[0]["content"] == "system"
