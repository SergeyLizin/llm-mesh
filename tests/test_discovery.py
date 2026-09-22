"""Validate discovery wire paths, response dialects, and error handling."""
import httpx
import pytest
import respx

from llm_mesh import list_models, LLMAuthError, LLMError, LLMValidationError


@pytest.mark.parametrize("base", ["https://example.test/v1", "https://example.test/v1/chat/completions"])
@pytest.mark.parametrize("payload", [{"data": [{"id": "one"}, {"id": "two"}]},
                                     {"models": [{"name": "one"}, {"name": "two"}]}])
@respx.mock
def test_discovery_preserves_order_and_authentication(base, payload):
    route = respx.get("https://example.test/v1/models").respond(200, json=payload)
    assert list_models(base, "test-key") == ["one", "two"]
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("status,error", [(401, LLMAuthError), (403, LLMAuthError), (500, LLMError)])
@respx.mock
def test_discovery_errors(status, error):
    respx.get("https://example.test/v1/models").respond(status)
    with pytest.raises(error):
        list_models("https://example.test/v1", "test-key")


@pytest.mark.parametrize("payload", [{"data": [{"unexpected": "value"}]},
                                    {"data": "invalid"}, ["invalid"]])
@respx.mock
def test_malformed_discovery(payload):
    respx.get("https://example.test/v1/models").respond(200, json=payload)
    with pytest.raises(LLMValidationError):
        list_models("https://example.test/v1", "test-key")
