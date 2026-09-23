"""Offline tests for GigaChat json_schema mode: request body shape, raw union schemas, content
parsing with fences or prose, and legacy function-call compatibility. Exercise body construction
and response parsing directly without HTTP. Construct clients inside async helpers.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest
import respx

from llm_mesh.types import LLMRequest, LLMResponse, LLMValidationError


def _run(coro):
    """Run a coroutine in a fresh event loop and close it afterward."""
    return asyncio.run(coro)


def _schema_with_oneof() -> dict[str, Any]:
    """Build a oneOf schema that legacy simplification would collapse; native json_schema must
    preserve it for constrained decoding.
    """
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "category": {
                "oneOf": [
                    {"type": "string", "const": "factual"},
                    {"type": "string", "const": "speculative"},
                ]
            },
        },
        "required": ["answer", "category"],
    }


def _make_request(*, mode: str = "function_call", schema: dict | None = None) -> LLMRequest:
    return LLMRequest(
        system="sys",
        user="usr",
        function_name="build_answer",
        function_description="Build answer.",
        schema=schema or _schema_with_oneof(),
        max_tokens=1024,
        temperature=0,
        mode=mode,
    )


async def _make_client_async(**kwargs):
    """Construct a GigaChat client inside a coroutine."""
    from llm_mesh.gigachat.client import GigaChatAsyncClient as GigaChatClient
    for k in list(os.environ):
        if k.startswith("GIGACHAT_") or k.startswith("GIGAPERS_") or k.startswith("GIGACORP_"):
            os.environ.pop(k, None)
    values = dict(
        tool_choice="auto",
        use_model_token_limits=False,
        model="GigaChat-3-Ultra",
        credentials="k",
        api_url="http://x",
        scope="s",
    )
    values.update(kwargs)
    return GigaChatClient(**values)


def _make_client(**kwargs):
    return _run(_make_client_async(**kwargs))


# --- Request body shape -----------------------------------------------------


class TestBuildBodyShape:
    """Keep structured modes mutually exclusive: json_schema uses response_format without legacy
    function fields, and function_call uses legacy fields without response_format.
    """

    def test_function_call_mode_builds_legacy_body(self):
        """The default function_call mode sends functions and function_call without
        response_format.
        """
        c = _make_client()
        body = c._build_body(_make_request(mode="function_call"))
        assert "functions" in body
        assert "function_call" in body
        assert body["function_call"] == {"name": "build_answer"}
        assert "response_format" not in body
        # Legacy schema simplification collapses oneOf.
        params = body["functions"][0]["parameters"]
        assert "oneOf" not in params.get("properties", {}).get("category", {})

    def test_json_schema_mode_builds_response_format(self):
        """json_schema sends response_format without legacy function fields and preserves the raw
        oneOf schema.
        """
        c = _make_client()
        body = c._build_body(_make_request(mode="json_schema"))
        assert "response_format" in body
        assert "functions" not in body
        assert "function_call" not in body
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["strict"] is True
        # Preserve oneOf in the raw schema.
        assert "oneOf" in rf["schema"]["properties"]["category"]

    def test_json_schema_mode_with_empty_schema_safe(self):
        """Use an empty object schema safely when json_schema has no supplied schema."""
        c = _make_client()
        req = LLMRequest(system="s", user="u", function_name="fn", mode="json_schema")
        body = c._build_body(req)
        assert body["response_format"]["schema"] == {"type": "object", "properties": {}}

    def test_both_modes_share_common_fields(self):
        """Both modes preserve common model, message, token, and temperature fields."""
        c = _make_client()
        for mode in ("function_call", "json_schema"):
            body = c._build_body(_make_request(mode=mode))
            assert body["model"] == "GigaChat-3-Ultra"
            assert body["max_tokens"] == 1024
            assert len(body["messages"]) == 2  # system + user
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"


# ───────────────────────── _parse_response: json_schema path ─────────────────────────


class TestParseJsonSchemaResponse:
    """Parse json_schema output from message.content through the JSON salvage helper, without
    requiring function_call.
    """

    def _payload(self, content: str, *, finish_reason: str = "stop") -> dict:
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "index": 0,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "GigaChat-3-Ultra",
        }

    def test_plain_json_content(self):
        """Parse plain JSON content directly."""
        c = _make_client()
        resp = c._parse_response(
            self._payload('{"answer": "Paris", "category": "factual"}'),
            _make_request(mode="json_schema"),
        )
        assert isinstance(resp, LLMResponse)
        assert resp.arguments == {"answer": "Paris", "category": "factual"}
        assert resp.function_name == "build_answer"  # Take this value from the request, not the response.
        assert resp.text == '{"answer": "Paris", "category": "factual"}'

    def test_markdown_fenced_json(self):
        """Recover JSON from a Markdown fence."""
        c = _make_client()
        content = '```json\n{"answer": "42", "category": "factual"}\n```'
        resp = c._parse_response(
            self._payload(content), _make_request(mode="json_schema")
        )
        assert resp.arguments == {"answer": "42", "category": "factual"}

    def test_prose_wrapped_json(self):
        """Recover the JSON object from surrounding prose."""
        c = _make_client()
        content = 'Response: {"answer": "yes", "category": "speculative"} something like that.'
        resp = c._parse_response(
            self._payload(content), _make_request(mode="json_schema")
        )
        assert resp.arguments == {"answer": "yes", "category": "speculative"}

    def test_invalid_json_raises(self):
        """Raise LLMValidationError for unparseable content."""
        c = _make_client()
        with pytest.raises(LLMValidationError, match="unparseable JSON"):
            c._parse_response(
                self._payload("not JSON"),
                _make_request(mode="json_schema"),
            )

    def test_empty_content_raises(self):
        """Raise LLMValidationError for empty content."""
        c = _make_client()
        with pytest.raises(LLMValidationError):
            c._parse_response(
                self._payload(""), _make_request(mode="json_schema")
            )

    def test_usage_and_finish_reason_parsed(self):
        """Parse usage and finish_reason consistently across modes."""
        c = _make_client()
        resp = c._parse_response(
            self._payload('{"a": 1}', finish_reason="length"),
            _make_request(mode="json_schema"),
        )
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.finish_reason == "length"


# ───────────────────────── Regression: legacy function_call path ─────────────────────────


class TestLegacyFunctionCallRegression:
    """Preserve legacy argument parsing and the missing-function-call error that triggers retries."""

    def test_function_call_response_parsed_as_before(self):
        """message.function_call.arguments → LLMResponse.arguments."""
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "build_answer",
                        "arguments": {"answer": "London", "category": "speculative"},
                    },
                },
                "finish_reason": "function_call",
                "index": 0,
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "model": "GigaChat-3-Ultra",
        }
        resp = c._parse_response(payload, _make_request(mode="function_call"))
        assert resp.arguments == {"answer": "London", "category": "speculative"}
        # Use the model-selected response function name rather than the requested fallback name.
        assert resp.function_name == "build_answer"

    def test_no_function_call_raises_legacy_error(self):
        """A missing legacy function call retains the validation error recognized by the retry
        loop.
        """
        c = _make_client()
        payload = {
            "choices": [{
                "message": {"role": "assistant", "content": "plain text"},
                "finish_reason": "stop",
                "index": 0,
            }],
            "usage": {},
            "model": "GigaChat-3-Ultra",
        }
        with pytest.raises(LLMValidationError, match="missing function_call"):
            c._parse_response(payload, _make_request(mode="function_call"))

    def test_function_call_arguments_as_json_string(self):
        """Continue accepting legacy arguments serialized as a JSON string."""
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "function_call": {
                        "name": "fn",
                        "arguments": '{"answer": "X", "category": "factual"}',
                    },
                },
                "finish_reason": "function_call",
                "index": 0,
            }],
            "usage": {},
            "model": "GigaChat-3-Ultra",
        }
        resp = c._parse_response(payload, _make_request(mode="function_call"))
        assert resp.arguments == {"answer": "X", "category": "factual"}


# ───────────────────────── Multi-tool (request.tools → FC auto) ─────────────


_TOOLS = [
    {"name": "registry_lookup", "description": "lookup",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}},
                   "required": ["id"]}},
    {"name": "final_answer", "description": "done",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"]}},
]


def _tools_request() -> LLMRequest:
    return LLMRequest(
        system="sys", user="usr", tools=_TOOLS,
        function_name="final_answer", max_tokens=512, temperature=0,
    )


class TestMultiToolLegacyFunctions:
    """Convert request.tools to legacy functions with auto selection, without modern tool fields."""

    def test_history_converts_tool_calls_to_legacy_function(self):
        """Convert modern tool history into legacy function turns with object arguments; GigaChat
        rejects string arguments in history.
        """
        c = _make_client()
        req = LLMRequest(
            system="sys",
            user="continue",
            tools=_TOOLS,
            history=[
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{
                     "id": "c1", "type": "function",
                     "function": {"name": "registry_lookup",
                                  "arguments": '{"id": "R-1"}'},
                 }]},
                {"role": "tool", "tool_call_id": "c1", "name": "registry_lookup",
                 "content": '{"ok": true}'},
            ],
            function_name="final_answer",
        )
        body = c._build_body(req)
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user", "assistant", "function", "user"]
        fc = body["messages"][2]["function_call"]
        assert fc["name"] == "registry_lookup"
        assert fc["arguments"] == {"id": "R-1"}
        assert isinstance(fc["arguments"], dict)
        assert body["messages"][3]["name"] == "registry_lookup"
        assert body["messages"][3]["content"] == '{"ok": true}'

    def test_history_legacy_function_call_string_args_become_object(self):
        c = _make_client()
        req = LLMRequest(
            system="sys",
            user="continue",
            tools=_TOOLS,
            history=[
                {"role": "assistant", "content": None,
                 "function_call": {"name": "registry_lookup",
                                  "arguments": '{"id": "R-2"}'}},
                {"role": "function", "name": "registry_lookup",
                 "content": "{}"},
            ],
            function_name="final_answer",
        )
        body = c._build_body(req)
        assert body["messages"][1]["function_call"]["arguments"] == {"id": "R-2"}

    def test_empty_user_omitted_after_function_turn(self):
        c = _make_client()
        req = LLMRequest(
            system="sys",
            user="",
            tools=_TOOLS,
            history=[
                {"role": "assistant", "content": None,
                 "function_call": {"name": "registry_lookup", "arguments": "{}"}},
                {"role": "function", "name": "registry_lookup", "content": "{}"},
            ],
            function_name="final_answer",
        )
        body = c._build_body(req)
        assert [m["role"] for m in body["messages"]] == [
            "system", "assistant", "function",
        ]

    def test_build_body_uses_auto_and_tool_catalog(self):
        c = _make_client()
        body = c._build_body(_tools_request())
        assert body["function_call"] == "auto"
        assert [f["name"] for f in body["functions"]] == [
            "registry_lookup", "final_answer",
        ]
        assert "response_format" not in body

    def test_parse_legacy_function_call(self):
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "function_call": {
                        "name": "registry_lookup",
                        "arguments": '{"id": "R-1"}',
                    },
                },
                "finish_reason": "function_call",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name == "registry_lookup"
        assert resp.arguments == {"id": "R-1"}

    def test_parse_tool_calls_fallback(self):
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer": "x"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name == "final_answer"
        assert resp.arguments == {"answer": "x"}

    def test_parse_pseudo_content_newline(self):
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "content": 'final_answer\n{"answer": "no data"}',
                },
                "finish_reason": "stop",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name == "final_answer"
        assert resp.arguments == {"answer": "no data"}

    def test_parse_pseudo_content_equals(self):
        c = _make_client()
        payload = {
            "choices": [{
                "message": {
                    "content": 'registry_lookup={"id": "R-88"}',
                },
                "finish_reason": "stop",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name == "registry_lookup"
        assert resp.arguments == {"id": "R-88"}

    def test_no_call_returns_empty_choice_not_raise(self):
        """Return an empty function selection for an auto turn without a call."""
        c = _make_client()
        payload = {
            "choices": [{
                "message": {"content": "plain text without a call"},
                "finish_reason": "stop",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name is None
        assert resp.arguments == {}
        assert resp.text == "plain text without a call"

    def test_unknown_pseudo_name_rejected(self):
        c = _make_client()
        payload = {
            "choices": [{
                "message": {"content": 'not_a_tool\n{"x": 1}'},
                "finish_reason": "stop",
            }],
            "usage": {},
            "model": "GigaChat-2",
        }
        resp = c._parse_response(payload, _tools_request())
        assert resp.function_name is None


# ───────────────────────── _parse_json_content unit ─────────────────────────


class TestParseJsonContentHelper:
    """Table-driven tests for JSON content salvage, independent of client state."""

    @pytest.mark.parametrize("content,expected", [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 2}  ', {"a": 2}),
        ('```json\n{"a": 3}\n```', {"a": 3}),
        ('```\n{"a": 4}\n```', {"a": 4}),
        ('Response: {"a": 5} end.', {"a": 5}),
        # Treat two separated JSON objects as ambiguous: their widest brace span is invalid
        # JSON, so return None instead of silently selecting one.
        ('noise {"a": 6} more {"b": 7}', None),
        ('', None),
        ('not JSON', None),
        ('[1, 2, 3]', None),  # An array is not an object; return None.
    ])
    def test_variants(self, content, expected):
        from llm_mesh.gigachat.client import GigaChatAsyncClient as GigaChatClient
        assert GigaChatClient._parse_json_content(content) == expected


def test_no_degrade_rejects_fenced_json_schema():
    """A fence is salvage. no_degrade must not accept it as native JSON."""
    client = _make_client(no_degrade=True)
    content = '```json\n{"answer": "42", "category": "factual"}\n```'
    payload = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {},
        "model": "GigaChat-3-Ultra",
    }
    with pytest.raises(LLMValidationError, match="forbids salvage"):
        client._parse_response(payload, _make_request(mode="json_schema"))


def test_no_degrade_accepts_raw_json_schema():
    client = _make_client(no_degrade=True)
    content = '{"answer": "42", "category": "factual"}'
    payload = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {},
        "model": "GigaChat-3-Ultra",
    }
    response = client._parse_response(payload, _make_request(mode="json_schema"))
    assert response.arguments == {"answer": "42", "category": "factual"}


def test_preserve_ignores_a_prose_function_envelope():
    """Prose that looks like a call is not a function_call under preserve."""
    client = _make_client(fallback_policy="preserve")
    payload = {
        "choices": [{
            "message": {"content": 'final_answer\n{"answer": "no data"}'},
            "finish_reason": "stop",
        }],
        "usage": {},
        "model": "GigaChat-2",
    }
    response = client._parse_response(payload, _tools_request())
    assert response.function_name is None
    assert response.text == 'final_answer\n{"answer": "no data"}'


def test_extra_body_does_not_replace_model_or_response_format(monkeypatch):
    monkeypatch.setenv(
        "LLM_EXTRA_BODY",
        '{"repetition_penalty": 1.1, "model": "nope", "response_format": {"type": "text"}}',
    )
    client = _make_client()
    body = client._build_body(_make_request(mode="json_schema"))
    assert body["repetition_penalty"] == 1.1
    assert body["model"] == "GigaChat-3-Ultra"
    assert body["response_format"]["type"] == "json_schema"


@respx.mock
def test_extra_body_and_headers_are_sent_on_generate_text(monkeypatch):
    monkeypatch.setenv(
        "LLM_EXTRA_BODY",
        '{"repetition_penalty": 1.05, "model": "nope"}',
    )
    monkeypatch.setenv("LLM_EXTRA_HEADERS", '{"X-Project": "p"}')
    route = respx.post("http://x/chat/completions").respond(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
        "model": "GigaChat-3-Ultra",
    })

    async def go():
        from llm_mesh.gigachat.client import GigaChatAsyncClient
        client = GigaChatAsyncClient(
            token="t",
            model="GigaChat-3-Ultra",
            api_url="http://x",
            use_model_token_limits=False,
        )
        try:
            await client.generate_text(
                LLMRequest(system="s", user="u", max_tokens=8),
            )
        finally:
            await client.aclose()

    _run(go())
    sent = json.loads(route.calls.last.request.content)
    assert sent["repetition_penalty"] == 1.05
    assert sent["model"] == "GigaChat-3-Ultra"
    assert route.calls.last.request.headers["X-Project"] == "p"
    assert route.calls.last.request.headers["Authorization"] == "Bearer t"


def test_invalid_extra_body_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", "not-json")
    client = _make_client()
    body = client._build_body(_make_request())
    assert "repetition_penalty" not in body


def test_extra_headers_do_not_replace_auth(monkeypatch):
    monkeypatch.setenv(
        "LLM_EXTRA_HEADERS",
        '{"X-Project": "p", "Authorization": "Bearer stolen"}',
    )
    client = _make_client()
    headers = client._chat_headers("real-token", "rq-1", Accept="text/event-stream")
    assert headers["Authorization"] == "Bearer real-token"
    assert headers["RqUID"] == "rq-1"
    assert headers["X-Project"] == "p"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"


def test_unknown_fallback_policy_is_rejected():
    with pytest.raises(ValueError, match="fallback_policy"):
        _make_client(fallback_policy="drop")
