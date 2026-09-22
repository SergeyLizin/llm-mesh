"""Fall back when a gateway rejects open-object schemas. Adding empty properties is not a valid
repair: constrained decoding may then force an empty object and silently lose arbitrary keys.
Detect the narrow schema restriction, cache it only for matching schemas, and report the
affected route and nodes.
"""

from __future__ import annotations

from llm_mesh.models_catalog import load_catalog_data

import httpx
import pytest
import respx

from llm_mesh.openai.client import (
    OpenAIClient,
    _open_object_paths,
    _request_has_open_object,
    _request_open_object_paths,
    _schema_has_open_object,
)
from llm_mesh.types import LLMRequest, LLMValidationError


BASE = "https://host/v1"
URL = "https://host/v1/chat/completions"

_REJECTION = {
    "error": {
        "message": (
            "Object fields require at least one of: 'properties' or 'anyOf' "
            "with a list of possible properties."
        ),
        "type": "invalid_request_error",
        "code": "client_error",
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LLM_BASE_URL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_API_KEY",
        "OPENROUTER_API_KEY", "LLM_DISABLE_TOOLS", "LLM_TOOL_CHOICE_PREF",
        "LLM_RESPONSE_FORMAT", "LLM_MAX_RETRIES", "LLM_NO_DEGRADE",
        "LLM_OPEN_OBJECT_SCHEMAS",
    ):
        monkeypatch.delenv(var, raising=False)


def _open_schema() -> dict:
    """Build a schema with an unrestricted object patch."""
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "patch": {"type": "object", "additionalProperties": True},
        },
        "required": ["kind"],
    }


def _closed_schema() -> dict:
    return {
        "type": "object",
        "properties": {"kind": {"type": "string"}},
        "required": ["kind"],
    }


def _request(schema: dict, **over) -> LLMRequest:
    kw = dict(
        system="You dispatch fragments",
        user="Add a tax ID field to the form",
        tools=[{"name": "emit", "description": "emit", "parameters": schema}],
        schema=schema,
        function_name="emit",
        mode="function_call",
    )
    kw.update(over)
    return LLMRequest(**kw)


def _client(monkeypatch) -> OpenAIClient:
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "k")
    return OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")


def _text_answer(payload: str) -> dict:
    return {
        "model": "cerebras/gemma-4-31b-it",
        "choices": [{"message": {"role": "assistant", "content": payload}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


# --- Schema-shape detection -------------------------------------------------


def test_detector_finds_open_object_at_any_depth():
    assert _schema_has_open_object({"type": "object", "additionalProperties": True})
    assert _schema_has_open_object(_open_schema())
    assert _schema_has_open_object(
        {"type": "array", "items": {"a": [{"type": "object"}]}},
    )


def test_detector_ignores_declared_and_composite_objects():
    """Ignore declared properties and composite schemas, which are distinct from unsupported
    unrestricted objects.
    """
    assert not _schema_has_open_object(_closed_schema())
    assert not _schema_has_open_object({"type": "object", "properties": {}})
    assert not _schema_has_open_object(
        {"type": "object", "anyOf": [{"type": "object", "properties": {}}]},
    )
    assert not _schema_has_open_object({"$ref": "#/$defs/X", "type": "object"})


def test_detector_reads_both_schema_carriers():
    """Inspect both the single schema and multi-tool parameter schemas."""
    assert _request_has_open_object(_request(_open_schema()))
    assert not _request_has_open_object(_request(_closed_schema()))
    mixed = _request(_closed_schema())
    mixed.tools = [{"name": "emit", "parameters": _open_schema()}]
    assert _request_has_open_object(mixed)


# --- Reactive fallback after schema rejection -------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_gateway_rejection_degrades_to_text_instead_of_raising():
    """Recover the recognized gateway rejection through text instead of propagating a raw
    OpenAIError.
    """
    calls = {"n": 0}

    def _route(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = req.content.decode()
        if '"tools"' in body:
            return httpx.Response(400, json=_REJECTION)
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")

    resp = await client.generate_structured(_request(_open_schema()))

    assert resp.arguments == {"kind": "forms"}
    assert client._last_served_tier == "text"
    # Cache only the open-object restriction, not a global multi-tool failure.
    assert client._open_objects_unsupported is True
    assert calls["n"] >= 2


@pytest.mark.asyncio
@respx.mock
async def test_learned_rejection_spares_closed_schemas():
    """A learned restriction must not move unrelated closed schemas to text."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
    client._open_objects_unsupported = True  # Simulate state after a recognized 400.

    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "cerebras/gemma-4-31b-it",
        "choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "emit", "arguments": '{"kind": "js_handler"}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }))

    resp = await client.generate_structured(_request(_closed_schema()))
    assert resp.arguments == {"kind": "js_handler"}
    assert client._last_served_tier == "multi"


# --- Declared route restrictions --------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_declared_route_never_pays_the_400():
    """A declared unsupported open-object schema avoids tool requests entirely."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        seen: list[str] = []

        def _route(req: httpx.Request) -> httpx.Response:
            seen.append(req.content.decode())
            return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

        respx.post(URL).mock(side_effect=_route)
        resp = await client.generate_structured(_request(_open_schema()))
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    assert resp.arguments == {"kind": "forms"}
    assert len(seen) == 1, "unnecessary round-trip: predictable 400 was still sent"
    assert '"tools"' not in seen[0]


@pytest.mark.asyncio
async def test_declared_route_refuses_native_tool_loop_honestly():
    """Reject incompatible native loops before HTTP rather than substitute a text answer for a
    model decision.
    """
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with pytest.raises(LLMValidationError, match="open object"):
            await client.generate_structured(
                _request(_open_schema(), tools_required=True),
            )
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)


def test_catalog_declares_the_field_only_where_measured():
    """Declare restrictions only on measured routes; let reactive detection handle routes whose
    behavior could not be verified.
    """
    import json
    from pathlib import Path

    catalog = load_catalog_data()
    models = catalog["models"] if isinstance(catalog, dict) else catalog
    declared = {
        m["id"] for m in models
        for r in m.get("providers", [])
        if r.get("open_object_schemas") == "unsupported"
    }
    assert declared == {
        "gemma4-31b-ron", "gemma4-31b-roff",
        "gpt-oss-120b-rlow", "gpt-oss-120b-ron",
    }
    for m in models:
        for r in m.get("providers", []):
            if r.get("open_object_schemas"):
                assert r["name"] == "llmgw-cerebras", (
                    "field declared on an unmeasured route: "
                    f"{m['id']}/{r['name']}"
                )


# --- Fallback warnings must identify the route and affected schema nodes -----


def test_paths_name_the_offending_nodes_not_just_the_fact():
    """Report the specific offending node paths so operators can fix the schema."""
    paths = _open_object_paths(_open_schema())
    assert paths == ["properties.patch"], paths
    nested = {"$defs": {"FormsParams": {"properties": {
        "full_schema": {"anyOf": [{"type": "object", "additionalProperties": True},
                                  {"type": "null"}]}}}}}
    assert _open_object_paths(nested) == [
        "$defs.FormsParams.properties.full_schema.anyOf[0]",
    ]
    assert _open_object_paths({"type": "object", "additionalProperties": True}) == [
        "<schema root>",
    ]


def test_request_paths_cover_both_carriers():
    req = _request(_closed_schema())
    req.tools = [{"name": "emit", "parameters": _open_schema()}]
    assert _request_open_object_paths(req) == ["tools[0].parameters.properties.patch"]


@pytest.mark.asyncio
@respx.mock
async def test_reactive_degrade_is_loud(caplog):
    """Verify both successful fallback and a warning identifying its route and node."""
    def _route(req: httpx.Request) -> httpx.Response:
        if '"tools"' in req.content.decode():
            return httpx.Response(400, json=_REJECTION)
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")

    with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
        resp = await client.generate_structured(_request(_open_schema()))

    assert resp.arguments == {"kind": "forms"}          # Fallback occurred.
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warns, "fallback occurred silently — invalid schema warning was lost"
    msg = warns[-1].getMessage()
    assert "llmgw-cerebras" in msg                      # The route is identified.
    assert "cerebras/gemma-4-31b-it" in msg
    assert "properties.patch" in msg                    # The node is identified.
    assert "models.json" in msg                         # Identify where to persist the restriction.


@pytest.mark.asyncio
@respx.mock
async def test_declared_bypass_is_loud_too(caplog):
    """A declared bypass avoids a request but does not make the schema compatible; keep the warning
    visible.
    """
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        respx.post(URL).mock(
            return_value=httpx.Response(200, json=_text_answer('{"kind": "forms"}')),
        )
        with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
            resp = await client.generate_structured(_request(_open_schema()))
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    assert resp.arguments == {"kind": "forms"}
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warns, "declared bypass occurred silently"
    assert any("properties.patch" in m and "llmgw-cerebras" in m for m in warns), warns


@pytest.mark.asyncio
async def test_native_loop_refusal_names_route_and_node():
    """Native-loop refusal must name the route and node, not merely report unavailable native
    support.
    """
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with pytest.raises(LLMValidationError) as err:
            await client.generate_structured(
                _request(_open_schema(), tools_required=True),
            )
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)
    text = str(err.value)
    assert "properties.patch" in text and "cerebras/gemma-4-31b-it" in text, text


@pytest.mark.asyncio
@respx.mock
async def test_quiet_when_schema_is_clean(caplog):
    """Do not emit open-object warnings for compatible closed schemas."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
    respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "cerebras/gemma-4-31b-it",
        "choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "emit", "arguments": '{"kind": "js_handler"}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }))
    with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
        await client.generate_structured(_request(_closed_schema()))
    assert not [r for r in caplog.records
                if "open object" in r.getMessage()]


# --- No-degrade results must not depend on catalog declarations --------------


@pytest.mark.asyncio
@respx.mock
async def test_no_degrade_declared_and_reactive_agree():
    """Declared and reactively learned restrictions must both reject implicit text fallback under
    no-degrade.
    """
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_NO_DEGRADE"] = "1"

    def _route(req: httpx.Request) -> httpx.Response:
        if '"tools"' in req.content.decode():
            return httpx.Response(400, json=_REJECTION)
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    try:
        # Reactive path: discover the restriction from a gateway 400.
        reactive = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with pytest.raises(LLMValidationError) as reactive_err:
            await reactive.generate_structured(_request(_open_schema()))

        # Declared path: the route already reports the same restriction.
        os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
        declared = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with pytest.raises(LLMValidationError) as declared_err:
            await declared.generate_structured(_request(_open_schema()))
    finally:
        os.environ.pop("LLM_NO_DEGRADE", None)
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    assert "LLM_NO_DEGRADE" in str(reactive_err.value)
    assert "LLM_NO_DEGRADE" in str(declared_err.value), (
        "declared route bypassed strict measurement mode — verdict "
        "depends on models.json, rather than the gateway"
    )


@pytest.mark.asyncio
@respx.mock
async def test_without_no_degrade_declared_bypass_still_serves_text():
    """Outside no-degrade, the declared bypass still serves text normally."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ.pop("LLM_NO_DEGRADE", None)
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        respx.post(URL).mock(
            return_value=httpx.Response(200, json=_text_answer('{"kind": "forms"}')),
        )
        resp = await client.generate_structured(_request(_open_schema()))
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)
    assert resp.arguments == {"kind": "forms"}
    assert client._last_served_tier == "text"


# --- Reactive restriction handling in native loops --------------------------


@pytest.mark.asyncio
@respx.mock
async def test_native_loop_400_is_caught_learned_and_named():
    """Catch, name, and cache native-loop schema rejection instead of repeating raw 400 failures on
    every turn.
    """
    import os
    from llm_mesh.openai.client import OpenAIError

    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)
    calls = {"n": 0}

    def _route(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json=_REJECTION)

    respx.post(URL).mock(side_effect=_route)
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")

    with pytest.raises(LLMValidationError) as err:
        await client.generate_structured(_request(_open_schema(), tools_required=True))

    # Distinguish an unsupported schema from an unrelated transport 400.
    assert not isinstance(err.value, OpenAIError)
    assert "properties.patch" in str(err.value)
    assert "cerebras/gemma-4-31b-it" in str(err.value)
    # The learned restriction avoids another failing HTTP request.
    assert client._open_objects_unsupported is True
    before = calls["n"]
    with pytest.raises(LLMValidationError):
        await client.generate_structured(_request(_open_schema(), tools_required=True))
    assert calls["n"] == before, "second turn of the native-loop incurred another 400"


@pytest.mark.asyncio
@respx.mock
async def test_native_loop_400_does_not_become_a_text_answer():
    """A native-loop schema rejection must not become a synthetic text answer."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    def _route(req: httpx.Request) -> httpx.Response:
        if '"tools"' in req.content.decode():
            return httpx.Response(400, json=_REJECTION)
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
    with pytest.raises(LLMValidationError):
        await client.generate_structured(_request(_open_schema(), tools_required=True))
    assert client._last_served_tier != "text"


@pytest.mark.asyncio
@respx.mock
async def test_native_loop_passes_other_400s_through_unchanged():
    """Preserve unrelated native-loop 400 errors as OpenAIError."""
    import os
    from llm_mesh.openai.client import OpenAIError

    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)
    respx.post(URL).mock(return_value=httpx.Response(400, json={
        "error": {"message": "context length exceeded", "type": "invalid_request_error"},
    }))
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
    with pytest.raises(OpenAIError):
        await client.generate_structured(_request(_open_schema(), tools_required=True))
    assert client._open_objects_unsupported is False


# --- Explicit text mode takes precedence over implicit bypass ---------------


@pytest.mark.asyncio
@respx.mock
async def test_ordered_text_mode_wins_over_the_bypass():
    """Serve explicitly requested text even with no-degrade and a declared open-object restriction."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    os.environ["LLM_DISABLE_TOOLS"] = "true"
    os.environ["LLM_NO_DEGRADE"] = "1"
    seen: list[str] = []

    def _route(req: httpx.Request) -> httpx.Response:
        seen.append(req.content.decode())
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        resp = await client.generate_structured(_request(_open_schema()))
    finally:
        for var in ("LLM_OPEN_OBJECT_SCHEMAS", "LLM_DISABLE_TOOLS", "LLM_NO_DEGRADE"):
            os.environ.pop(var, None)

    assert resp.arguments == {"kind": "forms"}
    assert client._last_served_tier == "text"
    assert len(seen) == 1 and '"tools"' not in seen[0]


@pytest.mark.asyncio
@respx.mock
async def test_no_degrade_still_refuses_when_text_mode_was_not_ordered():
    """Without explicitly disabled tools, no-degrade must still reject the implicit text bypass."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    os.environ["LLM_NO_DEGRADE"] = "1"
    os.environ.pop("LLM_DISABLE_TOOLS", None)
    respx.post(URL).mock(
        return_value=httpx.Response(200, json=_text_answer('{"kind": "forms"}')),
    )
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with pytest.raises(LLMValidationError, match="LLM_NO_DEGRADE"):
            await client.generate_structured(_request(_open_schema()))
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)
        os.environ.pop("LLM_NO_DEGRADE", None)


# --- Diagnostics must identify the actual source of a restriction -----------


@pytest.mark.asyncio
@respx.mock
async def test_learned_flag_does_not_claim_a_models_json_entry(caplog):
    """A reactively learned restriction must not claim that a catalog declaration exists."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    def _route(req: httpx.Request) -> httpx.Response:
        if '"tools"' in req.content.decode():
            return httpx.Response(400, json=_REJECTION)
        return httpx.Response(200, json=_text_answer('{"kind": "forms"}'))

    respx.post(URL).mock(side_effect=_route)
    client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
    await client.generate_structured(_request(_open_schema()))   # Learn from the initial 400.
    assert client._open_objects_unsupported and not client._open_objects_declared

    with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
        await client.generate_structured(_request(_open_schema()))  # Make the second call.
    msg = [r.getMessage() for r in caplog.records if "open object" in r.getMessage()][-1]
    assert "learned reactively" in msg, msg
    assert "not declared" in msg, msg
    assert "declared open_object_schemas" not in msg

    # The same client's native-loop refusal must report the same actual source.
    with pytest.raises(LLMValidationError) as err:
        await client.generate_structured(_request(_open_schema(), tools_required=True))
    assert "learned reactively" in str(err.value), str(err.value)


@pytest.mark.asyncio
@respx.mock
async def test_declared_flag_still_points_at_models_json(caplog):
    """When a declaration exists, continue pointing diagnostics to models.json."""
    import os
    os.environ["LLM_BASE_URL"] = BASE
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_OPEN_OBJECT_SCHEMAS"] = "unsupported"
    respx.post(URL).mock(
        return_value=httpx.Response(200, json=_text_answer('{"kind": "forms"}')),
    )
    try:
        client = OpenAIClient(model="cerebras/gemma-4-31b-it", label="llmgw-cerebras")
        with caplog.at_level("WARNING", logger="llm_mesh.openai.client"):
            await client.generate_structured(_request(_open_schema()))
        with pytest.raises(LLMValidationError) as err:
            await client.generate_structured(
                _request(_open_schema(), tools_required=True),
            )
    finally:
        os.environ.pop("LLM_OPEN_OBJECT_SCHEMAS", None)

    msg = [r.getMessage() for r in caplog.records if "open object" in r.getMessage()][-1]
    assert "declared open_object_schemas: unsupported in models.json" in msg
    assert "learned reactively" not in msg
    assert "models.json" in str(err.value)


