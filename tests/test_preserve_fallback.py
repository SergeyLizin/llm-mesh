"""Preservation exposes provider failures without trying another generation tier."""
import json
from contextlib import aclosing

import httpx
import pytest
import respx

from llm_mesh import LLMRequest, OpenAIClient, LLMValidationError
from llm_mesh.openai import OpenAIError

URL = 'https://example.invalid/v1/chat/completions'
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['ok']}},
          'required': ['answer']}


@pytest.mark.asyncio
@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('failure', ['corrupt', 'open_object', 'gateway'])
async def test_preserve_does_not_retry_in_another_tier(monkeypatch, multiple, failure):
    monkeypatch.setenv('LLM_OPTIONS', '{"max_retries":0,"open_object_schemas":"unsupported"}')
    if failure == 'corrupt':
        reply = httpx.Response(200, json={'choices': [{'message': {'tool_calls': [{
            'id': 'c', 'function': {'name': 'f', 'arguments': 'not json'}}]}, 'finish_reason': 'stop'}]})
    else:
        reply = httpx.Response(400 if failure == 'open_object' else 502,
                              text='additionalProperties is not supported for open object schemas')
    with respx.mock as router:
        route = router.post(URL).mock(return_value=reply)
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k',
                                         fallback_policy='preserve', tool_choice_pref='auto')) as client:
            request = LLMRequest(system='s', user='u', schema={'type': 'object'}, function_name='f',
                                 tools=[{'name': 'f', 'parameters': {'type': 'object'}}] if multiple else None)
            with pytest.raises(LLMValidationError if failure == 'corrupt' else OpenAIError):
                await client.generate_structured(request)
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)['tools']


@pytest.mark.asyncio
@pytest.mark.parametrize('validate', [False, True])
async def test_preserve_schema_violation_never_regenerates(monkeypatch, validate):
    monkeypatch.setenv('LLM_OPTIONS', '{}')
    with respx.mock as router:
        route = router.post(URL).mock(return_value=httpx.Response(200, json={
            'choices': [{'message': {'tool_calls': [{'function': {
                'name': 'f', 'arguments': '{"answer":"invalid"}'}}]}, 'finish_reason': 'stop'}]}))
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k',
                                         fallback_policy='preserve', validate_schema=validate)) as client:
            request = LLMRequest(system='s', user='u', schema=SCHEMA, function_name='f')
            if validate:
                with pytest.raises(LLMValidationError):
                    await client.generate_structured(request)
            else:
                assert (await client.generate_structured(request)).arguments == {'answer': 'invalid'}
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_preserve_allows_native_choice_negotiation(monkeypatch):
    monkeypatch.setenv('LLM_OPTIONS', '{}')
    with respx.mock as router:
        route = router.post(URL).mock(side_effect=[
            httpx.Response(404, text="No endpoints found that support the provided 'tool_choice'"),
            httpx.Response(200, json={'choices': [{'message': {'content': '{"answer":"ok"}'}}]}),
        ])
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k',
                                         fallback_policy='preserve')) as client:
            result = await client.generate_structured(LLMRequest(system='s', user='u', schema=SCHEMA))
    assert result.arguments == {'answer': 'ok'}
    assert route.call_count == 2
    assert json.loads(route.calls[1].request.content)['tool_choice'] == 'required'


@pytest.mark.asyncio
@pytest.mark.parametrize('multiple', [False, True])
async def test_preserve_does_not_replace_text_with_another_call(monkeypatch, multiple):
    monkeypatch.setenv('LLM_OPTIONS', '{}')
    with respx.mock as router:
        route = router.post(URL).mock(return_value=httpx.Response(200, json={
            'choices': [{'message': {'content': 'original answer'}, 'finish_reason': 'stop'}]}))
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k',
                                         fallback_policy='preserve')) as client:
            req = LLMRequest(system='s', user='u', schema=SCHEMA,
                             tools=[{'name': 'f', 'parameters': SCHEMA}] if multiple else None)
            if multiple:
                result = await client.generate_structured(req)
                assert result.text == 'original answer'
                assert result.tool_calls == []
            else:
                with pytest.raises(LLMValidationError):
                    await client.generate_structured(req)
    assert route.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"disable_tools": True}, {"tool_choice_pref": "text"}])
async def test_preserve_allows_explicit_text(monkeypatch, options):
    monkeypatch.setenv("LLM_OPTIONS", json.dumps(options))
    with respx.mock as router:
        route = router.post(URL).mock(return_value=httpx.Response(200, json={
            'choices': [{'message': {'content': '{"answer":"ok"}'}, 'finish_reason': 'stop'}]}))
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k',
                                         fallback_policy='preserve')) as client:
            result = await client.generate_structured(LLMRequest(system='s', user='u', schema=SCHEMA))
    assert result.arguments == {'answer': 'ok'}
    assert route.call_count == 1
    assert 'tools' not in json.loads(route.calls[0].request.content)
