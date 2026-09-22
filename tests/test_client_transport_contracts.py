"""Observe retry delays, usage, and truncation budgets through HTTP responses."""
import json
from contextlib import aclosing
import logging

import httpx
import pytest
import respx

from llm_mesh import GigaChatAsyncClient, LLMRequest, OpenAIClient

BASE = 'https://example.invalid/v1'
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
USAGE = {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14,
         'completion_tokens_details': {'reasoning_tokens': 3},
         'prompt_tokens_details': {'cached_tokens': 6}}


def client_for(provider):
    if provider == 'openai':
        return OpenAIClient(model='m', base_url=BASE, api_key='dummy')
    return GigaChatAsyncClient(model='m', api_url=BASE, token='dummy')


def payload():
    return {'choices': [{'message': {
        'content': '{"answer":"ok"}',
        'function_call': {'name': 'f', 'arguments': {'answer': 'ok'}},
        'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
            'name': 'f', 'arguments': '{"answer":"ok"}'}}],
    }, 'finish_reason': 'stop'}], 'usage': USAGE}


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['openai', 'gigachat'])
@pytest.mark.parametrize('failure', ['network', '500', '429'])
async def test_transient_retries_sleep_with_jitter(monkeypatch, provider, failure):
    monkeypatch.setenv('LLM_OPTIONS', '{"max_retries":2,"retry_backoff_s":1}')
    # Fix randomness, not the backoff helper, so a plain exponential sleep fails.
    monkeypatch.setattr('llm_mesh._retry.random.uniform', lambda low, high: high)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr('asyncio.sleep', sleep)
    attempts = 0

    def reply(request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            if failure == 'network':
                raise httpx.ConnectError('temporary failure', request=request)
            return httpx.Response(int(failure), text='temporarily unavailable')
        return httpx.Response(200, json=payload())

    with respx.mock as router:
        router.post(BASE + '/chat/completions').mock(side_effect=reply)
        async with aclosing(client_for(provider)) as client:
            result = await client.generate_text(LLMRequest(system='s', user='u'))
    assert result.text == '{"answer":"ok"}'
    assert attempts == 3
    assert delays == [1.25, 2.5]


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,mode', [
    ('openai', 'text'), ('openai', 'structured'), ('openai', 'multi'),
    ('openai', 'response_format'), ('gigachat', 'text'),
    ('gigachat', 'structured'), ('gigachat', 'json_schema'),
])
async def test_response_usage_and_truncation_match_wire(monkeypatch, caplog, provider, mode):
    monkeypatch.setenv('LLM_OPTIONS', json.dumps({
        'max_output_tokens': 64, 'length_retries': 0,
        'response_format': 'json_object' if mode == 'response_format' else '',
    }))
    monkeypatch.setenv('LLM_NO_DEGRADE', 'false')
    bodies = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        if mode == 'response_format' and body.get('tools'):
            return httpx.Response(404, json={'error': {'message': "No endpoints found that support the provided 'tool_choice'"}})
        data = payload()
        data['choices'][0]['finish_reason'] = 'length'
        return httpx.Response(200, json=data)

    request = LLMRequest(system='s', user='u', max_tokens=128, function_name='f', schema=SCHEMA,
                         mode='json_schema' if mode == 'json_schema' else 'function_call',
                         tools=[{'name': 'f', 'parameters': SCHEMA}] if mode == 'multi' else None)
    with respx.mock as router, caplog.at_level(logging.WARNING):
        router.post(BASE + '/chat/completions').mock(side_effect=reply)
        async with aclosing(client_for(provider)) as client:
            result = await (client.generate_text(request) if mode == 'text'
                            else client.generate_structured(request))
    assert all(body['max_tokens'] == 64 for body in bodies)
    if mode == 'response_format':
        assert bodies[-1]['response_format']['type'] == 'json_object'
    warnings = [r.getMessage() for r in caplog.records if 'finish_reason=length' in r.getMessage()]
    assert len(warnings) == 1
    assert 'max_tokens=64' in warnings[0]
    assert 'max_tokens=128' not in warnings[0]
    assert result.finish_reason == 'length'
    assert (result.usage.prompt_tokens, result.usage.completion_tokens,
            result.usage.total_tokens, result.usage.reasoning_tokens,
            result.usage.cache_hit_tokens, result.usage.cache_miss_tokens) == (10, 4, 14, 3, 6, 4)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['openai', 'gigachat'])
async def test_stream_usage_normalizes_reasoning_and_cache(monkeypatch, provider):
    monkeypatch.setenv('LLM_OPTIONS', '{}')
    data = {'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': USAGE}
    with respx.mock as router:
        router.post(BASE + '/chat/completions').mock(return_value=httpx.Response(
            200, text='data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n'))
        async with aclosing(client_for(provider)) as client:
            chunks = [chunk async for chunk in client.generate_stream(LLMRequest(system='s', user='u'))]
    usage = chunks[-1].usage
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
            usage.reasoning_tokens, usage.cache_hit_tokens, usage.cache_miss_tokens) == (10, 4, 14, 3, 6, 4)
