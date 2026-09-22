"""Reasoning request and response fields survive every generation mode."""
import json
from contextlib import aclosing

import httpx
import pytest
import respx

from llm_mesh import LLMRequest, OpenAIClient


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['text', 'structured', 'emulated'])
@pytest.mark.parametrize('effort', [None, 'high'])
async def test_reasoning_roundtrip(monkeypatch, mode, effort):
    monkeypatch.setenv('LLM_OPTIONS', json.dumps({'disable_tools': mode == 'emulated'}))
    monkeypatch.setenv('LLM_NO_DEGRADE', 'false')
    body = {}

    def reply(request):
        body.update(json.loads(request.content))
        message = {'content': '{"answer":"ok"}'}
        if effort:
            message['reasoning_content'] = 'thought'
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': 'stop'}]})

    with respx.mock as router:
        route = router.post('https://example.invalid/v1/chat/completions').mock(side_effect=reply)
        async with aclosing(OpenAIClient(model='m', base_url='https://example.invalid/v1', api_key='k')) as client:
            request = LLMRequest(system='s', user='u', reasoning_effort=effort,
                                 schema={'type':'object','properties':{'answer':{'type':'string'}}})
            result = await (client.generate_text(request) if mode == 'text'
                            else client.generate_structured(request))
    assert route.call_count == 1
    if effort:
        assert body['reasoning_effort'] == effort
        assert result.reasoning_content == 'thought'
    else:
        assert 'reasoning_effort' not in body
        assert result.reasoning_content is None
    assert bool(body.get('tools')) == (mode == 'structured')
