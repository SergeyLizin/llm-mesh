"""Image attachments on the current user turn.

An empty list keeps the text-only body. Providers that cannot carry an
image refuse before HTTP. The request guard sees the attachments, and a
refusal does not log the bytes.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx
from pydantic import ValidationError

from llm_mesh import (
    AnthropicClient,
    GeminiClient,
    ImageAttachment,
    LLMRequest,
    LLMRequestBlocked,
    LLMValidationError,
    OpenAIClient,
)
from llm_mesh.gigachat.batch import GigaChatBatchClient
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL, GigaChatClient
from llm_mesh.hooks import configure_request_hook

OPENAI_URL = "https://host/v1/chat/completions"
ANTHROPIC_URL = "https://anthropic.test/v1/messages"
GEMINI_URL = "https://gemini.test/v1beta/models/gemini-2.5-flash:generateContent"
GIGACHAT_URL = f"{GIGACHAT_BASE_URL}/chat/completions"
PNG = b"\x89PNG-secret-image"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def _openai_ok() -> dict:
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_ok() -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _gemini_ok() -> dict:
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }


def test_attachment_requires_one_source_and_an_allowlisted_type():
    image = ImageAttachment(data=PNG)
    assert image.media_type == "image/png"
    assert image.url is None
    with pytest.raises(ValidationError):
        ImageAttachment(url="https://cdn.example/a.png", data=PNG)
    with pytest.raises(ValidationError):
        ImageAttachment()
    with pytest.raises(ValidationError):
        ImageAttachment(data=PNG, media_type="image/jpg")
    url_only = ImageAttachment(url="https://cdn.example/a.png")
    assert url_only.media_type is None


@respx.mock
@pytest.mark.asyncio
async def test_openai_data_image_and_url_on_the_current_user_turn():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    request = LLMRequest(
        system="s",
        user="look",
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ],
        images=[
            ImageAttachment(data=PNG),
            ImageAttachment(url="https://cdn.example/a.png", media_type="image/png"),
        ],
    )
    try:
        await client.generate_text(request)
    finally:
        await client.aclose()
    messages = json.loads(route.calls[0].request.content)["messages"]
    assert messages[0]["content"] == "s" or messages[0]["role"] == "system"
    history_user = next(message for message in messages if message["content"] == "earlier")
    assert isinstance(history_user["content"], str)
    assert messages[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://cdn.example/a.png"},
            },
        ],
    }


@respx.mock
@pytest.mark.asyncio
async def test_openai_empty_images_keep_a_string_user_turn():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        await client.generate_text(LLMRequest(system="s", user="hi"))
        await client.generate_text(LLMRequest(system="s", user="hi", images=[]))
    finally:
        await client.aclose()
    first = route.calls[0].request.content
    second = route.calls[1].request.content
    assert first == second
    user = json.loads(first)["messages"][-1]
    assert user == {"role": "user", "content": "hi"}


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_data_image_and_url():
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_anthropic_ok()))
    client = AnthropicClient(model="claude", api_key="k", base_url="https://anthropic.test")
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="look",
            images=[
                ImageAttachment(data=PNG, media_type="image/jpeg"),
                ImageAttachment(url="https://cdn.example/a.png"),
            ],
        ))
        await client.generate_text(LLMRequest(system="s", user="hi"))
        await client.generate_text(LLMRequest(system="s", user="hi", images=[]))
    finally:
        await client.aclose()
    content = json.loads(route.calls[0].request.content)["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "look"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": PNG_B64,
            },
        },
        {
            "type": "image",
            "source": {"type": "url", "url": "https://cdn.example/a.png"},
        },
    ]
    assert route.calls[1].request.content == route.calls[2].request.content
    plain = json.loads(route.calls[1].request.content)["messages"][-1]
    assert plain == {"role": "user", "content": "hi"}


@respx.mock
@pytest.mark.asyncio
async def test_gemini_inline_data_and_url_rejection():
    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=_gemini_ok()))
    client = GeminiClient(
        model="gemini-2.5-flash", api_key="k", base_url="https://gemini.test/v1beta",
    )
    try:
        await client.generate_text(LLMRequest(
            system="s", user="look", images=[ImageAttachment(data=PNG)],
        ))
        with pytest.raises(LLMValidationError, match="public-URL image"):
            await client.generate_text(LLMRequest(
                system="s",
                user="look",
                images=[ImageAttachment(url="https://cdn.example/a.png")],
            ))
        await client.generate_text(LLMRequest(system="s", user="hi"))
        await client.generate_text(LLMRequest(system="s", user="hi", images=[]))
    finally:
        await client.aclose()
    assert route.call_count == 3
    parts = json.loads(route.calls[0].request.content)["contents"][-1]["parts"]
    assert parts == [
        {"text": "look"},
        {"inlineData": {"mimeType": "image/png", "data": PNG_B64}},
    ]
    assert route.calls[1].request.content == route.calls[2].request.content
    plain = json.loads(route.calls[1].request.content)["contents"][-1]
    assert plain == {"role": "user", "parts": [{"text": "hi"}]}


GIGACHAT_FILES = f"{GIGACHAT_BASE_URL}/files"


def _mock_gigachat_files(file_id: str = "file-1"):
    upload = respx.post(GIGACHAT_FILES).mock(
        return_value=httpx.Response(200, json={"id": file_id, "object": "file", "purpose": "general"}),
    )
    delete = respx.post(f"{GIGACHAT_FILES}/{file_id}/delete").mock(
        return_value=httpx.Response(200, json={"id": file_id, "deleted": True}),
    )
    return upload, delete


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_uploads_image_and_attaches_it_to_the_user_turn():
    upload, delete = _mock_gigachat_files("file-1")
    chat = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat-2-Max")
    try:
        response = await client.generate_text(LLMRequest(
            system="s", user="look", mode="text",
            images=[ImageAttachment(data=PNG, media_type="image/jpeg")],
        ))
    finally:
        await client.aclose()
    assert response.text == "hi"
    assert upload.call_count == 1
    sent = upload.calls[0].request
    assert sent.headers["content-type"].startswith("multipart/form-data")
    assert sent.headers["authorization"] == "Bearer dummy"
    assert b'name="purpose"' in sent.content and b"general" in sent.content
    assert b'filename="image0.jpg"' in sent.content and PNG in sent.content
    user = json.loads(chat.calls[0].request.content)["messages"][-1]
    assert user == {"role": "user", "content": "look", "attachments": ["file-1"]}
    assert PNG_B64 not in chat.calls[0].request.content.decode()
    assert delete.call_count == 1  # uploaded file is removed after the call


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_text_only_bodies_unchanged():
    upload, _ = _mock_gigachat_files()
    route = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat")
    try:
        await client.generate_text(LLMRequest(system="s", user="hi", mode="text"))
        await client.generate_text(LLMRequest(system="s", user="hi", mode="text", images=[]))
    finally:
        await client.aclose()
    assert upload.call_count == 0
    assert route.calls[0].request.content == route.calls[1].request.content
    user = json.loads(route.calls[0].request.content)["messages"][-1]
    assert user == {"role": "user", "content": "hi"}


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_stream_with_image_attaches_and_cleans_up():
    upload, delete = _mock_gigachat_files("file-9")
    sse = (
        'data: {"choices":[{"delta":{"content":"Схема"},"index":0}]}\n\n'
        'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    chat = respx.post(GIGACHAT_URL).mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}),
    )
    client = GigaChatClient(token="dummy", model="GigaChat-2-Max")
    try:
        chunks = [c async for c in client.generate_stream(LLMRequest(
            system="s", user="look", mode="text", images=[ImageAttachment(data=PNG)],
        ))]
    finally:
        await client.aclose()
    assert "".join(c.delta_text or "" for c in chunks) == "Схема"
    user = json.loads(chat.calls[0].request.content)["messages"][-1]
    assert user["attachments"] == ["file-9"]
    assert upload.call_count == 1 and delete.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_rejects_image_urls_and_structured_images_before_http():
    upload, _ = _mock_gigachat_files()
    chat = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat")
    try:
        with pytest.raises(LLMValidationError, match="image URLs are not supported"):
            await client.generate_text(LLMRequest(
                system="s", user="look", mode="text",
                images=[ImageAttachment(url="https://img.example/a.png")],
            ))
        with pytest.raises(LLMValidationError, match="no image input"):
            await client.generate_structured(LLMRequest(
                system="s", user="look", schema={"type": "object", "properties": {}},
                images=[ImageAttachment(data=PNG)],
            ))
    finally:
        await client.aclose()
    assert upload.call_count == 0 and chat.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_upload_refreshes_token_on_401():
    auth = respx.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth").mock(
        return_value=httpx.Response(200, json={"access_token": "fresh", "expires_at": 4102444800000}),
    )
    upload = respx.post(GIGACHAT_FILES).mock(side_effect=[
        httpx.Response(401, text="expired"),
        httpx.Response(200, json={"id": "file-2"}),
    ])
    respx.post(f"{GIGACHAT_FILES}/file-2/delete").mock(return_value=httpx.Response(200, json={}))
    chat = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(credentials="Y3JlZHM=", model="GigaChat-2-Max")
    try:
        await client.generate_text(LLMRequest(
            system="s", user="look", mode="text", images=[ImageAttachment(data=PNG)],
        ))
    finally:
        await client.aclose()
    assert upload.call_count == 2
    assert upload.calls[1].request.headers["authorization"] == "Bearer fresh"
    assert json.loads(chat.calls[0].request.content)["messages"][-1]["attachments"] == ["file-2"]
    assert auth.call_count >= 1


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_failed_upload_sends_no_chat_and_removes_earlier_uploads():
    upload = respx.post(GIGACHAT_FILES).mock(side_effect=[
        httpx.Response(200, json={"id": "file-a"}),
        httpx.Response(400, text="bad image"),
    ])
    delete = respx.post(f"{GIGACHAT_FILES}/file-a/delete").mock(return_value=httpx.Response(200, json={}))
    chat = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat-2-Max")
    try:
        with pytest.raises(Exception, match="file upload 400"):
            await client.generate_text(LLMRequest(
                system="s", user="look", mode="text",
                images=[ImageAttachment(data=PNG), ImageAttachment(data=PNG)],
            ))
    finally:
        await client.aclose()
    assert upload.call_count == 2 and chat.call_count == 0 and delete.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_cleanup_failure_does_not_fail_the_call(caplog):
    respx.post(GIGACHAT_FILES).mock(return_value=httpx.Response(200, json={"id": "file-3"}))
    respx.post(f"{GIGACHAT_FILES}/file-3/delete").mock(return_value=httpx.Response(500, text="oops"))
    respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat-2-Max")
    try:
        with caplog.at_level("WARNING"):
            response = await client.generate_text(LLMRequest(
                system="s", user="look", mode="text", images=[ImageAttachment(data=PNG)],
            ))
    finally:
        await client.aclose()
    assert response.text == "hi"
    assert "file-3" in caplog.text
    assert "secret-image" not in caplog.text


@respx.mock
@pytest.mark.asyncio
async def test_guard_can_block_on_images_without_logging_bytes(caplog):
    def block_images(request: LLMRequest) -> LLMRequest | None:
        if request.images:
            return None
        return request

    configure_request_hook(check_request=block_images)
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        with caplog.at_level("WARNING"):
            with pytest.raises(LLMRequestBlocked):
                await client.generate_text(LLMRequest(
                    system="s", user="look", images=[ImageAttachment(data=PNG)],
                ))
    finally:
        await client.aclose()
    assert route.call_count == 0
    assert "REQUEST_BLOCKED" in caplog.text
    assert PNG_B64 not in caplog.text
    assert "secret-image" not in caplog.text


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_guard_sees_images_before_the_provider_limit():
    route = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))

    def strip_images(request: LLMRequest) -> LLMRequest | None:
        if request.images:
            return request.model_copy(update={"images": []})
        return request

    configure_request_hook(check_request=strip_images)
    client = GigaChatClient(token="dummy", model="GigaChat")
    try:
        await client.generate_text(LLMRequest(
            system="s", user="look", mode="text", images=[ImageAttachment(data=PNG)],
        ))
    finally:
        await client.aclose()
    assert route.call_count == 1
    user = json.loads(route.calls[0].request.content)["messages"][-1]
    assert user == {"role": "user", "content": "look"}
    assert PNG_B64 not in route.calls[0].request.content.decode()


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_guard_can_block_images():
    def block_images(request: LLMRequest) -> LLMRequest | None:
        if request.images:
            return None
        return request

    configure_request_hook(check_request=block_images)
    route = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat")
    try:
        with pytest.raises(LLMRequestBlocked):
            await client.generate_text(LLMRequest(
                system="s", user="look", images=[ImageAttachment(data=PNG)],
            ))
    finally:
        await client.aclose()
    assert route.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_batch_rejects_images_before_upload():
    route = respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1"}),
    )
    client = GigaChatBatchClient(token="dummy")
    try:
        with pytest.raises(LLMValidationError, match="no image input"):
            await client.run_chat_batch([LLMRequest(
                system="s", user="look", images=[ImageAttachment(data=PNG)],
            )])
    finally:
        await client.aclose()
    assert route.call_count == 0
