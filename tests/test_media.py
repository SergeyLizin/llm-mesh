"""Audio, video, and document attachments on the current user turn.

Empty lists keep the text-only body. A client that cannot carry a kind
refuses before HTTP.
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
    AudioAttachment,
    DocumentAttachment,
    GeminiClient,
    LLMRequest,
    LLMValidationError,
    OpenAIClient,
    VideoAttachment,
)
from llm_mesh.gigachat.batch import GigaChatBatchClient
from llm_mesh.gigachat.client import GIGACHAT_BASE_URL, GigaChatClient
from llm_mesh.openai.batch import OpenAIBatchClient

OPENAI_URL = "https://host/v1/chat/completions"
ANTHROPIC_URL = "https://anthropic.test/v1/messages"
GEMINI_URL = "https://gemini.test/v1beta/models/gemini-2.5-flash:generateContent"
GIGACHAT_URL = f"{GIGACHAT_BASE_URL}/chat/completions"
WAV = b"RIFF-secret-audio"
PDF = b"%PDF-secret"
WAV_B64 = base64.b64encode(WAV).decode("ascii")
PDF_B64 = base64.b64encode(PDF).decode("ascii")


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


def test_media_requires_one_source_and_an_allowlisted_type():
    audio = AudioAttachment(data=WAV, media_type="audio/wav")
    assert audio.media_type == "audio/wav"
    with pytest.raises(ValidationError):
        AudioAttachment(data=WAV)
    with pytest.raises(ValidationError):
        AudioAttachment(data=WAV, media_type="audio/midi")
    with pytest.raises(ValidationError):
        VideoAttachment(url="https://cdn.example/a.mp4", data=b"x", media_type="video/mp4")
    document = DocumentAttachment(url="https://cdn.example/a.pdf")
    assert document.media_type is None


@respx.mock
@pytest.mark.asyncio
async def test_openai_audio_and_pdf_parts_and_video_refusal():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="listen",
            audio=[AudioAttachment(data=WAV, media_type="audio/wav")],
            documents=[DocumentAttachment(data=PDF, media_type="application/pdf")],
        ))
        with pytest.raises(LLMValidationError, match="no video input"):
            await client.generate_text(LLMRequest(
                system="s",
                user="watch",
                video=[VideoAttachment(data=b"mp4", media_type="video/mp4")],
            ))
        with pytest.raises(LLMValidationError, match="wav or mp3"):
            await client.generate_text(LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(data=WAV, media_type="audio/flac")],
            ))
        with pytest.raises(LLMValidationError, match="audio input is base64"):
            await client.generate_text(LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(url="https://cdn.example/a.wav", media_type="audio/wav")],
            ))
        await client.generate_text(LLMRequest(system="s", user="hi"))
        await client.generate_text(LLMRequest(
            system="s", user="hi", audio=[], video=[], documents=[],
        ))
    finally:
        await client.aclose()
    content = json.loads(route.calls[0].request.content)["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "listen"},
        {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}},
        {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:application/pdf;base64,{PDF_B64}",
            },
        },
    ]
    assert route.call_count == 3
    assert route.calls[1].request.content == route.calls[2].request.content


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_pdf_and_text_and_audio_refusal():
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=_anthropic_ok()))
    client = AnthropicClient(model="claude", api_key="k", base_url="https://anthropic.test")
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="read",
            documents=[
                DocumentAttachment(data=PDF, media_type="application/pdf"),
                DocumentAttachment(data=b"hello", media_type="text/plain"),
                DocumentAttachment(url="https://cdn.example/a.pdf"),
            ],
        ))
        with pytest.raises(LLMValidationError, match="no audio input"):
            await client.generate_text(LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(data=WAV, media_type="audio/wav")],
            ))
        with pytest.raises(LLMValidationError, match="no video input"):
            await client.generate_text(LLMRequest(
                system="s",
                user="watch",
                video=[VideoAttachment(data=b"mp4", media_type="video/mp4")],
            ))
    finally:
        await client.aclose()
    content = json.loads(route.calls[0].request.content)["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "read"},
        {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
        },
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "hello"},
        },
        {
            "type": "document",
            "source": {"type": "url", "url": "https://cdn.example/a.pdf"},
        },
    ]
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_gemini_inline_audio_video_and_pdf():
    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json=_gemini_ok()))
    client = GeminiClient(
        model="gemini-2.5-flash", api_key="k", base_url="https://gemini.test/v1beta",
    )
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="look",
            audio=[AudioAttachment(data=WAV, media_type="audio/wav")],
            video=[VideoAttachment(data=b"mp4", media_type="video/mp4")],
            documents=[DocumentAttachment(data=PDF, media_type="application/pdf")],
        ))
        with pytest.raises(LLMValidationError, match="public-URL audio"):
            await client.generate_text(LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(url="https://cdn.example/a.wav", media_type="audio/wav")],
            ))
        with pytest.raises(LLMValidationError, match="unsupported audio type"):
            await client.generate_text(LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(data=WAV, media_type="audio/opus")],
            ))
    finally:
        await client.aclose()
    parts = json.loads(route.calls[0].request.content)["contents"][-1]["parts"]
    assert parts == [
        {"text": "look"},
        {"inlineData": {"mimeType": "audio/wav", "data": WAV_B64}},
        {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(b"mp4").decode("ascii")}},
        {"inlineData": {"mimeType": "application/pdf", "data": PDF_B64}},
    ]
    assert route.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_uploads_audio_and_pdf_and_rejects_video():
    uploads = []

    def handler(request: httpx.Request) -> httpx.Response:
        uploads.append(request.content)
        file_id = f"file-{len(uploads)}"
        return httpx.Response(200, json={"id": file_id})

    respx.post(f"{GIGACHAT_BASE_URL}/files").mock(side_effect=handler)
    respx.post(url__regex=rf"{GIGACHAT_BASE_URL}/files/.*/delete").mock(
        return_value=httpx.Response(200, json={"deleted": True}),
    )
    chat = respx.post(GIGACHAT_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = GigaChatClient(token="dummy", model="GigaChat-2")
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="listen",
            mode="text",
            audio=[AudioAttachment(data=WAV, media_type="audio/mpeg")],
            documents=[DocumentAttachment(data=PDF, media_type="application/pdf")],
        ))
        with pytest.raises(LLMValidationError, match="video input is not supported"):
            await client.generate_text(LLMRequest(
                system="s",
                user="watch",
                mode="text",
                video=[VideoAttachment(data=b"mp4", media_type="video/mp4")],
            ))
    finally:
        await client.aclose()
    assert len(uploads) == 2
    assert b'filename="audio0.mp3"' in uploads[0] and WAV in uploads[0]
    assert b'filename="document0.pdf"' in uploads[1] and PDF in uploads[1]
    user = json.loads(chat.calls[0].request.content)["messages"][-1]
    assert user == {
        "role": "user",
        "content": "listen",
        "attachments": ["file-1", "file-2"],
    }
    assert chat.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_gigachat_batch_rejects_audio_before_upload():
    route = respx.post(f"{GIGACHAT_BASE_URL}/batches").mock(
        return_value=httpx.Response(200, json={"id": "b1"}),
    )
    client = GigaChatBatchClient(token="dummy")
    try:
        with pytest.raises(LLMValidationError, match="batches have no media input"):
            await client.run_chat_batch([LLMRequest(
                system="s",
                user="listen",
                audio=[AudioAttachment(data=WAV, media_type="audio/wav")],
            )])
    finally:
        await client.aclose()
    assert route.call_count == 0


_GIGACHAT_WIDE_AUDIO = {
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".weba",
    "audio/ogg": ".ogg",
    "audio/x-ogg": ".ogg",
    "audio/opus": ".opus",
}
_GIGACHAT_TEXT_PATHS = ("generate_text", "generate_stream", "generate_stream_events")
_GIGACHAT_SSE = (
    'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


async def _gigachat_call(client: GigaChatClient, method: str, request: LLMRequest):
    if method == "generate_text":
        return await client.generate_text(request)
    return [item async for item in getattr(client, method)(request)]


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("method", _GIGACHAT_TEXT_PATHS)
@pytest.mark.parametrize("media_type,ext", list(_GIGACHAT_WIDE_AUDIO.items()))
async def test_gigachat_uploads_audio_openai_chat_rejects(method, media_type, ext):
    uploads = []

    def handler(request: httpx.Request) -> httpx.Response:
        uploads.append(request.content)
        return httpx.Response(200, json={"id": f"file-{len(uploads)}"})

    respx.post(f"{GIGACHAT_BASE_URL}/files").mock(side_effect=handler)
    respx.post(url__regex=rf"{GIGACHAT_BASE_URL}/files/.*/delete").mock(
        return_value=httpx.Response(200, json={"deleted": True}),
    )
    if method == "generate_text":
        chat_response = httpx.Response(200, json=_openai_ok())
    else:
        chat_response = httpx.Response(
            200, text=_GIGACHAT_SSE, headers={"content-type": "text/event-stream"},
        )
    chat = respx.post(GIGACHAT_URL).mock(return_value=chat_response)
    payload = b"wide-audio"
    client = GigaChatClient(token="dummy", model="GigaChat-2")
    try:
        await _gigachat_call(client, method, LLMRequest(
            system="s",
            user="listen",
            mode="text",
            audio=[AudioAttachment(data=payload, media_type=media_type)],
        ))
    finally:
        await client.aclose()
    assert len(uploads) == 1
    assert f'filename="audio0{ext}"'.encode() in uploads[0]
    assert payload in uploads[0]
    user = json.loads(chat.calls[0].request.content)["messages"][-1]
    assert user == {"role": "user", "content": "listen", "attachments": ["file-1"]}
    assert chat.call_count == 1


@respx.mock
@pytest.mark.asyncio
async def test_openai_plain_text_is_a_text_part_not_a_file():
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_openai_ok()))
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    note = "plain note"
    try:
        await client.generate_text(LLMRequest(
            system="s",
            user="read",
            documents=[DocumentAttachment(data=note.encode(), media_type="text/plain")],
        ))
        with pytest.raises(LLMValidationError, match="inline UTF-8"):
            await client.generate_text(LLMRequest(
                system="s",
                user="read",
                documents=[DocumentAttachment(url="https://cdn.example/a.txt", media_type="text/plain")],
            ))
        with pytest.raises(LLMValidationError, match="must be UTF-8"):
            await client.generate_text(LLMRequest(
                system="s",
                user="read",
                documents=[DocumentAttachment(data=b"\xff", media_type="text/plain")],
            ))
    finally:
        await client.aclose()
    content = json.loads(route.calls[0].request.content)["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "read"},
        {"type": "text", "text": note},
    ]
    assert route.call_count == 1


def test_openai_batch_line_inlines_plain_text_and_keeps_pdf():
    client = OpenAIClient(model="m", api_key="k", base_url="https://host/v1")
    batch = OpenAIBatchClient(client=client)
    lines = batch.build_chat_lines([LLMRequest(
        system="s",
        user="read",
        documents=[
            DocumentAttachment(data=b"plain note", media_type="text/plain"),
            DocumentAttachment(data=PDF, media_type="application/pdf"),
        ],
    )])
    content = lines[0]["body"]["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "read"},
        {"type": "text", "text": "plain note"},
        {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:application/pdf;base64,{PDF_B64}",
            },
        },
    ]
    assert "document.txt" not in json.dumps(content)
