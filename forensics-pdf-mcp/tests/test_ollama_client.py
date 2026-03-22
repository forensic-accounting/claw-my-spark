"""Tests for ollama_client.py — Ollama API wrapper with mocked HTTP."""

import asyncio
import base64
import json
import time

import httpx
import pytest
import respx

from ollama_client import _VISION_SEMAPHORE, generate_summary, transcribe_page_image

OLLAMA_URL = "http://localhost:11434"
CHAT_URL = f"{OLLAMA_URL}/api/chat"


def _chat_response(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


@pytest.mark.asyncio
async def test_transcribe_success():
    """Successful transcription returns the model's message content."""
    with respx.mock:
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json=_chat_response("Invoice text here"))
        )
        result = await transcribe_page_image(b"png_bytes", "llama3.2-vision:90b", OLLAMA_URL)
    assert result == "Invoice text here"


@pytest.mark.asyncio
async def test_transcribe_falls_back_on_http_error():
    """When primary model returns 4xx, the fallback model is tried."""
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        if body["model"] == "llama3.2-vision:90b":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json=_chat_response("fallback result"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=side_effect)
        result = await transcribe_page_image(
            b"png_bytes",
            "llama3.2-vision:90b",
            OLLAMA_URL,
            fallback_model="llama3.2-vision:11b",
        )
    assert result == "fallback result"
    assert call_count == 2


@pytest.mark.asyncio
async def test_transcribe_raises_when_no_fallback():
    """Without a fallback, an HTTP error is propagated to the caller."""
    with respx.mock:
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(503, json={"error": "unavailable"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await transcribe_page_image(b"png", "llama3.2-vision:90b", OLLAMA_URL)


@pytest.mark.asyncio
async def test_generate_summary():
    """generate_summary returns the model's message content."""
    with respx.mock:
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json=_chat_response("• Document type: bank statement"))
        )
        result = await generate_summary("page text", "qwen3:32b", OLLAMA_URL)
    assert "bank statement" in result


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """No more than 3 vision requests run concurrently."""
    max_concurrent = 0
    current = 0
    lock = asyncio.Lock()

    async def counting_handler(request):
        nonlocal max_concurrent, current
        async with lock:
            current += 1
            if current > max_concurrent:
                max_concurrent = current
        await asyncio.sleep(0.05)  # hold the slot briefly
        async with lock:
            current -= 1
        return httpx.Response(200, json=_chat_response("text"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=counting_handler)
        tasks = [
            transcribe_page_image(b"png", "llama3.2-vision:90b", OLLAMA_URL)
            for _ in range(6)
        ]
        await asyncio.gather(*tasks)

    assert max_concurrent <= 3
