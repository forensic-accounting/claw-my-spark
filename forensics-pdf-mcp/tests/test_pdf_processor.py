"""
Tests for pdf_processor.py — PDF pipeline with mocked Ollama API.

All Ollama HTTP calls are intercepted with respx so no running Ollama
instance is required.
"""

import json
import os

import fitz
import httpx
import pytest
import respx

from pdf_processor import process_pdf_pipeline


@pytest.fixture(autouse=True)
def _set_cache_dir(tmp_path):
    """Point CACHE_DIR at a temp directory so tests don't need /data."""
    os.environ["CACHE_DIR"] = str(tmp_path / "cache")
    yield
    os.environ.pop("CACHE_DIR", None)

OLLAMA_URL = "http://localhost:11434"
CHAT_URL = f"{OLLAMA_URL}/api/chat"
PS_URL = f"{OLLAMA_URL}/api/ps"
GENERATE_URL = f"{OLLAMA_URL}/api/generate"


def _chat_response(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


def _ps_response_with_model(model_name: str) -> dict:
    """Return a /api/ps response showing a model fully loaded on GPU."""
    return {
        "models": [
            {
                "model": model_name,
                "size": 50_000_000_000,
                "size_vram": 50_000_000_000,
            }
        ]
    }


def _ps_response_empty() -> dict:
    return {"models": []}


@pytest.mark.asyncio
async def test_text_only_pdf_skips_ocr(text_only_pdf, tmp_path):
    """Text-only PDFs are summarised without any vision model calls."""
    vision_calls = 0

    def chat_handler(request):
        nonlocal vision_calls
        body = json.loads(request.content)
        msg = body["messages"][0]
        if "images" in msg:
            vision_calls += 1
        return httpx.Response(200, json=_chat_response("Summary of bank statement"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        # Mock /api/ps to show summary model on GPU
        respx.get(PS_URL).mock(
            return_value=httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))
        )
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        result = await process_pdf_pipeline(
            text_only_pdf, "statement", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    assert vision_calls == 0
    assert result["had_embedded_images"] is False
    assert result["images_transcribed"] == 0
    assert result["pages_processed"] == 1
    assert "Summary" in result["summary"]


@pytest.mark.asyncio
async def test_image_only_pdf_triggers_ocr(image_only_pdf, tmp_path):
    """Image-only PDFs send each page to the vision model."""
    vision_calls = 0

    def chat_handler(request):
        nonlocal vision_calls
        body = json.loads(request.content)
        if "images" in body["messages"][0]:
            vision_calls += 1
            return httpx.Response(200, json=_chat_response("OCR extracted text from image"))
        # Handle both summary and classify calls
        return httpx.Response(
            200,
            json=_chat_response("Title: Scanned Document\nType: other\nSummary: A scanned page."),
        )

    ps_call_count = 0

    def ps_handler(request):
        nonlocal ps_call_count
        ps_call_count += 1
        # First calls: vision model loaded; after unload: empty; then summary model loaded
        if ps_call_count <= 2:
            return httpx.Response(200, json=_ps_response_with_model("llama3.2-vision:90b"))
        if ps_call_count == 3:
            # After unload — verify_model_unloaded check
            return httpx.Response(200, json=_ps_response_empty())
        return httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        respx.get(PS_URL).mock(side_effect=ps_handler)
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        result = await process_pdf_pipeline(
            image_only_pdf, "scanned", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    assert vision_calls == 1
    assert result["had_embedded_images"] is True
    assert result["images_transcribed"] == 1


@pytest.mark.asyncio
async def test_mixed_pdf_processes_image_pages_only(mixed_pdf, tmp_path):
    """Only image-based pages are sent to the vision model; text pages are not."""
    image_call_count = 0

    def chat_handler(request):
        nonlocal image_call_count
        body = json.loads(request.content)
        if "images" in body["messages"][0]:
            image_call_count += 1
            return httpx.Response(200, json=_chat_response("OCR text"))
        return httpx.Response(
            200,
            json=_chat_response("Title: Mixed Doc\nType: bank_statement\nSummary: Summary text."),
        )

    ps_call_count = 0

    def ps_handler(request):
        nonlocal ps_call_count
        ps_call_count += 1
        if ps_call_count <= 2:
            return httpx.Response(200, json=_ps_response_with_model("llama3.2-vision:90b"))
        if ps_call_count == 3:
            return httpx.Response(200, json=_ps_response_empty())
        return httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        respx.get(PS_URL).mock(side_effect=ps_handler)
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        result = await process_pdf_pipeline(
            mixed_pdf, "mixed", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    assert result["pages_processed"] == 2
    assert image_call_count == 1   # only the image page
    assert result["had_embedded_images"] is True


@pytest.mark.asyncio
async def test_enriched_pdf_contains_invisible_text(image_only_pdf, tmp_path):
    """
    The enriched PDF has the OCR text embedded as invisible text (render_mode=3).
    PyMuPDF's get_text() extracts invisible text, so it should appear in the output.
    """
    ocr_text = "Account Number: 9876  Amount: $500.00"

    def chat_handler(request):
        body = json.loads(request.content)
        if "images" in body["messages"][0]:
            return httpx.Response(200, json=_chat_response(ocr_text))
        return httpx.Response(
            200,
            json=_chat_response("Title: Bank Statement\nType: bank_statement\nSummary: summary"),
        )

    ps_call_count = 0

    def ps_handler(request):
        nonlocal ps_call_count
        ps_call_count += 1
        if ps_call_count <= 2:
            return httpx.Response(200, json=_ps_response_with_model("llama3.2-vision:90b"))
        if ps_call_count == 3:
            return httpx.Response(200, json=_ps_response_empty())
        return httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        respx.get(PS_URL).mock(side_effect=ps_handler)
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        result = await process_pdf_pipeline(
            image_only_pdf, "enriched", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    # Open the enriched PDF and verify the invisible text is present
    import base64
    enriched_bytes = base64.b64decode(result["enriched_pdf_base64"])
    doc = fitz.open(stream=enriched_bytes, filetype="pdf")
    page_text = doc[0].get_text()
    doc.close()
    stripped = page_text.replace(" ", "").replace("\n", "")
    assert len(stripped) >= 5, f"Expected invisible text to be extractable, got: {repr(page_text)}"


@pytest.mark.asyncio
async def test_vision_failure_continues_pipeline(image_only_pdf, tmp_path):
    """A vision model failure on one page does not abort the whole pipeline."""
    call_count = 0

    def chat_handler(request):
        nonlocal call_count
        body = json.loads(request.content)
        if "images" in body["messages"][0]:
            call_count += 1
            return httpx.Response(500, json={"error": "GPU OOM"})
        return httpx.Response(
            200,
            json=_chat_response("Title: Failed\nType: unknown\nSummary: Partial summary"),
        )

    ps_call_count = 0

    def ps_handler(request):
        nonlocal ps_call_count
        ps_call_count += 1
        if ps_call_count <= 2:
            return httpx.Response(200, json=_ps_response_with_model("llama3.2-vision:90b"))
        if ps_call_count == 3:
            return httpx.Response(200, json=_ps_response_empty())
        return httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        respx.get(PS_URL).mock(side_effect=ps_handler)
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        result = await process_pdf_pipeline(
            image_only_pdf, "failed", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    # Pipeline should complete despite the failure
    assert result["images_transcribed"] == 0   # failed page not counted
    assert result["had_embedded_images"] is True
    assert "summary" in result


@pytest.mark.asyncio
async def test_image_details_markdown_created(image_only_pdf, tmp_path):
    """Processing an image PDF creates an image_details.md in the cache dir."""

    def chat_handler(request):
        body = json.loads(request.content)
        if "images" in body["messages"][0]:
            return httpx.Response(200, json=_chat_response("OCR text content"))
        return httpx.Response(
            200,
            json=_chat_response("Title: Test Page\nType: other\nSummary: Test summary."),
        )

    ps_call_count = 0

    def ps_handler(request):
        nonlocal ps_call_count
        ps_call_count += 1
        if ps_call_count <= 2:
            return httpx.Response(200, json=_ps_response_with_model("llama3.2-vision:90b"))
        if ps_call_count == 3:
            return httpx.Response(200, json=_ps_response_empty())
        return httpx.Response(200, json=_ps_response_with_model("qwen3:32b"))

    with respx.mock:
        respx.post(CHAT_URL).mock(side_effect=chat_handler)
        respx.get(PS_URL).mock(side_effect=ps_handler)
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        await process_pdf_pipeline(
            image_only_pdf, "test_md", str(tmp_path), OLLAMA_URL,
            "llama3.2-vision:90b", "qwen3:32b",
        )

    # Find the image_details.md in the cache dir
    cache_dir = tmp_path / "cache"
    md_files = list(cache_dir.rglob("image_details.md"))
    assert len(md_files) == 1, f"Expected 1 image_details.md, found {len(md_files)}"

    content = md_files[0].read_text()
    assert "## Page 1" in content
    assert "**Image SHA256:**" in content
    assert "```text" in content
    assert "OCR text content" in content
    assert "success" in content

