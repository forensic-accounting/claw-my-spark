"""
Async Ollama API wrapper.

Two public functions:
    transcribe_page_image  — sends a rendered page PNG to a vision model
    generate_summary       — sends concatenated text to a reasoning model

A module-level asyncio.Semaphore(3) caps concurrent vision requests to
avoid GPU memory pressure when processing multi-page documents.
"""

import asyncio
import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Cap simultaneous vision calls to avoid OOM on the 90b model
_VISION_SEMAPHORE = asyncio.Semaphore(3)

VISION_TIMEOUT = 900.0   # seconds — 90b model on DGX Spark takes ~10–12 min per page
SUMMARY_TIMEOUT = 600.0

VISION_PROMPT = (
    "You are a forensic document analyst and OCR assistant. "
    "This image is a financial document — it may be a bank statement, "
    "cancelled check, wire transfer record, or invoice. Your task:\n\n"
    "1. Extract ALL text exactly as it appears, including account numbers, "
    "routing numbers, dates, dollar amounts, payee names, memo fields, and "
    "any signatures or handwritten annotations.\n"
    "2. Preserve the structure: use whitespace and line breaks to reflect the "
    "original layout. Reproduce tables as tab-separated rows.\n"
    "3. Do not interpret, summarize, or omit any text.\n"
    "4. If a field is partially legible, include it with a [?] marker.\n\n"
    "Return only the extracted text. No preamble or commentary."
)

SUMMARY_PROMPT = (
    "/no_think\n"
    "You are a financial forensics analyst. The following is OCR-extracted "
    "text from a financial document. Produce a structured summary including:\n\n"
    "- Document type (bank statement, check, invoice, wire transfer, etc.)\n"
    "- Issuing institution or vendor name\n"
    "- Account holder name(s) if present\n"
    "- Account or reference numbers (show only last 4 digits)\n"
    "- Date range or transaction date\n"
    "- Total amounts, balances, or invoice totals\n"
    "- Key individual transactions or line items (top 10 by amount if more exist)\n"
    "- Any anomalies, handwritten annotations, or irregularities noted\n\n"
    "Be factual and precise. Use bullet points."
)


async def transcribe_page_image(
    png_bytes: bytes,
    model: str,
    ollama_base_url: str,
    fallback_model: Optional[str] = None,
) -> str:
    """
    Render a page image through a vision model and return the extracted text.

    Uses the module-level semaphore (max 3 concurrent calls).
    Falls back to fallback_model if the primary returns an HTTP error.
    """
    b64_image = base64.b64encode(png_bytes).decode()
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": VISION_PROMPT,
                "images": [b64_image],
            }
        ],
    }

    async with _VISION_SEMAPHORE:
        # Use timeout=None on the httpx client and wrap with asyncio.wait_for so
        # the limit is a true wall-clock total, not a per-chunk read timeout.
        # (httpx's read timeout resets on each received chunk, so chunked ollama
        # responses can silently run for many minutes past the intended limit.)
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                resp = await asyncio.wait_for(
                    client.post(
                        f"{ollama_base_url.rstrip('/')}/api/chat", json=payload
                    ),
                    timeout=VISION_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except httpx.HTTPStatusError as exc:
                if fallback_model and fallback_model != model:
                    logger.warning(
                        "Primary vision model %s failed (%s), trying fallback %s",
                        model,
                        exc.response.status_code,
                        fallback_model,
                    )
                    payload["model"] = fallback_model
                    resp = await asyncio.wait_for(
                        client.post(
                            f"{ollama_base_url.rstrip('/')}/api/chat", json=payload
                        ),
                        timeout=VISION_TIMEOUT,
                    )
                    resp.raise_for_status()
                    return resp.json()["message"]["content"]
                raise


async def generate_summary(
    full_text: str,
    model: str,
    ollama_base_url: str,
) -> str:
    """Send concatenated document text to a reasoning model for a forensic summary."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": f"{SUMMARY_PROMPT}\n\n---\n\n{full_text}",
            }
        ],
    }
    async with httpx.AsyncClient(timeout=SUMMARY_TIMEOUT) as client:
        resp = await client.post(
            f"{ollama_base_url.rstrip('/')}/api/chat", json=payload
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
