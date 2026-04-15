"""
PDF forensics processing pipeline — two-phase model execution.

Phase 1 (vision model):
  1. Page classification — detect image-based vs text pages
  2. Page rendering     — render image pages to PNG at 2x scale
  3. OCR transcription  — send PNGs sequentially to vision model
  4. Text embedding     — write invisible text (PDF TR3) onto each page
  Results are written incrementally to image_details.md for resumability.
  If vision and summary models differ, vision model is unloaded after this phase.

Phase 2 (summary model):
  5. Summary generation — structured forensic summary from all text
  6. Image enrichment   — title, type, and per-image summary
  Summary model is unloaded after this phase.

When vision_model == summary_model (e.g. gemma4:31b), the model stays
loaded across both phases — no unload/reload cycle.

Entry point: process_pdf_pipeline()
"""

import asyncio
import base64
import logging
import os
import pathlib
from typing import Optional

import fitz  # PyMuPDF

from image_details import ImageDetailsDoc, ImagePageDetail, sha256_of_bytes
from ollama_client import (
    assert_model_on_gpu,
    generate_summary,
    transcribe_page_image,
    unload_model,
    verify_model_unloaded,
)
from pdf_cache import sha256_of

logger = logging.getLogger(__name__)

# A page is considered "image-based" if it has embedded images AND fewer than
# this many characters of native text.
IMAGE_PAGE_TEXT_THRESHOLD = 50

# Prompt sent to qwen3:32b to classify each image page
_IMAGE_CLASSIFY_PROMPT = (
    "/no_think\n"
    "You are a forensic document analyst. Given the following OCR-extracted text "
    "from a single page of a financial document, provide:\n\n"
    "1. **Title**: A short descriptive title for this page (e.g., "
    "\"Chase Bank Statement - Page 1\", \"Cancelled Check #1042\").\n"
    "2. **Type**: The document type — one of: bank_statement, check, invoice, "
    "wire_transfer, tax_form, receipt, contract, letter, table, other.\n"
    "3. **Summary**: A 2-3 sentence forensic summary of the page content.\n\n"
    "Respond in exactly this format (no extra text):\n"
    "Title: <title>\n"
    "Type: <type>\n"
    "Summary: <summary>\n\n"
    "---\n\n"
)


async def process_pdf_pipeline(
    pdf_bytes: bytes,
    output_stem: str,
    workspace_dir: str,
    ollama_base_url: str,
    vision_model: str,
    summary_model: str,
    progress_callback=None,  # async callable(done: int, total: int) or None
    status_callback=None,    # async callable(message: str) or None
) -> dict:
    """
    Run the full forensics pipeline on a PDF.

    Returns a dict with keys:
        summary, enriched_pdf_base64, enriched_pdf_path,
        had_embedded_images, pages_processed, images_transcribed
    """
    fallback_model = (
        "llama3.2-vision:11b" if "90b" in vision_model else None
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Detect re-submission of an already-enriched PDF
    if "forensics-enriched" in (doc.metadata.get("keywords") or ""):
        doc.close()
        return {
            "summary": "This document has already been forensically processed.",
            "enriched_pdf_base64": base64.b64encode(pdf_bytes).decode(),
            "enriched_pdf_path": "",
            "had_embedded_images": False,
            "pages_processed": 0,
            "images_transcribed": 0,
        }

    total_pages = doc.page_count
    pdf_hash = sha256_of(pdf_bytes)
    cache_dir = os.environ.get("CACHE_DIR", "/data/pdf_cache")

    if status_callback:
        await status_callback(f"Classifying {total_pages} pages...")

    # ── Stage 1: classify pages ──────────────────────────────────────
    image_page_indices = []
    text_parts: dict[int, str] = {}  # page_index → native text

    for i, page in enumerate(doc):
        native_text = page.get_text().strip()
        has_images = len(page.get_images(full=True)) > 0
        if has_images and len(native_text) < IMAGE_PAGE_TEXT_THRESHOLD:
            image_page_indices.append(i)
        else:
            if native_text:
                text_parts[i] = native_text

    had_embedded_images = len(image_page_indices) > 0
    images_transcribed = 0

    if status_callback:
        n_image = len(image_page_indices)
        n_text = total_pages - n_image
        if had_embedded_images:
            await status_callback(
                f"Classification complete: {n_image} image page(s), {n_text} text page(s). Starting OCR..."
            )
        else:
            await status_callback(
                f"Classification complete: {n_text} native-text page(s), no image pages."
            )

    # ── Phase 1: Vision extraction (sequential, resumable) ───────────
    if had_embedded_images:
        # Set up image details markdown for resumability
        # Page numbers in the markdown are 1-indexed for readability
        details_doc = ImageDetailsDoc(
            cache_dir=cache_dir,
            pdf_sha256=pdf_hash,
            filename=output_stem,
            total_pages=total_pages,
            image_page_indices=[i + 1 for i in image_page_indices],
        )
        completed_pages = details_doc.get_completed_pages()

        if completed_pages:
            logger.info(
                "Resuming: %d/%d image pages already completed",
                len(completed_pages), len(image_page_indices),
            )
            if status_callback:
                await status_callback(
                    f"Resuming: {len(completed_pages)}/{len(image_page_indices)} "
                    f"image pages already processed, picking up where we left off."
                )

        # Ensure vision model is on GPU
        await assert_model_on_gpu(vision_model, ollama_base_url)

        # Process each image page sequentially
        for idx, page_index in enumerate(image_page_indices):
            page_number = page_index + 1  # 1-indexed for display/markdown

            # Resumability: skip already-completed pages
            if page_number in completed_pages:
                # Still need to load the text into text_parts for summary
                existing = details_doc.load()
                if page_number in existing and existing[page_number].extraction_status == "success":
                    text_parts[page_index] = existing[page_number].text_content
                    images_transcribed += 1
                if progress_callback:
                    await progress_callback(idx + 1, len(image_page_indices))
                continue

            if status_callback:
                await status_callback(
                    f"Processing image page {page_number} "
                    f"({idx + 1}/{len(image_page_indices)})..."
                )

            # Stage 2: Render page to PNG
            page = doc[page_index]
            mat = fitz.Matrix(2.0, 2.0)  # ~144 DPI
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            image_sha = sha256_of_bytes(png_bytes)

            # Stage 3: OCR transcription (sequential — one page at a time)
            try:
                text = await transcribe_page_image(
                    png_bytes, vision_model, ollama_base_url, fallback_model
                )
                if text and not text.startswith("__FAILED__"):
                    extraction_status = "success"

                    # Stage 4: embed invisible text
                    page.insert_text(
                        fitz.Point(0, 0),
                        text,
                        fontsize=1,
                        render_mode=3,   # PDF TR3 — invisible text
                        color=(0, 0, 0),
                    )
                    text_parts[page_index] = text
                    images_transcribed += 1
                else:
                    extraction_status = "failure"
                    text = text or ""
                    logger.warning("Transcription failed for page %d", page_number)
            except asyncio.TimeoutError:
                extraction_status = "failure"
                text = f"Timeout on page {page_number}"
                logger.warning("Transcription timed out for page %d", page_number)
            except Exception as exc:
                extraction_status = "failure"
                text = f"Error: {exc}"
                logger.warning("Transcription error for page %d: %s", page_number, exc)

            # Write result to markdown immediately (crash-safe)
            detail = ImagePageDetail(
                page_number=page_number,
                image_sha256=image_sha,
                text_content=text,
                extraction_status=extraction_status,
            )
            details_doc.append_page(detail)

            if progress_callback:
                await progress_callback(idx + 1, len(image_page_indices))

        # Phase 1 complete — unload vision model unless same as summary model
        same_model = vision_model == summary_model
        if same_model:
            if status_callback:
                await status_callback(
                    "Vision extraction complete. Same model used for summary — keeping loaded."
                )
        else:
            if status_callback:
                await status_callback("Vision extraction complete. Unloading vision model...")
            await unload_model(vision_model, ollama_base_url)
            await verify_model_unloaded(vision_model, ollama_base_url)
            if status_callback:
                await status_callback("Vision model unloaded and verified. Loading summary model...")

    # ── Phase 2: Summary & enrichment ────────────────────────────────

    # Collect all text for summary
    MAX_SUMMARY_CHARS = 40_000
    all_text_parts = [
        f"[Page {i + 1}]\n{text}"
        for i, text in sorted(text_parts.items())
    ]
    full_text = "\n\n".join(all_text_parts)
    if len(full_text) > MAX_SUMMARY_CHARS:
        logger.warning(
            "Document text (%d chars) exceeds limit; truncating to %d chars for summary",
            len(full_text), MAX_SUMMARY_CHARS,
        )
        full_text = full_text[:MAX_SUMMARY_CHARS] + "\n\n[... document truncated for summary ...]"

    if full_text.strip():
        if status_callback:
            await status_callback(
                f"Generating summary from {len(text_parts)} page(s) "
                f"({len(full_text):,} chars) using {summary_model}..."
            )
        try:
            await assert_model_on_gpu(summary_model, ollama_base_url)
            summary = await generate_summary(full_text, summary_model, ollama_base_url)
        except Exception as exc:
            logger.error("Summary generation failed: %s: %s", type(exc).__name__, exc)
            summary = f"Summary generation failed: {type(exc).__name__}: {exc}"
    else:
        summary = "No extractable text content found in this document."

    # Phase 2b: Enrich each image page with title/type/summary via qwen3:32b
    if had_embedded_images:
        if status_callback:
            await status_callback("Classifying and summarizing individual image pages...")

        for page_index in image_page_indices:
            page_number = page_index + 1
            page_text = text_parts.get(page_index, "")
            if not page_text or page_text.startswith("__FAILED__"):
                details_doc.update_page_fields(
                    page_number,
                    image_title="N/A — extraction failed",
                    image_type="unknown",
                    summarization_status="failure",
                )
                continue

            try:
                classification = await _classify_image_page(
                    page_text, summary_model, ollama_base_url
                )
                details_doc.update_page_fields(
                    page_number,
                    image_title=classification["title"],
                    image_type=classification["type"],
                    summarization_status="success",
                )
            except Exception as exc:
                logger.warning("Image classification failed for page %d: %s", page_number, exc)
                details_doc.update_page_fields(
                    page_number,
                    image_title="_error_",
                    image_type="_error_",
                    summarization_status="failure",
                )

        # Unload summary model
        await unload_model(summary_model, ollama_base_url)

    # ── Finalize enriched PDF ────────────────────────────────────────
    from datetime import datetime, timezone as _tz
    processed_at = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = f"Forensically enriched  |  {processed_at}"
    first_page = doc[0]
    footer_y = first_page.rect.height - 6
    first_page.insert_text(
        fitz.Point(4, footer_y),
        footer,
        fontsize=6,
        color=(0.55, 0.55, 0.55),
    )

    existing_keywords = (doc.metadata.get("keywords") or "").strip()
    doc.set_metadata({
        **doc.metadata,
        "keywords": f"forensics-enriched {existing_keywords}".strip(),
    })

    out_path = pathlib.Path(workspace_dir) / f"{output_stem}.pdf"
    enriched_bytes = doc.tobytes(garbage=4, deflate=True)
    out_path.write_bytes(enriched_bytes)
    doc.close()

    return {
        "summary": summary,
        "enriched_pdf_base64": base64.b64encode(enriched_bytes).decode(),
        "enriched_pdf_path": str(out_path),
        "had_embedded_images": had_embedded_images,
        "pages_processed": total_pages,
        "images_transcribed": images_transcribed,
    }


async def _classify_image_page(
    page_text: str,
    model: str,
    ollama_base_url: str,
) -> dict:
    """Ask qwen3:32b to classify a single image page.

    Returns dict with keys: title, type, summary.
    """
    import httpx

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": f"{_IMAGE_CLASSIFY_PROMPT}{page_text}",
            }
        ],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ollama_base_url.rstrip('/')}/api/chat", json=payload
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]

    # Parse structured response
    result = {"title": "_error_", "type": "unknown", "summary": ""}
    for line in content.strip().splitlines():
        if line.lower().startswith("title:"):
            result["title"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("type:"):
            result["type"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("summary:"):
            result["summary"] = line.split(":", 1)[1].strip()
    return result
