"""
PDF forensics processing pipeline.

Stages:
  1. Page classification — detect image-based vs text pages
  2. Page rendering     — render image pages to PNG at 2x scale
  3. OCR transcription  — send PNGs to llama3.2-vision via ollama_client
  4. Text embedding     — write invisible text (PDF TR3) onto each page
  5. Summary generation — send all text to qwen3:32b for a forensic summary

Entry point: process_pdf_pipeline()
"""

import asyncio
import base64
import logging
import pathlib
from typing import Optional

import fitz  # PyMuPDF

from ollama_client import generate_summary, transcribe_page_image

logger = logging.getLogger(__name__)

# A page is considered "image-based" if it has embedded images AND fewer than
# this many characters of native text (catches pages that are entirely scanned
# but have a small header/footer in native text).
IMAGE_PAGE_TEXT_THRESHOLD = 50

# Process image pages in batches to bound memory usage on large documents
BATCH_SIZE = 10


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

    if status_callback:
        await status_callback(f"Classifying {total_pages} pages...")

    # Stage 1: classify pages
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

    if had_embedded_images:
        # Stages 2+3+4: render, transcribe, embed in batches
        for batch_start in range(0, len(image_page_indices), BATCH_SIZE):
            batch = image_page_indices[batch_start : batch_start + BATCH_SIZE]
            tasks = [
                _safe_transcribe(doc, pi, vision_model, ollama_base_url, fallback_model)
                for pi in batch
            ]
            results = await asyncio.gather(*tasks)

            for idx, (pi, transcribed_text) in enumerate(results):
                if transcribed_text and not transcribed_text.startswith("__FAILED__"):
                    # Stage 4: embed invisible text
                    page = doc[pi]
                    page.insert_text(
                        fitz.Point(0, 0),
                        transcribed_text,
                        fontsize=1,
                        render_mode=3,   # PDF TR3 — invisible text
                        color=(0, 0, 0),
                    )
                    text_parts[pi] = transcribed_text
                    images_transcribed += 1
                elif transcribed_text.startswith("__FAILED__"):
                    logger.warning("Transcription failed for page %d", pi + 1)

                # Report per-page progress
                if progress_callback:
                    done = batch_start + idx + 1
                    await progress_callback(done, len(image_page_indices))

    # Stage 5: generate summary from all collected text
    # Cap input to ~40,000 chars to keep within model context/timeout bounds.
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
            summary = await generate_summary(full_text, summary_model, ollama_base_url)
        except Exception as exc:
            logger.error("Summary generation failed: %s: %s", type(exc).__name__, exc)
            summary = f"Summary generation failed: {type(exc).__name__}: {exc}"
    else:
        summary = "No extractable text content found in this document."

    # Stamp page 1 with a small footer and set document metadata
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

    # Save enriched PDF
    out_path = pathlib.Path(workspace_dir) / f"{output_stem}.pdf"
    enriched_bytes = doc.tobytes(garbage=4, deflate=True)
    out_path.write_bytes(enriched_bytes)
    doc.close()

    return {
        "summary": summary,
        "enriched_pdf_base64": base64.b64encode(enriched_bytes).decode(),
        "enriched_pdf_path": str(out_path),  # {output_stem}.pdf (no suffix — metadata inside identifies it)
        "had_embedded_images": had_embedded_images,
        "pages_processed": total_pages,
        "images_transcribed": images_transcribed,
    }


async def _safe_transcribe(
    doc: fitz.Document,
    page_index: int,
    vision_model: str,
    ollama_base_url: str,
    fallback_model: Optional[str],
) -> tuple[int, str]:
    """Render a page to PNG and transcribe it; return (page_index, text_or_failure_marker)."""
    try:
        page = doc[page_index]
        mat = fitz.Matrix(2.0, 2.0)  # ~144 DPI — sufficient for OCR accuracy
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        text = await transcribe_page_image(
            png_bytes, vision_model, ollama_base_url, fallback_model
        )
        return page_index, text
    except asyncio.TimeoutError:
        return page_index, f"__FAILED__: timeout on page {page_index + 1}"
    except Exception as exc:
        return page_index, f"__FAILED__: {exc}"
