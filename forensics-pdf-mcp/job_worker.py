"""
Background worker that processes queued Drive sync jobs.

Runs as an asyncio task inside the FastAPI server.  Claims one job at a
time, walks the Google Drive folders, downloads each PDF, runs it through
``process_pdf_pipeline()`` directly (no MCP round-trip), and stores the
result in the ``PdfCache``.

Reuses Google Drive helpers from ``drive_sync``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pathlib
from typing import Any

from job_queue import JobQueue
from pdf_cache import PdfCache, sha256_of, slugify
from pdf_processor import process_pdf_pipeline

# Import Drive helpers from drive_sync (download, list, service builder)
from drive_sync import _build_drive_service_from_file, _list_pdfs_recursively, _download_pdf

logger = logging.getLogger(__name__)


async def run_job_worker(
    queue: JobQueue,
    cache: PdfCache,
    *,
    ollama_base_url: str,
    vision_model: str,
    summary_model: str,
    workspace_dir: str,
    poll_interval: float = 5.0,
) -> None:
    """Main worker loop — runs forever, claiming and processing jobs."""
    logger.info("Job worker started")
    while True:
        job = queue.claim()
        if job is None:
            await asyncio.sleep(poll_interval)
            continue

        job_id = job["job_id"]
        logger.info("Processing job %s", job_id)
        try:
            await _process_job(
                queue=queue,
                cache=cache,
                job=job,
                ollama_base_url=ollama_base_url,
                vision_model=vision_model,
                summary_model=summary_model,
                workspace_dir=workspace_dir,
            )
            queue.complete(job_id)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            queue.fail(job_id, str(exc))
        finally:
            # Clean up credentials file
            creds_path = job.get("creds_path")
            if creds_path:
                try:
                    pathlib.Path(creds_path).unlink(missing_ok=True)
                except Exception:
                    pass


async def _process_job(
    queue: JobQueue,
    cache: PdfCache,
    job: dict[str, Any],
    ollama_base_url: str,
    vision_model: str,
    summary_model: str,
    workspace_dir: str,
) -> None:
    """Walk Drive folders, download PDFs, process each through the pipeline."""
    job_id = job["job_id"]
    creds_path = job["creds_path"]
    folders = job["folders"]

    drive_service = _build_drive_service_from_file(creds_path)

    # Phase 1: enumerate all PDFs across all folders
    all_files: list[tuple[str, dict]] = []  # (section, drive_file_metadata)
    for i, folder in enumerate(folders):
        section = folder["section"]
        folder_id = folder["folder_id"]
        logger.info("  Listing %s (folder %s...)", section, folder_id[:12])
        files = _list_pdfs_recursively(drive_service, folder_id)
        for f in files:
            all_files.append((section, f))
        queue.update(job_id, folders_done=i + 1)

    queue.update(job_id, files_total=len(all_files))
    logger.info("  Found %d total PDFs across %d folders", len(all_files), len(folders))

    # Phase 2: process each PDF
    for idx, (section, df) in enumerate(all_files):
        drive_path = df.get("_drive_path", df["name"])
        queue.update(
            job_id,
            current_file=drive_path,
            current_file_progress="downloading",
        )

        try:
            pdf_bytes = _download_pdf(drive_service, df["id"])

            # Check cache — skip if already processed
            cached = cache.get(pdf_bytes)
            if cached is not None:
                logger.info("  Cache hit: %s", drive_path)
                queue.update(
                    job_id,
                    files_done=idx + 1,
                    files_cached=queue.get(job_id)["files_cached"] + 1,
                    current_file_progress="cached",
                )
                continue

            queue.update(job_id, current_file_progress="processing")

            # Progress callback updates the job's current_file_progress
            async def _on_progress(done: int, total: int, _jid: str = job_id) -> None:
                queue.update(_jid, current_file_progress=f"page {done}/{total}")

            async def _on_status(message: str) -> None:
                logger.info("    %s: %s", drive_path, message)

            output_stem = pathlib.Path(df["name"]).stem
            result = await process_pdf_pipeline(
                pdf_bytes=pdf_bytes,
                output_stem=output_stem,
                workspace_dir=workspace_dir,
                ollama_base_url=ollama_base_url,
                vision_model=vision_model,
                summary_model=summary_model,
                progress_callback=_on_progress,
                status_callback=_on_status,
            )

            # Store in cache
            try:
                cache.put(
                    pdf_bytes=pdf_bytes,
                    filename=output_stem,
                    summary=result["summary"],
                    enriched_pdf=base64.b64decode(result["enriched_pdf_base64"]),
                    had_embedded_images=result["had_embedded_images"],
                    pages_processed=result["pages_processed"],
                    images_transcribed=result["images_transcribed"],
                )
            except Exception:
                logger.exception("Failed to cache %s (non-fatal)", drive_path)

            queue.update(job_id, files_done=idx + 1)
            logger.info("  Processed: %s (%d pages)", drive_path, result["pages_processed"])

        except Exception as exc:
            logger.exception("  Error processing %s", drive_path)
            # Record per-file error but continue
            job_state = queue.get(job_id)
            errors = job_state["errors"] if job_state else []
            errors.append(f"{drive_path}: {exc}")
            queue.update(
                job_id,
                files_done=idx + 1,
                files_errors=(job_state["files_errors"] + 1) if job_state else 1,
                errors_json=json.dumps(errors),
            )
