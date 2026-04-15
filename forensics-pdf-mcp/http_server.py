"""
forensics-pdf-mcp HTTP server.

Exposes the MCP Streamable HTTP transport (2025-03-26 spec) at /mcp via
FastMCP, wrapped in a FastAPI app with ECDSA authentication middleware.

All /mcp requests require four X-Auth-* headers. /health is exempt.

Environment variables:
    OLLAMA_BASE_URL   default http://localhost:11434
    VISION_MODEL      default llama3.2-vision:90b
    SUMMARY_MODEL     default qwen3:32b
    WORKSPACE_DIR     default /workspace
    DB_PATH           default /data/keys.db
    CACHE_DIR         default /data/pdf_cache
    HTTP_PORT         default 18790
"""

import asyncio
import base64
import json
import logging
import os
import pathlib
import uuid

import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastmcp import Context, FastMCP

from auth.key_registry import KeyRegistry
from auth.middleware import ECDSAAuthMiddleware
from job_queue import JobQueue, JobBusyError
from job_worker import run_job_worker
from pdf_cache import PdfCache, slugify
from pdf_processor import process_pdf_pipeline
from drive_sync import run_sync, get_synced_documents, get_document_enriched_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=__import__("sys").stdout,
)
# Force stdout to flush immediately so docker logs shows output in real time
__import__("sys").stdout.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-vision:90b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen3:32b")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
DB_PATH = os.getenv("DB_PATH", "/data/keys.db")
CACHE_DIR = os.getenv("CACHE_DIR", "/data/pdf_cache")
HTTP_PORT = int(os.getenv("HTTP_PORT", "18790"))

# --- Ensure workspace exists ---
pathlib.Path(WORKSPACE_DIR).mkdir(parents=True, exist_ok=True)

# --- Key registry (initialised before app creation so middleware can use it) ---
registry = KeyRegistry(DB_PATH)
registry.init_db()

# --- PDF result cache ---
cache = PdfCache(CACHE_DIR)

# --- Job queue ---
JOBS_DIR = os.getenv("JOBS_DIR", "/data/jobs")
JOBS_DB = os.getenv("JOBS_DB", "/data/jobs.db")
pathlib.Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)
job_queue = JobQueue(JOBS_DB)

# --- FastMCP server ---
mcp = FastMCP("forensics-pdf-mcp")


@mcp.tool()
async def process_pdf(
    ctx: Context,
    file_path: str | None = None,
    file_base64: str | None = None,
    filename: str | None = None,
) -> str:
    """
    Process a PDF through the forensic OCR pipeline.

    Provide either:
        file_path   — absolute path to a PDF inside /workspace
        file_base64 — base64-encoded PDF content (also supply filename)

    Returns a JSON string with keys:
        summary, enriched_pdf_base64, enriched_pdf_path,
        had_embedded_images, pages_processed, images_transcribed
    """
    if file_path:
        pdf_path = pathlib.Path(file_path)
        if not pdf_path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})
        pdf_bytes = pdf_path.read_bytes()
        output_stem = pdf_path.stem
    elif file_base64 and filename:
        try:
            pdf_bytes = base64.b64decode(file_base64)
        except Exception as exc:
            return json.dumps({"error": f"Invalid base64: {exc}"})
        output_stem = pathlib.Path(filename).stem
    else:
        return json.dumps(
            {"error": "Provide either file_path or both file_base64 and filename"}
        )

    # --- Cache lookup ---
    cached = cache.get(pdf_bytes)
    if cached is not None:
        await ctx.info(f"Cache hit — returning stored result for {cached.filename}")
        return json.dumps({
            "summary": cached.summary,
            "enriched_pdf_base64": base64.b64encode(cached.enriched_pdf).decode(),
            "enriched_pdf_path": f"{WORKSPACE_DIR}/{slugify(cached.filename)}.pdf",
            "had_embedded_images": cached.had_embedded_images,
            "pages_processed": cached.pages_processed,
            "images_transcribed": cached.images_transcribed,
        })

    async def _on_progress(done: int, total: int) -> None:
        await ctx.report_progress(done, total)

    async def _on_status(message: str) -> None:
        await ctx.info(message)

    try:
        result = await process_pdf_pipeline(
            pdf_bytes=pdf_bytes,
            output_stem=output_stem,
            workspace_dir=WORKSPACE_DIR,
            ollama_base_url=OLLAMA_BASE_URL,
            vision_model=VISION_MODEL,
            summary_model=SUMMARY_MODEL,
            progress_callback=_on_progress,
            status_callback=_on_status,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        return json.dumps({"error": str(exc)})

    # --- Store in cache ---
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
        logger.exception("Failed to store result in cache (non-fatal)")

    return json.dumps(result)


@mcp.tool()
async def sync_drive(
    ctx: Context,
    full: bool = False,
) -> str:
    """
    Sync PDFs from Google Drive through the forensic OCR pipeline to S3.

    Args:
        full: If True, clear all sync state and re-process every file.
              If False (default), only process new/modified files.

    Returns a JSON summary with keys: synced, skipped, errors.
    """
    async def _on_status(message: str) -> None:
        await ctx.info(message)

    try:
        totals = await run_sync(
            full=full,
            status_callback=_on_status,
        )
        return json.dumps(totals)
    except Exception as exc:
        logger.exception("Drive sync error")
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def list_synced_documents(
    ctx: Context,
    section: str | None = None,
) -> str:
    """
    Return metadata for all synced documents. Instant — reads from cache.

    Args:
        section: Optional filter (HOA, Condo1, Condo2, Condo3, Condo4).
                 If omitted, returns all sections.

    Returns JSON array of document objects with keys:
        drive_file_id, drive_name, section, s3_filename, s3_path,
        description, last_synced_at
    """
    docs = get_synced_documents(section=section)
    return json.dumps(docs)


@mcp.tool()
async def get_document_pdf(
    ctx: Context,
    drive_file_id: str,
) -> str:
    """
    Return the enriched (OCR'd) PDF for a specific document.

    Args:
        drive_file_id: Google Drive file ID of the document.

    Returns JSON with enriched_pdf_base64, or error.
    """
    pdf_bytes = get_document_enriched_pdf(drive_file_id)
    if pdf_bytes is None:
        return json.dumps({"error": f"Document not found: {drive_file_id}"})
    return json.dumps({
        "enriched_pdf_base64": base64.b64encode(pdf_bytes).decode(),
    })


@mcp.tool()
async def submit_sync_job(
    ctx: Context,
    google_credentials_json: str,
    folders: list[dict],
) -> str:
    """
    Submit a Drive sync job for background processing.

    The server walks the specified Google Drive folders, downloads each PDF,
    and runs it through the forensic OCR pipeline.  Returns immediately with
    a job_id that can be polled via ``get_job_status``.

    Only one job may run at a time — submitting while one is active returns
    an error with the active job's ID.

    Args:
        google_credentials_json: Service-account JSON string.
        folders: List of {"section": "HOA", "folder_id": "..."} dicts.

    Returns JSON: {"job_id": "...", "status": "pending"} or
                  {"error": "busy", "active_job_id": "..."}.
    """
    job_id = str(uuid.uuid4())

    # Write credentials to a temp file
    creds_path = os.path.join(JOBS_DIR, f"{job_id}_creds.json")
    pathlib.Path(creds_path).write_text(google_credentials_json)

    try:
        job = job_queue.submit(job_id, folders, creds_path)
    except JobBusyError as exc:
        # Clean up the creds file we just wrote
        pathlib.Path(creds_path).unlink(missing_ok=True)
        return json.dumps({
            "error": "busy",
            "active_job_id": exc.active_job_id,
        })

    await ctx.info(f"Job {job_id} submitted — {len(folders)} folders queued")
    return json.dumps({"job_id": job_id, "status": job["status"]})


@mcp.tool()
async def get_job_status(
    ctx: Context,
    job_id: str,
) -> str:
    """
    Poll the status of a sync job submitted via ``submit_sync_job``.

    Args:
        job_id: The job ID returned by submit_sync_job.

    Returns JSON with: job_id, status, folders_total, folders_done,
    files_total, files_done, files_cached, files_errors, current_file,
    current_file_progress, errors, started_at, completed_at.
    """
    job = job_queue.get(job_id)
    if job is None:
        return json.dumps({"error": f"Job not found: {job_id}"})

    # Don't expose the credentials path to the client
    job.pop("creds_path", None)
    return json.dumps(job)


# --- FastAPI app ---
# Capture the mcp ASGI app first so we can compose its lifespan with ours.
# FastMCP's StreamableHTTPSessionManager requires the lifespan to run its
# internal task group; without it every request gets a 500.
mcp_app = mcp.http_app(path="/")
_mcp_lifespan = mcp_app.lifespan


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app_instance):
    """Compose FastMCP lifespan with the background job worker."""
    worker_task = asyncio.create_task(
        run_job_worker(
            queue=job_queue,
            cache=cache,
            ollama_base_url=OLLAMA_BASE_URL,
            vision_model=VISION_MODEL,
            summary_model=SUMMARY_MODEL,
            workspace_dir=WORKSPACE_DIR,
        )
    )
    logger.info("Background job worker started")
    async with _mcp_lifespan(app_instance):
        yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Background job worker stopped")


app = FastAPI(title="forensics-pdf-mcp", lifespan=_lifespan)

# Add auth middleware first (becomes outermost wrapper due to LIFO)
app.add_middleware(ECDSAAuthMiddleware, registry=registry)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- REST endpoints for job queue (no MCP/SSE overhead) ---

@app.post("/jobs/submit")
async def rest_submit_job(request: fastapi.Request):
    """Submit a Drive sync job. Returns immediately with job_id."""
    body = await request.json()
    google_credentials_json = body.get("google_credentials_json", "")
    folders = body.get("folders", [])
    if not google_credentials_json or not folders:
        return JSONResponse(
            {"error": "google_credentials_json and folders are required"},
            status_code=400,
        )
    job_id = str(uuid.uuid4())
    creds_path = os.path.join(JOBS_DIR, f"{job_id}_creds.json")
    pathlib.Path(creds_path).write_text(google_credentials_json)
    try:
        job = job_queue.submit(job_id, folders, creds_path)
    except JobBusyError as exc:
        pathlib.Path(creds_path).unlink(missing_ok=True)
        return JSONResponse({"error": "busy", "active_job_id": exc.active_job_id})
    return {"job_id": job_id, "status": job["status"]}


@app.get("/jobs/{job_id}")
async def rest_job_status(job_id: str):
    """Poll job status. Instant JSON response."""
    job = job_queue.get(job_id)
    if job is None:
        return JSONResponse({"error": f"Job not found: {job_id}"}, status_code=404)
    job.pop("creds_path", None)
    return job


# Mount FastMCP at /mcp — must happen after middleware is registered
app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    logger.info("Starting forensics-pdf-mcp on port %d", HTTP_PORT)
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")
