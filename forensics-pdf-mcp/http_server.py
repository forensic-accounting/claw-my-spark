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
    HTTP_PORT         default 18790
"""

import base64
import json
import logging
import os
import pathlib

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastmcp import FastMCP

from auth.key_registry import KeyRegistry
from auth.middleware import ECDSAAuthMiddleware
from pdf_processor import process_pdf_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-vision:90b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen3:32b")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
DB_PATH = os.getenv("DB_PATH", "/data/keys.db")
HTTP_PORT = int(os.getenv("HTTP_PORT", "18790"))

# --- Ensure workspace exists ---
pathlib.Path(WORKSPACE_DIR).mkdir(parents=True, exist_ok=True)

# --- Key registry (initialised before app creation so middleware can use it) ---
registry = KeyRegistry(DB_PATH)
registry.init_db()

# --- FastMCP server ---
mcp = FastMCP("forensics-pdf-mcp")


@mcp.tool()
async def process_pdf(
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

    try:
        result = await process_pdf_pipeline(
            pdf_bytes=pdf_bytes,
            output_stem=output_stem,
            workspace_dir=WORKSPACE_DIR,
            ollama_base_url=OLLAMA_BASE_URL,
            vision_model=VISION_MODEL,
            summary_model=SUMMARY_MODEL,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        return json.dumps({"error": str(exc)})

    return json.dumps(result)


# --- FastAPI app ---
app = FastAPI(title="forensics-pdf-mcp")

# Add auth middleware first (becomes outermost wrapper due to LIFO)
app.add_middleware(ECDSAAuthMiddleware, registry=registry)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount FastMCP at /mcp — must happen after middleware is registered
app.mount("/mcp", mcp.http_app(path="/"))


if __name__ == "__main__":
    logger.info("Starting forensics-pdf-mcp on port %d", HTTP_PORT)
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")
