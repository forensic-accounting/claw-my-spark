# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claw-my-spark is a forensic PDF processing pipeline for DGX Spark infrastructure. It combines a Python MCP server (`forensics-pdf-mcp`) for PDF OCR and forensics analysis, a Node.js AI agent gateway (`openclaw`) with Ollama integration, and an ExpressVPN sidecar for network isolation—all orchestrated via Docker Compose.

## Build & Development Commands

```bash
make setup    # Create venv + install forensics-pdf-mcp in editable mode
make build    # Build all Docker images
make up       # Start all services (docker compose up)
make down     # Stop all services
make test     # Run integration test (requires running services + active venv)
```

### Running Unit Tests

```bash
source venv/bin/activate
python3 -m pytest forensics-pdf-mcp/tests/ -v

# Single test file:
python3 -m pytest forensics-pdf-mcp/tests/test_signing.py -v

# Single test:
python3 -m pytest forensics-pdf-mcp/tests/test_signing.py::test_function_name -v
```

Unit tests use `pytest` + `pytest-asyncio` + `respx` (mocked HTTP, no live Ollama needed). Integration test (`tests/test.py`) requires running services.

## Architecture

### Services (docker-compose.yaml)

- **ollama** — GPU-accelerated LLM inference (NVIDIA). External volume `ollama-models` must pre-exist.
- **forensics-pdf-mcp** — Python MCP server on host network (port 18790). Handles PDF processing.
- **openclaw** — Node.js agent gateway sharing expressvpn's network namespace.
- **expressvpn** — OpenVPN sidecar (NET_ADMIN capability). Exposes openclaw on port 18789.
- **caddy** — HTTPS reverse proxy (ports 80/443) forwarding to openclaw via expressvpn.

### forensics-pdf-mcp (primary component)

**Entry point:** `http_server.py` — FastAPI + FastMCP server exposing a single MCP tool `process_pdf`.

**PDF Processing Pipeline** (`pdf_processor.py`, 5 stages):
1. **Page Classification** — Detect image vs text pages (threshold: 50 chars)
2. **Page Rendering** — Render image pages to PNG at 144 DPI via PyMuPDF
3. **Vision Transcription** — OCR via Ollama vision model (90b, fallback to 11b). Max 3 concurrent, 900s timeout per page, batches of 10.
4. **Invisible Text Embedding** — Embed extracted text using TR3 render mode (searchable PDF, no visual change)
5. **Summary Generation** — Structured forensic summary via Ollama text model (40k char cap)

**Authentication** (`auth/`): ECDSA P-256 per-request signing. Four required headers on `/mcp/*`: `X-Auth-Key-ID`, `X-Auth-Timestamp` (±60s), `X-Auth-Nonce` (single-use UUID), `X-Auth-Signature`. Nonce replay defense via SQLite with 5-minute TTL.

**Caching** (`pdf_cache.py`): SHA-256 of input PDF bytes → `/data/pdf_cache/{hash}/` containing `summary.txt`, `enriched.pdf`, `meta.json`.

**Client** (`client/forensics_client.py`): Async context manager using httpx. Reads config from `~/.forensics-pdf-mcp/` or environment. Returns `ProcessResult` dataclass.

**Key admin tools:** `admin/keygen.py` (generate P-256 keypairs), `admin/register_client.py` (register/revoke/list keys in SQLite).

### Key Models (configured via env vars)

- `VISION_MODEL` — llama3.2-vision:90b (OCR)
- `SUMMARY_MODEL` — qwen3:32b (summarization)
- `DEFAULT_MODEL` — qwen3:14b (OpenClaw default)

## Key Dependencies

- **fastmcp>=2.2.0** — MCP protocol (HTTP/SSE transport, spec 2025-03-26)
- **PyMuPDF>=1.24.0** — PDF rendering, text embedding, manipulation
- **httpx>=0.27.0** — Async HTTP (Ollama API + client library)
- **cryptography>=42.0.0** — ECDSA P-256 signing/verification
- **respx>=0.21.0** — Test mocking for httpx

## Deployment

Code is deployed to DGX Spark via rsync + docker rebuild:
```bash
rsync -av --exclude='__pycache__' forensics-pdf-mcp/ claude@dgx-spark-claude:~/projects/claw-my-spark/forensics-pdf-mcp/
docker compose build forensics-pdf-mcp && docker compose restart forensics-pdf-mcp
```
