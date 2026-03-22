# forensics-pdf-mcp

An MCP (Model Context Protocol) server that processes PDF files through a vision-based OCR pipeline using locally hosted LLMs on the DGX Spark. Designed for financial forensics — bank statements, scanned checks, invoices, and similar documents.

---

## Overview

`forensics-pdf-mcp` accepts a PDF file, detects whether it contains embedded images (i.e., is scanned rather than digitally generated), transcribes those images using the `llama3.2-vision:90b` model, embeds the extracted text invisibly back into the PDF as a searchable layer, and returns both the enriched PDF and a structured summary of the document's contents.

The service runs as a Docker container on the DGX Spark alongside the existing Ollama stack. It exposes a single HTTP/SSE transport over `http://dgx-spark-claude:FORENSICS_PDF_MCP_PORT`. All requests are authenticated using Elliptic Curve Digital Signature Algorithm (ECDSA). Private keys never traverse the network.

---

## Architecture

```
┌─────────────────────────────────┐
│    Any MCP Client (remote)      │
│    forensics_client.py          │
│    private_key.pem (P-256)      │
└─────────────┬───────────────────┘
              │ HTTP/SSE
              │ ECDSA-signed requests
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    forensics-pdf-mcp container                          │
│                                                                         │
│   http_server.py     ← FastAPI HTTP/SSE MCP server + auth middleware   │
│       │                                                                 │
│   auth/                                                                 │
│     ecc_auth.py      ← ECDSA signature verification                     │
│     key_registry.py  ← SQLite public key store + nonce replay defense  │
│       │                                                                 │
│   pdf_processor.py   ← PDF pipeline orchestration                      │
│   ollama_client.py   ← Async Ollama API wrapper                        │
│                                                                         │
│   /data/keys.db      ← SQLite (persisted via Docker volume)            │
│   /workspace/        ← PDF working directory (Docker volume)           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP (host network)
                                ▼
                   ┌────────────────────────┐
                   │    ollama container    │
                   │   llama3.2-vision:90b  │
                   │   qwen3:32b            │
                   └────────────────────────┘
```

---

## Directory Layout

```
forensics-pdf-mcp/
├── FORENSIC-PDF-DESIGN.md       # This document
├── Dockerfile                   # Container image definition
├── requirements.txt             # Server Python dependencies
│
├── http_server.py               # FastAPI HTTP/SSE MCP server (sole entrypoint)
├── pdf_processor.py             # PDF processing pipeline
├── ollama_client.py             # Async Ollama API wrapper
│
├── auth/
│   ├── __init__.py
│   ├── ecc_auth.py              # ECDSA signature verification middleware
│   ├── key_registry.py          # SQLite key store (public keys + nonces)
│   └── schema.sql               # Database schema
│
├── admin/
│   ├── keygen.py                # Generate a client EC keypair
│   └── register_client.py       # Register a client public key with the server
│
└── client/
    ├── requirements.txt         # Client-only Python dependencies
    └── forensics_client.py      # Python MCP client library
```

---

## Container Design

### Base Image

`python:3.12-slim` — minimal footprint, matches the host Python version.

### Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /workspace /data

ENTRYPOINT ["python3", "http_server.py"]
```

### docker-compose.yaml Addition

```yaml
forensics-pdf-mcp:
  build:
    context: ./forensics-pdf-mcp
    dockerfile: Dockerfile
  container_name: forensics-pdf-mcp
  restart: unless-stopped
  network_mode: host
  ports:
    - "${FORENSICS_PDF_MCP_PORT:-18790}:18790"
  volumes:
    - /tmp/forensics-pdf-mcp:/workspace
    - forensics-pdf-keys:/data
  environment:
    OLLAMA_BASE_URL:   "http://localhost:11434"
    VISION_MODEL:      "llama3.2-vision:90b"
    SUMMARY_MODEL:     "qwen3:32b"
    WORKSPACE_DIR:     "/workspace"
    DB_PATH:           "/data/keys.db"
    HTTP_PORT:         "${FORENSICS_PDF_MCP_PORT:-18790}"
  depends_on:
    ollama:
      condition: service_healthy
  stdin_open: true
  tty: false
```

Add the volume to the top-level `volumes:` block:

```yaml
volumes:
  forensics-pdf-keys:
    name: forensics-pdf-keys
```

---

## Transport: HTTP/SSE

`http_server.py` runs a FastAPI application that serves the MCP protocol over HTTP with Server-Sent Events. The server listens on `0.0.0.0:18790` (configurable via `HTTP_PORT`).

### MCP over HTTP/SSE

| Endpoint | Method | Purpose |
|---|---|---|
| `/mcp/sse` | GET | SSE stream — server pushes MCP messages to client |
| `/mcp/messages` | POST | Client sends MCP messages (tool calls, etc.) |
| `/health` | GET | Unauthenticated health check |

All `/mcp/*` endpoints require a valid ECDSA authentication header set (see Authentication section below).

---

## Authentication: ECDSA Request Signing

### Design Principles

- The API key **never appears on the network** — only a signature over request metadata does.
- Each client has a unique P-256 EC private key stored locally.
- The server stores only the corresponding public keys in SQLite.
- Every HTTP request is signed. Replayed or tampered requests are rejected.

### Elliptic Curve Parameters

| Parameter | Value |
|---|---|
| Curve | P-256 (secp256r1, NIST) |
| Hash | SHA-256 |
| Signature scheme | ECDSA |
| Key format | PEM (PKCS#8 private, SubjectPublicKeyInfo public) |

P-256 is the standard choice for modern ECDSA: widely supported, 128-bit security level, compact signatures (~71 bytes DER-encoded).

### SQLite Schema (`auth/schema.sql`)

```sql
-- Registered client public keys
CREATE TABLE IF NOT EXISTS keys (
    key_id      TEXT PRIMARY KEY,          -- UUID v4, identifies the client
    client_name TEXT NOT NULL,             -- human-readable label
    public_key  TEXT NOT NULL,             -- PEM-encoded SubjectPublicKeyInfo
    created_at  TEXT NOT NULL,             -- ISO-8601 UTC
    active      INTEGER NOT NULL DEFAULT 1 -- 0 = revoked
);

-- Seen nonces — prevents replay attacks
-- Rows older than 5 minutes are pruned on each request
CREATE TABLE IF NOT EXISTS nonces (
    nonce      TEXT PRIMARY KEY,           -- UUID v4, single-use
    key_id     TEXT NOT NULL,
    used_at    TEXT NOT NULL               -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_nonces_used_at ON nonces(used_at);
```

### Per-Request Signing Protocol

#### Client side (`auth/ecc_auth.py` — signing half)

1. Compute `body_hash = SHA-256(raw request body bytes).hex()`
2. Assemble the canonical string (newline-separated, no trailing newline):
   ```
   <key_id>
   <timestamp>        ← ISO-8601 UTC, e.g. 2026-03-22T14:05:00.123456Z
   <nonce>            ← fresh UUID v4
   <body_hash>
   ```
3. Sign the UTF-8 encoding of the canonical string with ECDSA + SHA-256:
   ```python
   signature_der = private_key.sign(canonical_bytes, ec.ECDSA(hashes.SHA256()))
   signature_b64 = base64.urlsafe_b64encode(signature_der).decode()
   ```
4. Attach headers to the HTTP request:
   ```
   X-Auth-Key-ID:    <key_id>
   X-Auth-Timestamp: <timestamp>
   X-Auth-Nonce:     <nonce>
   X-Auth-Signature: <signature_b64>
   ```

#### Server side (`auth/ecc_auth.py` — verification half)

1. Extract the four `X-Auth-*` headers. Return `401` if any are missing.
2. Parse timestamp; reject if `|now - timestamp| > 60 seconds`. Return `401`.
3. Look up `key_id` in SQLite `keys` table. Return `401` if not found or inactive.
4. Check `nonce` is not in the `nonces` table. Return `401` if already seen (replay).
5. Read raw request body bytes; compute `body_hash = SHA-256(body).hex()`.
6. Reconstruct the canonical string identically to the client.
7. Verify ECDSA signature using the stored public key:
   ```python
   public_key.verify(signature_der, canonical_bytes, ec.ECDSA(hashes.SHA256()))
   ```
   Return `401` if verification raises `InvalidSignature`.
8. Insert nonce into SQLite with current timestamp. Prune nonces older than 5 minutes.
9. Allow request to proceed.

### Security Properties

| Threat | Defense |
|---|---|
| API key interception | Private key never sent; only a signature of the request is transmitted |
| Request replay | Per-request UUID nonce stored in SQLite; rejected on second use |
| Request tampering | Body hash is part of the signed canonical string; any modification invalidates the signature |
| Timestamp forgery | Server enforces ±60s window; clocks must be roughly synchronized |
| Key compromise | Admin can set `active = 0` in SQLite to revoke a key instantly |
| Brute-force key recovery | Computationally infeasible against P-256 |

---

## Key Management

### Generating a Client Keypair (`admin/keygen.py`)

Run once per client machine. Outputs two PEM files.

```bash
# On the client machine
python3 admin/keygen.py --client-name "my-workstation" --output-dir ~/.forensics-pdf-mcp/
```

Produces:
- `~/.forensics-pdf-mcp/private_key.pem` — EC private key (chmod 600)
- `~/.forensics-pdf-mcp/public_key.pem` — public key to register with the server
- `~/.forensics-pdf-mcp/key_id.txt` — the UUID assigned to this keypair

### Registering a Public Key (`admin/register_client.py`)

Run inside the container (or via `docker exec`) to add a client's public key to the server's database.

```bash
# On the DGX Spark
docker exec -i forensics-pdf-mcp python3 admin/register_client.py \
    --key-id "$(cat ~/.forensics-pdf-mcp/key_id.txt)" \
    --client-name "my-workstation" \
    --public-key-file /path/to/public_key.pem
```

### Revoking a Key

```bash
docker exec -i forensics-pdf-mcp python3 -c \
  "from auth.key_registry import KeyRegistry; KeyRegistry('/data/keys.db').revoke('<key_id>')"
```

---

## Python MCP Client (`client/forensics_client.py`)

A self-contained Python library that any script or application can import to submit PDFs to the forensics pipeline.

### Client Configuration

The client reads from `~/.forensics-pdf-mcp/config.json` by default:

```json
{
  "server_url": "http://dgx-spark-claude:18790",
  "key_id_file": "~/.forensics-pdf-mcp/key_id.txt",
  "private_key_file": "~/.forensics-pdf-mcp/private_key.pem"
}
```

All fields can be overridden programmatically or via environment variables (`FORENSICS_SERVER_URL`, `FORENSICS_KEY_ID_FILE`, `FORENSICS_PRIVATE_KEY_FILE`).

### Client API

```python
from client.forensics_client import ForensicsClient, ProcessResult

async with ForensicsClient() as client:

    # Submit a local PDF file
    result: ProcessResult = await client.process_pdf_file("/path/to/statement.pdf")

    # Or submit raw bytes
    result: ProcessResult = await client.process_pdf_bytes(pdf_bytes, filename="invoice.pdf")

    print(result.summary)
    result.save_enriched("/path/to/statement_ocr.pdf")
```

### `ProcessResult` Fields

| Field | Type | Description |
|---|---|---|
| `summary` | `str` | Structured forensics summary from qwen3:32b |
| `enriched_pdf` | `bytes` | PDF with invisible OCR text layer embedded |
| `had_embedded_images` | `bool` | Whether image-based pages were found |
| `pages_processed` | `int` | Total pages in document |
| `images_transcribed` | `int` | Pages processed through vision model |
| `save_enriched(path)` | method | Writes `enriched_pdf` to disk |

### Client Dependencies (`client/requirements.txt`)

```
httpx>=0.27
cryptography>=42.0
```

---

## MCP Tool: `process_pdf`

The server exposes a single tool.

**Inputs** (one file source required):

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | one of | Absolute path to PDF inside `/workspace` |
| `file_base64` | string | one of | Base64-encoded PDF content |
| `filename` | string | with `file_base64` | Original filename for output naming |

**Outputs:**

| Field | Type | Description |
|---|---|---|
| `summary` | string | Structured narrative summary |
| `enriched_pdf_base64` | string | Base64-encoded enriched PDF |
| `enriched_pdf_path` | string | Path inside `/workspace` where enriched PDF was saved |
| `had_embedded_images` | bool | Whether image-based pages were detected |
| `pages_processed` | int | Total page count |
| `images_transcribed` | int | Pages sent through the vision model |

---

## Processing Pipeline

### Stage 1 — Image Detection

Open the PDF with PyMuPDF (`fitz`). For each page:

1. `page.get_images(full=True)` — detect embedded image XREFs.
2. `page.get_text()` — if empty or near-empty alongside images, page is image-based.
3. Classify each page: `text_page`, `image_page`, or `mixed_page`.

If the PDF has no image pages, skip Stages 2–4 and jump directly to Stage 5 (summary).

### Stage 2 — Page Rendering

Each image-bearing page is rendered to a full-page PNG (preserves spatial layout):

```python
mat = fitz.Matrix(2.0, 2.0)    # ~144 DPI
pix = page.get_pixmap(matrix=mat)
png_bytes = pix.tobytes("png")
```

Full-page rendering is used instead of per-XREF extraction so that column alignment, table structure, and context between adjacent text and images is preserved.

### Stage 3 — Vision Transcription (`llama3.2-vision:90b`)

```
POST http://localhost:11434/api/chat
{
  "model": "llama3.2-vision:90b",
  "stream": false,
  "messages": [{
    "role": "user",
    "content": "<see prompt below>",
    "images": ["<base64 PNG>"]
  }]
}
```

**Prompt:**
```
You are a forensic document analyst and OCR assistant. This image is a financial
document — it may be a bank statement, cancelled check, wire transfer record, or
invoice. Your task:

1. Extract ALL text exactly as it appears, including account numbers, routing
   numbers, dates, dollar amounts, payee names, memo fields, and any signatures
   or handwritten annotations.
2. Preserve the structure: use whitespace and line breaks to reflect the original
   layout. Reproduce tables as tab-separated rows.
3. Do not interpret, summarize, or omit any text.
4. If a field is partially legible, include it with a [?] marker.

Return only the extracted text. No preamble or commentary.
```

- Concurrency cap: **3 simultaneous vision requests** (asyncio semaphore)
- Per-page timeout: **180 seconds**
- Fallback: if `llama3.2-vision:90b` is unavailable, retry with `llama3.2-vision:11b`

### Stage 4 — Invisible Text Embedding

Using PyMuPDF's PDF invisible text rendering mode (TR3):

```python
page.insert_text(
    point=fitz.Point(0, page.rect.height),
    text=extracted_text,
    fontsize=1,
    render_mode=3,     # PDF spec TR3 — text present in stream, not painted
    color=(0, 0, 0),
)
```

Output saved to `/workspace/<filename>_forensics.pdf`.

### Stage 5 — Summary Generation (`qwen3:32b`)

All text (OCR-extracted + native) is sent to `qwen3:32b`:

```
You are a financial forensics analyst. Produce a structured summary including:
- Document type
- Issuing institution or vendor
- Account holder name(s)
- Account/reference numbers (show only last 4 digits)
- Date range or transaction date
- Totals, balances, or invoice amounts
- Top 10 individual transactions or line items by amount
- Any anomalies, handwritten annotations, or irregularities

Be factual and precise. Use bullet points.
```

---

## Edge Case Handling

| Scenario | Behavior |
|---|---|
| PDF has no images, only text | Skip Stages 2–4; summarize existing text |
| Mixed PDF | Image pages through vision; text pages use native extraction |
| Vision model timeout | Mark page `transcription_failed`; continue remaining pages |
| Corrupt or password-protected PDF | Return structured MCP error |
| PDF > 50 pages | Process in batches of 10; emit MCP log progress messages |
| `llama3.2-vision:90b` unavailable | Fallback to `:11b` with warning in response |
| Nonce already seen | HTTP 401; client should generate a new nonce and retry |
| Timestamp drift > 60s | HTTP 401; check system clock synchronization |

---

## Development and Deployment Workflow

### Initial Setup

```bash
# 1. Sync source to DGX Spark
rsync -av ./forensics-pdf-mcp/ claude@dgx-spark-claude:~/projects/claw-my-spark/forensics-pdf-mcp/

# 2. Build and start
ssh claude@dgx-spark-claude "cd ~/projects/claw-my-spark && \
  docker compose build forensics-pdf-mcp && \
  docker compose up -d forensics-pdf-mcp"

# 3. Initialize the database
ssh claude@dgx-spark-claude "docker exec forensics-pdf-mcp python3 -c \
  'from auth.key_registry import KeyRegistry; KeyRegistry(\"/data/keys.db\").init()'"

# 4. Generate client keypair (on local machine)
python3 forensics-pdf-mcp/admin/keygen.py --client-name "$(hostname)" \
  --output-dir ~/.forensics-pdf-mcp/

# 5. Register public key with server
docker exec -i forensics-pdf-mcp python3 admin/register_client.py \
  --key-id "$(cat ~/.forensics-pdf-mcp/key_id.txt)" \
  --client-name "$(hostname)" \
  --public-key "$(cat ~/.forensics-pdf-mcp/public_key.pem)"
```

### Iterative Development

```bash
rsync -av ./forensics-pdf-mcp/ claude@dgx-spark-claude:~/projects/claw-my-spark/forensics-pdf-mcp/
ssh claude@dgx-spark-claude "cd ~/projects/claw-my-spark && \
  docker compose build forensics-pdf-mcp && \
  docker compose restart forensics-pdf-mcp"
docker logs -f forensics-pdf-mcp
```

### Test the Pipeline Directly

```bash
# Copy a test PDF into the workspace
scp test.pdf claude@dgx-spark-claude:/tmp/forensics-pdf-mcp/

# Run the pipeline
ssh claude@dgx-spark-claude "docker exec forensics-pdf-mcp python3 -c \
  'from pdf_processor import process_pdf; import asyncio; \
   r = asyncio.run(process_pdf(\"/workspace/test.pdf\")); print(r[\"summary\"])'"
```

### Test the HTTP Client

```bash
python3 - <<'EOF'
import asyncio
from forensics-pdf-mcp.client.forensics_client import ForensicsClient

async def main():
    async with ForensicsClient() as c:
        result = await c.process_pdf_file("test.pdf")
        print(result.summary)
        result.save_enriched("test_forensics.pdf")

asyncio.run(main())
EOF
```

---

## Models Used

| Model | Role | Why |
|---|---|---|
| `llama3.2-vision:90b` | Image-to-text transcription | Highest on-device accuracy; critical for financial figures and account numbers |
| `qwen3:32b` | Document summarization | Strong structured reasoning; appropriate for forensic analysis tasks |

Both are already pulled in the local Ollama instance. No internet required at runtime.

---

## Security Considerations

| Area | Detail |
|---|---|
| Private key storage | Stored only on the client machine at `~/.forensics-pdf-mcp/private_key.pem` (chmod 600). Never sent to the server. |
| Key transport | Public keys are registered out-of-band via `docker exec` (requires SSH access to DGX). |
| Network exposure | HTTP port (18790) is on the LAN only. Not routed through ExpressVPN. Not exposed to the internet by the existing stack. |
| Financial data | All LLM inference is on-device. Extracted text and PDFs never leave the local network. |
| Workspace files | `/tmp/forensics-pdf-mcp` on host — cleared on reboot. The `forensics-pdf-keys` volume persists key registrations across container restarts. |
| Key revocation | Set `active = 0` in SQLite immediately stops accepting requests from that key without container restart. |
