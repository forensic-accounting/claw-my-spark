# forensics-pdf-mcp

An MCP (Model Context Protocol) server that processes PDF files through a vision-based OCR pipeline using locally hosted LLMs on the DGX Spark. Designed for financial forensics — bank statements, scanned checks, invoices, and similar documents.

---

## Overview

`forensics-pdf-mcp` accepts a PDF file, detects whether it contains embedded images (i.e., is scanned rather than digitally generated), transcribes those images using the `llama3.2-vision:90b` model, embeds the extracted text invisibly back into the PDF as a searchable layer, and returns both the enriched PDF and a structured summary of the document's contents.

The entire service runs as a Docker container on the DGX Spark alongside the existing Ollama stack. It communicates with Claude Code via the MCP stdio transport tunneled over SSH.

---

## Architecture

```
Claude Code (local)
    │
    │  MCP stdio over SSH
    ▼
ssh claude@dgx-spark-claude
    │
    │  docker exec / stdio
    ▼
┌─────────────────────────────────┐
│   forensics-pdf-mcp container   │
│                                 │
│   server.py  (MCP stdio)        │
│       │                         │
│   pdf_processor.py              │
│       │                         │
│   ollama_client.py              │
│       │                         │
└───────┼─────────────────────────┘
        │  HTTP  (host network)
        ▼
┌───────────────────┐
│  ollama container │
│  llama3.2-vision  │
│  :90b / :11b      │
│  qwen3:32b        │
└───────────────────┘
```

The container runs on the Docker host network so it can reach Ollama at `http://localhost:11434` without any additional networking configuration.

---

## Directory Layout

```
forensics-pdf-mcp/
├── DESIGN.md                  # This document
├── Dockerfile                 # Container image definition
├── docker-compose.yaml        # Service definition (integrates with existing stack)
├── requirements.txt           # Python dependencies
├── server.py                  # MCP server entrypoint (stdio transport)
├── pdf_processor.py           # Core PDF processing pipeline
└── ollama_client.py           # Async Ollama API wrapper
```

---

## Container Design

### Base Image

`python:3.12-slim` — minimal footprint, matches the Python version already on the DGX Spark host.

### Build

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "server.py"]
```

### Runtime

| Setting | Value |
|---|---|
| Network mode | `host` |
| Restart policy | `unless-stopped` |
| Volumes | `/tmp/forensics-pdf-mcp` mounted at `/workspace` for temp file I/O |
| Environment | `OLLAMA_BASE_URL=http://localhost:11434` |

### docker-compose.yaml (addition to existing stack)

The service is added as a new entry in the existing `docker-compose.yaml` so it shares the same lifecycle as Ollama:

```yaml
forensics-pdf-mcp:
  build:
    context: ./forensics-pdf-mcp
    dockerfile: Dockerfile
  container_name: forensics-pdf-mcp
  restart: unless-stopped
  network_mode: host
  volumes:
    - /tmp/forensics-pdf-mcp:/workspace
  environment:
    OLLAMA_BASE_URL: "http://localhost:11434"
    VISION_MODEL: "llama3.2-vision:90b"
    SUMMARY_MODEL: "qwen3:32b"
    WORKSPACE_DIR: "/workspace"
  depends_on:
    ollama:
      condition: service_healthy
  stdin_open: true
  tty: false
```

`stdin_open: true` keeps the container alive waiting for MCP stdio connections. Claude Code connects by running `docker exec -i forensics-pdf-mcp python3 server.py` via SSH.

---

## MCP Interface

### Transport

stdio, tunneled through SSH:

```json
{
  "mcpServers": {
    "forensics-pdf-mcp": {
      "command": "ssh",
      "args": [
        "claude@dgx-spark-claude",
        "docker", "exec", "-i", "forensics-pdf-mcp",
        "python3", "server.py"
      ]
    }
  }
}
```

### Tool: `process_pdf`

The server exposes a single tool.

**Inputs** (one of the two file inputs is required):

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | one of | Absolute path to a PDF file accessible inside the container's `/workspace` volume |
| `file_base64` | string | one of | Base64-encoded PDF file content |
| `filename` | string | when using `file_base64` | Original filename, used for naming the output file |

**Outputs:**

| Field | Type | Description |
|---|---|---|
| `summary` | string | Structured narrative summary of the document |
| `enriched_pdf_base64` | string | Base64-encoded PDF with invisible OCR text layer embedded |
| `enriched_pdf_path` | string | Path inside `/workspace` where the enriched PDF was saved |
| `had_embedded_images` | bool | Whether the PDF contained image-based pages |
| `pages_processed` | int | Total number of pages in the document |
| `images_transcribed` | int | Number of pages processed through the vision model |

---

## Processing Pipeline

### Stage 1 — Image Detection

Open the PDF with PyMuPDF (`fitz`). For each page:

1. Call `page.get_images(full=True)` — returns all image XREFs embedded on the page.
2. Attempt `page.get_text()` — if this returns empty or near-empty text alongside present images, the page is image-based.
3. Build a per-page classification: `text_page`, `image_page`, or `mixed_page`.

A PDF is considered image-bearing if **any** page is classified as `image_page` or `mixed_page`.

If the PDF has no embedded images and has extractable text, the pipeline skips to Stage 4 (summary only).

### Stage 2 — Image Extraction and Rendering

Rather than extracting individual image XREFs (which may be partial regions or decorative elements), each image-bearing page is **rendered to a full-page PNG** using:

```python
mat = fitz.Matrix(2.0, 2.0)   # 2x scale = ~144 DPI, sufficient for OCR
pix = page.get_pixmap(matrix=mat)
png_bytes = pix.tobytes("png")
```

Rendering the full page preserves spatial relationships between text blocks, tables, and figures — which is critical for financial documents where column alignment and row context carry meaning.

### Stage 3 — Vision Transcription

For each rendered page image, an async HTTP POST is made to Ollama:

```
POST http://localhost:11434/api/chat
{
  "model": "llama3.2-vision:90b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "<prompt>",
      "images": ["<base64-encoded PNG>"]
    }
  ]
}
```

**Prompt**:
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

Pages are processed with a concurrency cap of **3 simultaneous vision requests** (asyncio semaphore) to avoid GPU memory contention on the 90b model.

Per-page timeout: **180 seconds** (the 90b model on a GB10 takes 30–90s per page depending on density).

### Stage 4 — Invisible Text Embedding

The original PDF is reopened in write mode. For each page that was transcribed:

```python
page.insert_text(
    point=fitz.Point(0, page.rect.height),  # bottom-left anchor
    text=extracted_text,
    fontsize=1,                              # 1pt — effectively invisible
    render_mode=3,                           # PDF invisible text mode (TR3)
    color=(0, 0, 0),
)
```

`render_mode=3` is the PDF specification's "invisible text" rendering mode — the text is present in the content stream and fully selectable/searchable but not painted on screen. This is the same technique used by Tesseract, OCRmyPDF, and Adobe Acrobat's OCR feature.

The enriched PDF is saved to `/workspace/<original_filename>_forensics.pdf`.

### Stage 5 — Summary Generation

All extracted text (plus any pre-existing extractable text from non-image pages) is concatenated and sent to `qwen3:32b` for a structured summary:

```
You are a financial forensics analyst. The following is OCR-extracted text from
a financial document. Produce a structured summary including:

- Document type (bank statement, check, invoice, wire transfer, etc.)
- Issuing institution or vendor name
- Account holder name(s) if present
- Account or reference numbers (redact last 4 digits if a full number is shown)
- Date range or transaction date
- Total amounts, balances, or invoice totals
- Key individual transactions or line items (top 10 by amount if more exist)
- Any anomalies, handwritten annotations, or irregularities noted

Be factual and precise. Use bullet points.
```

---

## Edge Case Handling

| Scenario | Behavior |
|---|---|
| PDF has no images, only text | Skip Stages 2–4; still generate summary from extracted text |
| Mixed PDF (some text pages, some image pages) | Image pages go through vision; text pages use native extraction; both feed summary |
| Multi-image page (e.g., a page with a check image and a header logo) | Full page render captures all elements in context |
| Vision model timeout on a page | Log warning, mark page as `transcription_failed`, continue with remaining pages |
| Corrupt or password-protected PDF | Return structured MCP error with diagnostic message |
| Very large PDF (>50 pages) | Process in batches of 10 pages; stream progress via MCP log messages |
| `llama3.2-vision:90b` not available | Automatically fall back to `llama3.2-vision:11b` with a warning in the response |

---

## Development and Deployment Workflow

### Initial Setup

1. **Write code locally** in `claw-my-spark/forensics-pdf-mcp/`
2. **Push or rsync to remote**:
   ```bash
   rsync -av ./forensics-pdf-mcp/ claude@dgx-spark-claude:~/projects/claw-my-spark/forensics-pdf-mcp/
   ```
3. **Build and start the container** on the remote:
   ```bash
   ssh claude@dgx-spark-claude "cd ~/projects/claw-my-spark && docker compose build forensics-pdf-mcp && docker compose up -d forensics-pdf-mcp"
   ```

### Iterative Development

- Edit code locally
- Rsync changed files
- Rebuild container: `docker compose build forensics-pdf-mcp && docker compose restart forensics-pdf-mcp`
- Tail logs: `docker logs -f forensics-pdf-mcp`

### Testing the Pipeline Directly

```bash
ssh claude@dgx-spark-claude \
  "docker exec -i forensics-pdf-mcp python3 -c \
   'from pdf_processor import process_pdf; import asyncio; print(asyncio.run(process_pdf(\"/workspace/test.pdf\")))'"
```

### Configuring Claude Code

Add to `~/.claude/settings.json` on the local workstation:

```json
{
  "mcpServers": {
    "forensics-pdf-mcp": {
      "command": "ssh",
      "args": [
        "claude@dgx-spark-claude",
        "docker", "exec", "-i", "forensics-pdf-mcp",
        "python3", "server.py"
      ]
    }
  }
}
```

Restart Claude Code after saving. The `forensics-pdf-mcp` tool will appear in the tool list.

---

## Models Used

| Model | Role | Why |
|---|---|---|
| `llama3.2-vision:90b` | Image-to-text transcription | Highest accuracy available on the DGX; critical for financial figures and account numbers |
| `qwen3:32b` | Document summarization | Strong reasoning and structured output; appropriate for analytical summary tasks |

Both models are already pulled and available in the local Ollama instance. No internet access is required at runtime.

---

## Security Considerations

- The container runs with no elevated privileges.
- No ports are exposed — communication is exclusively via stdio through SSH.
- PDF files processed via `file_base64` are written to `/workspace` inside the container, which maps to `/tmp/forensics-pdf-mcp` on the host. Files in `/tmp` are cleared on host reboot.
- Sensitive financial data (account numbers, transaction history) never leaves the local network — all LLM inference happens on-device.
- The Ollama endpoint (`localhost:11434`) is only accessible within the host network; it is not exposed to the internet by the existing stack configuration.
