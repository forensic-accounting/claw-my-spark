# forensics-pdf-mcp

PDF forensics pipeline running on the DGX Spark. Accepts scanned financial documents (bank statements, checks, invoices), extracts embedded images via `llama3.2-vision:90b`, embeds the OCR text invisibly back into the PDF so it becomes searchable, and returns a structured forensic summary.

All HTTP requests are authenticated with ECDSA P-256 signatures — your private key never leaves your machine.

---

## How It Works

1. **Image detection** — PyMuPDF scans each page; pages with embedded images and no native text are flagged as scanned
2. **OCR** — each scanned page is rendered to PNG and sent to `llama3.2-vision:90b` via Ollama
3. **Text embedding** — extracted text is written back into the PDF using PDF invisible text mode (TR3), making the document full-text searchable in any viewer
4. **Summary** — all text is passed to `qwen3:32b` for a structured forensic summary (document type, account info, amounts, anomalies)

---

## Setup

### 1. Start the server (DGX Spark)

```bash
cd ~/projects/claw-my-spark
docker compose up -d forensics-pdf-mcp
```

Verify it's running:

```bash
curl http://dgx-spark-claude:18790/health
# {"status":"ok"}
```

### 2. Generate a client keypair (your local machine)

```bash
python3 forensics-pdf-mcp/admin/keygen.py \
    --client-name "$(hostname)" \
    --output-dir ~/.forensics-pdf-mcp/
```

This writes three files to `~/.forensics-pdf-mcp/`:
- `private_key.pem` — keep this secret, chmod 600
- `public_key.pem` — register this with the server
- `key_id.txt` — UUID that identifies your key

### 3. Register your public key with the server

```bash
ssh claude@dgx-spark-claude \
  "docker exec forensics-pdf-mcp python3 admin/register_client.py \
    --key-id $(cat ~/.forensics-pdf-mcp/key_id.txt) \
    --client-name $(hostname) \
    --public-key-file /dev/stdin" \
  < ~/.forensics-pdf-mcp/public_key.pem
```

Confirm registration:

```bash
ssh claude@dgx-spark-claude \
  "docker exec forensics-pdf-mcp python3 admin/register_client.py --list"
```

### 4. Install client dependencies (local)

```bash
pip install httpx cryptography
```

---

## Using the Python Client

### Basic usage

```python
import asyncio
from forensics_pdf_mcp.client.forensics_client import ForensicsClient

async def main():
    async with ForensicsClient() as client:
        result = await client.process_pdf_file("statement.pdf")

        print(result.summary)
        result.save_enriched("statement_forensics.pdf")

asyncio.run(main())
```

The client reads your key and server URL automatically from `~/.forensics-pdf-mcp/`. The enriched PDF is written to disk and the original is unchanged.

---

### Explicit configuration

```python
from forensics_pdf_mcp.client.forensics_client import ForensicsClient

async with ForensicsClient(
    server_url="http://dgx-spark-claude:18790",
    key_id=open("~/.forensics-pdf-mcp/key_id.txt").read().strip(),
    private_key_pem=open("~/.forensics-pdf-mcp/private_key.pem").read(),
) as client:
    result = await client.process_pdf_file("invoice.pdf")
```

Or via environment variables (useful in scripts and CI):

```bash
export FORENSICS_SERVER_URL=http://dgx-spark-claude:18790
export FORENSICS_KEY_ID=$(cat ~/.forensics-pdf-mcp/key_id.txt)
export FORENSICS_PRIVATE_KEY=$(cat ~/.forensics-pdf-mcp/private_key.pem)
```

---

### Submit raw bytes instead of a file path

```python
with open("statement.pdf", "rb") as f:
    pdf_bytes = f.read()

async with ForensicsClient() as client:
    result = await client.process_pdf_bytes(pdf_bytes, filename="statement.pdf")
```

---

### Process a directory of PDFs

```python
import asyncio
from pathlib import Path
from forensics_pdf_mcp.client.forensics_client import ForensicsClient

async def process_all(input_dir: str, output_dir: str):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    async with ForensicsClient() as client:
        for pdf in Path(input_dir).glob("*.pdf"):
            print(f"Processing {pdf.name}...")
            result = await client.process_pdf_file(pdf)

            enriched_path = output / f"{pdf.stem}_forensics.pdf"
            result.save_enriched(enriched_path)

            summary_path = output / f"{pdf.stem}_summary.txt"
            summary_path.write_text(result.summary)

            print(f"  Pages: {result.pages_processed}")
            print(f"  Images transcribed: {result.images_transcribed}")
            print(f"  Had scanned content: {result.had_embedded_images}")
            print(f"  Enriched PDF: {enriched_path}")
            print()

asyncio.run(process_all("./statements/", "./output/"))
```

---

### Inspect the result object

```python
async with ForensicsClient() as client:
    result = await client.process_pdf_file("check.pdf")

# Structured summary from qwen3:32b
print(result.summary)

# Save the enriched (searchable) PDF
result.save_enriched("check_searchable.pdf")

# Or work with the bytes directly (e.g. upload to S3)
enriched_bytes: bytes = result.enriched_pdf

# Metadata
print(f"Document had embedded images: {result.had_embedded_images}")
print(f"Total pages: {result.pages_processed}")
print(f"Pages OCR'd via vision model: {result.images_transcribed}")
print(f"Server-side path: {result.enriched_pdf_path}")
```

Example summary output for a bank statement:

```
• Document type: Bank statement
• Issuing institution: First National Bank
• Account holder: John A. Smith
• Account number: ****7842
• Date range: 2026-02-01 to 2026-02-28
• Ending balance: $4,312.00
• Key transactions:
  - 2026-02-03  Direct Deposit - Employer     +$3,500.00
  - 2026-02-07  Mortgage payment - CitiBank    -$1,850.00
  - 2026-02-14  Transfer to savings            -$500.00
  - 2026-02-19  ACH - Utility Company          -$142.00
  - 2026-02-22  POS - Grocery Store            -$87.43
• No anomalies or handwritten annotations detected
```

---

### Error handling

```python
import httpx
from forensics_pdf_mcp.client.forensics_client import ForensicsClient

async with ForensicsClient() as client:
    try:
        result = await client.process_pdf_file("document.pdf")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Authentication failed — check your key_id and private_key_pem")
        else:
            print(f"Server error: {e.response.status_code} {e.response.text}")
    except RuntimeError as e:
        # Server returned an error inside the MCP response
        print(f"Processing error: {e}")
    except FileNotFoundError:
        print("PDF file not found")
```

---

## Key Management

### List registered keys

```bash
ssh claude@dgx-spark-claude \
  "docker exec forensics-pdf-mcp python3 admin/register_client.py --list"
```

### Revoke a key

```bash
ssh claude@dgx-spark-claude \
  "docker exec forensics-pdf-mcp python3 admin/register_client.py \
    --revoke --key-id <uuid>"
```

Revocation is instant — the key is marked inactive in SQLite and all subsequent requests using it return 401 without requiring a container restart.

---

## Configuration

All server settings are environment variables in `docker-compose.yaml`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `VISION_MODEL` | `llama3.2-vision:90b` | Model for image OCR |
| `SUMMARY_MODEL` | `qwen3:32b` | Model for document summarization |
| `WORKSPACE_DIR` | `/workspace` | Temp directory for enriched PDFs |
| `DB_PATH` | `/data/keys.db` | SQLite database path |
| `FORENSICS_PDF_MCP_PORT` | `18790` | HTTP port (set in `.env`) |

If `llama3.2-vision:90b` is unavailable, the server automatically falls back to `llama3.2-vision:11b`.

---

## Rebuilding After Code Changes

```bash
# From your local machine
rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    ./forensics-pdf-mcp/ \
    claude@dgx-spark-claude:~/projects/claw-my-spark/forensics-pdf-mcp/

ssh claude@dgx-spark-claude "cd ~/projects/claw-my-spark && \
    docker compose build forensics-pdf-mcp && \
    docker compose restart forensics-pdf-mcp"
```

## Running Tests

```bash
ssh claude@dgx-spark-claude "cd ~/projects/claw-my-spark && \
    docker compose run --rm --entrypoint python3 forensics-pdf-mcp \
    -m pytest tests/ -v"
```

Tests run entirely in-container with no Ollama calls (all mocked with respx).
