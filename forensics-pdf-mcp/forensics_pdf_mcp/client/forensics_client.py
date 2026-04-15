"""
Self-contained Python client for forensics-pdf-mcp.

Does NOT import from the server's auth/ module.
Communicates via raw JSON-RPC POST to /mcp (MCP Streamable HTTP transport).

Quick start:
    from forensics_pdf_mcp.client.forensics_client import ForensicsClient

    async with ForensicsClient(
        server_url="http://dgx-spark-claude:18790",
        key_id=open("~/.forensics-pdf-mcp/key_id.txt").read().strip(),
        private_key_pem=open("~/.forensics-pdf-mcp/private_key.pem").read(),
    ) as client:
        result = await client.process_pdf_file("statement.pdf")
        print(result.summary)
        result.save_enriched("statement_forensics.pdf")

Configuration is read from ~/.forensics-pdf-mcp/ by default, or overridden
via constructor args or environment variables:
    FORENSICS_SERVER_URL
    FORENSICS_KEY_ID          (UUID string)
    FORENSICS_PRIVATE_KEY     (PEM content)
"""

import base64
import hashlib
import json
import logging
import os
import pathlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


# ---------------------------------------------------------------------------
# Signing helpers (self-contained copy of auth/signing.py logic)
# ---------------------------------------------------------------------------

def _build_canonical(key_id: str, timestamp: str, nonce: str, body_bytes: bytes) -> bytes:
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    return f"{key_id}\n{timestamp}\n{nonce}\n{body_hash}".encode("utf-8")


def _sign_request(private_key_pem: str, key_id: str, body_bytes: bytes) -> dict:
    """Return the four X-Auth-* request headers."""
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
        password=None,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    nonce = str(uuid.uuid4())
    canonical = _build_canonical(key_id, timestamp, nonce, body_bytes)
    sig_der = private_key.sign(canonical, ec.ECDSA(hashes.SHA256()))
    sig_b64 = base64.urlsafe_b64encode(sig_der).decode()
    return {
        "X-Auth-Key-ID": key_id,
        "X-Auth-Timestamp": timestamp,
        "X-Auth-Nonce": nonce,
        "X-Auth-Signature": sig_b64,
    }


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    summary: str
    enriched_pdf: bytes
    had_embedded_images: bool
    pages_processed: int
    images_transcribed: int
    enriched_pdf_path: str

    def save_enriched(self, path: str | pathlib.Path) -> None:
        """Write the enriched PDF bytes to disk."""
        pathlib.Path(path).write_bytes(self.enriched_pdf)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ForensicsClient:
    """
    Async context manager for the forensics-pdf-mcp HTTP MCP server.

    Args:
        server_url:      Base URL of the server, e.g. http://dgx-spark-claude:18790
        key_id:          UUID of the registered key
        private_key_pem: PEM-encoded PKCS8 EC private key
        config_dir:      Directory to read defaults from (default: ~/.forensics-pdf-mcp/)
        timeout:         Request timeout in seconds (default: 300)
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        config_dir: str | pathlib.Path = "~/.forensics-pdf-mcp/",
        timeout: float = 700.0,
    ) -> None:
        config_dir = pathlib.Path(config_dir).expanduser()

        self._server_url = (
            server_url
            or os.environ.get("FORENSICS_SERVER_URL")
            or self._read_config(config_dir, "server_url", "http://dgx-spark-claude:18790")
        )

        self._key_id = (
            key_id
            or os.environ.get("FORENSICS_KEY_ID")
            or self._read_file(config_dir / "key_id.txt")
        )

        self._private_key_pem = (
            private_key_pem
            or os.environ.get("FORENSICS_PRIVATE_KEY")
            or self._read_file(config_dir / "private_key.pem")
        )

        if not self._key_id or not self._private_key_pem:
            raise ValueError(
                "key_id and private_key_pem are required. "
                "Run admin/keygen.py to generate a keypair."
            )

        self._http = httpx.AsyncClient(
            base_url=self._server_url.rstrip("/"),
            timeout=timeout,
        )
        self._session_id: Optional[str] = None

    # --- Context manager ---

    async def __aenter__(self) -> "ForensicsClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self._http.aclose()

    # --- Public API ---

    async def process_pdf_file(
        self,
        path: str | pathlib.Path,
        progress_callback=None,
        status_callback=None,
    ) -> ProcessResult:
        """Submit a local PDF file for forensic processing.

        Args:
            path:              Path to the PDF file.
            progress_callback: Optional callable(done, total) — fired after each
                               image page is OCR'd.
            status_callback:   Optional callable(message) — fired for text status
                               updates (classification complete, summary started, etc.).
        """
        path = pathlib.Path(path)
        return await self.process_pdf_bytes(
            path.read_bytes(),
            filename=path.name,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

    async def process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        progress_callback=None,
        status_callback=None,
    ) -> ProcessResult:
        """Submit raw PDF bytes for forensic processing.

        Args:
            pdf_bytes:         Raw PDF content.
            filename:          Original filename (used server-side for output naming).
            progress_callback: Optional callable(done, total) — see process_pdf_file.
            status_callback:   Optional callable(message) — see process_pdf_file.
        """
        result_dict = await self._call_tool(
            "process_pdf",
            {
                "file_base64": base64.b64encode(pdf_bytes).decode(),
                "filename": filename,
            },
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        if "error" in result_dict:
            raise RuntimeError(f"Server error: {result_dict['error']}")
        return ProcessResult(
            summary=result_dict["summary"],
            enriched_pdf=base64.b64decode(result_dict["enriched_pdf_base64"]),
            had_embedded_images=result_dict["had_embedded_images"],
            pages_processed=result_dict["pages_processed"],
            images_transcribed=result_dict["images_transcribed"],
            enriched_pdf_path=result_dict["enriched_pdf_path"],
        )

    async def submit_sync_job(
        self,
        google_credentials_json: str,
        folders: list[dict[str, str]],
    ) -> dict:
        """Submit a Drive sync job for background processing.

        Uses the REST endpoint (not MCP/SSE) for instant response.

        Args:
            google_credentials_json: Service-account JSON string.
            folders: List of {"section": "HOA", "folder_id": "..."} dicts.

        Returns dict with job_id and status, or error info if busy.
        """
        body = json.dumps({
            "google_credentials_json": google_credentials_json,
            "folders": folders,
        }).encode()
        auth_headers = _sign_request(self._private_key_pem, self._key_id, body)
        auth_headers["Content-Type"] = "application/json"
        resp = await self._http.post("/jobs/submit", content=body, headers=auth_headers)
        resp.raise_for_status()
        return resp.json()

    async def get_job_status(self, job_id: str) -> dict:
        """Poll the status of a sync job.

        Uses the REST endpoint (not MCP/SSE) for instant response.

        Args:
            job_id: The job ID returned by submit_sync_job.

        Returns dict with job status, progress, and errors.
        """
        # GET request — sign an empty body
        auth_headers = _sign_request(self._private_key_pem, self._key_id, b"")
        resp = await self._http.get(f"/jobs/{job_id}", headers=auth_headers)
        resp.raise_for_status()
        return resp.json()

    # --- Internal ---

    async def _initialize_session(self) -> None:
        """Send MCP initialize handshake and store the session ID."""
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "forensics-pdf-mcp-client", "version": "0.1.0"},
            },
        }).encode()
        auth_headers = _sign_request(self._private_key_pem, self._key_id, payload)
        auth_headers["Content-Type"] = "application/json"
        auth_headers["Accept"] = "application/json, text/event-stream"
        resp = await self._http.post("/mcp/", content=payload, headers=auth_headers)
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id")

    async def _call_tool(
        self, tool_name: str, arguments: dict,
        progress_callback=None, status_callback=None,
        _retried: bool = False,
    ) -> dict:
        """Call an MCP tool, streaming SSE events as they arrive.

        Lazily initializes the MCP session on first use.

        Progress notifications (``notifications/progress``) are forwarded to
        *progress_callback(done, total)* if provided.  The final JSON-RPC
        result is returned once the stream ends.
        """
        if self._session_id is None:
            await self._initialize_session()
        progress_token = str(uuid.uuid4())
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {"progressToken": progress_token},
            },
        }).encode()

        auth_headers = _sign_request(self._private_key_pem, self._key_id, payload)
        auth_headers["Content-Type"] = "application/json"
        auth_headers["Accept"] = "application/json, text/event-stream"
        if self._session_id:
            auth_headers["Mcp-Session-Id"] = self._session_id

        async with self._http.stream(
            "POST", "/mcp/", content=payload, headers=auth_headers
        ) as resp:
            if resp.status_code == 401 and not _retried:
                # Session may have expired — re-initialize and retry once
                await resp.aread()
                logging.getLogger(__name__).warning(
                    "Got 401, re-initializing MCP session and retrying"
                )
                await self._initialize_session()
                return await self._call_tool(
                    tool_name, arguments,
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                    _retried=True,
                )
            resp.raise_for_status()
            result_data = None
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())

                # Progress notification — fire callback and keep reading
                if event.get("method") == "notifications/progress":
                    if progress_callback:
                        params = event.get("params", {})
                        progress_callback(
                            params.get("progress", 0),
                            params.get("total", 0),
                        )
                    continue

                # Status/log notification from ctx.info()
                if event.get("method") == "notifications/message":
                    if status_callback:
                        params = event.get("params", {})
                        data = params.get("data", "")
                        # FastMCP wraps the message as {"msg": "...", "extra": ...}
                        if isinstance(data, dict):
                            data = data.get("msg", str(data))
                        status_callback(data)
                    continue

                # Final result or error
                if "result" in event or "error" in event:
                    result_data = event
                    break

        if result_data is None:
            raise RuntimeError("No result received from MCP server")
        if "error" in result_data:
            raise RuntimeError(f"MCP error: {result_data['error']}")

        content = result_data.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            text = content[0]["text"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Tool returned non-JSON text (likely an error message)
                is_error = result_data.get("result", {}).get("isError", False)
                if is_error:
                    raise RuntimeError(f"Tool error: {text}")
                raise RuntimeError(f"Tool returned non-JSON text: {text!r}")
        raise RuntimeError(f"Unexpected MCP response structure: {result_data}")

    @staticmethod
    def _read_file(path: pathlib.Path) -> Optional[str]:
        try:
            return path.read_text().strip()
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_config(config_dir: pathlib.Path, key: str, default: str) -> str:
        config_file = config_dir / "config.json"
        try:
            config = json.loads(config_file.read_text())
            return config.get(key, default)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
