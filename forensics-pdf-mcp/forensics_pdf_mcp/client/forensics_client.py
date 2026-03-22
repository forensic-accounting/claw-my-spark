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
        timeout: float = 300.0,
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
        await self._initialize_session()
        return self

    async def __aexit__(self, *_) -> None:
        await self._http.aclose()

    # --- Public API ---

    async def process_pdf_file(self, path: str | pathlib.Path) -> ProcessResult:
        """Submit a local PDF file for forensic processing."""
        path = pathlib.Path(path)
        return await self.process_pdf_bytes(path.read_bytes(), filename=path.name)

    async def process_pdf_bytes(
        self, pdf_bytes: bytes, filename: str = "document.pdf"
    ) -> ProcessResult:
        """Submit raw PDF bytes for forensic processing."""
        result_dict = await self._call_tool(
            "process_pdf",
            {
                "file_base64": base64.b64encode(pdf_bytes).decode(),
                "filename": filename,
            },
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
        # Consume the initialize result (required by some server implementations)
        self._parse_mcp_response(resp)

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }).encode()

        auth_headers = _sign_request(self._private_key_pem, self._key_id, payload)
        auth_headers["Content-Type"] = "application/json"
        auth_headers["Accept"] = "application/json, text/event-stream"
        if self._session_id:
            auth_headers["Mcp-Session-Id"] = self._session_id

        resp = await self._http.post("/mcp/", content=payload, headers=auth_headers)
        resp.raise_for_status()

        data = self._parse_mcp_response(resp)
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")

        # MCP tool result is in data["result"]["content"][0]["text"]
        content = data.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        raise RuntimeError(f"Unexpected MCP response structure: {data}")

    @staticmethod
    def _parse_mcp_response(resp: httpx.Response) -> dict:
        """Parse a JSON or SSE response from the MCP server."""
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            # Extract JSON from the first `data:` line in the SSE stream
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[len("data:"):].strip())
            raise RuntimeError(f"No data line in SSE response: {resp.text[:200]}")
        return resp.json()

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
