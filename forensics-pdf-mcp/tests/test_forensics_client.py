"""
Tests for client/forensics_client.py — Python MCP client.

The server's /mcp endpoint is mocked with respx so no running server
is required.
"""

import base64
import hashlib
import json
import uuid

import httpx
import pytest
import respx
from cryptography.exceptions import InvalidSignature

from auth.signing import verify_signature
from client.forensics_client import ForensicsClient, ProcessResult

SERVER_URL = "http://test-server:18790"
MCP_URL = f"{SERVER_URL}/mcp"

SAMPLE_RESULT = {
    "summary": "Bank statement for account ending 1234. Balance: $1,500.00.",
    "enriched_pdf_base64": base64.b64encode(b"%PDF-1.4 enriched").decode(),
    "enriched_pdf_path": "/workspace/test_forensics.pdf",
    "had_embedded_images": True,
    "pages_processed": 3,
    "images_transcribed": 2,
}

MCP_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [{"type": "text", "text": json.dumps(SAMPLE_RESULT)}]
    },
}


def make_client(ec_keypair, tmp_path):
    key_id, private_pem, _ = ec_keypair
    return ForensicsClient(
        server_url=SERVER_URL,
        key_id=key_id,
        private_key_pem=private_pem,
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_process_pdf_bytes_returns_result(ec_keypair, tmp_path):
    """process_pdf_bytes returns a properly populated ProcessResult."""
    with respx.mock:
        respx.post(MCP_URL).mock(return_value=httpx.Response(200, json=MCP_RESPONSE))
        async with make_client(ec_keypair, tmp_path) as client:
            result = await client.process_pdf_bytes(b"%PDF-1.4 test", "test.pdf")

    assert isinstance(result, ProcessResult)
    assert result.summary == SAMPLE_RESULT["summary"]
    assert result.had_embedded_images is True
    assert result.pages_processed == 3
    assert result.images_transcribed == 2
    assert result.enriched_pdf == b"%PDF-1.4 enriched"


@pytest.mark.asyncio
async def test_auth_headers_present(ec_keypair, tmp_path):
    """Every request carries all four X-Auth-* headers."""
    captured_request = None

    def capture(request):
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=MCP_RESPONSE)

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=capture)
        async with make_client(ec_keypair, tmp_path) as client:
            await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    assert captured_request is not None
    for header in ["X-Auth-Key-ID", "X-Auth-Timestamp", "X-Auth-Nonce", "X-Auth-Signature"]:
        assert header in captured_request.headers, f"Missing header: {header}"


@pytest.mark.asyncio
async def test_signature_is_valid_over_request_body(ec_keypair, tmp_path):
    """The X-Auth-Signature in the request is a valid ECDSA signature over the body."""
    key_id, private_pem, public_pem = ec_keypair
    captured_request = None

    def capture(request):
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=MCP_RESPONSE)

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=capture)
        async with ForensicsClient(
            server_url=SERVER_URL, key_id=key_id, private_key_pem=private_pem
        ) as client:
            await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    h = captured_request.headers
    # Should not raise
    verify_signature(
        public_pem,
        h["X-Auth-Key-ID"],
        h["X-Auth-Timestamp"],
        h["X-Auth-Nonce"],
        captured_request.content,
        h["X-Auth-Signature"],
    )


@pytest.mark.asyncio
async def test_save_enriched_writes_file(ec_keypair, tmp_path):
    """save_enriched writes the enriched PDF bytes to the given path."""
    with respx.mock:
        respx.post(MCP_URL).mock(return_value=httpx.Response(200, json=MCP_RESPONSE))
        async with make_client(ec_keypair, tmp_path) as client:
            result = await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    out_path = tmp_path / "output.pdf"
    result.save_enriched(out_path)
    assert out_path.exists()
    assert out_path.read_bytes() == b"%PDF-1.4 enriched"


@pytest.mark.asyncio
async def test_server_error_raises(ec_keypair, tmp_path):
    """An HTTP 500 from the server raises a RuntimeError."""
    with respx.mock:
        respx.post(MCP_URL).mock(return_value=httpx.Response(500, json={"error": "internal"}))
        async with make_client(ec_keypair, tmp_path) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")
