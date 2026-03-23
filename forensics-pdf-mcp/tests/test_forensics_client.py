"""
Tests for client/forensics_client.py — Python MCP client.

The server's /mcp endpoint is mocked with respx so no running server
is required.
"""

import base64
import json
import uuid

import httpx
import pytest
import respx

from auth.signing import verify_signature
from client.forensics_client import ForensicsClient, ProcessResult

SERVER_URL = "http://test-server:18790"
MCP_URL = f"{SERVER_URL}/mcp/"

SAMPLE_RESULT = {
    "summary": "Bank statement for account ending 1234. Balance: $1,500.00.",
    "enriched_pdf_base64": base64.b64encode(b"%PDF-1.4 enriched").decode(),
    "enriched_pdf_path": "/workspace/test_forensics.pdf",
    "had_embedded_images": True,
    "pages_processed": 3,
    "images_transcribed": 2,
}

# SSE-formatted final tool result
_TOOL_SSE = (
    "event: message\n"
    "data: " + json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(SAMPLE_RESULT)}]},
    }) + "\n\n"
)

# SSE-formatted initialize response
_INIT_SSE = (
    "event: message\n"
    "data: " + json.dumps({
        "jsonrpc": "2.0",
        "id": 0,
        "result": {"protocolVersion": "2025-03-26", "capabilities": {}},
    }) + "\n\n"
)

_INIT_HEADERS = {
    "content-type": "text/event-stream",
    "mcp-session-id": "test-session-123",
}
_TOOL_HEADERS = {"content-type": "text/event-stream"}


def _sse_response(text, headers):
    return httpx.Response(200, text=text, headers=headers)


def _mock_mcp(route):
    """Side-effect: return init response for the first call, tool response for subsequent."""
    call_count = 0

    def handler(request):
        nonlocal call_count
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return _sse_response(_INIT_SSE, _INIT_HEADERS)
        return _sse_response(_TOOL_SSE, _TOOL_HEADERS)

    route.mock(side_effect=handler)


def make_client(ec_keypair):
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
        _mock_mcp(respx.post(MCP_URL))
        async with make_client(ec_keypair) as client:
            result = await client.process_pdf_bytes(b"%PDF-1.4 test", "test.pdf")

    assert isinstance(result, ProcessResult)
    assert result.summary == SAMPLE_RESULT["summary"]
    assert result.had_embedded_images is True
    assert result.pages_processed == 3
    assert result.images_transcribed == 2
    assert result.enriched_pdf == b"%PDF-1.4 enriched"


@pytest.mark.asyncio
async def test_auth_headers_present(ec_keypair):
    """Every request carries all four X-Auth-* headers."""
    tool_request = None

    def handler(request):
        nonlocal tool_request
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return _sse_response(_INIT_SSE, _INIT_HEADERS)
        tool_request = request
        return _sse_response(_TOOL_SSE, _TOOL_HEADERS)

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=handler)
        async with make_client(ec_keypair) as client:
            await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    assert tool_request is not None
    for header in ["X-Auth-Key-ID", "X-Auth-Timestamp", "X-Auth-Nonce", "X-Auth-Signature"]:
        assert header in tool_request.headers, f"Missing header: {header}"


@pytest.mark.asyncio
async def test_signature_is_valid_over_request_body(ec_keypair):
    """The X-Auth-Signature in the request is a valid ECDSA signature over the body."""
    key_id, private_pem, public_pem = ec_keypair
    tool_request = None

    def handler(request):
        nonlocal tool_request
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return _sse_response(_INIT_SSE, _INIT_HEADERS)
        tool_request = request
        return _sse_response(_TOOL_SSE, _TOOL_HEADERS)

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=handler)
        async with ForensicsClient(
            server_url=SERVER_URL, key_id=key_id, private_key_pem=private_pem
        ) as client:
            await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    h = tool_request.headers
    verify_signature(
        public_pem,
        h["X-Auth-Key-ID"],
        h["X-Auth-Timestamp"],
        h["X-Auth-Nonce"],
        tool_request.content,
        h["X-Auth-Signature"],
    )


@pytest.mark.asyncio
async def test_save_enriched_writes_file(ec_keypair, tmp_path):
    """save_enriched writes the enriched PDF bytes to the given path."""
    with respx.mock:
        _mock_mcp(respx.post(MCP_URL))
        async with make_client(ec_keypair) as client:
            result = await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")

    out_path = tmp_path / "output.pdf"
    result.save_enriched(out_path)
    assert out_path.exists()
    assert out_path.read_bytes() == b"%PDF-1.4 enriched"


@pytest.mark.asyncio
async def test_server_error_raises(ec_keypair):
    """An HTTP 500 from the server raises HTTPStatusError."""
    def handler(request):
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return _sse_response(_INIT_SSE, _INIT_HEADERS)
        return httpx.Response(500, json={"error": "internal"})

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=handler)
        async with make_client(ec_keypair) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.process_pdf_bytes(b"%PDF-1.4", "doc.pdf")


@pytest.mark.asyncio
async def test_progress_callback_fired(ec_keypair):
    """Progress notifications in the SSE stream are forwarded to the callback."""
    progress_events = []

    progress_sse = (
        "event: message\n"
        "data: " + json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": "tok", "progress": 1, "total": 3},
        }) + "\n\n"
        "event: message\n"
        "data: " + json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": "tok", "progress": 2, "total": 3},
        }) + "\n\n"
        "event: message\n"
        "data: " + json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": json.dumps(SAMPLE_RESULT)}]},
        }) + "\n\n"
    )

    def handler(request):
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return _sse_response(_INIT_SSE, _INIT_HEADERS)
        return _sse_response(progress_sse, _TOOL_HEADERS)

    with respx.mock:
        respx.post(MCP_URL).mock(side_effect=handler)
        async with make_client(ec_keypair) as client:
            await client.process_pdf_bytes(
                b"%PDF-1.4", "doc.pdf",
                progress_callback=lambda done, total: progress_events.append((done, total)),
            )

    assert progress_events == [(1, 3), (2, 3)]
