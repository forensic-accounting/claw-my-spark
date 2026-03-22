"""
Integration tests for http_server.py — FastAPI app with ECDSA auth middleware.

Uses httpx.AsyncClient with ASGITransport for fully async in-process testing,
which correctly handles BaseHTTPMiddleware without sync/async context issues.
"""

import base64
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from httpx import ASGITransport

from auth.key_registry import KeyRegistry
from auth.middleware import ECDSAAuthMiddleware
from auth.signing import sign_request
from fastapi import FastAPI
from fastmcp import FastMCP


def make_test_app(registry: KeyRegistry):
    """Build a minimal test version of the FastAPI app with the given registry."""
    mcp = FastMCP("test-forensics")

    @mcp.tool()
    async def process_pdf(
        file_path: str | None = None,
        file_base64: str | None = None,
        filename: str | None = None,
    ) -> str:
        return json.dumps({
            "summary": "Test summary",
            "enriched_pdf_base64": base64.b64encode(b"%PDF-1.4 test").decode(),
            "enriched_pdf_path": "/workspace/test_forensics.pdf",
            "had_embedded_images": False,
            "pages_processed": 1,
            "images_transcribed": 0,
        })

    app = FastAPI()
    app.add_middleware(ECDSAAuthMiddleware, registry=registry)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.mount("/mcp", mcp.http_app(path="/"))
    return app


@pytest.fixture
def app_with_registry(tmp_db, ec_keypair):
    """Test app + populated registry. Returns (app, registry, key_id, private_pem)."""
    key_id, private_pem, public_pem = ec_keypair
    registry = KeyRegistry(tmp_db)
    registry.init_db()
    registry.register_key(key_id, "test-client", public_pem)
    app = make_test_app(registry)
    return app, registry, key_id, private_pem


def _signed_headers(key_id: str, private_pem: str, payload: bytes) -> dict:
    headers = sign_request(private_pem, key_id, payload)
    headers["Content-Type"] = "application/json"
    return headers


# ---------------------------------------------------------------------------
# Tests — all async using httpx.AsyncClient + ASGITransport
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_no_auth(app_with_registry):
    """GET /health returns 200 without any authentication headers."""
    app, *_ = app_with_registry
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_mcp_requires_auth(app_with_registry):
    """POST /mcp without auth headers returns 401."""
    app, _, key_id, private_pem = app_with_registry
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", content=payload, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_mcp_missing_individual_header_rejected(app_with_registry):
    """Each missing X-Auth-* header individually causes 401."""
    app, _, key_id, private_pem = app_with_registry
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    headers = _signed_headers(key_id, private_pem, payload)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for header_to_drop in ["X-Auth-Key-ID", "X-Auth-Timestamp", "X-Auth-Nonce", "X-Auth-Signature"]:
            partial = {k: v for k, v in headers.items() if k != header_to_drop}
            resp = await client.post("/mcp", content=payload, headers=partial)
            assert resp.status_code == 401, f"Expected 401 when {header_to_drop} is missing"


@pytest.mark.asyncio
async def test_expired_timestamp_rejected(app_with_registry):
    """A request with a timestamp 120 seconds in the past is rejected."""
    app, _, key_id, private_pem = app_with_registry
    payload = b"{}"
    headers = sign_request(private_pem, key_id, payload)
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    headers["X-Auth-Timestamp"] = old_ts
    headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", content=payload, headers=headers)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_nonce_replay_rejected(app_with_registry):
    """The second request with the same nonce is rejected."""
    app, _, key_id, private_pem = app_with_registry
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    headers = _signed_headers(key_id, private_pem, payload)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/mcp", content=payload, headers=headers)
        assert resp1.status_code != 401, f"First request unexpectedly got 401: {resp1.text}"
        resp2 = await client.post("/mcp", content=payload, headers=headers)
        assert resp2.status_code == 401


@pytest.mark.asyncio
async def test_mcp_accepts_valid_auth(app_with_registry):
    """A correctly signed request passes authentication and reaches FastMCP."""
    app, _, key_id, private_pem = app_with_registry
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    headers = _signed_headers(key_id, private_pem, payload)
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/mcp", content=payload, headers=headers)
    assert resp.status_code != 401, f"Valid auth got 401: {resp.text}"
