"""
Starlette/FastAPI middleware that verifies ECDSA P-256 request signatures.

Every request to /mcp/* must carry four headers:
    X-Auth-Key-ID        UUID of the registered key
    X-Auth-Timestamp     ISO-8601 UTC timestamp (e.g. 2026-03-22T14:05:00.123456Z)
    X-Auth-Nonce         UUID v4, single-use
    X-Auth-Signature     base64url(ECDSA_SHA256(canonical_string))

Canonical string = "{key_id}\\n{timestamp}\\n{nonce}\\n{sha256(body).hex()}"

/health is exempt from authentication.

Body-reading note: Starlette caches await request.body() in request._body
on first call. The middleware reads it here; FastMCP re-reads from the same
cache without stream exhaustion.
"""

import logging
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .key_registry import KeyRegistry
from .signing import verify_signature

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/"}
TIMESTAMP_WINDOW_SECONDS = 120


class ECDSAAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, registry: KeyRegistry) -> None:
        super().__init__(app)
        self._registry = registry

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # --- Extract required headers ---
        key_id = request.headers.get("X-Auth-Key-ID")
        timestamp = request.headers.get("X-Auth-Timestamp")
        nonce = request.headers.get("X-Auth-Nonce")
        signature_b64 = request.headers.get("X-Auth-Signature")

        if not all([key_id, timestamp, nonce, signature_b64]):
            logger.warning("Auth rejected: missing headers (path=%s)", request.url.path)
            return JSONResponse(
                {"detail": "Missing authentication headers"},
                status_code=401,
            )

        # --- Validate timestamp window ---
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            delta = abs((datetime.now(timezone.utc) - ts).total_seconds())
            if delta > TIMESTAMP_WINDOW_SECONDS:
                logger.warning("Auth rejected: timestamp delta %.1fs (path=%s)", delta, request.url.path)
                return JSONResponse(
                    {"detail": "Timestamp out of acceptable window"},
                    status_code=401,
                )
        except (ValueError, AttributeError):
            logger.warning("Auth rejected: invalid timestamp format (path=%s)", request.url.path)
            return JSONResponse({"detail": "Invalid timestamp format"}, status_code=401)

        # --- Look up public key ---
        public_key_pem = self._registry.get_active_key(key_id)
        if not public_key_pem:
            logger.warning("Auth rejected: unknown key %s (path=%s)", key_id, request.url.path)
            return JSONResponse({"detail": "Unknown or inactive key"}, status_code=401)

        # --- Check nonce has not been used ---
        if self._registry.has_nonce(nonce):
            logger.warning("Auth rejected: nonce replay (path=%s)", request.url.path)
            return JSONResponse({"detail": "Nonce already used"}, status_code=401)

        # --- Read and hash the body (Starlette caches this in request._body) ---
        body = await request.body()

        # --- Verify ECDSA signature ---
        try:
            verify_signature(public_key_pem, key_id, timestamp, nonce, body, signature_b64)
        except (InvalidSignature, ValueError, Exception) as exc:
            logger.warning("Auth rejected: invalid signature for key %s: %s (path=%s)", key_id, exc, request.url.path)
            return JSONResponse({"detail": "Invalid signature"}, status_code=401)

        # --- Record nonce (also prunes old ones) ---
        try:
            self._registry.record_nonce(nonce, key_id)
        except Exception:
            # Duplicate insert race — treat as replay
            return JSONResponse({"detail": "Nonce already used"}, status_code=401)

        return await call_next(request)
