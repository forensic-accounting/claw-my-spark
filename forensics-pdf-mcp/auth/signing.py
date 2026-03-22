"""
Pure ECDSA P-256 signing and verification.

No HTTP or framework dependencies — importable by both the server middleware
and any standalone client that copies this module.

Canonical string format (UTF-8, newline-separated, no trailing newline):
    {key_id}\\n{timestamp}\\n{nonce}\\n{sha256(body_bytes).hexdigest()}
"""

import base64
import hashlib
import uuid
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature  # re-exported for callers
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

__all__ = ["sign_request", "verify_signature", "build_canonical", "InvalidSignature"]


def build_canonical(key_id: str, timestamp: str, nonce: str, body_bytes: bytes) -> bytes:
    """Return the UTF-8 canonical string that is signed/verified."""
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    canonical = f"{key_id}\n{timestamp}\n{nonce}\n{body_hash}"
    return canonical.encode("utf-8")


def sign_request(private_key_pem: str, key_id: str, body_bytes: bytes) -> dict:
    """
    Sign a request body and return the four X-Auth-* headers.

    Args:
        private_key_pem: PEM-encoded PKCS8 EC private key
        key_id: UUID identifying the key in the server registry
        body_bytes: raw request body bytes (use b"" for GET requests)

    Returns:
        dict with keys: X-Auth-Key-ID, X-Auth-Timestamp, X-Auth-Nonce, X-Auth-Signature
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
        password=None,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    nonce = str(uuid.uuid4())
    canonical = build_canonical(key_id, timestamp, nonce, body_bytes)
    signature_der = private_key.sign(canonical, ec.ECDSA(hashes.SHA256()))
    signature_b64 = base64.urlsafe_b64encode(signature_der).decode()
    return {
        "X-Auth-Key-ID": key_id,
        "X-Auth-Timestamp": timestamp,
        "X-Auth-Nonce": nonce,
        "X-Auth-Signature": signature_b64,
    }


def verify_signature(
    public_key_pem: str,
    key_id: str,
    timestamp: str,
    nonce: str,
    body_bytes: bytes,
    signature_b64: str,
) -> None:
    """
    Verify an ECDSA signature over the canonical string.

    Raises:
        cryptography.exceptions.InvalidSignature on failure
        ValueError on malformed inputs
    """
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode() if isinstance(public_key_pem, str) else public_key_pem,
    )
    canonical = build_canonical(key_id, timestamp, nonce, body_bytes)
    # urlsafe_b64decode is tolerant of missing padding
    signature_der = base64.urlsafe_b64decode(signature_b64 + "==")
    public_key.verify(signature_der, canonical, ec.ECDSA(hashes.SHA256()))
