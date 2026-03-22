"""Tests for auth/signing.py — pure ECDSA sign/verify logic."""

import hashlib
import uuid

import pytest
from cryptography.exceptions import InvalidSignature

from auth.signing import build_canonical, sign_request, verify_signature


def test_round_trip(ec_keypair):
    """A signature produced by sign_request passes verify_signature."""
    key_id, private_pem, public_pem = ec_keypair
    body = b'{"jsonrpc":"2.0","method":"tools/call"}'
    headers = sign_request(private_pem, key_id, body)
    # Should not raise
    verify_signature(
        public_pem,
        headers["X-Auth-Key-ID"],
        headers["X-Auth-Timestamp"],
        headers["X-Auth-Nonce"],
        body,
        headers["X-Auth-Signature"],
    )


def test_tampered_body_rejected(ec_keypair):
    """Changing the body after signing invalidates the signature."""
    key_id, private_pem, public_pem = ec_keypair
    body = b"original body"
    headers = sign_request(private_pem, key_id, body)
    with pytest.raises(InvalidSignature):
        verify_signature(
            public_pem,
            headers["X-Auth-Key-ID"],
            headers["X-Auth-Timestamp"],
            headers["X-Auth-Nonce"],
            b"tampered body",
            headers["X-Auth-Signature"],
        )


def test_tampered_key_id_rejected(ec_keypair):
    """Changing key_id in the verification call invalidates the signature."""
    key_id, private_pem, public_pem = ec_keypair
    body = b"body"
    headers = sign_request(private_pem, key_id, body)
    with pytest.raises(InvalidSignature):
        verify_signature(
            public_pem,
            str(uuid.uuid4()),   # different key_id
            headers["X-Auth-Timestamp"],
            headers["X-Auth-Nonce"],
            body,
            headers["X-Auth-Signature"],
        )


def test_empty_body_canonical(ec_keypair):
    """Empty body uses SHA-256 of b'' in the canonical string."""
    key_id, private_pem, public_pem = ec_keypair
    body = b""
    headers = sign_request(private_pem, key_id, body)
    canonical = build_canonical(
        key_id,
        headers["X-Auth-Timestamp"],
        headers["X-Auth-Nonce"],
        body,
    )
    expected_hash = hashlib.sha256(b"").hexdigest()
    assert canonical.decode().endswith(expected_hash)
    # Signature still valid for empty body
    verify_signature(
        public_pem,
        key_id,
        headers["X-Auth-Timestamp"],
        headers["X-Auth-Nonce"],
        body,
        headers["X-Auth-Signature"],
    )


def test_canonical_string_structure(ec_keypair):
    """Canonical string has exactly 4 newline-separated fields in the correct order."""
    key_id, private_pem, _ = ec_keypair
    body = b"test"
    headers = sign_request(private_pem, key_id, body)
    canonical = build_canonical(
        headers["X-Auth-Key-ID"],
        headers["X-Auth-Timestamp"],
        headers["X-Auth-Nonce"],
        body,
    ).decode()
    parts = canonical.split("\n")
    assert len(parts) == 4
    assert parts[0] == key_id
    assert parts[1] == headers["X-Auth-Timestamp"]
    assert parts[2] == headers["X-Auth-Nonce"]
    assert parts[3] == hashlib.sha256(body).hexdigest()
