"""Tests for auth/key_registry.py — SQLite key store and nonce replay defense."""

import sqlite3
import uuid

import pytest

from auth.key_registry import KeyRegistry


def test_init_creates_tables(tmp_db):
    """init_db creates both tables and the nonce index."""
    registry = KeyRegistry(tmp_db)
    registry.init_db()
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "keys" in tables
    assert "nonces" in tables
    conn.close()


def test_register_and_retrieve(populated_registry):
    """A registered key can be retrieved by key_id."""
    registry, key_id, _, public_pem = populated_registry
    result = registry.get_active_key(key_id)
    assert result == public_pem


def test_revoke_key(populated_registry):
    """A revoked key returns None from get_active_key."""
    registry, key_id, _, _ = populated_registry
    registry.revoke_key(key_id)
    assert registry.get_active_key(key_id) is None


def test_nonce_replay_detection(populated_registry):
    """has_nonce returns True after record_nonce is called."""
    registry, key_id, _, _ = populated_registry
    nonce = str(uuid.uuid4())
    assert registry.has_nonce(nonce) is False
    registry.record_nonce(nonce, key_id)
    assert registry.has_nonce(nonce) is True


def test_nonce_uniqueness_enforced(populated_registry):
    """record_nonce raises on a duplicate nonce."""
    registry, key_id, _, _ = populated_registry
    nonce = str(uuid.uuid4())
    registry.record_nonce(nonce, key_id)
    with pytest.raises(Exception):  # sqlite3.IntegrityError
        registry.record_nonce(nonce, key_id)


def test_inactive_key_not_returned(tmp_db, ec_keypair):
    """A key with active=0 is not returned by get_active_key."""
    key_id, _, public_pem = ec_keypair
    registry = KeyRegistry(tmp_db)
    registry.init_db()
    registry.register_key(key_id, "test", public_pem)
    registry.revoke_key(key_id)
    assert registry.get_active_key(key_id) is None


def test_unknown_key_returns_none(tmp_db):
    """get_active_key for an unknown key_id returns None."""
    registry = KeyRegistry(tmp_db)
    registry.init_db()
    assert registry.get_active_key(str(uuid.uuid4())) is None
