"""
SQLite-backed public key store and nonce replay defense.

All methods are synchronous. When called from async code (e.g. the auth
middleware), run them via anyio.to_thread.run_sync to avoid blocking the
event loop.

Thread safety: a single connection is held per instance, protected by a
threading.Lock.
"""

import pathlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

NONCE_TTL_MINUTES = 5


class KeyRegistry:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

    def init_db(self) -> None:
        """Create tables from schema.sql if they do not already exist."""
        schema_path = pathlib.Path(__file__).parent / "schema.sql"
        with self._lock:
            self._conn.executescript(schema_path.read_text())
            self._conn.commit()

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def register_key(self, key_id: str, client_name: str, public_key_pem: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO keys (key_id, client_name, public_key, created_at, active)"
                " VALUES (?, ?, ?, ?, 1)",
                (key_id, client_name, public_key_pem, now),
            )
            self._conn.commit()

    def get_active_key(self, key_id: str) -> str | None:
        """Return the public key PEM for an active key, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT public_key FROM keys WHERE key_id = ? AND active = 1",
                (key_id,),
            ).fetchone()
        return row[0] if row else None

    def revoke_key(self, key_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE keys SET active = 0 WHERE key_id = ?", (key_id,)
            )
            self._conn.commit()

    def list_keys(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_id, client_name, created_at, active FROM keys ORDER BY created_at"
            ).fetchall()
        return [
            {"key_id": r[0], "client_name": r[1], "created_at": r[2], "active": bool(r[3])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Nonce replay defense
    # ------------------------------------------------------------------

    def has_nonce(self, nonce: str) -> bool:
        """Return True if this nonce has been used before."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM nonces WHERE nonce = ?", (nonce,)
            ).fetchone()
        return row is not None

    def record_nonce(self, nonce: str, key_id: str) -> None:
        """
        Insert a nonce and prune expired nonces in a single transaction.
        Raises sqlite3.IntegrityError if the nonce already exists.
        """
        now = datetime.now(timezone.utc).isoformat()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=NONCE_TTL_MINUTES)
        ).isoformat()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM nonces WHERE used_at < ?", (cutoff,)
                )
                self._conn.execute(
                    "INSERT INTO nonces (nonce, key_id, used_at) VALUES (?, ?, ?)",
                    (nonce, key_id, now),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
