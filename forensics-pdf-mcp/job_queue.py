"""
SQLite-backed single-job queue for async Drive sync processing.

Only one job may be pending or processing at a time. Attempting to submit
a second job while one is active raises ``JobBusyError``.

Thread safety: a single connection protected by ``threading.Lock``,
same pattern as ``auth/key_registry.py``.
"""

import json
import logging
import pathlib
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobBusyError(Exception):
    """Raised when a job is submitted while another is active."""

    def __init__(self, active_job_id: str) -> None:
        self.active_job_id = active_job_id
        super().__init__(f"A job is already active: {active_job_id}")


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    job_id              TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'pending',
    folders_json        TEXT,
    creds_path          TEXT,
    folders_total       INTEGER DEFAULT 0,
    folders_done        INTEGER DEFAULT 0,
    files_total         INTEGER DEFAULT 0,
    files_done          INTEGER DEFAULT 0,
    files_cached        INTEGER DEFAULT 0,
    files_errors        INTEGER DEFAULT 0,
    current_file        TEXT,
    current_file_progress TEXT,
    errors_json         TEXT DEFAULT '[]',
    created_at          TEXT,
    started_at          TEXT,
    completed_at        TEXT
);
"""


class JobQueue:
    """SQLite-backed job queue that enforces at-most-one active job."""

    def __init__(self, db_path: str = "/data/jobs.db") -> None:
        self._lock = threading.Lock()
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        # On startup, reset any "processing" jobs back to "pending"
        # so they get picked up again by the worker.
        self._recover()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        job_id: str,
        folders: list[dict[str, str]],
        creds_path: str,
    ) -> dict[str, Any]:
        """Insert a new job.  Raises ``JobBusyError`` if one is already active."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            active = self._active_row()
            if active:
                raise JobBusyError(active["job_id"])
            self._conn.execute(
                "INSERT INTO jobs (job_id, status, folders_json, creds_path,"
                " folders_total, created_at)"
                " VALUES (?, 'pending', ?, ?, ?, ?)",
                (
                    job_id,
                    json.dumps(folders),
                    creds_path,
                    len(folders),
                    now,
                ),
            )
            self._conn.commit()
        logger.info("Job %s submitted (%d folders)", job_id, len(folders))
        return self.get(job_id)  # type: ignore[return-value]

    def claim(self) -> Optional[dict[str, Any]]:
        """Atomically claim the oldest pending job (set to processing)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'pending'"
                " ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE jobs SET status = 'processing', started_at = ?"
                " WHERE job_id = ?",
                (now, row["job_id"]),
            )
            self._conn.commit()
        logger.info("Claimed job %s", row["job_id"])
        return self.get(row["job_id"])

    def update(self, job_id: str, **fields: Any) -> None:
        """Update arbitrary progress fields on a job."""
        if not fields:
            return
        allowed = {
            "folders_done", "files_total", "files_done", "files_cached",
            "files_errors", "current_file", "current_file_progress",
            "errors_json",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Cannot update fields: {bad}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values())
        vals.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {sets} WHERE job_id = ?", vals
            )
            self._conn.commit()

    def complete(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = 'completed', completed_at = ?,"
                " current_file = NULL, current_file_progress = NULL"
                " WHERE job_id = ?",
                (now, job_id),
            )
            self._conn.commit()
        logger.info("Job %s completed", job_id)

    def fail(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # Append error to errors_json
            row = self._conn.execute(
                "SELECT errors_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            errors = json.loads(row["errors_json"]) if row else []
            errors.append(error)
            self._conn.execute(
                "UPDATE jobs SET status = 'failed', completed_at = ?,"
                " errors_json = ?, current_file = NULL,"
                " current_file_progress = NULL"
                " WHERE job_id = ?",
                (now, json.dumps(errors), job_id),
            )
            self._conn.commit()
        logger.info("Job %s failed: %s", job_id, error)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return full job state as a dict, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def active_job(self) -> Optional[dict[str, Any]]:
        """Return the currently active (pending or processing) job, or None."""
        with self._lock:
            row = self._active_row()
        if not row:
            return None
        return self._row_to_dict(row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active_row(self) -> Optional[sqlite3.Row]:
        """Must be called with self._lock held."""
        return self._conn.execute(
            "SELECT * FROM jobs WHERE status IN ('pending', 'processing')"
            " ORDER BY created_at LIMIT 1"
        ).fetchone()

    def _recover(self) -> None:
        """Reset processing → pending on startup so the worker retries."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = 'pending', started_at = NULL"
                " WHERE status = 'processing'"
            )
            self._conn.commit()
        if cursor.rowcount:
            logger.info("Recovered %d interrupted job(s)", cursor.rowcount)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        # Parse JSON fields for the caller
        if d.get("folders_json"):
            d["folders"] = json.loads(d["folders_json"])
        else:
            d["folders"] = []
        if d.get("errors_json"):
            d["errors"] = json.loads(d["errors_json"])
        else:
            d["errors"] = []
        # Drop raw JSON columns from the public dict
        d.pop("folders_json", None)
        d.pop("errors_json", None)
        return d

    def close(self) -> None:
        with self._lock:
            self._conn.close()
