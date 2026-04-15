"""
Google Drive → Forensics → S3 sync pipeline.

Downloads PDFs from Google Drive, sends them to the forensics-pdf-mcp
server for OCR processing (populating the server-side cache), and
optionally uploads enriched PDFs to S3.

Sync state is persisted to a JSON file so the MCP ``list_synced_documents``
tool can return results instantly without touching Drive or S3.

Environment variables:
    GOOGLE_APPLICATION_CREDENTIALS  Path to service-account JSON
    DRIVE_FOLDER_HOA                Google Drive folder ID
    DRIVE_FOLDER_CONDO1             Google Drive folder ID
    DRIVE_FOLDER_CONDO2             Google Drive folder ID
    DRIVE_FOLDER_CONDO3             Google Drive folder ID
    DRIVE_FOLDER_CONDO4             Google Drive folder ID
    S3_ENDPOINT                     e.g. minio, s3.amazonaws.com
    S3_PORT                         e.g. 9000, 443
    S3_ACCESS_KEY / S3_SECRET_KEY
    S3_USE_SSL                      true/false
    S3_REGION                       default us-east-1
    BUCKET_PREFIX                   optional prefix for bucket names
    SYNC_STATE_PATH                 default /data/sync_state.json
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from minio import Minio

from client.forensics_client import ForensicsClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

SYNC_STATE_PATH = os.getenv("SYNC_STATE_PATH", "/data/sync_state.json")


def _get_folder_ids() -> dict[str, str]:
    """Return configured Drive folder IDs, raising if any are missing."""
    ids = {
        "HOA": os.getenv("DRIVE_FOLDER_HOA", ""),
        "Condo1": os.getenv("DRIVE_FOLDER_CONDO1", ""),
        "Condo2": os.getenv("DRIVE_FOLDER_CONDO2", ""),
        "Condo3": os.getenv("DRIVE_FOLDER_CONDO3", ""),
        "Condo4": os.getenv("DRIVE_FOLDER_CONDO4", ""),
    }
    missing = [k for k, v in ids.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing Drive folder IDs for: {', '.join(missing)}. "
            "Set DRIVE_FOLDER_HOA, DRIVE_FOLDER_CONDO1, etc."
        )
    return ids


def _get_s3_client() -> Minio:
    provider = os.getenv("STORAGE_PROVIDER", "local")
    if provider == "local":
        return Minio(
            endpoint=f"{os.getenv('S3_ENDPOINT', 'localhost')}:{os.getenv('S3_PORT', '9000')}",
            access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            secure=os.getenv("S3_USE_SSL", "false").lower() == "true",
            region=os.getenv("S3_REGION", "us-east-1"),
        )
    return Minio(
        endpoint=f"{os.getenv('S3_ENDPOINT')}:{os.getenv('S3_PORT', '443')}",
        access_key=os.getenv("S3_ACCESS_KEY"),
        secret_key=os.getenv("S3_SECRET_KEY"),
        secure=os.getenv("S3_USE_SSL", "true").lower() != "false",
        region=os.getenv("S3_REGION", "us-east-1"),
    )


def _bucket_name(section: str) -> str:
    prefix = os.getenv("BUCKET_PREFIX", "")
    return f"{prefix}{section.lower()}".lstrip("-")


def _slugify_path(drive_path: str) -> str:
    """Convert a Drive path like ``subdir/My File.pdf`` to ``subdir/my-file.pdf``."""
    segments = drive_path.split("/")
    slugged = []
    for seg in segments:
        slug = seg.lower()
        slug = re.sub(r"\.pdf$", "", slug)
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-") or "document"
        slugged.append(slug)
    return "/".join(slugged) + ".pdf"


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------

class SyncState:
    """JSON-file backed sync state, keyed by Drive file ID."""

    def __init__(self, path: str = SYNC_STATE_PATH) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, drive_file_id: str) -> dict[str, Any] | None:
        return self._data.get(drive_file_id)

    def put(
        self,
        drive_file_id: str,
        *,
        drive_modified_time: str,
        drive_name: str,
        section: str,
        s3_filename: str,
        s3_path: str,
        description: str,
    ) -> None:
        self._data[drive_file_id] = {
            "drive_file_id": drive_file_id,
            "drive_modified_time": drive_modified_time,
            "drive_name": drive_name,
            "section": section,
            "s3_filename": s3_filename,
            "s3_path": s3_path,
            "description": description,
            "last_synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def all_documents(self, section: str | None = None) -> list[dict[str, Any]]:
        docs = list(self._data.values())
        if section:
            docs = [d for d in docs if d.get("section") == section]
        return docs

    def clear(self) -> None:
        self._data = {}


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

def _build_drive_service():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set")
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds)


def _list_pdfs_recursively(
    drive_service, folder_id: str, path: str = ""
) -> list[dict[str, Any]]:
    """Walk a Drive folder tree and return all PDF file metadata."""
    results: list[dict[str, Any]] = []
    page_token = None

    while True:
        resp = (
            drive_service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false "
                  f"and (mimeType='application/pdf' or mimeType='application/vnd.google-apps.folder')",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, description)",
                pageToken=page_token,
            )
            .execute()
        )

        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                sub_path = f"{path}/{f['name']}" if path else f["name"]
                logger.info("  Walking subfolder: %s", sub_path)
                results.extend(
                    _list_pdfs_recursively(drive_service, f["id"], sub_path)
                )
            else:
                f["_drive_path"] = f"{path}/{f['name']}" if path else f["name"]
                results.append(f)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def _download_pdf(drive_service, file_id: str) -> bytes:
    """Download a file from Google Drive and return its bytes."""
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _ensure_bucket(s3: Minio, bucket: str) -> None:
    if not s3.bucket_exists(bucket):
        s3.make_bucket(bucket, os.getenv("S3_REGION", "us-east-1"))
        logger.info("Created bucket: %s", bucket)


def _upload_to_s3(s3: Minio, bucket: str, key: str, data: bytes, content_type: str = "application/pdf") -> None:
    _ensure_bucket(s3, bucket)
    s3.put_object(
        bucket,
        key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Main sync pipeline
# ---------------------------------------------------------------------------

async def run_cache(
    status_callback=None,
) -> dict[str, int]:
    """
    Download PDFs from Google Drive and send them to the MCP server for
    processing, populating the server-side cache.  Does not upload to S3
    or update sync state.
    """

    async def _status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            await status_callback(msg)

    folder_ids = _get_folder_ids()
    drive_service = _build_drive_service()
    totals = {"cached": 0, "errors": 0}

    async with ForensicsClient() as client:
        for section, folder_id in folder_ids.items():
            await _status(f"\nCaching {section} (folder {folder_id[:12]}...)...")

            drive_files = _list_pdfs_recursively(drive_service, folder_id)
            await _status(f"  Found {len(drive_files)} PDFs in Drive")

            for df in drive_files:
                drive_path = df.get("_drive_path", df["name"])
                try:
                    await _status(f"  Processing {drive_path}...")

                    pdf_bytes = _download_pdf(drive_service, df["id"])
                    await _status(f"    Downloaded {len(pdf_bytes) / 1024:.1f} KB")

                    await _status("    Sending to MCP server...")
                    result = await client.process_pdf_bytes(
                        pdf_bytes,
                        filename=df["name"],
                        progress_callback=lambda done, total: logger.info(
                            "    Page %d/%d", done, total
                        ),
                        status_callback=lambda m: logger.info("    %s", m),
                    )

                    totals["cached"] += 1
                    await _status(
                        f"    Done — {result.pages_processed} pages, "
                        f"{result.images_transcribed} images transcribed"
                    )

                except Exception as exc:
                    logger.exception("Error caching %s", drive_path)
                    await _status(f"    ERROR: {exc}")
                    totals["errors"] += 1

    await _status(
        f"\nCache complete: {totals['cached']} processed, "
        f"{totals['errors']} errors"
    )
    return totals


async def run_sync(
    full: bool = False,
    status_callback=None,
) -> dict[str, int]:
    """
    Run the full Google Drive → MCP server → S3 sync.

    Downloads PDFs from Drive, sends them to the MCP server for processing,
    uploads enriched PDFs to S3, and updates sync state.

    Args:
        full:               If True, clear sync state and re-process everything.
        status_callback:    Optional async callable(message: str) for log messages.

    Returns:
        Dict with keys: synced, skipped, errors
    """

    async def _status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            await status_callback(msg)

    folder_ids = _get_folder_ids()
    drive_service = _build_drive_service()
    s3 = _get_s3_client()
    state = SyncState()

    if full:
        await _status("Full sync: clearing sync state and S3 buckets...")
        state.clear()
        for section in folder_ids:
            bucket = _bucket_name(section)
            try:
                if s3.bucket_exists(bucket):
                    for obj in s3.list_objects(bucket, recursive=True):
                        s3.remove_object(bucket, obj.object_name)
                    await _status(f"  Cleared bucket {bucket}")
            except Exception as exc:
                await _status(f"  Warning: could not clear {bucket}: {exc}")

    totals = {"synced": 0, "skipped": 0, "errors": 0}

    async with ForensicsClient() as client:
        for section, folder_id in folder_ids.items():
            await _status(f"\nSyncing {section} (folder {folder_id[:12]}...)...")

            drive_files = _list_pdfs_recursively(drive_service, folder_id)
            await _status(f"  Found {len(drive_files)} PDFs in Drive")

            for df in drive_files:
                drive_path = df.get("_drive_path", df["name"])
                try:
                    # Check if unchanged
                    existing = state.get(df["id"])
                    if existing and not full:
                        drive_modified = df.get("modifiedTime", "")
                        if drive_modified <= existing.get("drive_modified_time", ""):
                            logger.debug("  Skipping %s (unchanged)", drive_path)
                            totals["skipped"] += 1
                            continue

                    await _status(f"  Processing {drive_path}...")

                    # Download from Drive
                    pdf_bytes = _download_pdf(drive_service, df["id"])
                    await _status(f"    Downloaded {len(pdf_bytes) / 1024:.1f} KB")

                    # Send to MCP server for processing (populates server cache)
                    await _status("    Sending to MCP server...")
                    result = await client.process_pdf_bytes(
                        pdf_bytes,
                        filename=df["name"],
                        progress_callback=lambda done, total: logger.info(
                            "    Page %d/%d", done, total
                        ),
                        status_callback=lambda m: logger.info("    %s", m),
                    )

                    enriched_pdf = result.enriched_pdf
                    description = result.summary

                    # Upload enriched PDF to S3
                    bucket = _bucket_name(section)
                    s3_filename = _slugify_path(drive_path)
                    s3_path = f"{bucket}/{s3_filename}"

                    _upload_to_s3(s3, bucket, s3_filename, enriched_pdf)
                    await _status(f"    Uploaded to {s3_path}")

                    # Fallback description if forensics didn't produce one
                    if not description:
                        description = (
                            df.get("description")
                            or df["name"].replace(".pdf", "").replace("_", " ")
                        )

                    # Update sync state
                    state.put(
                        df["id"],
                        drive_modified_time=df.get("modifiedTime", ""),
                        drive_name=df["name"],
                        section=section,
                        s3_filename=s3_filename,
                        s3_path=s3_path,
                        description=description,
                    )
                    state.save()

                    totals["synced"] += 1
                    await _status(f"    Synced {df['name']} → {s3_filename}")

                except Exception as exc:
                    logger.exception("Error syncing %s", drive_path)
                    await _status(f"    ERROR: {exc}")
                    totals["errors"] += 1

            # Rebuild catalog.json for this section
            try:
                section_docs = state.all_documents(section=section)
                catalog = {
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "bucket": _bucket_name(section),
                    "documents": [
                        {
                            "filename": d["s3_filename"],
                            "path": d["s3_path"],
                            "description": d.get("description", ""),
                        }
                        for d in section_docs
                    ],
                }
                catalog_bytes = json.dumps(catalog, indent=2).encode()
                _upload_to_s3(
                    s3, _bucket_name(section), "catalog.json", catalog_bytes, "application/json"
                )
                await _status(f"  Updated catalog.json ({len(section_docs)} entries)")
            except Exception as exc:
                await _status(f"  Warning: failed to write catalog.json: {exc}")

    await _status(
        f"\nSync complete: {totals['synced']} synced, "
        f"{totals['skipped']} skipped, {totals['errors']} errors"
    )
    return totals


def get_synced_documents(section: str | None = None) -> list[dict[str, Any]]:
    """Return all synced document metadata from the state file (instant)."""
    state = SyncState()
    return state.all_documents(section=section)


def get_document_enriched_pdf(drive_file_id: str) -> bytes | None:
    """Return enriched PDF bytes for a document, reading from S3."""
    state = SyncState()
    doc = state.get(drive_file_id)
    if not doc:
        return None
    s3 = _get_s3_client()
    bucket = _bucket_name(doc["section"])
    try:
        resp = s3.get_object(bucket, doc["s3_filename"])
        return resp.read()
    except Exception:
        logger.exception("Failed to read %s from S3", doc["s3_path"])
        return None
    finally:
        try:
            resp.close()
            resp.release_conn()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    full = "--full" in sys.argv
    dry_run = "--dry-run" in sys.argv
    cache_only = "--cache-only" in sys.argv

    if dry_run:
        # Just list what would be synced
        folder_ids = _get_folder_ids()
        drive_service = _build_drive_service()
        state = SyncState()
        for section, fid in folder_ids.items():
            print(f"\n{section} (folder {fid[:12]}...):")
            files = _list_pdfs_recursively(drive_service, fid)
            for f in files:
                existing = state.get(f["id"])
                status = "SKIP" if existing and f.get("modifiedTime", "") <= existing.get("drive_modified_time", "") else "SYNC"
                print(f"  [{status}] {f.get('_drive_path', f['name'])}")
        sys.exit(0)

    if cache_only:
        result = asyncio.run(run_cache())
        sys.exit(1 if result["errors"] > 0 else 0)

    result = asyncio.run(run_sync(full=full))
    sys.exit(1 if result["errors"] > 0 else 0)
