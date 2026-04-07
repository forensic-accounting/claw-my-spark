"""
Filesystem-based PDF processing cache.

Each processed PDF is stored under:
    {cache_dir}/{sha256}/
        summary.txt     — forensic summary text
        enriched.pdf    — enriched PDF bytes
        meta.json       — filename, flags, page counts, timestamp

Lookup is a single os.path.exists check on the hash directory.
Entries can be deleted by simply removing the directory.

Default cache_dir: /data/pdf_cache  (on the forensics-pdf-keys Docker volume)
"""

import hashlib
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    sha256: str
    filename: str
    summary: str
    enriched_pdf: bytes
    had_embedded_images: bool
    pages_processed: int
    images_transcribed: int


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify(name: str) -> str:
    """Convert a filename/stem to a URL-safe slug (matches website convention)."""
    slug = name.lower()
    slug = re.sub(r"\.pdf$", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "document"


class PdfCache:
    """Filesystem cache for processed PDFs, keyed by SHA-256 of raw input bytes."""

    def __init__(self, cache_dir: str = "/data/pdf_cache") -> None:
        self._root = pathlib.Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("PDF cache directory: %s", self._root)

    def _entry(self, digest: str) -> pathlib.Path:
        return self._root / digest

    def get(self, pdf_bytes: bytes) -> Optional[CachedResult]:
        digest = sha256_of(pdf_bytes)
        entry = self._entry(digest)
        if not entry.exists():
            return None
        try:
            meta = json.loads((entry / "meta.json").read_text())
            summary = (entry / "summary.txt").read_text()
            enriched_pdf = (entry / f"{slugify(meta['filename'])}.pdf").read_bytes()
        except Exception:
            logger.exception("Cache entry %s is corrupt — ignoring", digest[:12])
            return None
        logger.info("Cache hit for %s (sha256=%s…)", meta.get("filename"), digest[:12])
        return CachedResult(
            sha256=digest,
            filename=meta["filename"],
            summary=summary,
            enriched_pdf=enriched_pdf,
            had_embedded_images=meta["had_embedded_images"],
            pages_processed=meta["pages_processed"],
            images_transcribed=meta["images_transcribed"],
        )

    def put(
        self,
        pdf_bytes: bytes,
        filename: str,
        summary: str,
        enriched_pdf: bytes,
        had_embedded_images: bool,
        pages_processed: int,
        images_transcribed: int,
    ) -> None:
        digest = sha256_of(pdf_bytes)
        entry = self._entry(digest)
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "summary.txt").write_text(summary)
        (entry / f"{slugify(filename)}.pdf").write_bytes(enriched_pdf)
        (entry / "meta.json").write_text(json.dumps({
            "filename": filename,
            "had_embedded_images": had_embedded_images,
            "pages_processed": pages_processed,
            "images_transcribed": images_transcribed,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }))
        logger.info("Cached result for %s (sha256=%s…)", filename, digest[:12])
