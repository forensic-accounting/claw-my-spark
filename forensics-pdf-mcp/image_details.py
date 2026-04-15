"""
Per-PDF image details markdown — resumable intermediate state.

Each processed PDF gets an image_details.md in its cache directory that
records per-page vision extraction results.  If a job is cancelled or
restarted, already-completed pages are skipped.

Phase 1 (vision model) writes: sha256, page_number, text_content,
extraction_status.  Phase 2 (summary model) fills in: image_title,
image_type, summarization_status.
"""

import hashlib
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_PENDING = "_pending_"


@dataclass
class ImagePageDetail:
    page_number: int
    image_sha256: str
    text_content: str
    extraction_status: str  # "success" or "failure"
    image_title: str = _PENDING
    image_type: str = _PENDING
    summarization_status: str = _PENDING


class ImageDetailsDoc:
    """Read/write the per-PDF image_details.md file."""

    def __init__(
        self,
        cache_dir: str,
        pdf_sha256: str,
        filename: str,
        total_pages: int,
        image_page_indices: list[int],
    ) -> None:
        self._dir = pathlib.Path(cache_dir) / pdf_sha256
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "image_details.md"
        self.pdf_sha256 = pdf_sha256
        self.filename = filename
        self.total_pages = total_pages
        self.image_page_indices = image_page_indices

    @property
    def path(self) -> pathlib.Path:
        return self._path

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _render_header(self) -> str:
        return (
            f"# Image Details — {self.filename}\n\n"
            f"**PDF SHA256:** `{self.pdf_sha256}`\n"
            f"**Total Pages:** {self.total_pages}\n"
            f"**Image Pages:** {len(self.image_page_indices)}\n"
        )

    @staticmethod
    def _render_page_section(d: ImagePageDetail) -> str:
        lines = [
            f"\n---\n",
            f"## Page {d.page_number}\n",
            f"- **Image SHA256:** `{d.image_sha256}`",
            f"- **Page Number:** {d.page_number}",
            f"- **Image Extraction:** {d.extraction_status}",
            f"- **Image Title:** {d.image_title}",
            f"- **Image Type:** {d.image_type}",
            f"- **Textual Summarization:** {d.summarization_status}",
            f"",
            f"### Extracted Text",
            f"",
            f"```text",
            d.text_content,
            f"```",
        ]
        return "\n".join(lines) + "\n"

    def ensure_header(self) -> None:
        """Write the header if the file does not yet exist."""
        if not self._path.exists():
            self._path.write_text(self._render_header(), encoding="utf-8")
            logger.info("Created image_details.md for %s", self.filename)

    def append_page(self, detail: ImagePageDetail) -> None:
        """Append one page section to the markdown (crash-safe incremental write)."""
        self.ensure_header()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(self._render_page_section(detail))
        logger.info("Appended page %d to image_details.md", detail.page_number)

    def update_page_fields(
        self,
        page_number: int,
        *,
        image_title: Optional[str] = None,
        image_type: Optional[str] = None,
        summarization_status: Optional[str] = None,
    ) -> None:
        """Update qwen3-filled fields for a specific page in the markdown."""
        if not self._path.exists():
            return
        content = self._path.read_text(encoding="utf-8")
        if image_title is not None:
            content = re.sub(
                rf"(## Page {page_number}\n(?:.*\n)*?- \*\*Image Title:\*\*) [^\n]+",
                rf"\1 {image_title}",
                content,
            )
        if image_type is not None:
            content = re.sub(
                rf"(## Page {page_number}\n(?:.*\n)*?- \*\*Image Type:\*\*) [^\n]+",
                rf"\1 {image_type}",
                content,
            )
        if summarization_status is not None:
            content = re.sub(
                rf"(## Page {page_number}\n(?:.*\n)*?- \*\*Textual Summarization:\*\*) [^\n]+",
                rf"\1 {summarization_status}",
                content,
            )
        self._path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Reading / resumability
    # ------------------------------------------------------------------

    def get_completed_pages(self) -> set[int]:
        """Return set of page numbers already successfully recorded."""
        if not self._path.exists():
            return set()
        content = self._path.read_text(encoding="utf-8")
        completed: set[int] = set()
        for m in re.finditer(
            r"## Page (\d+)\n.*?- \*\*Image Extraction:\*\* (success|failure)",
            content,
            re.DOTALL,
        ):
            completed.add(int(m.group(1)))
        return completed

    def load(self) -> dict[int, ImagePageDetail]:
        """Parse the markdown and return page_number → ImagePageDetail."""
        if not self._path.exists():
            return {}

        content = self._path.read_text(encoding="utf-8")
        details: dict[int, ImagePageDetail] = {}

        # Split on page section headers
        sections = re.split(r"(?=## Page \d+)", content)
        for section in sections:
            m_header = re.match(r"## Page (\d+)", section)
            if not m_header:
                continue
            page_num = int(m_header.group(1))

            def _field(name: str) -> str:
                m = re.search(rf"- \*\*{re.escape(name)}:\*\* (.+)", section)
                return m.group(1).strip() if m else _PENDING

            # Extract text content from ```text ... ``` block
            m_text = re.search(r"```text\n(.*?)```", section, re.DOTALL)
            text_content = m_text.group(1).rstrip("\n") if m_text else ""

            details[page_num] = ImagePageDetail(
                page_number=page_num,
                image_sha256=_field("Image SHA256").strip("`"),
                text_content=text_content,
                extraction_status=_field("Image Extraction"),
                image_title=_field("Image Title"),
                image_type=_field("Image Type"),
                summarization_status=_field("Textual Summarization"),
            )
        return details


def sha256_of_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()
