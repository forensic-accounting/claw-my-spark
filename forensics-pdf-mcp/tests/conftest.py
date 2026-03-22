"""
Shared pytest fixtures.

PDF fixtures are generated programmatically using PyMuPDF so no binary
test files need to be committed to the repository.
"""

import io
import uuid

import fitz  # PyMuPDF
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from PIL import Image

from auth.key_registry import KeyRegistry


# ---------------------------------------------------------------------------
# Cryptographic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ec_keypair():
    """Generate a real P-256 keypair. Returns (key_id, private_pem, public_pem)."""
    key_id = str(uuid.uuid4())
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return key_id, private_pem, public_pem


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a fresh SQLite database."""
    return str(tmp_path / "test_keys.db")


@pytest.fixture
def populated_registry(tmp_db, ec_keypair):
    """
    A KeyRegistry with one active key pre-registered.
    Returns (registry, key_id, private_pem, public_pem).
    """
    key_id, private_pem, public_pem = ec_keypair
    registry = KeyRegistry(tmp_db)
    registry.init_db()
    registry.register_key(key_id, "test-client", public_pem)
    return registry, key_id, private_pem, public_pem


# ---------------------------------------------------------------------------
# PDF fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def text_only_pdf():
    """A minimal single-page PDF containing only native text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        fitz.Point(72, 100),
        "Test Bank Statement\nAccount: 1234567890\nBalance: $1,500.00\n"
        "Date: 2026-03-01\nTransaction: Payment to ABC Corp  -$250.00",
        fontsize=12,
        color=(0, 0, 0),
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def image_only_pdf():
    """A single-page PDF whose content is entirely an embedded PNG image."""
    # Create a simple white PNG with some black pixels (simulates a scanned page)
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(50, 50, 545, 750), stream=png_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mixed_pdf(text_only_pdf, image_only_pdf):
    """A two-page PDF: page 0 = text only, page 1 = image only."""
    doc_text = fitz.open(stream=text_only_pdf, filetype="pdf")
    doc_img = fitz.open(stream=image_only_pdf, filetype="pdf")
    doc_out = fitz.open()
    doc_out.insert_pdf(doc_text)
    doc_out.insert_pdf(doc_img)
    pdf_bytes = doc_out.tobytes()
    doc_text.close()
    doc_img.close()
    doc_out.close()
    return pdf_bytes
