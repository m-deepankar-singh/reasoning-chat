"""
Unit tests for PDF utilities.
"""
import pytest
from pathlib import Path
from app.utilities.pdf_utils import pdf_has_images, convert_pdf_to_markdown
import fitz
import tempfile
import os

@pytest.fixture
def sample_text_pdf():
    """Create a sample PDF with only text."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "This is a test PDF with only text")
        doc.save(tmp.name)
        doc.close()
        yield tmp.name
        os.unlink(tmp.name)

@pytest.fixture
def sample_image_pdf():
    """Create a sample PDF with an image."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        doc = fitz.open()
        page = doc.new_page()
        # Create a simple rectangle as an image
        page.draw_rect((100, 100, 200, 200), color=(1, 0, 0))
        doc.save(tmp.name)
        doc.close()
        yield tmp.name
        os.unlink(tmp.name)

def test_pdf_has_images_with_text_only(sample_text_pdf):
    """Test pdf_has_images with a text-only PDF."""
    assert not pdf_has_images(sample_text_pdf)

def test_pdf_has_images_with_image(sample_image_pdf):
    """Test pdf_has_images with a PDF containing an image."""
    assert pdf_has_images(sample_image_pdf)

def test_convert_pdf_to_markdown(sample_text_pdf):
    """Test PDF to Markdown conversion."""
    markdown_content, title = convert_pdf_to_markdown(sample_text_pdf)
    assert "This is a test PDF with only text" in markdown_content
    assert isinstance(title, str)
    assert len(title) > 0

def test_invalid_pdf_path():
    """Test handling of invalid PDF path."""
    with pytest.raises(Exception):
        pdf_has_images("nonexistent.pdf")

def test_convert_invalid_pdf():
    """Test conversion of invalid PDF."""
    with pytest.raises(Exception):
        convert_pdf_to_markdown("nonexistent.pdf")
