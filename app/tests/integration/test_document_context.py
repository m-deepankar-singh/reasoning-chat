"""
Integration tests for document context functionality.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
from app.main import app
from app.core.config import get_settings
from app.services.document_context import document_context_service

client = TestClient(app)
settings = get_settings()

@pytest.fixture
def setup_test_files(tmp_path):
    """Setup test files for document processing."""
    # Create test directories
    raw_dir = tmp_path / "processed_files" / "raw"
    md_dir = tmp_path / "processed_files"
    raw_dir.mkdir(parents=True)
    
    # Create test files
    raw_file = raw_dir / "test.txt"
    raw_file.write_text("This is a test raw document.")
    
    md_file = md_dir / "test.md"
    md_file.write_text("# Test Document\nThis is a test markdown document.")
    
    # Update settings paths
    settings.RAW_FILES_DIR = str(raw_dir)
    settings.PROCESSED_DIR = str(md_dir)
    
    yield
    
    # Cleanup
    raw_file.unlink()
    md_file.unlink()

@pytest.mark.asyncio
async def test_raw_document_processing(setup_test_files):
    """Test processing of raw documents."""
    settings.ENABLE_DOC_CONTEXT = True
    settings.ENABLE_REASONING = False
    
    response = await document_context_service.process_raw_documents("test query")
    assert response is not None

@pytest.mark.asyncio
async def test_markdown_document_processing(setup_test_files):
    """Test processing of markdown documents."""
    settings.ENABLE_DOC_CONTEXT = True
    settings.ENABLE_REASONING = True
    
    response = await document_context_service.process_markdown_documents("test query")
    assert response is not None

def test_chat_with_doc_context_disabled():
    """Test chat endpoint with document context disabled."""
    settings.ENABLE_DOC_CONTEXT = False
    
    response = client.post(
        "/api/v1/chat",
        json={
            "prompt": "test query",
            "conversation_id": "test",
            "enable_reasoning": False
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_chat_with_doc_context_enabled(setup_test_files):
    """Test chat endpoint with document context enabled."""
    settings.ENABLE_DOC_CONTEXT = True
    
    response = client.post(
        "/api/v1/chat",
        json={
            "prompt": "test query",
            "conversation_id": "test",
            "enable_reasoning": True
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
