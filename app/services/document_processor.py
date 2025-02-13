"""
Document processing and handling services.
"""
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader, PdfWriter
import os
import uuid
import shutil
from pathlib import Path
from app.core.config import get_settings
from io import BytesIO
import subprocess
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Any, Union
from app.utilities.pdf_utils import pdf_has_images, convert_pdf_to_markdown
import logging

logger = logging.getLogger(__name__)

settings = get_settings()

class DocumentProcessor:
    """Handles document processing operations."""
    
    @staticmethod
    async def process_document(file_id: str) -> dict:
        """Process document by file ID and return processed file info."""
        file_path = file_handler.get_file_path(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        try:
            if file_ext == '.pdf':
                # Check if PDF contains images
                has_images = pdf_has_images(file_path)
                
                if has_images:
                    # File is already in raw directory, just use it
                    file_type = "pdf"
                    processed_file_path = file_path
                else:
                    # Convert to markdown and store in processed directory
                    markdown_content, title = convert_pdf_to_markdown(file_path)
                    processed_file_id = str(uuid.uuid4())
                    processed_file_path = Path(settings.PROCESSED_DIR) / f"{processed_file_id}.md"
                    processed_file_path.write_text(markdown_content, encoding='utf-8')
                    file_type = "markdown"
                    
                    # Delete the original file from raw directory since we don't need it
                    try:
                        Path(file_path).unlink()
                        logger.info(f"Deleted raw file {file_path} after successful markdown conversion")
                    except Exception as e:
                        logger.warning(f"Failed to delete raw file {file_path}: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            return {
                "processed_file_id": Path(processed_file_path).stem,
                "original_name": file_handler.file_metadata[file_id]["original_name"],
                "processed_path": str(processed_file_path),
                "file_type": file_type,
                "has_images": has_images if file_ext == '.pdf' else None
            }
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


    @staticmethod
    async def validate_file_type(filename: str):
        """Validate file type based on extension."""
        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    @staticmethod
    async def validate_file_size(file: UploadFile):
        """Validate file size."""
        if await file.read(settings.MAX_FILE_SIZE + 1) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        file.file.seek(0)  # Reset file pointer

    @staticmethod
    def process_pdf(content: bytes) -> str:
        """Extract text from PDF."""
        try:
            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")


class FileHandler:
    """Handles file operations."""
    
    def __init__(self):
        """Initialize file handler."""
        self.file_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Create necessary directories
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        os.makedirs(settings.RAW_FILES_DIR, exist_ok=True)

    async def save_raw_file(self, file: UploadFile) -> str:
        """Save uploaded file and return file ID."""
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Validate content type
        if file.content_type != settings.ALLOWED_FILE_TYPES[file_ext]:
            raise HTTPException(status_code=400, detail="Invalid content type")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_path = Path(settings.RAW_FILES_DIR) / f"{file_id}{file_ext}"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save file content
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Store metadata
            self.file_metadata[file_id] = {
                "original_name": file.filename,
                "content_type": file.content_type,
                "file_path": str(file_path)
            }
            
            return file_id
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path from file ID."""
        metadata = self.file_metadata.get(file_id)
        return metadata["file_path"] if metadata else None

    def get_mime_type(self, file_id: str) -> Optional[str]:
        """Get MIME type from file ID."""
        metadata = self.file_metadata.get(file_id)
        return metadata["content_type"] if metadata else None

# Global instances
document_processor = DocumentProcessor()
file_handler = FileHandler()
