"""
File upload and processing routes.
"""
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
from typing import Optional
from app.services.document_processor import file_handler, document_processor
from pathlib import Path

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    conversation_id: Optional[str] = None
):
    """Handle file uploads."""
    try:
        # Save raw file and get file ID
        file_id = await file_handler.save_raw_file(file)
        
        # Process document automatically
        processed_file = await document_processor.process_document(file_id)
        
        return {
            "file_id": file_id,
            "original_name": file.filename,
            "content_type": file.content_type,
            "processed_file": processed_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download processed file by ID."""
    try:
        file_path = file_handler.get_file_path(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=Path(file_path).name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
