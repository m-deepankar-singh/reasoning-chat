"""
Request models for API endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Chat completion request model."""
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)
    conversation_id: Optional[str] = None
    enable_search: bool = False
    enable_doc_context: bool = False
    enable_reasoning: bool = False
    file_id: Optional[str] = None
    images: Optional[List[str]] = None

class GenerateRequest(BaseModel):
    """Text generation request model."""
    prompt: str
    max_tokens: Optional[int] = Field(default=400, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    enable_search: Optional[bool] = False
    enable_doc_context: Optional[bool] = False
    enable_reasoning: Optional[bool] = False
