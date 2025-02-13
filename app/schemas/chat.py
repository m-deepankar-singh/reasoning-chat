"""
Chat request and response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=150)
    temperature: float = Field(default=0.2)
    enable_reasoning: bool = Field(default=False)
    enable_doc_context: bool = Field(default=True)
    enable_search: bool = Field(default=False)
    conversation_id: Optional[str] = None
    images: Optional[List[str]] = None

class ChatResponse(BaseModel):
    content: str
    conversation_id: str
    reasoning: Optional[str] = None
