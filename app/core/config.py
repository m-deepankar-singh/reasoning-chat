"""
Core configuration settings for the application.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Dict
import secrets


class Settings(BaseSettings):
    """Application settings."""
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI OpenAI Project"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Keys
    DEEPSEEK_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    SERPAPI_KEY: Optional[str] = None
    
    # File Settings
    UPLOAD_DIR: str = "uploaded_files"
    PROCESSED_DIR: str = "processed_files"
    RAW_FILES_DIR: str = "processed_files/raw"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES: Dict[str, str] = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    LIBREOFFICE_PATH: str = "soffice"
    
    # Database
    DATABASE_PATH: str = "conversations.db"
    
    # Model Settings
    MODEL_PATH: str = "qwen2.5-3b-instruct-q6_k.gguf"
    ENABLE_DOC_CONTEXT: bool = True
    ENABLE_REASONING: bool = False
    GEMINI_MODELS: Dict[str, str] = {
        "reasoning": "gemini-2.0-flash-thinking-exp-01-21",
        "default": "gemini-2.0-flash"
    }
    DEEPSEEK_MODELS: Dict[str, str] = {
        "reasoning": "deepseek-reasoner",
        "default": "deepseek-chat"
    }
    
    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
