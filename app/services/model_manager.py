"""
Model management and initialization.
"""
from llama_cpp import Llama
from openai import OpenAI
from google import genai
import os
import logging
from typing import Optional
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages different AI model clients."""
    
    def __init__(self):
        """Initialize model clients."""
        self.llm = Llama(
            model_path=settings.MODEL_PATH,
            n_ctx=4096,
            n_threads=4,
        )
        
        # Initialize clients
        self.clients = {}
        
        # Initialize Deepseek client
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            try:
                self.clients["deepseek"] = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"  # Updated base URL
                )
                logger.info("Deepseek client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Deepseek client: {str(e)}")
                self.clients["deepseek"] = None
        else:
            logger.warning("DEEPSEEK_API_KEY not found in environment")
            self.clients["deepseek"] = None
        
        # Initialize Gemini client
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                self.clients["gemini"] = genai.Client(
                    api_key=google_api_key,
                    http_options={'api_version':'v1alpha'}
                )
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                self.clients["gemini"] = None
        else:
            logger.warning("GOOGLE_API_KEY not found in environment")
            self.clients["gemini"] = None
    
    def get_client(self, name: str):
        """Get a specific model client."""
        client = self.clients.get(name)
        if client is None:
            logger.warning(f"Client '{name}' is not available")
        return client

# Global instance
model_manager = ModelManager()
