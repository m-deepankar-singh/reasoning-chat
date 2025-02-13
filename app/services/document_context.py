"""
Service for handling document context processing.
"""
from typing import Dict, List, Optional
from pathlib import Path
import os
import google.generativeai as genai
from app.core.config import get_settings
from app.utilities.pdf_utils import read_pdf_bytes
from app.services.reasoning_engine import reasoning_engine
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class DocumentContextService:
    """Service for processing and managing document context."""
    
    def get_raw_documents(self) -> List[str]:
        """Get content from raw document files."""
        raw_dir = settings.RAW_FILES_DIR
        if not os.path.exists(raw_dir):
            return []
        return [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]

    def get_markdown_documents(self) -> Dict[str, str]:
        """Get content from markdown document files with filenames as keys."""
        md_dir = Path(settings.PROCESSED_DIR)
        if not md_dir.exists():
            logger.warning(f"Markdown directory {md_dir} does not exist")
            return {}
            
        contents = {}
        for file in md_dir.glob("*.md"):
            if file.is_file():
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            contents[file.stem] = content
                            logger.debug(f"Successfully read markdown file: {file.stem}")
                except Exception as e:
                    logger.error(f"Error reading markdown file {file}: {e}")
        
        if not contents:
            logger.warning("No markdown documents found")
        return contents

    def format_document_context(self, documents: Dict[str, str]) -> str:
        """Format document contents into a structured context string."""
        if not documents:
            return ""
            
        context_parts = []
        for filename, content in documents.items():
            context_parts.append(f"# Document: {filename}\n\n{content}")
        
        formatted_context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Formatted context from {len(documents)} documents")
        return formatted_context

    async def process_raw_documents(self, query: str) -> str:
        """
        Process raw documents using Gemini model.
        
        Args:
            query (str): Query to process documents with
            
        Returns:
            str: Generated response from Gemini
        """
        try:
            # Initialize Gemini client
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            
            # Select model based on reasoning state
            model_name = settings.GEMINI_MODELS["reasoning"] if settings.ENABLE_REASONING else settings.GEMINI_MODELS["default"]
            model = genai.GenerativeModel(model_name)
            
            # Get list of PDF files
            pdf_files = self.get_raw_documents()
            if not pdf_files:
                return "No PDF documents found in raw directory."
            
            responses = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(settings.RAW_FILES_DIR, pdf_file)
                try:
                    # Read PDF as bytes
                    pdf_bytes = read_pdf_bytes(pdf_path)
                    
                    # Create content parts for Gemini
                    response = model.generate_content(
                        contents=[{
                            "mime_type": "application/pdf",
                            "data": pdf_bytes
                        }, query]
                    )
                    
                    if response.text:
                        responses.append(f"\n{response.text}")
                    
                except FileNotFoundError as e:
                    logger.error(f"File not found error: {str(e)}")
                    continue
                except IOError as e:
                    logger.error(f"IO error while reading PDF: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
                    continue
            
            if not responses:
                return "Failed to process any documents."
            
            return "\n\n".join(responses)
            
        except Exception as e:
            logger.error(f"Error in process_raw_documents: {str(e)}")
            return f"Error processing documents: {str(e)}"

    async def process_markdown_documents(self, query: str) -> Optional[str]:
        """Process markdown documents using Deepseek."""
        try:
            # Get and format document contents
            documents = self.get_markdown_documents()
            if not documents:
                logger.warning("No markdown documents available for context")
                return None
                
            # Format document context
            context = self.format_document_context(documents)
            logger.info(f"Processing query with context from {len(documents)} documents")
            
            # Create a structured prompt that combines context and query
            structured_prompt = f"""Based on the following document context, please answer this question: {query}

Document Context:
{context}

Please provide a relevant answer based on the document content above."""
            
            # Process with Deepseek reasoner
            response = await reasoning_engine.get_reasoning(
                prompt=structured_prompt,
                enable_reasoning=settings.ENABLE_REASONING,
                max_tokens=500,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing markdown documents: {e}")
            return None

document_context_service = DocumentContextService()
