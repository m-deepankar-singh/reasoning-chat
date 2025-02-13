from fastapi import FastAPI, HTTPException, status, UploadFile, Request, types
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
from llama_cpp import Llama
import os
from serpapi import GoogleSearch
from typing import Dict, List, Optional, Any, Union
import uuid
import sqlite3
from contextlib import contextmanager
import shutil
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import asyncio
from pypdf import PdfReader
from docx import Document
import mammoth
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
from logging_config import setup_logger
from fastapi.responses import JSONResponse

# Configuration and Environment Setup
load_dotenv()

# Setup logging
logger = setup_logger(log_level="INFO")

class Config:
    UPLOAD_DIR = "uploaded_files"
    PROCESSED_DIR = "processed_files"
    RAW_FILES_DIR = os.path.join(PROCESSED_DIR, "raw")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    DATABASE_PATH = "conversations.db"
    MODEL_PATH = "qwen2.5-3b-instruct-q6_k.gguf"
    REQUIRED_KEYS = ["SERPAPI_KEY", "GOOGLE_API_KEY"]
    ALLOWED_EXTENSIONS = {'.pdf', '.docx'}
    GEMINI_MODELS = {
        "reasoning": "gemini-2.0-flash-thinking-exp-01-21",
        "default": "gemini-2.0-flash"
    }
    
    @classmethod
    def validate_environment(cls):
        missing = [k for k in cls.REQUIRED_KEYS if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.RAW_FILES_DIR, exist_ok=True)

# Initialize Configuration
Config.validate_environment()
Config.create_directories()

# Initialize FastAPI app with logging middleware
app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request details
        logger.info("Request started", extra={
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None,
            "query_params": str(request.query_params)
        })
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response details
            logger.info("Request completed", extra={
                "status_code": response.status_code,
                "process_time": f"{process_time:.3f}s"
            })
            
            return response
        except Exception as exc:
            process_time = time.time() - start_time
            logger.error("Request failed", extra={
                "error": str(exc),
                "process_time": f"{process_time:.3f}s"
            })
            raise

app.add_middleware(LoggingMiddleware)

# Error handling with logging
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP Exception", extra={
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": request.url.path
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception", extra={
        "error": str(exc),
        "type": type(exc).__name__,
        "path": request.url.path
    }, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Model Initialization and Service Clients
class ModelManager:
    def __init__(self):
        self.llm = Llama(
            model_path=Config.MODEL_PATH,
            n_ctx=4096,
            n_threads=4,
        )
        
        self.clients = {
            "deepseek": OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            ) if os.getenv("DEEPSEEK_API_KEY") else None,
            "gemini": genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY"),
                http_options={'api_version':'v1alpha'}
            ) if os.getenv("GOOGLE_API_KEY") else None
        }
    
    def get_client(self, name: str):
        return self.clients.get(name)

# Initialize Model Manager
model_manager = ModelManager()

# Database Management
class DatabaseManager:
    @staticmethod
    @contextmanager
    def get_connection():
        conn = sqlite3.connect(Config.DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    @staticmethod
    def init_database():
        with DatabaseManager.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    processed_path TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

# Initialize Database
DatabaseManager.init_database()

# Core Components
class ReasoningEngine:
    @staticmethod
    async def get_reasoning(prompt: str, images: Optional[List[str]] = None) -> str:
        try:
            if images:
                return await ReasoningEngine._process_with_gemini_thinking(prompt, images)
            if model_manager.get_client("deepseek"):
                return await ReasoningEngine._process_with_deepseek_reasoner(prompt)
            return await ReasoningEngine._process_with_fallback(prompt)
        except Exception as e:
            print(f"Reasoning error: {e}")
            return await ReasoningEngine._process_with_fallback(prompt)

    @staticmethod
    async def _process_with_deepseek_reasoner(prompt: str):
        try:
            deepseek_client = model_manager.get_client("deepseek")
            if not deepseek_client:
                logger.warning("Deepseek client not available, falling back to Gemini")
                return await ReasoningEngine._process_with_gemini_thinking(prompt, [])
            
            response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert at reasoning and analysis. Break down complex problems step by step."},
                    {"role": "user", "content": f"Analyze this query step by step:\n{prompt}"}
                ],
                temperature=0.3,
                max_tokens=1
            )
            
            # Extract reasoning_content from response
            try:
                reasoning = response.choices[0].message.reasoning_content
                if not reasoning:
                    logger.warning("Empty reasoning_content received from deepseek-reasoner")
                    return await ReasoningEngine._process_with_gemini_thinking(prompt, [])
                return reasoning
            except AttributeError:
                logger.error("Missing reasoning_content in deepseek-reasoner response")
                return await ReasoningEngine._process_with_gemini_thinking(prompt, [])
                
        except Exception as e:
            logger.error(f"Deepseek reasoning failed: {str(e)}")
            return await ReasoningEngine._process_with_gemini_thinking(prompt, [])

    @staticmethod
    async def _process_with_gemini_thinking(prompt: str, images: List[str]):
        try:
            gemini_client = model_manager.get_client("gemini")
            if not gemini_client:
                return await ReasoningEngine._process_with_fallback(prompt)
            
            sys_instruct = "You are an expert at reasoning and analysis. Break down complex problems step by step."
            response = gemini_client.models.generate_content(
                model=Config.GEMINI_MODELS["reasoning"],
                config=types.GenerateContentConfig(system_instruction=sys_instruct),
                contents=[prompt]
            )
            # Return Gemini response directly
            return response.candidates[0].text
        except Exception as e:
            logger.error(f"Gemini thinking failed: {str(e)}")
            return await ReasoningEngine._process_with_fallback(prompt)

    @staticmethod
    async def _process_with_fallback(prompt: str) -> str:
        return "Reasoning unavailable. Proceeding with standard processing."

class ContentHandler:
    MAX_LENGTH = 45000
    
    @staticmethod
    def process_content(content: str) -> str:
        if len(content) > ContentHandler.MAX_LENGTH:
            return content[:ContentHandler.MAX_LENGTH] + "..."
        return content
    
    @staticmethod
    def get_document_context() -> Optional[str]:
        import glob
        md_files = sorted(glob.glob(os.path.join(Config.PROCESSED_DIR, "*.md")), key=os.path.getmtime, reverse=True)
        if md_files:
            with open(md_files[0], "r", encoding="utf-8") as f:
                return f.read()
        return None

class DocumentStore:
    def __init__(self, expiration_time: int = 3600):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.expiration_time = expiration_time
    
    def store_document(self, conversation_id: str, content: str):
        self.documents[conversation_id] = {
            'content': content,
            'timestamp': time.time()
        }
    
    def get_document(self, conversation_id: str) -> Optional[str]:
        doc_data = self.documents.get(conversation_id)
        if not doc_data:
            return None
        
        if time.time() - doc_data['timestamp'] > self.expiration_time:
            del self.documents[conversation_id]
            return None
        
        return doc_data['content']
    
    def cleanup_expired(self):
        current_time = time.time()
        expired_keys = [k for k, v in self.documents.items() 
                       if current_time - v['timestamp'] > self.expiration_time]
        for k in expired_keys:
            del self.documents[k]

# Initialize Document Store
document_store = DocumentStore()

# Document Processing Pipeline
class DocumentProcessor:
    @staticmethod
    async def process_document(file: UploadFile) -> dict:
        await DocumentProcessor.validate_file_size(file)
        content = await file.read()
        ext = Path(file.filename).suffix.lower()
        
        processors = {
            '.pdf': DocumentProcessor.process_pdf,
            '.docx': DocumentProcessor.process_docx,
            '.txt': DocumentProcessor.process_txt
        }
        
        if ext not in processors:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return await processors[ext](content)

    @staticmethod
    async def validate_file_size(file: UploadFile):
        chunk = await file.read(Config.MAX_FILE_SIZE + 1)
        await file.seek(0)
        if len(chunk) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

    @staticmethod
    async def process_pdf(content: bytes) -> dict:
        # PDF processing implementation
        images = []
        markdown_content = []
        pdf = PdfReader(BytesIO(content))
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            markdown_content.append(text)
            
            for image in page.images:
                image_data = base64.b64encode(image.data).decode()
                images.append({
                    'data': image_data,
                    'position': f'page_{page_num}',
                    'index': len(images)
                })
        
        return {
            'markdown': '\n'.join(markdown_content),
            'images': images
        }

    @staticmethod
    async def process_docx(content: bytes) -> dict:
        # DOCX processing implementation
        doc = Document(BytesIO(content))
        images = []
        
        result = mammoth.convert_to_markdown(BytesIO(content))
        markdown_text = result.value
        
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_data = base64.b64encode(rel.target_part.blob).decode()
                images.append({
                    'data': image_data,
                    'position': f'paragraph_{len(images)}',
                    'index': len(images)
                })
        
        return {
            'markdown': markdown_text,
            'images': images
        }

    @staticmethod
    async def process_txt(content: bytes) -> dict:
        return {
            'markdown': content.decode('utf-8'),
            'images': []
        }

# File Handling
class FileHandler:
    def __init__(self):
        self.file_metadata = {}

    async def save_raw_file(self, file: UploadFile) -> str:
        """Save uploaded file to raw files directory and return file ID"""
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            )

        # Generate unique file ID and path
        file_id = str(uuid.uuid4())
        file_path = Path(Config.RAW_FILES_DIR) / f"{file_id}{file_extension}"

        # Save file metadata
        self.file_metadata[file_id] = {
            "original_name": file.filename,
            "file_path": str(file_path),
            "mime_type": file.content_type,
            "timestamp": time.time()
        }

        # Save file
        try:
            contents = await file.read()
            if len(contents) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum limit of {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            return file_id
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file: {str(e)}"
            )

    def get_file_path(self, file_id: str) -> str:
        """Get file path from file ID"""
        if file_id not in self.file_metadata:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        return self.file_metadata[file_id]["file_path"]

    def get_mime_type(self, file_id: str) -> str:
        """Get MIME type from file ID"""
        if file_id not in self.file_metadata:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        return self.file_metadata[file_id]["mime_type"]

# Initialize File Handler
file_handler = FileHandler()

# Conversation Management
class Conversation:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.search_cache: Dict[str, dict] = {}
    
    def add_message(self, role: str, content: str):
        with DatabaseManager.get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (self.id, role, content)
            )
            conn.commit()
    
    def get_messages(self) -> List[dict]:
        with DatabaseManager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (self.id,)
            )
            return [dict(row) for row in cursor.fetchall()]

# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.4
    conversation_id: Optional[str] = None
    enable_search: bool = False
    enable_doc_context: bool = False
    enable_reasoning: bool = False
    file_id: Optional[str] = None
    images: Optional[List[str]] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 400
    temperature: Optional[float] = 0.7
    enable_search: Optional[bool] = False
    enable_doc_context: Optional[bool] = False
    enable_reasoning: Optional[bool] = False

# API Endpoints
@app.post("/chat")
async def chat_completion(request: ChatRequest):
    logger.info("Chat completion request received", extra={
        "conversation_id": request.conversation_id,
        "enable_search": request.enable_search,
        "enable_doc_context": request.enable_doc_context,
        "enable_reasoning": request.enable_reasoning,
        "file_id": request.file_id
    })
    
    try:
        conversation = conversations.get(request.conversation_id)
        if not conversation:
            conversation = Conversation()
            conversations[conversation.id] = conversation
            logger.info("New conversation created", extra={"conversation_id": conversation.id})
            
        # Initialize response components
        search_results = ""
        doc_context = ""
        final_prompt = request.prompt

        # Handle file processing if file_id is provided
        if request.file_id:
            try:
                file_path = file_handler.get_file_path(request.file_id)
                mime_type = file_handler.get_mime_type(request.file_id)
                
                # Read file bytes
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                # Get Gemini client
                gemini_client = model_manager.get_client("gemini")
                if not gemini_client:
                    raise HTTPException(
                        status_code=500,
                        detail="Gemini client not initialized"
                    )

                # Select model based on reasoning flag
                model = Config.GEMINI_MODELS["reasoning"] if request.enable_reasoning else Config.GEMINI_MODELS["default"]

                # Create content parts
                contents = [
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=mime_type
                    ),
                    request.prompt
                ]

                # Generate response
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction="You are an expert at analyzing documents and providing detailed, accurate responses."
                    )
                )

                if not response.candidates:
                    raise HTTPException(
                        status_code=500,
                        detail="No response generated from Gemini"
                    )

                return {
                    "conversation_id": conversation.id,
                    "response": response.text,
                    "search_results": search_results,
                    "doc_context": doc_context
                }

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file with Gemini: {str(e)}"
                )

        # Continue with existing chat logic for non-file requests
        final_prompt = request.prompt
        search_query = None
        doc_context = None
        reasoning_content = None

        # Build context chain
        if request.enable_search:
            search_query = await generate_search_query(request.prompt, conversation.id)
            if search_query not in conversation.search_cache:
                conversation.search_cache[search_query] = web_search(search_query)
                if len(conversation.search_cache) > 3:
                    del conversation.search_cache[next(iter(conversation.search_cache))]
            search_context = conversation.search_cache[search_query]["context"]
            final_prompt = f"Web Context:\n{search_context}\n\n{final_prompt}"

        if request.enable_doc_context:
            doc_context = ContentHandler.get_document_context()
            if doc_context:
                if not request.enable_reasoning:
                    processed = False
                    # Try Deepseek first
                    deepseek_client = model_manager.get_client("deepseek")
                    if deepseek_client:
                        try:
                            deepseek_response = deepseek_client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "You are an expert at analyzing documents and providing concise, relevant summaries."},
                                    {"role": "user", "content": f"Given this document context:\n{doc_context}\n\nProvide a concise summary focusing on the most relevant information for this user query: {request.prompt}"}
                                ],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            doc_context = deepseek_response.choices[0].message.content
                            processed = True
                        except Exception as e:
                            print(f"Deepseek chat failed: {str(e)}")
                    
                    # If Deepseek failed or wasn't available, use Gemini flash
                    if not processed:
                        gemini_client = model_manager.get_client("gemini")
                        if gemini_client:
                            try:
                                gemini_response = gemini_client.models.generate_content(
                                    model=Config.GEMINI_MODELS["default"],
                                    config=types.GenerateContentConfig(
                                        system_instruction="You are an expert at analyzing documents and providing concise, relevant summaries."
                                    ),
                                    contents=[f"Given this document context:\n{doc_context}\n\nProvide a concise summary focusing on the most relevant information for this user query: {request.prompt}"]
                                )
                                doc_context = gemini_response.candidates[0].text
                            except Exception as e:
                                print(f"Gemini flash failed: {str(e)}")
                
                final_prompt = f"Document Context:\n{doc_context}\n\n{final_prompt}"
        if request.enable_reasoning:
            reasoning_content = await ReasoningEngine.get_reasoning(final_prompt, request.images)
            if reasoning_content:
                final_prompt = f"{final_prompt}\n\nReasoning Context: {reasoning_content}"
                # Return Gemini response directly for reasoning
                gemini_client = model_manager.get_client("gemini")
                if not gemini_client:
                    raise HTTPException(
                        status_code=500,
                        detail="Gemini client not initialized"
                    )
                gemini_response = gemini_client.models.generate_content(
                    model=Config.GEMINI_MODELS["reasoning"],
                    config=types.GenerateContentConfig(
                        system_instruction="You are an expert at analyzing documents and providing detailed, accurate responses."
                    ),
                    contents=[final_prompt]
                )
                return {
                    "response": gemini_response.candidates[0].text,
                    "reasoning": reasoning_content,
                    "search_query": search_query if request.enable_search else None,
                    "document_context_present": bool(doc_context) if request.enable_doc_context else None,
                    "conversation_id": conversation.id,
                    "history": conversation.get_messages()
                }
            # Fallback to local LLM if reasoning failed
            output = model_manager.llm(
                final_prompt + "\n\nUsing the context provided, give a concise answer.",
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            response_text = output["choices"][0]["text"]
        elif request.images and len(request.images) > 0:
            sys_instruct = "You are a expert document analyser and give analysis of documents uploaded"
            gemini_client = model_manager.get_client("gemini")
            if not gemini_client:
                raise HTTPException(
                    status_code=500,
                    detail="Gemini client not initialized"
                )
            gemini_response = gemini_client.models.generate_content(
                model=Config.GEMINI_MODELS["default"],
                config=types.GenerateContentConfig(system_instruction=sys_instruct),
                contents=[final_prompt]
            )
            response_text = gemini_response.candidates[0].text
        else:
            output = model_manager.llm(
                final_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            response_text = output["choices"][0]["text"]

        # Update conversation
        conversation.add_message("user", request.prompt)
        conversation.add_message("assistant", response_text)

        logger.info("Chat completion successful", extra={
            "conversation_id": conversation.id,
            "response_length": len(str(response_text))
        })
        return {
            "response": response_text,
            "reasoning": reasoning_content if request.enable_reasoning else None,
            "search_query": search_query if request.enable_search else None,
            "document_context_present": bool(doc_context) if request.enable_doc_context else None,
            "conversation_id": conversation.id,
            "history": conversation.get_messages()
        }

    except Exception as e:
        logger.error("Chat completion failed", extra={
            "error": str(e),
            "conversation_id": request.conversation_id
        }, exc_info=True)
        raise

@app.post("/upload")
async def upload_file(
    file: UploadFile,
    conversation_id: Optional[str] = None
):
    logger.info("File upload started", extra={
        "filename": file.filename,
        "content_type": file.content_type,
        "conversation_id": conversation_id
    })
    
    try:
        # Save file using FileHandler
        file_id = await file_handler.save_raw_file(file)
        
        logger.info("File upload successful", extra={
            "file_id": file_id,
            "conversation_id": conversation_id,
            "size_bytes": os.path.getsize(file_handler.get_file_path(file_id))
        })
        return {
            "status": "success",
            "file_id": file_id,
            "original_filename": file.filename,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error("File upload failed", extra={
            "filename": file.filename,
            "error": str(e)
        }, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.post("/process-document")
async def process_document(
    file: UploadFile,
    conversation_id: Optional[str] = None
):
    try:
        content = await DocumentProcessor.process_document(file)
        
        md_filename = os.path.join(Config.PROCESSED_DIR, f"{os.path.splitext(file.filename)[0]}_{int(time.time())}.md")
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(content['markdown'])
        
        return {
            "status": "success",
            "document_info": {
                "filename": file.filename,
                "stored_file": md_filename,
                "char_count": len(content['markdown']),
                "image_count": len(content.get('images', []))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    with DatabaseManager.get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conversation_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": messages
    }

# Search functionality
async def generate_search_query(prompt: str, conversation_id: str) -> str:
    """Generate optimized search query based on prompt and conversation context"""
    system_message = {
        "role": "system",
        "content": """Generate a concise web search query based on the current prompt and conversation history. 
        Consider the context from previous messages but focus on the latest prompt. Never mention the year.
        Return ONLY the query text."""
    }
    
    try:
        if not model_manager.get_client("deepseek"):
            raise HTTPException(
                status_code=400,
                detail="Deepseek API required for search"
            )
            
        conv = conversations.get(conversation_id)
        if conv:
            history = conv.get_messages()
            messages = [system_message] + [{"role": m["role"], "content": m["content"]} for m in history]
        else:
            messages = [system_message]
            
        messages.append({"role": "user", "content": prompt})
        
        response = model_manager.get_client("deepseek").chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=30,
            temperature=0.3
        )
        return response.choices[0].message.content.strip('"')
    except Exception as e:
        print(f"Search query generation error: {e}")
        return prompt

def web_search(query: str, num_results: int = 3) -> dict:
    """Perform web search using SerpAPI"""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": num_results
        }
        results = GoogleSearch(params).get_dict().get('organic_results', [])
        return {
            "context": "\n".join([f"[{res.get('title')}]: {res.get('snippet')}" 
                                for res in results]),
            "raw": results
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return {"context": "", "raw": []}

# Global state
conversations: Dict[str, Conversation] = {}

class FallbackHandler:
    """Handles fallback scenarios when primary models are unavailable"""
    
    @staticmethod
    async def process_with_fallback(content: str, 
                                  max_tokens: int = 400, 
                                  temperature: float = 0.7) -> str:
        """Process content using local model as fallback"""
        try:
            output = model_manager.llm(
                content,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return output["choices"][0]["text"]
        except Exception as e:
            print(f"Fallback processing error: {e}")
            raise HTTPException(
                status_code=500,
                detail="All processing options failed"
            )

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Text generation endpoint with configurable options"""
    try:
        final_prompt = request.prompt

        # Handle search context
        if request.enable_search:
            search_results = web_search(request.prompt)
            if search_results["context"]:
                final_prompt = f"Web Context:\n{search_results['context']}\n\n{final_prompt}"

        # Handle document context
        if request.enable_doc_context:
            doc_context = ContentHandler.get_document_context()
            if doc_context:
                processed = False
                # Try Deepseek first
                deepseek_client = model_manager.get_client("deepseek")
                if deepseek_client:
                    try:
                        deepseek_response = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "You are an expert at analyzing documents and providing concise, relevant summaries."},
                                {"role": "user", "content": f"Given this document context:\n{doc_context}\n\nProvide a concise summary focusing on the most relevant information for this user query: {request.prompt}"}
                            ],
                            temperature=0.3,
                            max_tokens=1000
                        )
                        doc_context = deepseek_response.choices[0].message.content
                        processed = True
                    except Exception as e:
                        print(f"Deepseek processing failed: {str(e)}")
                
                # If Deepseek failed or wasn't available, try Gemini
                if not processed:
                    gemini_client = model_manager.get_client("gemini")
                    if gemini_client:
                        try:
                            gemini_response = gemini_client.models.generate_content(
                                model=Config.GEMINI_MODELS["default"],
                                config=types.GenerateContentConfig(
                                    system_instruction="You are an expert at analyzing documents and providing concise, relevant summaries."
                                ),
                                contents=[f"Given this document context:\n{doc_context}\n\nProvide a concise summary focusing on the most relevant information for this user query: {request.prompt}"]
                            )
                            doc_context = gemini_response.candidates[0].text
                            processed = True
                        except Exception as e:
                            print(f"Gemini processing failed: {str(e)}")
                
                final_prompt = f"Document Context:\n{doc_context}\n\n{final_prompt}"

        # Handle reasoning if enabled
        if request.enable_reasoning:
            reasoning = await ReasoningEngine.get_reasoning(final_prompt)
            final_prompt = f"{final_prompt}\n\nReasoning:\n{reasoning}"

        # Generate response using local model
        output = await FallbackHandler.process_with_fallback(
            final_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return {
            "generated_text": output,
            "prompt_length": len(final_prompt),
            "output_length": len(output)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)