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

# Configuration and Environment Setup
load_dotenv()

class Config:
    UPLOAD_DIR = "uploaded_files"
    PROCESSED_DIR = "processed_files"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    DATABASE_PATH = "conversations.db"
    MODEL_PATH = "qwen2.5-3b-instruct-q6_k.gguf"
    REQUIRED_KEYS = ["SERPAPI_KEY"]
    
    @classmethod
    def validate_environment(cls):
        missing = [k for k in cls.REQUIRED_KEYS if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)

# Initialize Configuration
Config.validate_environment()
Config.create_directories()

app = FastAPI()

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
    async def _process_with_gemini_thinking(prompt: str, images: List[str]) -> str:
        gemini = model_manager.get_client("gemini")
        if not gemini:
            raise HTTPException(status_code=400, detail="Gemini API required")
        
        pil_images = [Image.open(BytesIO(base64.b64decode(img))).resize((256, 256)) 
                     for img in images]
        
        response = gemini.models.generate_content(
            model='gemini-2.0-flash-thinking-exp-01-21',
            contents=pil_images + [prompt],
            config={'thinking_config': {'include_thoughts': True}}
        )
        
        reasoning = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                    reasoning.append(part.text)
                else:
                    reasoning.append(part.text)
        return "\n".join(reasoning) if reasoning else "No reasoning generated"

    @staticmethod
    async def _process_with_deepseek_reasoner(prompt: str) -> str:
        deepseek = model_manager.get_client("deepseek")
        response = deepseek.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

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
    try:
        # Initialize conversation
        conv = conversations.get(request.conversation_id) or Conversation()
        if not conv.id in conversations:
            conversations[conv.id] = conv

        final_prompt = request.prompt
        search_query = None
        doc_context = None
        reasoning_content = None

        # Build context chain
        if request.enable_search:
            search_query = await generate_search_query(request.prompt, conv.id)
            if search_query not in conv.search_cache:
                conv.search_cache[search_query] = web_search(search_query)
                if len(conv.search_cache) > 3:
                    del conv.search_cache[next(iter(conv.search_cache))]
            search_context = conv.search_cache[search_query]["context"]
            final_prompt = f"Web Context:\n{search_context}\n\n{final_prompt}"

        if request.enable_doc_context:
            doc_context = ContentHandler.get_document_context()
            if doc_context:
                final_prompt = f"Document Context:\n{doc_context}\n\n{final_prompt}"
        if request.enable_reasoning:
            reasoning_content = await ReasoningEngine.get_reasoning(final_prompt, request.images)
            if reasoning_content:
                final_prompt = f"{final_prompt}\n\nReasoning Context: {reasoning_content}"
            output = model_manager.llm(
                final_prompt + "\n\nUsing the context provided, give a concise answer.",
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            response_text = output["choices"][0]["text"]
        elif request.images and len(request.images) > 0:
            sys_instruct = "You are a expert document analyser and give analysis of documents uploaded"
            gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
            gemini_response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
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
        conv.add_message("user", request.prompt)
        conv.add_message("assistant", response_text)

        return {
            "response": response_text,
            "reasoning": reasoning_content if request.enable_reasoning else None,
            "search_query": search_query if request.enable_search else None,
            "document_context_present": bool(doc_context) if request.enable_doc_context else None,
            "conversation_id": conv.id,
            "history": conv.get_messages()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            raise HTTPException(status_code=400, detail="Deepseek API required for search")
            
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