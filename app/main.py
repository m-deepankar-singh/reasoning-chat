"""
FastAPI application entry point.
"""
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, files
from app.db.sqlite_db import init_db
from app.middleware.upload_config import UploadConfigMiddleware
from app.core.application import app as base_app
from app.core.config import get_settings

# Load environment variables
load_dotenv()

# Initialize database
init_db()

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add upload configuration middleware
app.add_middleware(UploadConfigMiddleware)

# Include routers
app.include_router(chat.router)
app.include_router(files.router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)