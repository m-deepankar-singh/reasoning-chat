"""
Core FastAPI application instance and configuration.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.core.middleware import LoggingMiddleware

settings = get_settings()

def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    # Add middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app

app = create_application()
