"""
File upload configuration middleware.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

class UploadConfigMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and "/upload" in request.url.path:
            if not hasattr(request.state, "upload_started"):
                request.state.upload_started = True
                # Configure upload settings
                request._max_files = 10  # Adjust this value as needed
                request._max_fields = 10
                request._max_file_size = 100 * 1024 * 1024  # 100MB, adjust as needed
        
        response = await call_next(request)
        return response
