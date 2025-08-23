import contextvars
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ContextMiddleware(BaseHTTPMiddleware):
    """Adds a context to the current FastAPI request"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request.state.ctx = contextvars.copy_context()

        response = await call_next(request)

        return response
