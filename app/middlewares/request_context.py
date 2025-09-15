import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.request_context import RequestContext
from app.common.vars import request_context_var


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Unified middleware for request context management.
    Replaces both ContextMiddleware and CorrelationIdMiddleware.
    """

    def __init__(self, app, correlation_header: str = 'X-Correlation-ID'):
        super().__init__(app)
        self.correlation_header = correlation_header

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Create or use existing correlation ID
        correlation_id = request.headers.get(self.correlation_header)
        if not correlation_id:
            correlation_id = uuid.uuid4().hex

        # Create request context
        context = RequestContext(correlation_id=correlation_id, path=str(request.url.path), method=request.method)

        # Store in request state for access by dependencies
        request.state.request_context = context

        # Set context variable and process request
        request_context_var.set(context)
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.correlation_header] = context.correlation_id

        return response
