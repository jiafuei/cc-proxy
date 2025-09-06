import contextvars
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
        # Create request context
        correlation_id = request.headers.get(self.correlation_header)
        if not correlation_id:
            # For FastAPI requests, use normal UUID (32 characters)
            correlation_id = uuid.uuid4().hex
        
        context = RequestContext(
            correlation_id=correlation_id,
            path=str(request.url.path),
            method=request.method
        )
        
        # Store in request state for access by dependencies
        request.state.request_context = context
        
        # Capture current context and run request within it
        ctx = contextvars.copy_context()
        request.state.ctx = ctx
        
        # Run request with context set
        response = await ctx.run(self._run_with_context, request, call_next, context)
        
        # Add correlation ID to response headers
        response.headers[self.correlation_header] = context.correlation_id
        
        return response
    
    async def _run_with_context(
        self,
        request: Request,
        call_next: Callable,
        context: RequestContext
    ) -> Response:
        """Run request with context set in ContextVar."""
        request_context_var.set(context)
        return await call_next(request)