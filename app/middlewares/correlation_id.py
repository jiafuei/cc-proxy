import contextvars
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.utils import generate_correlation_id
from app.common.vars import correlation_id as cid_var


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to inject correlation ID into requests for tracing."""

    def __init__(self, app, header_name: str = 'X-Correlation-ID'):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = generate_correlation_id()

        ctx: contextvars.Context = request.state.ctx

        response = await ctx.run(inject_and_call_next, request, call_next, correlation_id)
        response.headers[self.header_name] = correlation_id

        return response


async def inject_and_call_next(request: Request, call_next: Callable, correlation_id: str):
    cid_var.set(correlation_id)
    return await call_next(request)
