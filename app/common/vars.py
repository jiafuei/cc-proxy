from contextvars import ContextVar

from .request_context import RequestContext

# Single ContextVar for entire request context
request_context_var: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())


def get_correlation_id() -> str:
    """Get correlation ID from request context."""
    ctx = request_context_var.get()
    return ctx.correlation_id


# New context accessors
def get_request_context() -> RequestContext:
    """Get current request context."""
    return request_context_var.get()


def set_request_context(context: RequestContext) -> None:
    """Set request context."""
    request_context_var.set(context)
