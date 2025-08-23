"""Error formatter implementation for formatting exceptions."""

import traceback
import orjson
from typing import Optional, Tuple

from app.services.sse_formatter.interfaces import SseFormatter

from .exceptions import (
    AuthenticationException,
    AuthorizationException,
    ExternalApiException,
    ExternalApiNotFoundException,
    ExternalApiOverloadedException,
    ExternalApiServerException,
    HttpClientException,
    PipelineException,
    RateLimitException,
    RequestTooLargeException,
    TransformerException,
)
from .interfaces import ErrorFormatter
from .models import ClaudeError, ClaudeErrorDetail


class ApiErrorFormatter(ErrorFormatter):
    """Formats exceptions for API responses."""

    def __init__(self):
        pass

    def format_for_sse(self, exc: PipelineException) -> Tuple[ClaudeError, str]:
        """Format exception for SSE response."""
        error_type = self.get_error_type(exc)

        # Create structured error data
        error_data = ClaudeError(error=ClaudeErrorDetail(type=error_type, message='\n'.join(traceback.format_exception(exc))))

        if exc.correlation_id:
            error_data.request_id = exc.correlation_id

        json_data = orjson.dumps(error_data.to_dict()).decode('utf-8')
        sse_event = f'event: error\ndata: {json_data}\n\n'

        return error_data, sse_event

    def get_error_type(self, exc: PipelineException) -> str:
        """Map domain exceptions to API error types."""
        match exc:
            case AuthenticationException():
                return 'authentication_error'
            case AuthorizationException():
                return 'permission_error'
            case ExternalApiNotFoundException():
                return 'not_found_error'
            case RequestTooLargeException():
                return 'request_too_large'
            case RateLimitException():
                return 'rate_limit_error'
            case ExternalApiServerException():
                return 'api_error'
            case ExternalApiOverloadedException():
                return 'overloaded_error'
            case TransformerException():
                return 'invalid_request_error'
            case HttpClientException():
                return 'api_error'
            case ExternalApiException():
                return 'api_error'
            case _:
                return 'api_error'
