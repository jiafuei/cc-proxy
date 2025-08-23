"""Exception mapper implementation for converting external exceptions to domain exceptions."""

from typing import Optional

import httpx
from fastapi import status as Status

from app.common.utils import get_correlation_id

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
)
from .interfaces import ExceptionMapper


class HttpExceptionMapper(ExceptionMapper):
    """Maps HTTP exceptions to domain exceptions."""

    def map_httpx_exception(self, exc: httpx.HTTPError, correlation_id: Optional[str] = None) -> PipelineException:
        """Convert httpx exceptions to domain exceptions."""
        if not correlation_id:
            correlation_id = get_correlation_id()

        if isinstance(exc, httpx.HTTPStatusError):
            return self._map_status_error(exc, correlation_id)
        elif isinstance(exc, httpx.RequestError):
            return HttpClientException(f'HTTP client error: {str(exc)}', correlation_id=correlation_id)
        else:
            return HttpClientException(f'Unknown HTTP error: {str(exc)}', correlation_id=correlation_id)

    def _map_status_error(self, exc: httpx.HTTPStatusError, correlation_id: Optional[str] = None) -> ExternalApiException:
        """Map HTTP status errors to specific domain exceptions."""
        response_text = exc.response.text if exc.response and exc.response.is_stream_consumed else 'server error'
        status_code = exc.response.status_code if exc.response else 0

        error_message = f'External API error: {response_text}' if response_text else str(exc)

        match status_code:
            case Status.HTTP_401_UNAUTHORIZED:
                return AuthenticationException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case Status.HTTP_403_FORBIDDEN:
                return AuthorizationException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case Status.HTTP_404_NOT_FOUND:
                return ExternalApiNotFoundException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case Status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
                return RequestTooLargeException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case Status.HTTP_429_TOO_MANY_REQUESTS:
                return RateLimitException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case Status.HTTP_500_INTERNAL_SERVER_ERROR:
                return ExternalApiServerException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case 529:  # Service overloaded
                return ExternalApiOverloadedException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
            case _:
                return ExternalApiException(error_message, status_code=status_code, response_body=response_text, correlation_id=correlation_id)
