"""Error handling service for mapping exceptions to HTTP responses."""

import json
import uuid
from typing import Dict, Optional, Tuple

import httpx
from fastapi import status as Status

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


class ErrorHandlingService:
    """Service for handling errors and mapping them to appropriate responses."""

    def convert_httpx_exception(self, exc: httpx.HTTPError, correlation_id: Optional[str] = None) -> PipelineException:
        """Convert httpx exceptions to domain exceptions."""

        if isinstance(exc, httpx.HTTPStatusError):
            return self._map_status_error(exc, correlation_id)
        elif isinstance(exc, httpx.RequestError):
            return HttpClientException(f'HTTP client error: {str(exc)}', correlation_id=correlation_id)
        else:
            return HttpClientException(f'Unknown HTTP error: {str(exc)}', correlation_id=correlation_id)

    def _map_status_error(self, exc: httpx.HTTPStatusError, correlation_id: Optional[str] = None) -> ExternalApiException:
        """Map HTTP status errors to specific domain exceptions."""

        response_text = exc.response.text if exc.response else ''
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

    def get_error_response_data(self, exc: PipelineException) -> Tuple[Dict[str, str], str]:
        """Get structured error response data for SSE streams."""

        error_type = self._get_error_type(exc)

        # Create structured error data
        error_data = {'type': 'error', 'error': {'type': error_type, 'message': exc.message}}

        if exc.correlation_id:
            error_data['correlation_id'] = exc.correlation_id

        # Format as SSE event with proper JSON encoding
        json_data = json.dumps(error_data, ensure_ascii=False)
        sse_event = f'event: error\ndata: {json_data}'

        return error_data, sse_event

    def _get_error_type(self, exc: PipelineException) -> str:
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
                return 'transformation_error'
            case HttpClientException():
                return 'connection_error'
            case ExternalApiException():
                return 'external_api_error'
            case _:
                return 'pipeline_error'

    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID for request tracing."""
        return uuid.uuid4().hex
