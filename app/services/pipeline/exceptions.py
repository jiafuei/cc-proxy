"""Pipeline service domain exceptions."""

from typing import Optional


class PipelineException(Exception):
    """Base exception for pipeline operations."""

    def __init__(self, message: str, correlation_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.correlation_id = correlation_id


class ExternalApiException(PipelineException):
    """Exception for external API communication errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, correlation_id)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationException(ExternalApiException):
    """Authentication failed with external API."""

    pass


class AuthorizationException(ExternalApiException):
    """Authorization failed with external API."""

    pass


class RateLimitException(ExternalApiException):
    """Rate limit exceeded on external API."""

    pass


class RequestTooLargeException(ExternalApiException):
    """Request payload too large for external API."""

    pass


class ExternalApiNotFoundException(ExternalApiException):
    """External API endpoint not found."""

    pass


class ExternalApiServerException(ExternalApiException):
    """External API server error."""

    pass


class ExternalApiOverloadedException(ExternalApiException):
    """External API is overloaded."""

    pass


class TransformerException(PipelineException):
    """Exception during request/response transformation."""

    pass


class HttpClientException(PipelineException):
    """HTTP client communication error."""

    pass
