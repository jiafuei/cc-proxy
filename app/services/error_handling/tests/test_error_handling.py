"""Tests for error handling service."""

from unittest.mock import Mock

import httpx
import pytest

from app.services.error_handling.error_formatter import ApiErrorFormatter
from app.services.error_handling.exception_mapper import HttpExceptionMapper
from app.services.error_handling.exceptions import (
    AuthenticationException,
    HttpClientException,
    RateLimitException,
)


@pytest.fixture
def exception_mapper():
    return HttpExceptionMapper()


@pytest.fixture
def error_formatter():
    return ApiErrorFormatter()


def test_map_401_to_authentication_exception(exception_mapper):
    """Test mapping 401 status to AuthenticationException."""
    response = Mock(status_code=401, text='Unauthorized', is_stream_consumed=True)
    httpx_error = httpx.HTTPStatusError('401', request=Mock(), response=response)

    result = exception_mapper.map_httpx_exception(httpx_error, 'test-id')

    assert isinstance(result, AuthenticationException)
    assert result.correlation_id == 'test-id'
    assert result.status_code == 401


def test_map_429_to_rate_limit_exception(exception_mapper):
    """Test mapping 429 status to RateLimitException."""
    response = Mock(status_code=429, text='Too Many Requests', is_stream_consumed=True)
    httpx_error = httpx.HTTPStatusError('429', request=Mock(), response=response)

    result = exception_mapper.map_httpx_exception(httpx_error, 'test-id')

    assert isinstance(result, RateLimitException)
    assert result.status_code == 429


def test_map_request_error_to_http_client_exception(exception_mapper):
    """Test mapping request errors to HttpClientException."""
    httpx_error = httpx.RequestError('Connection failed')

    result = exception_mapper.map_httpx_exception(httpx_error, 'test-id')

    assert isinstance(result, HttpClientException)
    assert 'HTTP client error' in result.message


def test_get_error_type_for_authentication(error_formatter):
    """Test getting error type for authentication exception."""
    exc = AuthenticationException('Unauthorized', correlation_id='test-id')

    error_type = error_formatter.get_error_type(exc)

    assert error_type == 'authentication_error'


def test_get_error_type_for_rate_limit(error_formatter):
    """Test getting error type for rate limit exception."""
    exc = RateLimitException('Rate limit exceeded', correlation_id='test-id')

    error_type = error_formatter.get_error_type(exc)

    assert error_type == 'rate_limit_error'


def test_format_exception_for_sse(error_formatter):
    """Test formatting exception for SSE response."""
    exc = AuthenticationException('Unauthorized', correlation_id='test-id')

    error_data, sse_event = error_formatter.format_for_sse(exc)

    assert error_data.error.type == 'authentication_error'
    assert error_data.request_id == 'test-id'
    assert 'event: error' in sse_event
    assert 'authentication_error' in sse_event
