"""Tests for error handling service."""

import json
from unittest.mock import Mock

import httpx
import pytest
from fastapi import status as Status

from app.services.pipeline.error_handler import ErrorHandlingService
from app.services.pipeline.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ExternalApiNotFoundException,
    ExternalApiOverloadedException,
    ExternalApiServerException,
    HttpClientException,
    RateLimitException,
    RequestTooLargeException,
)


class TestErrorHandlingService:
    @pytest.fixture
    def error_handler(self):
        return ErrorHandlingService()

    @pytest.fixture
    def correlation_id(self):
        return 'test-correlation-id'

    def test_generate_correlation_id(self, error_handler):
        correlation_id = error_handler.generate_correlation_id()

        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 32  # UUID hex string length

    def test_convert_httpx_request_error(self, error_handler, correlation_id):
        original_error = httpx.RequestError('Connection failed')

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, HttpClientException)
        assert 'HTTP client error: Connection failed' in result.message
        assert result.correlation_id == correlation_id

    def test_convert_httpx_status_error_401(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_401_UNAUTHORIZED
        response.text = 'Authentication failed'

        original_error = httpx.HTTPStatusError('401', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, AuthenticationException)
        assert result.status_code == 401
        assert result.response_body == 'Authentication failed'
        assert result.correlation_id == correlation_id

    def test_convert_httpx_status_error_403(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_403_FORBIDDEN
        response.text = 'Permission denied'

        original_error = httpx.HTTPStatusError('403', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, AuthorizationException)
        assert result.status_code == 403

    def test_convert_httpx_status_error_404(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_404_NOT_FOUND
        response.text = 'Not found'

        original_error = httpx.HTTPStatusError('404', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, ExternalApiNotFoundException)

    def test_convert_httpx_status_error_413(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        response.text = 'Request too large'

        original_error = httpx.HTTPStatusError('413', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, RequestTooLargeException)

    def test_convert_httpx_status_error_429(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_429_TOO_MANY_REQUESTS
        response.text = 'Rate limit exceeded'

        original_error = httpx.HTTPStatusError('429', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, RateLimitException)

    def test_convert_httpx_status_error_500(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = Status.HTTP_500_INTERNAL_SERVER_ERROR
        response.text = 'Internal server error'

        original_error = httpx.HTTPStatusError('500', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, ExternalApiServerException)

    def test_convert_httpx_status_error_529(self, error_handler, correlation_id):
        response = Mock()
        response.status_code = 529
        response.text = 'Service overloaded'

        original_error = httpx.HTTPStatusError('529', request=Mock(), response=response)

        result = error_handler.convert_httpx_exception(original_error, correlation_id)

        assert isinstance(result, ExternalApiOverloadedException)

    def test_get_error_response_data(self, error_handler):
        exception = AuthenticationException('Authentication failed', status_code=401, correlation_id='test-id')

        error_data, sse_event = error_handler.get_error_response_data(exception)

        assert error_data['type'] == 'error'
        assert error_data['error']['type'] == 'authentication_error'
        assert error_data['error']['message'] == 'Authentication failed'
        assert error_data['correlation_id'] == 'test-id'

        # Verify SSE format
        assert sse_event.startswith('event: error\ndata: ')

        # Verify JSON parsing
        json_part = sse_event[len('event: error\ndata: ') :]
        parsed_json = json.loads(json_part)
        assert parsed_json == error_data

    def test_get_error_type_mapping(self, error_handler):
        test_cases = [
            (AuthenticationException('test'), 'authentication_error'),
            (AuthorizationException('test'), 'permission_error'),
            (ExternalApiNotFoundException('test'), 'not_found_error'),
            (RequestTooLargeException('test'), 'request_too_large'),
            (RateLimitException('test'), 'rate_limit_error'),
            (ExternalApiServerException('test'), 'api_error'),
            (ExternalApiOverloadedException('test'), 'overloaded_error'),
            (HttpClientException('test'), 'connection_error'),
        ]

        for exception, expected_type in test_cases:
            error_type = error_handler._get_error_type(exception)
            assert error_type == expected_type
