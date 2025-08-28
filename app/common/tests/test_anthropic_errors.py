"""Tests for anthropic_errors utility module."""

from unittest.mock import Mock

from app.common.anthropic_errors import extract_error_message, map_http_status_to_anthropic_error


class TestMapHttpStatusToAnthropicError:
    """Test HTTP status code to Anthropic error type mapping."""

    def test_known_error_codes(self):
        """Test mapping for known error codes."""
        assert map_http_status_to_anthropic_error(400) == 'invalid_request_error'
        assert map_http_status_to_anthropic_error(401) == 'authentication_error'
        assert map_http_status_to_anthropic_error(403) == 'permission_error'
        assert map_http_status_to_anthropic_error(404) == 'not_found_error'
        assert map_http_status_to_anthropic_error(413) == 'request_too_large'
        assert map_http_status_to_anthropic_error(429) == 'rate_limit_error'
        assert map_http_status_to_anthropic_error(500) == 'api_error'
        assert map_http_status_to_anthropic_error(529) == 'overloaded_error'

    def test_unknown_error_codes(self):
        """Test fallback for unknown error codes."""
        assert map_http_status_to_anthropic_error(422) == 'api_error'
        assert map_http_status_to_anthropic_error(503) == 'api_error'
        assert map_http_status_to_anthropic_error(999) == 'api_error'


class TestExtractErrorMessage:
    """Test error message extraction from HTTP responses."""

    def test_extract_from_json_response(self):
        """Test extracting message from JSON error response."""
        # Mock httpx.HTTPStatusError with JSON response
        mock_response = Mock()
        mock_response.text = '{"error": {"message": "Invalid API key"}}'
        mock_response.json.return_value = {'error': {'message': 'Invalid API key'}}
        mock_response.status_code = 401
        mock_response.reason_phrase = 'Unauthorized'

        mock_error = Mock()
        mock_error.response = mock_response

        result = extract_error_message(mock_error)
        assert result == 'Invalid API key'

    def test_extract_fallback_with_response_text(self):
        """Test fallback message with response text."""
        # Mock httpx.HTTPStatusError without JSON error structure
        mock_response = Mock()
        mock_response.text = 'Rate limit exceeded'
        mock_response.json.return_value = {'message': 'Rate limit exceeded'}  # No "error" key
        mock_response.status_code = 429
        mock_response.reason_phrase = 'Too Many Requests'

        mock_error = Mock()
        mock_error.response = mock_response

        result = extract_error_message(mock_error)
        assert result == 'HTTP 429: Too Many Requests - Rate limit exceeded'

    def test_extract_fallback_json_parsing_fails(self):
        """Test fallback when JSON parsing fails."""
        # Mock httpx.HTTPStatusError with invalid JSON
        mock_response = Mock()
        mock_response.text = 'Internal Server Error'
        mock_response.json.side_effect = Exception('Invalid JSON')
        mock_response.status_code = 500
        mock_response.reason_phrase = 'Internal Server Error'

        mock_error = Mock()
        mock_error.response = mock_response

        result = extract_error_message(mock_error)
        assert result == 'HTTP 500: Internal Server Error - Internal Server Error'

    def test_extract_fallback_no_response_text(self):
        """Test fallback when response text cannot be read."""
        # Mock httpx.HTTPStatusError with unreadable response text
        mock_response = Mock()

        # Create a property that raises an exception
        def text_property():
            raise Exception('Cannot read response')

        type(mock_response).text = property(lambda self: text_property())
        mock_response.status_code = 503
        mock_response.reason_phrase = 'Service Unavailable'

        mock_error = Mock()
        mock_error.response = mock_response

        result = extract_error_message(mock_error)
        assert result == 'HTTP 503: Service Unavailable - <unable to read response>'
