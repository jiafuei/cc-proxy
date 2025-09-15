import pytest
from unittest.mock import Mock

from app.common.anthropic_errors import extract_error_message, map_http_status_to_anthropic_error


class TestMapHttpStatusToAnthropicError:
    """Test HTTP status code to Anthropic error type mapping."""

    @pytest.mark.parametrize("status_code,expected_error_type", [
        # Known error codes
        (400, 'invalid_request_error'),
        (401, 'authentication_error'),
        (403, 'permission_error'),
        (404, 'not_found_error'),
        (413, 'request_too_large'),
        (429, 'rate_limit_error'),
        (500, 'api_error'),
        (529, 'overloaded_error'),
        # Unknown error codes - fallback to api_error
        (422, 'api_error'),
        (503, 'api_error'),
        (999, 'api_error'),
    ])
    def test_status_code_mapping(self, status_code, expected_error_type):
        """Test HTTP status code to error type mapping."""
        assert map_http_status_to_anthropic_error(status_code) == expected_error_type


class TestExtractErrorMessage:
    """Test error message extraction from HTTP responses."""

    @pytest.mark.parametrize("scenario,mock_setup,expected_message", [
        # JSON response with error message
        ("json_response",
         lambda: {
             'text': '{"error": {"message": "Invalid API key"}}',
             'json_return': {'error': {'message': 'Invalid API key'}},
             'status_code': 401,
             'reason_phrase': 'Unauthorized'
         },
         'Invalid API key'),
        # Fallback with response text (no error key)
        ("fallback_with_text",
         lambda: {
             'text': 'Rate limit exceeded',
             'json_return': {'message': 'Rate limit exceeded'},
             'status_code': 429,
             'reason_phrase': 'Too Many Requests'
         },
         'HTTP 429: Too Many Requests - Rate limit exceeded'),
        # JSON parsing fails
        ("json_parsing_fails",
         lambda: {
             'text': 'Internal Server Error',
             'json_exception': Exception('Invalid JSON'),
             'status_code': 500,
             'reason_phrase': 'Internal Server Error'
         },
         'HTTP 500: Internal Server Error - Internal Server Error'),
        # Response text unreadable
        ("text_unreadable",
         lambda: {
             'text_exception': Exception('Cannot read response'),
             'status_code': 503,
             'reason_phrase': 'Service Unavailable'
         },
         'HTTP 503: Service Unavailable - <unable to read response>')
    ])
    def test_extract_error_message_scenarios(self, scenario, mock_setup, expected_message):
        """Test error message extraction from HTTP responses."""
        setup = mock_setup()
        
        mock_response = Mock()
        mock_response.status_code = setup['status_code']
        mock_response.reason_phrase = setup['reason_phrase']
        
        # Handle text property
        if 'text_exception' in setup:
            type(mock_response).text = property(lambda self: (_ for _ in ()).throw(setup['text_exception']))
        else:
            mock_response.text = setup['text']
        
        # Handle JSON method
        if 'json_exception' in setup:
            mock_response.json.side_effect = setup['json_exception']
        elif 'json_return' in setup:
            mock_response.json.return_value = setup['json_return']
        
        mock_error = Mock()
        mock_error.response = mock_response
        
        result = extract_error_message(mock_error)
        assert result == expected_message
