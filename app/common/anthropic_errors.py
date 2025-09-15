"""Anthropic API error handling utilities."""

import httpx


def map_http_status_to_anthropic_error(status_code: int) -> str:
    """Map HTTP status codes to Anthropic error types."""
    mapping = {
        400: 'invalid_request_error',
        401: 'authentication_error',
        403: 'permission_error',
        404: 'not_found_error',
        413: 'request_too_large',
        429: 'rate_limit_error',
        500: 'api_error',
        529: 'overloaded_error',
    }
    return mapping.get(status_code, 'api_error')


def extract_error_message(http_error: httpx.HTTPStatusError) -> str:
    """Extract error message from HTTP response."""
    try:
        if hasattr(http_error.response, 'text') and http_error.response.text:
            response_json = http_error.response.json()
            if isinstance(response_json, dict) and 'error' in response_json:
                return response_json['error'].get('message', str(http_error))
    except Exception:
        pass

    # Fallback: include response body in the message, not just status
    response_text = ''
    try:
        response_text = http_error.response.text
    except Exception:
        response_text = '<unable to read response>'

    return f'HTTP {http_error.response.status_code}: {http_error.response.reason_phrase} - {response_text}'
