"""API error mapping helpers for channel routers."""

from __future__ import annotations

import httpx

ERROR_MAPPING = {
    400: 'invalid_request_error',
    401: 'authentication_error',
    403: 'permission_error',
    404: 'not_found_error',
    413: 'request_too_large',
    429: 'rate_limit_error',
    500: 'api_error',
    529: 'overloaded_error',
}


def map_http_status_to_anthropic_error(status_code: int) -> str:
    """Translate HTTP status codes into Anthropic error types."""

    return ERROR_MAPPING.get(status_code, 'api_error')


def extract_error_message(http_error: httpx.HTTPStatusError) -> str:
    """Pull a descriptive message out of a provider error."""

    try:
        if getattr(http_error.response, 'text', None):
            response_json = http_error.response.json()
            if isinstance(response_json, dict) and 'error' in response_json:
                return response_json['error'].get('message', str(http_error))
    except Exception:
        pass

    try:
        response_text = http_error.response.text
    except Exception:
        response_text = '<unable to read response>'

    return f'HTTP {http_error.response.status_code}: {http_error.response.reason_phrase} - {response_text}'


__all__ = ['extract_error_message', 'map_http_status_to_anthropic_error']
