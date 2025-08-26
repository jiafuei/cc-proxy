"""Anthropic transformers - pure passthrough implementations."""

from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from app.services.transformers.interfaces import RequestTransformer, ResponseTransformer


class AnthropicAuthTransformer(RequestTransformer):
    """Pure passthrough transformer for Anthropic requests.

    Since incoming requests are already in Claude/Anthropic format,
    no transformation is needed.
    """

    def __init__(self, api_key: str = '', base_url: str = 'https://api.anthropic.com/v1/messages'):
        """Initialize with API credentials.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for Anthropic API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.host = urlparse(base_url).hostname

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Pure passthrough - incoming format is already Anthropic format."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        final_headers = {
            k: v
            for k, v in headers.items()
            if any(
                (
                    k.startswith(prefix)
                    for prefix in (
                        'x-',
                        'anthropic',
                        'user-',
                    )
                )
            )
        }
        final_headers = final_headers | {'authorization': f'Bearer {self.api_key}'}
        return request, final_headers


class AnthropicResponseTransformer(ResponseTransformer):
    """Pure passthrough transformer for Anthropic responses."""

    async def transform_chunk(self, params: Dict[str, Any]) -> bytes:
        """Pure passthrough - response is already in correct format."""
        return params['chunk']

    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pure passthrough - response is already in correct format."""
        return params['response']
