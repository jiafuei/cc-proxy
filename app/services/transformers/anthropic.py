"""Anthropic transformers - pure passthrough implementations."""

from typing import Any, Dict, Mapping, Tuple

from fastapi import Request
from urllib.parse import urlparse

from app.config.user_models import ProviderConfig
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

    async def transform(self, request: Dict[str, Any], headers: Mapping[str, Any], config: ProviderConfig, original_request: Request) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Pure passthrough - incoming format is already Anthropic format."""
        headers = dict(headers | {'host': self.host, 'authorization': f'Bearer {self.api_key}'})
        headers.pop('content-length', None)
        return request, headers


class AnthropicResponseTransformer(ResponseTransformer):
    """Pure passthrough transformer for Anthropic responses."""

    async def transform_chunk(self, chunk: bytes) -> bytes:
        """Pure passthrough - response is already in correct format."""
        return chunk

    async def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Pure passthrough - response is already in correct format."""
        return response
