"""Generic transformers for common operations."""

from typing import Any, Dict, Tuple

from app.config.user_models import ProviderConfig
from app.services.transformers.interfaces import RequestTransformer


class UrlPathTransformer(RequestTransformer):
    """Generic URL path transformer that modifies provider config URL.

    Strips trailing slashes from the base URL and appends a user-defined path.
    """

    def __init__(self, logger, path: str):
        """Initialize transformer.

        Args:
            logger: Logger instance
            path: Path to append to the base URL (e.g., '/v1/chat/completions')
        """
        self.logger = logger
        self.path = path

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Modify provider config URL by appending configured path."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        if 'provider_config' in params:
            provider_config: ProviderConfig = params['provider_config']
            base_url = provider_config.url.strip('/')
            path = self.path if self.path.startswith('/') or not self.path else '/' + self.path
            provider_config.url = base_url + path

        return request, headers


class AuthHeaderTransformer(RequestTransformer):
    """Generic authentication header transformer for any provider.

    Adds configurable authentication header with API key from provider config.
    """

    def __init__(self, logger, header_name: str = 'authorization', value_prefix: str = 'Bearer '):
        """Initialize transformer.

        Args:
            logger: Logger instance
            header_name: Name of the auth header (default: 'authorization')
            value_prefix: Prefix for the auth value (default: 'Bearer ')
        """
        self.logger = logger
        self.header_name = header_name
        self.value_prefix = value_prefix

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Add authentication header with API key from provider config."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']
        provider_config: ProviderConfig = params['provider_config']

        # Get API key from provider config
        api_key = provider_config.api_key

        # Add auth header
        headers[self.header_name] = f'{self.value_prefix}{api_key}'

        return request, headers
