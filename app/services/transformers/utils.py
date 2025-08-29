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


class AddHeaderTransformer(RequestTransformer):
    """Generic header transformer that adds any header with configurable key, prefix, value, and suffix.

    Adds a header to the request with full control over its construction.
    """

    def __init__(self, logger, key: str, value: str, prefix: str = '', suffix: str = ''):
        """Initialize transformer.

        Args:
            logger: Logger instance
            key: Header name/key to add
            value: Header value (used literally, no resolution)
            prefix: Text to prepend to the value (default: '')
            suffix: Text to append to the value (default: '')
        """
        self.logger = logger
        self.key = key
        self.value = value
        self.prefix = prefix
        self.suffix = suffix

    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Add header with configured key, prefix, value, and suffix."""
        request: dict[str, Any] = params['request']
        headers: dict[str, str] = params['headers']

        # Construct header value with prefix and suffix
        header_value = f'{self.prefix}{self.value}{self.suffix}'

        # Add header
        headers[self.key] = header_value

        return request, headers
