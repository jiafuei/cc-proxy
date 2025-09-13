"""Tests for Gemini API key transformer utility."""

from unittest.mock import MagicMock

import pytest

from app.config.user_models import ProviderConfig
from app.services.transformers.utils import GeminiApiKeyTransformer


class TestGeminiApiKeyTransformer:
    """Test cases for GeminiApiKeyTransformer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.transformer = GeminiApiKeyTransformer(self.logger)

    @pytest.mark.parametrize(
        'api_key_source,expected_key', [('provider_config', 'test-api-key-123'), ('authorization_header', 'sk-test456'), ('x_goog_api_key_header', 'direct-api-key')]
    )
    @pytest.mark.asyncio
    async def test_adds_api_key_from_various_sources(self, api_key_source, expected_key):
        """Test adding API key from various sources as query parameter."""
        base_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent'

        if api_key_source == 'provider_config':
            provider_config = ProviderConfig(name='test_provider', url=base_url, api_key=expected_key)
            headers = {}
        elif api_key_source == 'authorization_header':
            provider_config = ProviderConfig(name='test_provider', url=base_url, api_key='')
            headers = {'authorization': f'Bearer {expected_key}'}
        elif api_key_source == 'x_goog_api_key_header':
            provider_config = ProviderConfig(name='test_provider', url=base_url, api_key='')
            headers = {'x-goog-api-key': expected_key}

        params = {'request': {'model': 'test'}, 'headers': headers, 'provider_config': provider_config}

        request, headers_result = await self.transformer.transform(params)

        # Request and headers should be unchanged
        assert request == {'model': 'test'}
        assert headers_result == headers

        # URL should now include API key as query parameter
        expected_url = f'{base_url}?key={expected_key}'
        assert provider_config.url == expected_url

    @pytest.mark.asyncio
    async def test_handles_url_with_existing_query_params(self):
        """Test handling URLs that already have query parameters."""
        provider_config = ProviderConfig(
            name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?alt=json', api_key='test-key'
        )

        params = {'request': {'model': 'test'}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Should preserve existing query params and add key
        assert 'alt=json' in provider_config.url
        assert 'key=test-key' in provider_config.url

    @pytest.mark.asyncio
    async def test_provider_config_priority_over_headers(self):
        """Test that provider config API key takes priority over headers."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='config-key')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer header-key'}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Should use provider config key, not header key
        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=config-key'
        assert provider_config.url == expected_url

    @pytest.mark.asyncio
    async def test_handles_missing_provider_config(self):
        """Test behavior when provider_config is missing."""
        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer test-key'}}

        request, headers = await self.transformer.transform(params)

        # Should return unchanged when no provider config
        assert request == {'model': 'test'}
        assert headers == {'authorization': 'Bearer test-key'}

    @pytest.mark.asyncio
    async def test_handles_missing_api_key(self):
        """Test behavior when no API key is found."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # URL should remain unchanged
        assert provider_config.url == 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent'

        # Should log warning
        self.logger.warning.assert_called_with('No API key found for Gemini authentication')

    @pytest.mark.parametrize('auth_value,expected_key', [('Bearer   sk-test-with-spaces   ', 'sk-test-with-spaces'), ('raw-api-key', 'raw-api-key')])
    @pytest.mark.asyncio
    async def test_handles_authorization_header_formats(self, auth_value, expected_key):
        """Test handling authorization header with and without Bearer prefix."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': auth_value}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Should extract key correctly regardless of Bearer prefix
        expected_url = f'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={expected_key}'
        assert provider_config.url == expected_url
