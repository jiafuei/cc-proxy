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

    @pytest.mark.asyncio
    async def test_adds_api_key_from_provider_config(self):
        """Test adding API key from provider config as query parameter."""
        provider_config = ProviderConfig(
            name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='test-api-key-123'
        )

        params = {'request': {'model': 'test'}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Request and headers should be unchanged
        assert request == {'model': 'test'}
        assert headers == {}

        # URL should now include API key as query parameter
        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=test-api-key-123'
        assert provider_config.url == expected_url

    @pytest.mark.asyncio
    async def test_adds_api_key_from_authorization_header(self):
        """Test adding API key from authorization header."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer sk-test456'}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=sk-test456'
        assert provider_config.url == expected_url

    @pytest.mark.asyncio
    async def test_adds_api_key_from_x_goog_api_key_header(self):
        """Test adding API key from x-goog-api-key header."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {'x-goog-api-key': 'direct-api-key'}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=direct-api-key'
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

    @pytest.mark.asyncio
    async def test_strips_bearer_prefix(self):
        """Test that Bearer prefix is properly stripped from authorization header."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'Bearer   sk-test-with-spaces   '}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Should strip Bearer prefix and whitespace
        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=sk-test-with-spaces'
        assert provider_config.url == expected_url

    @pytest.mark.asyncio
    async def test_handles_authorization_without_bearer(self):
        """Test handling authorization header without Bearer prefix."""
        provider_config = ProviderConfig(name='test_provider', url='https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent', api_key='')

        params = {'request': {'model': 'test'}, 'headers': {'authorization': 'raw-api-key'}, 'provider_config': provider_config}

        request, headers = await self.transformer.transform(params)

        # Should use raw value when no Bearer prefix
        expected_url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=raw-api-key'
        assert provider_config.url == expected_url
