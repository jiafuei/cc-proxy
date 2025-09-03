"""Tests for AnthropicHeadersTransformer."""

from unittest.mock import Mock

import pytest

from app.config.user_models import ProviderConfig
from app.services.transformers.anthropic import AnthropicHeadersTransformer


class TestAnthropicHeadersTransformer:
    """Test cases for AnthropicHeadersTransformer."""

    @pytest.fixture
    def transformer_x_api_key(self):
        """Create transformer instance with x-api-key auth header."""
        logger = Mock()
        return AnthropicHeadersTransformer(logger, 'x-api-key')

    @pytest.fixture
    def transformer_authorization(self):
        """Create transformer instance with authorization auth header."""
        logger = Mock()
        return AnthropicHeadersTransformer(logger, 'authorization')

    @pytest.mark.asyncio
    async def test_filter_headers_keeps_anthropic_prefixes(self, transformer_x_api_key):
        """Test that only Anthropic-compatible headers are kept."""
        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-123',
                'x-custom': 'value',
                'anthropic-version': '2023-06-01',
                'user-agent': 'test-client',
                'content-type': 'application/json',  # Should be filtered out
                'host': 'api.anthropic.com',  # Should be filtered out
                'authorization': 'Bearer sk-ant-456',
            },
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Should keep x-, anthropic-, user-, and authorization prefixed headers
        expected_headers = {
            'x-api-key': 'sk-ant-123',
            'x-custom': 'value',
            'anthropic-version': '2023-06-01',
            'user-agent': 'test-client',
            'authorization': 'Bearer sk-ant-456',
        }
        assert filtered_headers == expected_headers
        assert request == params['request']

    @pytest.mark.asyncio
    async def test_injects_x_api_key_from_provider_config(self, transformer_x_api_key):
        """Test that x-api-key is injected from provider config."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {
            'request': {'messages': []},
            'headers': {
                'x-custom': 'value',
                'user-agent': 'test-client',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # API key should be injected as x-api-key from config
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'
        assert 'authorization' not in filtered_headers

    @pytest.mark.asyncio
    async def test_injects_authorization_from_provider_config(self, transformer_authorization):
        """Test that authorization header is injected from provider config."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {
            'request': {'messages': []},
            'headers': {
                'x-custom': 'value',
                'user-agent': 'test-client',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_authorization.transform(params)

        # API key should be injected as authorization with Bearer prefix from config
        assert filtered_headers['authorization'] == 'Bearer sk-ant-config-key-123'
        assert 'x-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_removes_authorization_when_injecting_x_api_key(self, transformer_x_api_key):
        """Test that authorization header is removed when x-api-key is injected from config."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer sk-ant-old-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Config API key should replace authorization header with x-api-key
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'
        assert 'authorization' not in filtered_headers

    @pytest.mark.asyncio
    async def test_removes_x_api_key_when_injecting_authorization(self, transformer_authorization):
        """Test that x-api-key header is removed when authorization is injected from config."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-old-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_authorization.transform(params)

        # Config API key should replace x-api-key header with authorization
        assert filtered_headers['authorization'] == 'Bearer sk-ant-config-key-123'
        assert 'x-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_no_api_key_in_config_preserves_client_headers(self, transformer_x_api_key):
        """Test that client headers are preserved when no API key in config."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key='',  # Empty API key
        )

        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-client-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Client's x-api-key should be preserved
        assert filtered_headers['x-api-key'] == 'sk-ant-client-key'

    @pytest.mark.asyncio
    async def test_no_provider_config_preserves_headers(self, transformer_x_api_key):
        """Test behavior when provider_config is not in params."""
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer sk-ant-client-key',
                'x-custom': 'value',
            },
            # No provider_config
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Headers should be filtered but not transformed
        assert 'authorization' in filtered_headers
        assert filtered_headers['authorization'] == 'Bearer sk-ant-client-key'
        assert 'x-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_overrides_existing_x_api_key_with_config(self, transformer_x_api_key):
        """Test that config API key overrides existing x-api-key header."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-old-client-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config,
        }

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Config API key should override client's x-api-key
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'

    @pytest.mark.asyncio
    async def test_empty_headers_with_provider_config_x_api_key(self, transformer_x_api_key):
        """Test with empty headers but provider config with API key using x-api-key."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {'request': {'messages': []}, 'headers': {}, 'provider_config': provider_config}

        request, filtered_headers = await transformer_x_api_key.transform(params)

        # Should only contain the injected x-api-key
        assert filtered_headers == {'x-api-key': 'sk-ant-config-key-123'}

    @pytest.mark.asyncio
    async def test_empty_headers_with_provider_config_authorization(self, transformer_authorization):
        """Test with empty headers but provider config with API key using authorization."""
        provider_config = ProviderConfig(name='anthropic-test', url='https://api.anthropic.com/v1/messages', api_key='sk-ant-config-key-123')

        params = {'request': {'messages': []}, 'headers': {}, 'provider_config': provider_config}

        request, filtered_headers = await transformer_authorization.transform(params)

        # Should only contain the injected authorization header with Bearer prefix
        assert filtered_headers == {'authorization': 'Bearer sk-ant-config-key-123'}
