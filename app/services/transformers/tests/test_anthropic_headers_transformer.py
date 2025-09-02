"""Tests for AnthropicHeadersTransformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.anthropic import AnthropicHeadersTransformer
from app.config.user_models import ProviderConfig


class TestAnthropicHeadersTransformer:
    """Test cases for AnthropicHeadersTransformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        logger = Mock()
        return AnthropicHeadersTransformer(logger)

    @pytest.mark.asyncio
    async def test_filter_headers_keeps_anthropic_prefixes(self, transformer):
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

        request, filtered_headers = await transformer.transform(params)

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
    async def test_injects_api_key_from_provider_config(self, transformer):
        """Test that API key is injected from provider config."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key='sk-ant-config-key-123'
        )
        
        params = {
            'request': {'messages': []},
            'headers': {
                'x-custom': 'value',
                'user-agent': 'test-client',
            },
            'provider_config': provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # API key should be injected from config
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'
        assert 'authorization' not in filtered_headers

    @pytest.mark.asyncio
    async def test_removes_authorization_when_injecting_api_key(self, transformer):
        """Test that authorization header is removed when API key is injected from config."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key='sk-ant-config-key-123'
        )
        
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer sk-ant-old-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # Config API key should replace authorization header
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'
        assert 'authorization' not in filtered_headers

    @pytest.mark.asyncio
    async def test_no_api_key_in_config_preserves_client_headers(self, transformer):
        """Test that client headers are preserved when no API key in config."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key=''  # Empty API key
        )
        
        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-client-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # Client's x-api-key should be preserved
        assert filtered_headers['x-api-key'] == 'sk-ant-client-key'

    @pytest.mark.asyncio
    async def test_no_provider_config_preserves_headers(self, transformer):
        """Test behavior when provider_config is not in params."""
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer sk-ant-client-key',
                'x-custom': 'value',
            }
            # No provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # Headers should be filtered but not transformed
        assert 'authorization' in filtered_headers
        assert filtered_headers['authorization'] == 'Bearer sk-ant-client-key'
        assert 'x-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_overrides_existing_x_api_key_with_config(self, transformer):
        """Test that config API key overrides existing x-api-key header."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key='sk-ant-config-key-123'
        )
        
        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-old-client-key',
                'x-custom': 'value',
            },
            'provider_config': provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # Config API key should override client's x-api-key
        assert filtered_headers['x-api-key'] == 'sk-ant-config-key-123'

    @pytest.mark.asyncio
    async def test_empty_headers_with_provider_config(self, transformer):
        """Test with empty headers but provider config with API key."""
        provider_config = ProviderConfig(
            name='anthropic-test',
            url='https://api.anthropic.com/v1/messages',
            api_key='sk-ant-config-key-123'
        )
        
        params = {
            'request': {'messages': []},
            'headers': {},
            'provider_config': provider_config
        }

        request, filtered_headers = await transformer.transform(params)

        # Should only contain the injected API key
        assert filtered_headers == {'x-api-key': 'sk-ant-config-key-123'}

