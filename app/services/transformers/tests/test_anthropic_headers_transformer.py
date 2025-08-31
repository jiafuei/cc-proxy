"""Tests for AnthropicHeadersTransformer."""

import pytest
from unittest.mock import Mock

from app.services.transformers.anthropic import AnthropicHeadersTransformer


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
                'authorization': 'Bearer sk-ant-456'
            }
        }

        request, filtered_headers = await transformer.transform(params)

        expected_headers = {
            'x-api-key': 'sk-ant-123',
            'x-custom': 'value',
            'anthropic-version': '2023-06-01',
            'user-agent': 'test-client'
        }
        assert filtered_headers == expected_headers
        assert request == params['request']

    @pytest.mark.asyncio
    async def test_prefers_x_api_key_over_authorization(self, transformer):
        """Test that x-api-key is preferred over authorization header."""
        params = {
            'request': {'messages': []},
            'headers': {
                'x-api-key': 'sk-ant-123',
                'authorization': 'Bearer sk-ant-456'
            }
        }

        request, filtered_headers = await transformer.transform(params)

        # x-api-key should be kept, authorization should be removed
        assert 'x-api-key' in filtered_headers
        assert 'authorization' not in filtered_headers
        assert filtered_headers['x-api-key'] == 'sk-ant-123'

    @pytest.mark.asyncio
    async def test_converts_authorization_to_x_api_key_bearer_prefix(self, transformer):
        """Test conversion of Bearer authorization to x-api-key."""
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer sk-ant-456'
            }
        }

        request, filtered_headers = await transformer.transform(params)

        assert 'x-api-key' in filtered_headers
        assert 'authorization' not in filtered_headers
        assert filtered_headers['x-api-key'] == 'sk-ant-456'

    @pytest.mark.asyncio
    async def test_converts_authorization_to_x_api_key_case_insensitive(self, transformer):
        """Test case-insensitive Bearer prefix removal."""
        test_cases = [
            ('Bearer sk-ant-123', 'sk-ant-123'),
            ('bearer sk-ant-456', 'sk-ant-456'),
            ('BEARER sk-ant-789', 'sk-ant-789'),
            ('BeArEr sk-ant-abc', 'sk-ant-abc'),
        ]

        for auth_value, expected_key in test_cases:
            params = {
                'request': {'messages': []},
                'headers': {'authorization': auth_value}
            }

            request, filtered_headers = await transformer.transform(params)

            assert filtered_headers['x-api-key'] == expected_key

    @pytest.mark.asyncio
    async def test_converts_authorization_without_bearer_prefix(self, transformer):
        """Test conversion of raw API key in authorization header."""
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'sk-ant-raw-key'
            }
        }

        request, filtered_headers = await transformer.transform(params)

        assert 'x-api-key' in filtered_headers
        assert 'authorization' not in filtered_headers
        assert filtered_headers['x-api-key'] == 'sk-ant-raw-key'

    @pytest.mark.asyncio
    async def test_handles_bearer_with_extra_whitespace(self, transformer):
        """Test Bearer prefix removal with extra whitespace."""
        params = {
            'request': {'messages': []},
            'headers': {
                'authorization': 'Bearer  sk-ant-123  '
            }
        }

        request, filtered_headers = await transformer.transform(params)

        assert filtered_headers['x-api-key'] == 'sk-ant-123'

    @pytest.mark.asyncio
    async def test_no_authorization_header(self, transformer):
        """Test when no authorization header is present."""
        params = {
            'request': {'messages': []},
            'headers': {
                'x-custom': 'value',
                'user-agent': 'test-client'
            }
        }

        request, filtered_headers = await transformer.transform(params)

        expected_headers = {
            'x-custom': 'value',
            'user-agent': 'test-client'
        }
        assert filtered_headers == expected_headers
        assert 'x-api-key' not in filtered_headers

    @pytest.mark.asyncio
    async def test_empty_headers(self, transformer):
        """Test with empty headers."""
        params = {
            'request': {'messages': []},
            'headers': {}
        }

        request, filtered_headers = await transformer.transform(params)

        assert filtered_headers == {}