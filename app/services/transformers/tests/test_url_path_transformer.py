"""Tests for UrlPathTransformer."""

from unittest.mock import Mock

import pytest

from app.config.user_models import ProviderConfig
from app.services.transformers.utils import UrlPathTransformer


class TestUrlPathTransformer:
    """Test cases for the UrlPathTransformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with mock logger."""
        mock_logger = Mock()
        return UrlPathTransformer(mock_logger, '/v1/chat/completions')

    @pytest.fixture
    def provider_config(self):
        """Create a sample provider config."""
        return ProviderConfig(name='test_provider', url='https://api.example.com', api_key='test_key')

    @pytest.mark.asyncio
    async def test_strips_trailing_slash_and_appends_path(self, transformer, provider_config):
        """Test that trailing slashes are stripped and path is appended correctly."""
        provider_config.url = 'https://api.example.com/'

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}, 'provider_config': provider_config}

        request, headers = await transformer.transform(params)

        assert provider_config.url == 'https://api.example.com/v1/chat/completions'
        assert request == {'model': 'test'}
        assert headers == {'content-type': 'application/json'}

    @pytest.mark.asyncio
    async def test_handles_url_without_trailing_slash(self, transformer, provider_config):
        """Test URL without trailing slash."""
        provider_config.url = 'https://api.example.com'

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}, 'provider_config': provider_config}

        request, headers = await transformer.transform(params)

        assert provider_config.url == 'https://api.example.com/v1/chat/completions'

    @pytest.mark.asyncio
    async def test_handles_multiple_trailing_slashes(self, transformer, provider_config):
        """Test URL with multiple trailing slashes."""
        provider_config.url = 'https://api.example.com///'

        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}, 'provider_config': provider_config}

        request, headers = await transformer.transform(params)

        assert provider_config.url == 'https://api.example.com/v1/chat/completions'

    @pytest.mark.asyncio
    async def test_handles_missing_provider_config(self, transformer):
        """Test behavior when provider_config is missing from params."""
        params = {'request': {'model': 'test'}, 'headers': {'content-type': 'application/json'}}

        request, headers = await transformer.transform(params)

        # Should not raise error and return request/headers unchanged
        assert request == {'model': 'test'}
        assert headers == {'content-type': 'application/json'}

    @pytest.mark.asyncio
    async def test_custom_path_configuration(self, provider_config):
        """Test transformer with custom path configuration."""
        mock_logger = Mock()
        custom_transformer = UrlPathTransformer(mock_logger, '/api/v2/completions')
        provider_config.url = 'https://custom.api.com/'

        params = {'request': {'model': 'custom'}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await custom_transformer.transform(params)

        assert provider_config.url == 'https://custom.api.com/api/v2/completions'

    @pytest.mark.asyncio
    async def test_empty_path_configuration(self, provider_config):
        """Test transformer with empty path."""
        mock_logger = Mock()
        empty_path_transformer = UrlPathTransformer(mock_logger, '')
        provider_config.url = 'https://api.example.com/'

        params = {'request': {'model': 'test'}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await empty_path_transformer.transform(params)

        assert provider_config.url == 'https://api.example.com'

    @pytest.mark.asyncio
    async def test_path_starting_with_slash(self, provider_config):
        """Test that path starting with slash works correctly."""
        mock_logger = Mock()
        transformer = UrlPathTransformer(mock_logger, '/v1/models')
        provider_config.url = 'https://api.example.com'

        params = {'request': {}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await transformer.transform(params)

        assert provider_config.url == 'https://api.example.com/v1/models'

    @pytest.mark.asyncio
    async def test_path_without_leading_slash(self, provider_config):
        """Test path without leading slash."""
        mock_logger = Mock()
        transformer = UrlPathTransformer(mock_logger, 'v1/embeddings')
        provider_config.url = 'https://api.example.com/'

        params = {'request': {}, 'headers': {}, 'provider_config': provider_config}

        request, headers = await transformer.transform(params)

        assert provider_config.url == 'https://api.example.com/v1/embeddings'

    def test_constructor_parameters(self):
        """Test that constructor accepts and stores parameters correctly."""
        mock_logger = Mock()
        path = '/custom/endpoint'
        transformer = UrlPathTransformer(mock_logger, path)

        assert transformer.logger == mock_logger
        assert transformer.path == path
