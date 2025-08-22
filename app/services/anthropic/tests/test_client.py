import httpx
import pytest

from app.config.models import ConfigModel
from app.services.anthropic.client import AnthropicStreamingService


class TestAnthropicStreamingService:
    @pytest.fixture
    def config(self):
        return ConfigModel(anthropic_api_url='https://api.anthropic.com/v1/messages', anthropic_api_key='test-key')

    @pytest.fixture
    def service(self, config):
        client = httpx.AsyncClient()
        return AnthropicStreamingService(client, config)

    def test_initialization_with_config(self, config):
        client = httpx.AsyncClient()
        service = AnthropicStreamingService(client, config)

        assert service._api_url == 'https://api.anthropic.com/v1/messages'
        assert service._api_key == 'test-key'
        assert service._client == client

    def test_initialization_with_no_key(self):
        config = ConfigModel(anthropic_api_url='https://api.anthropic.com/v1/messages', anthropic_api_key=None)
        client = httpx.AsyncClient()
        service = AnthropicStreamingService(client, config)

        assert service._api_url == 'https://api.anthropic.com/v1/messages'
        assert service._api_key is None

    def test_initialization_with_custom_url(self):
        config = ConfigModel(anthropic_api_url='https://custom.anthropic-proxy.com/v1/messages', anthropic_api_key='test-key')
        client = httpx.AsyncClient()
        service = AnthropicStreamingService(client, config)

        assert service._api_url == 'https://custom.anthropic-proxy.com/v1/messages'
        assert service._api_key == 'test-key'
