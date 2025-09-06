"""Tests for Provider and ProviderManager classes."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request

from app.common.dumper import Dumper, DumpHandles
from app.common.models import AnthropicRequest
from app.config.user_models import ModelConfig, ProviderConfig
from app.services.provider import Provider, ProviderManager
from app.services.transformer_loader import TransformerLoader
from app.services.transformers import RequestTransformer, ResponseTransformer


class MockRequestTransformer(RequestTransformer):
    """Mock request transformer for testing."""

    def __init__(self, name='mock_request'):
        self.name = name

    async def transform(self, params):
        request = params['request']
        headers = params['headers']
        # Simple transformation - add a test header
        headers[f'{self.name}-processed'] = 'true'
        return request, headers


class MockResponseTransformer(ResponseTransformer):
    """Mock response transformer that tracks state preservation."""

    def __init__(self, name='mock_response'):
        self.name = name
        self.chunk_calls = []
        self.response_calls = []

    async def transform_chunk(self, params):
        """Track chunk processing and state preservation."""
        chunk = params['chunk']

        # Record this call for verification
        call_info = {'chunk_size': len(chunk), 'has_sse_state': 'sse_state' in params, 'state_id': id(params.get('sse_state', {})) if 'sse_state' in params else None}
        self.chunk_calls.append(call_info)

        # Initialize or preserve state
        if 'sse_state' not in params:
            params['sse_state'] = {'transformer': self.name, 'chunk_count': 0}

        params['sse_state']['chunk_count'] += 1

        # Add transformer identifier to chunk
        modified_chunk = chunk.replace(b'data: ', f'data: [{self.name}] '.encode())
        yield modified_chunk

    async def transform_response(self, params):
        """Track response processing."""
        response = params['response']
        self.response_calls.append({'response_id': response.get('id', 'unknown')})

        # Add transformer identifier - handle both string and list content
        if 'content' in response:
            content = response['content']
            if isinstance(content, list):
                # Modify text content blocks
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text' and 'text' in block:
                        block['text'] = f'[{self.name}] {block["text"]}'
            else:
                # Handle string content (legacy format)
                response['content'] = f'[{self.name}] {content}'

        return response


class TestProvider:
    """Test cases for the Provider class."""

    @pytest.fixture
    def mock_transformer_loader(self):
        """Create mock transformer loader."""
        loader = Mock(spec=TransformerLoader)
        loader.load_transformers.return_value = []
        return loader

    @pytest.fixture
    def mock_request_transformer(self):
        """Create mock request transformer."""
        return MockRequestTransformer('test_request')

    @pytest.fixture
    def mock_response_transformer(self):
        """Create mock response transformer."""
        return MockResponseTransformer('test_response')

    @pytest.fixture
    def provider_config(self):
        """Create test provider configuration."""
        return ProviderConfig(name='test-provider', url='https://api.test.com/v1/chat', timeout=30.0, transformers={'request': ['test_request'], 'response': ['test_response']})

    @pytest.fixture
    def provider(self, provider_config, mock_transformer_loader, mock_request_transformer, mock_response_transformer):
        """Create Provider instance for testing."""

        # Mock transformer loader to return our test transformers
        def mock_load_transformers(names):
            if names == ['test_request']:
                return [mock_request_transformer]
            elif names == ['test_response']:
                return [mock_response_transformer]
            else:
                return []

        mock_transformer_loader.load_transformers.side_effect = mock_load_transformers

        provider = Provider(provider_config, mock_transformer_loader)
        return provider

    @pytest.fixture
    def mock_anthropic_request(self):
        """Create mock Anthropic request."""
        return AnthropicRequest(model='claude-3-sonnet', messages=[{'role': 'user', 'content': 'Hello'}], stream=True)

    @pytest.fixture
    def mock_fastapi_request(self):
        """Create mock FastAPI request."""
        request = Mock(spec=Request)
        request.headers = {'authorization': 'Bearer test-token'}
        return request

    @pytest.fixture
    def mock_dumper(self):
        """Create mock dumper."""
        dumper = Mock(spec=Dumper)
        from app.common.dumper import DumpFiles

        handles = DumpHandles(files=DumpFiles(), correlation_id='test-correlation', base_path='/tmp')
        dumper.begin.return_value = handles
        return dumper

    @pytest.mark.asyncio
    async def test_process_request_returns_json_response(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test that process_request returns proper JSON response."""

        # Mock HTTP response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            'id': 'msg_test123',
            'model': 'claude-3-haiku',
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Hello world'}],
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        }

        # Mock the HTTP client
        provider.http_client.post = AsyncMock(return_value=mock_response)

        # Process the request
        handles = mock_dumper.begin.return_value
        result = await provider.process_request(mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles)

        # Verify we got the expected JSON response
        assert result['id'] == 'msg_test123'
        assert result['model'] == 'claude-3-haiku'
        assert result['content'][0]['text'] == '[test_response] Hello world'  # Transformed by mock transformer
        assert result['usage']['input_tokens'] == 10


class TestProviderManager:
    """Test cases for the ProviderManager class."""

    @pytest.fixture
    def providers_config(self):
        """Create test providers configuration."""
        return [ProviderConfig(name='provider1', url='https://api1.test.com', transformers={}), ProviderConfig(name='provider2', url='https://api2.test.com', transformers={})]

    @pytest.fixture
    def models_config(self):
        """Create test models configuration."""
        return [ModelConfig(alias='model1', provider='provider1', id='actual-model-1'), ModelConfig(alias='model2', provider='provider2', id='actual-model-2')]

    @pytest.fixture
    def mock_transformer_loader(self):
        """Create mock transformer loader."""
        loader = Mock(spec=TransformerLoader)
        loader.load_transformers.return_value = []
        return loader

    def test_provider_loading(self, providers_config, models_config, mock_transformer_loader):
        """Test that providers are loaded correctly."""
        manager = ProviderManager(providers_config, models_config, mock_transformer_loader)

        assert len(manager.providers) == 2
        assert 'provider1' in manager.providers
        assert 'provider2' in manager.providers

    def test_model_mapping(self, providers_config, models_config, mock_transformer_loader):
        """Test that model aliases are mapped correctly."""
        manager = ProviderManager(providers_config, models_config, mock_transformer_loader)

        # Test valid mapping
        provider, model_id = manager.get_provider_for_model('model1')
        assert provider is not None
        assert model_id == 'actual-model-1'

        # Test invalid mapping
        result = manager.get_provider_for_model('nonexistent')
        assert result is None

    def test_list_operations(self, providers_config, models_config, mock_transformer_loader):
        """Test listing providers and models."""
        manager = ProviderManager(providers_config, models_config, mock_transformer_loader)

        providers = manager.list_providers()
        assert set(providers) == {'provider1', 'provider2'}

        models = manager.list_models()
        assert set(models) == {'model1', 'model2'}

    def test_get_provider_by_name(self, providers_config, models_config, mock_transformer_loader):
        """Test getting provider by name."""
        manager = ProviderManager(providers_config, models_config, mock_transformer_loader)

        provider = manager.get_provider_by_name('provider1')
        assert provider is not None
        assert provider.name == 'provider1'

        provider = manager.get_provider_by_name('nonexistent')
        assert provider is None
