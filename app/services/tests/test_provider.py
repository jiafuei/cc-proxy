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
        return ProviderConfig(
            name='test-provider', 
            url='https://api.test.com/v1/chat', 
            timeout=30.0, 
            transformers={'request': ['test_request'], 'response': ['test_response']},
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
            ]
        )

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
    async def test_process_operation_returns_json_response(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test that process_operation returns proper JSON response."""

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

        # Process the request using the new method
        handles = mock_dumper.begin.return_value
        result = await provider.process_operation('messages', mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles)

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
        return [
            ProviderConfig(
                name='provider1', 
                url='https://api1.test.com', 
                transformers={},
                capabilities=[
                    {'operation': 'messages', 'class_name': 'MessagesCapability'},
                    {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
                ]
            ), 
            ProviderConfig(
                name='provider2', 
                url='https://api2.test.com', 
                transformers={},
                capabilities=[
                    {'operation': 'messages', 'class_name': 'MessagesCapability'},
                    {'operation': 'count_tokens', 'class_name': 'OpenAITokenCountCapability'}
                ]
            )
        ]

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


class TestProviderCapabilities:
    """Test capability functionality in Provider class."""

    @pytest.fixture
    def mock_transformer_loader(self):
        """Create mock transformer loader."""
        loader = Mock(spec=TransformerLoader)
        loader.load_transformers.return_value = []
        return loader

    @pytest.fixture
    def anthropic_provider_config(self):
        """Create Anthropic provider configuration."""
        return ProviderConfig(
            name='anthropic-provider', 
            url='https://api.anthropic.com/v1/messages', 
            api_key='test-key', 
            transformers={'request': [], 'response': []}, 
            timeout=30,
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
            ]
        )

    @pytest.fixture
    def openai_provider_config(self):
        """Create OpenAI provider configuration."""
        return ProviderConfig(
            name='openai-provider', 
            url='https://api.openai.com/v1/chat/completions', 
            api_key='test-key', 
            transformers={'request': [], 'response': []}, 
            timeout=30,
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'OpenAITokenCountCapability'}
            ]
        )

    def test_provider_loads_capabilities(self, anthropic_provider_config, mock_transformer_loader):
        """Test that Provider loads capabilities on initialization."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        # Should have loaded messages and count_tokens capabilities
        assert 'messages' in provider.capabilities
        assert 'count_tokens' in provider.capabilities
        assert len(provider.capabilities) == 2

    def test_anthropic_provider_gets_standard_capabilities(self, anthropic_provider_config, mock_transformer_loader):
        """Test that Anthropic provider gets standard capabilities."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        # Should get standard TokenCountCapability
        from app.services.capabilities import TokenCountCapability

        assert isinstance(provider.capabilities['count_tokens'], TokenCountCapability)

    def test_openai_provider_gets_openai_capabilities(self, openai_provider_config, mock_transformer_loader):
        """Test that OpenAI provider gets OpenAI-specific capabilities."""
        provider = Provider(openai_provider_config, mock_transformer_loader)

        # Should get OpenAITokenCountCapability
        from app.services.capabilities import OpenAITokenCountCapability

        assert isinstance(provider.capabilities['count_tokens'], OpenAITokenCountCapability)

    def test_supports_operation(self, anthropic_provider_config, mock_transformer_loader):
        """Test supports_operation method."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        assert provider.supports_operation('messages')
        assert provider.supports_operation('count_tokens')
        assert not provider.supports_operation('embeddings')
        assert not provider.supports_operation('nonexistent')

    def test_list_operations(self, anthropic_provider_config, mock_transformer_loader):
        """Test list_operations method."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        operations = provider.list_operations()
        assert set(operations) == {'messages', 'count_tokens'}

    @pytest.mark.asyncio
    async def test_process_operation_messages(self, anthropic_provider_config, mock_transformer_loader):
        """Test process_operation with messages operation."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        # Mock the _send_request method
        mock_response = Mock()
        mock_response.text = '{"id": "msg_123", "content": "test response"}'
        mock_response.json.return_value = {'id': 'msg_123', 'content': 'test response'}
        provider._send_request = AsyncMock(return_value=mock_response)

        # Create test data
        from app.common.models import AnthropicRequest

        request = AnthropicRequest(model='claude-3-haiku', messages=[{'role': 'user', 'content': 'Hello'}])

        mock_fastapi_request = Mock()
        mock_fastapi_request.headers = {}

        mock_dumper = Mock()
        from app.common.dumper import DumpFiles, DumpHandles

        handles = DumpHandles(files=DumpFiles(), correlation_id='test', base_path='/tmp')

        # Call process_operation
        result = await provider.process_operation('messages', request, mock_fastapi_request, 'default', mock_dumper, handles)

        # Should return the response
        assert result == {'id': 'msg_123', 'content': 'test response'}

        # Should have called _send_request with original URL (messages doesn't modify it)
        provider._send_request.assert_called_once()
        call_args = provider._send_request.call_args
        config_used = call_args[0][0]  # First argument is config
        assert config_used.url == 'https://api.anthropic.com/v1/messages'

    @pytest.mark.asyncio
    async def test_process_operation_count_tokens(self, anthropic_provider_config, mock_transformer_loader):
        """Test process_operation with count_tokens operation."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        # Mock the _send_request method
        mock_response = Mock()
        mock_response.text = '{"input_tokens": 10, "output_tokens": 0}'
        mock_response.json.return_value = {'input_tokens': 10, 'output_tokens': 0}
        provider._send_request = AsyncMock(return_value=mock_response)

        # Create test data
        from app.common.models import AnthropicRequest

        request = AnthropicRequest(model='claude-3-haiku', messages=[{'role': 'user', 'content': 'Hello'}])

        mock_fastapi_request = Mock()
        mock_fastapi_request.headers = {}

        mock_dumper = Mock()
        from app.common.dumper import DumpFiles, DumpHandles

        handles = DumpHandles(files=DumpFiles(), correlation_id='test', base_path='/tmp')

        # Call process_operation
        result = await provider.process_operation('count_tokens', request, mock_fastapi_request, 'default', mock_dumper, handles)

        # Should return the response
        assert result == {'input_tokens': 10, 'output_tokens': 0}

        # Should have called _send_request with modified URL (count_tokens adds suffix)
        provider._send_request.assert_called_once()
        call_args = provider._send_request.call_args
        config_used = call_args[0][0]  # First argument is config
        assert config_used.url == 'https://api.anthropic.com/v1/messages/count_tokens'

    @pytest.mark.asyncio
    async def test_process_operation_unsupported(self, anthropic_provider_config, mock_transformer_loader):
        """Test process_operation with unsupported operation."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        # Create test data
        from app.common.models import AnthropicRequest
        from app.services.capabilities import UnsupportedOperationError

        request = AnthropicRequest(model='claude-3-haiku', messages=[{'role': 'user', 'content': 'Hello'}])
        mock_fastapi_request = Mock()
        mock_dumper = Mock()
        from app.common.dumper import DumpFiles, DumpHandles

        handles = DumpHandles(files=DumpFiles(), correlation_id='test', base_path='/tmp')

        # Should raise UnsupportedOperationError
        with pytest.raises(UnsupportedOperationError) as exc_info:
            await provider.process_operation('embeddings', request, mock_fastapi_request, 'default', mock_dumper, handles)

        assert exc_info.value.operation == 'embeddings'
        assert exc_info.value.provider_name == 'anthropic-provider'


class TestProviderCapabilityValidation:
    """Test capability configuration validation and error handling."""

    @pytest.fixture
    def mock_transformer_loader(self):
        """Create mock transformer loader."""
        loader = Mock(spec=TransformerLoader)
        loader.load_transformers.return_value = []
        return loader

    def test_provider_without_capabilities_fails(self, mock_transformer_loader):
        """Test that provider without capabilities configuration fails."""
        from app.config.user_models import ProviderConfig
        
        # Create config without capabilities (empty list)
        config = ProviderConfig(
            name='no-capabilities-provider',
            url='https://api.test.com/v1',
            capabilities=[]
        )

        with pytest.raises(ValueError, match="has no capabilities configured"):
            Provider(config, mock_transformer_loader)

    def test_provider_with_invalid_capability_class_fails(self, mock_transformer_loader):
        """Test that invalid capability class names fail."""
        from app.config.user_models import ProviderConfig
        
        config = ProviderConfig(
            name='invalid-capability-provider',
            url='https://api.test.com/v1',
            capabilities=[
                {'operation': 'messages', 'class_name': 'NonExistentCapability'}
            ]
        )

        with pytest.raises(ValueError, match="Could not import capability class"):
            Provider(config, mock_transformer_loader)

    def test_provider_with_valid_custom_capability_succeeds(self, mock_transformer_loader):
        """Test that valid custom capability class works."""
        from app.config.user_models import ProviderConfig
        
        config = ProviderConfig(
            name='custom-capability-provider',
            url='https://api.test.com/v1',
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
            ]
        )

        # Should not raise any exception
        provider = Provider(config, mock_transformer_loader)
        
        # Verify capabilities were loaded
        assert 'messages' in provider.capabilities
        assert 'count_tokens' in provider.capabilities
        assert len(provider.capabilities) == 2

    def test_provider_capability_class_resolution(self, mock_transformer_loader):
        """Test capability class resolution for built-in classes."""
        from app.config.user_models import ProviderConfig
        from app.services.capabilities import MessagesCapability, TokenCountCapability, OpenAITokenCountCapability
        
        config = ProviderConfig(
            name='capability-resolution-provider',
            url='https://api.test.com/v1',
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'},
                {'operation': 'openai_count', 'class_name': 'OpenAITokenCountCapability'}
            ]
        )

        provider = Provider(config, mock_transformer_loader)
        
        # Verify correct capability types were instantiated
        assert isinstance(provider.capabilities['messages'], MessagesCapability)
        assert isinstance(provider.capabilities['count_tokens'], TokenCountCapability)
        assert isinstance(provider.capabilities['openai_count'], OpenAITokenCountCapability)

    def test_provider_capability_with_params(self, mock_transformer_loader):
        """Test capability instantiation with parameters."""
        from app.config.user_models import ProviderConfig
        
        config = ProviderConfig(
            name='parameterized-capability-provider',
            url='https://api.test.com/v1',
            capabilities=[
                {
                    'operation': 'messages', 
                    'class_name': 'MessagesCapability',
                    'params': {}  # No params for MessagesCapability
                },
                {
                    'operation': 'count_tokens',
                    'class_name': 'TokenCountCapability', 
                    'params': {}  # No params for TokenCountCapability
                }
            ]
        )

        # Should succeed even with empty params
        provider = Provider(config, mock_transformer_loader)
        assert len(provider.capabilities) == 2
