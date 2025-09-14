"""Tests for Provider and ProviderManager classes."""

from unittest.mock import AsyncMock, Mock

import httpx
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
    def anthropic_provider_config(self):
        """Create test Anthropic provider configuration."""
        return ProviderConfig(
            name='anthropic-provider',
            url='https://api.anthropic.com',
            api_key='test-key',
            type='anthropic',
            capabilities=['messages', 'count_tokens'],
            transformers={'request': ['test_request'], 'response': ['test_response']},
            timeout=30,
        )

    @pytest.fixture
    def openai_provider_config(self):
        """Create test OpenAI provider configuration."""
        return ProviderConfig(
            name='openai-provider',
            url='https://api.openai.com',
            api_key='test-key',
            type='openai',
            capabilities=['messages'],  # OpenAI doesn't support count_tokens
            transformers={'request': ['test_request'], 'response': ['test_response']},
            timeout=30,
        )

    @pytest.fixture
    def gemini_provider_config(self):
        """Create test Gemini provider configuration."""
        return ProviderConfig(
            name='gemini-provider',
            url='https://generativelanguage.googleapis.com/v1beta/models/gemini-pro',
            api_key='test-key',
            type='gemini',
            capabilities=['messages', 'count_tokens'],
            transformers={'request': ['test_request'], 'response': ['test_response']},
            timeout=30,
        )

    @pytest.fixture
    def provider(self, anthropic_provider_config, mock_transformer_loader, mock_request_transformer, mock_response_transformer):
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

        provider = Provider(anthropic_provider_config, mock_transformer_loader)
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

    def test_provider_initialization_anthropic(self, anthropic_provider_config, mock_transformer_loader):
        """Test Provider initialization with Anthropic type."""
        provider = Provider(anthropic_provider_config, mock_transformer_loader)

        assert provider.name == 'anthropic-provider'
        assert provider.capabilities == ['messages', 'count_tokens']
        assert provider.config.type == 'anthropic'
        assert provider.spec is not None

    def test_provider_initialization_openai(self, openai_provider_config, mock_transformer_loader):
        """Test Provider initialization with OpenAI type."""
        provider = Provider(openai_provider_config, mock_transformer_loader)

        assert provider.name == 'openai-provider'
        assert provider.capabilities == ['messages']  # Only messages supported
        assert provider.config.type == 'openai'

    def test_provider_initialization_invalid_type(self, mock_transformer_loader):
        """Test Provider initialization with invalid type."""
        config = ProviderConfig(name='invalid-provider', url='https://api.test.com', type='invalid', timeout=30)

        with pytest.raises(ValueError, match='Unknown provider type: invalid'):
            Provider(config, mock_transformer_loader)

    def test_provider_default_capabilities(self, mock_transformer_loader):
        """Test provider with default capabilities (None means all supported by type)."""
        config = ProviderConfig(
            name='default-provider',
            url='https://api.anthropic.com/v1/messages',
            type='anthropic',
            capabilities=None,  # Use all supported by type
            timeout=30,
        )

        provider = Provider(config, mock_transformer_loader)
        assert set(provider.capabilities) == {'messages', 'count_tokens'}

    def test_provider_filtered_capabilities(self, mock_transformer_loader):
        """Test provider with filtered capabilities."""
        config = ProviderConfig(
            name='filtered-provider',
            url='https://api.anthropic.com',
            type='anthropic',
            capabilities=['messages'],  # Only enable messages
            timeout=30,
        )

        provider = Provider(config, mock_transformer_loader)
        assert provider.capabilities == ['messages']

    def test_provider_unsupported_capability_error(self, mock_transformer_loader):
        """Test error for unsupported capability."""
        config = ProviderConfig(
            name='error-provider',
            url='https://api.openai.com',
            type='openai',
            capabilities=['messages', 'count_tokens'],  # count_tokens not supported by OpenAI
            timeout=30,
        )

        # Should raise ValueError for unsupported capabilities
        with pytest.raises(ValueError, match="doesn't support"):
            Provider(config, mock_transformer_loader)

    def test_supports_operation(self, provider):
        """Test supports_operation method."""
        assert provider.supports_operation('messages')
        assert provider.supports_operation('count_tokens')
        assert not provider.supports_operation('embeddings')
        assert not provider.supports_operation('nonexistent')

    def test_list_operations(self, provider):
        """Test list_operations method."""
        operations = provider.list_operations()
        assert operations == ['messages', 'count_tokens']

    @pytest.mark.asyncio
    async def test_process_operation_messages_anthropic(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test process_operation with messages operation for Anthropic."""

        # Mock HTTP response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.text = '{"id": "msg_test123", "model": "claude-3-haiku", "role": "assistant", "content": [{"type": "text", "text": "Hello world"}], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}'
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

        # Verify the correct URL was used (for messages, should be base URL)
        provider.http_client.post.assert_called_once()
        call_args = provider.http_client.post.call_args
        url_used = call_args[0][0]  # First positional argument is the URL
        assert url_used == 'https://api.anthropic.com/v1/messages'

    @pytest.mark.asyncio
    async def test_process_operation_count_tokens_anthropic(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test process_operation with count_tokens operation for Anthropic."""

        # Mock HTTP response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.text = '{"input_tokens": 10, "output_tokens": 0}'
        mock_response.json.return_value = {'input_tokens': 10, 'output_tokens': 0}

        # Mock the HTTP client
        provider.http_client.post = AsyncMock(return_value=mock_response)

        # Process the request
        handles = mock_dumper.begin.return_value
        result = await provider.process_operation('count_tokens', mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles)

        # Verify we got the expected JSON response
        assert result['input_tokens'] == 10
        assert result['output_tokens'] == 0

        # Verify the correct URL was used (for count_tokens, should append /count_tokens)
        provider.http_client.post.assert_called_once()
        call_args = provider.http_client.post.call_args
        url_used = call_args[0][0]
        assert url_used == 'https://api.anthropic.com/v1/messages/count_tokens'

    @pytest.mark.asyncio
    async def test_process_operation_unsupported(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test process_operation with unsupported operation."""

        # Create test data
        handles = mock_dumper.begin.return_value

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await provider.process_operation('embeddings', mock_anthropic_request, mock_fastapi_request, 'default', mock_dumper, handles)

        assert 'embeddings' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_operation_openai_messages(self, openai_provider_config, mock_transformer_loader, mock_request_transformer, mock_response_transformer):
        """Test process_operation with OpenAI messages operation."""

        # Set up transformer loader
        def mock_load_transformers(names):
            if names == ['test_request']:
                return [mock_request_transformer]
            elif names == ['test_response']:
                return [mock_response_transformer]
            else:
                return []

        mock_transformer_loader.load_transformers.side_effect = mock_load_transformers

        provider = Provider(openai_provider_config, mock_transformer_loader)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.text = '{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello!"}}]}'
        mock_response.json.return_value = {'id': 'chatcmpl-123', 'choices': [{'message': {'content': 'Hello!'}}]}

        provider.http_client.post = AsyncMock(return_value=mock_response)

        # Create test data
        request = AnthropicRequest(model='gpt-4', messages=[{'role': 'user', 'content': 'Hello'}])
        mock_fastapi_request = Mock()
        mock_fastapi_request.headers = {}
        mock_dumper = Mock()
        from app.common.dumper import DumpFiles, DumpHandles

        handles = DumpHandles(files=DumpFiles(), correlation_id='test', base_path='/tmp')

        # Process the request
        result = await provider.process_operation('messages', request, mock_fastapi_request, 'test-key', mock_dumper, handles)

        # Verify response
        assert result['id'] == 'chatcmpl-123'

        # Verify the correct URL was used (OpenAI should use /chat/completions)
        provider.http_client.post.assert_called_once()
        call_args = provider.http_client.post.call_args
        url_used = call_args[0][0]
        assert url_used == 'https://api.openai.com/v1/chat/completions'

    @pytest.mark.asyncio
    async def test_http_error_handling(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test HTTP error handling."""

        # Mock HTTP error response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(message='Bad Request', request=Mock(), response=Mock())
        mock_response.aread = AsyncMock()

        provider.http_client.post = AsyncMock(return_value=mock_response)

        handles = mock_dumper.begin.return_value

        # Should re-raise the HTTP error
        with pytest.raises(httpx.HTTPStatusError):
            await provider.process_operation('messages', mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles)

        # Should have called aread on error
        mock_response.aread.assert_called_once()


class TestProviderManager:
    """Test cases for the ProviderManager class."""

    @pytest.fixture
    def providers_config(self):
        """Create test providers configuration."""
        return [
            ProviderConfig(name='provider1', url='https://api1.test.com/v1/messages', type='anthropic', capabilities=['messages', 'count_tokens']),
            ProviderConfig(name='provider2', url='https://api2.test.com/v1', type='openai', capabilities=['messages']),
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


class TestProviderConfigValidation:
    """Test provider configuration validation and error handling."""

    @pytest.fixture
    def mock_transformer_loader(self):
        """Create mock transformer loader."""
        loader = Mock(spec=TransformerLoader)
        loader.load_transformers.return_value = []
        return loader

    def test_provider_without_type_fails(self, mock_transformer_loader):
        """Test that provider without type fails validation at config level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderConfig(
                name='no-type-provider',
                url='https://api.test.com/v1',
                # type is missing - should fail validation
            )

    def test_provider_with_invalid_capability_fails(self, mock_transformer_loader):
        """Test that provider with invalid capability fails validation at config level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match='Invalid capability'):
            ProviderConfig(
                name='invalid-capability-provider',
                url='https://api.test.com/v1',
                type='anthropic',
                capabilities=['invalid_operation'],  # Not in valid set
            )

    def test_transformer_override(self, mock_transformer_loader):
        """Test that provider can override default transformers."""
        # Mock different transformers for override test
        custom_transformers = [Mock(), Mock()]
        mock_transformer_loader.load_transformers.return_value = custom_transformers

        config = ProviderConfig(
            name='override-provider',
            url='https://api.anthropic.com/v1/messages',
            type='anthropic',
            transformers={'request': [{'class': 'custom.RequestTransformer'}], 'response': [{'class': 'custom.ResponseTransformer'}]},
        )

        provider = Provider(config, mock_transformer_loader)

        # Should have called load_transformers with custom config, not defaults
        assert mock_transformer_loader.load_transformers.call_count == 2  # request + response
        assert provider.request_transformers == custom_transformers
        assert provider.response_transformers == custom_transformers

    @pytest.mark.asyncio
    async def test_provider_cleanup(self, mock_transformer_loader):
        """Test provider cleanup."""
        config = ProviderConfig(name='cleanup-provider', url='https://api.anthropic.com/v1/messages', type='anthropic')

        provider = Provider(config, mock_transformer_loader)

        # Mock the http client close method
        provider.http_client.aclose = AsyncMock()

        await provider.close()
        provider.http_client.aclose.assert_called_once()
