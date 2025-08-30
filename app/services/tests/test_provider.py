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
        handles = DumpHandles(None, None, None, None, None, None, None, None, 'test-correlation')
        dumper.begin.return_value = handles
        return dumper

    @pytest.mark.asyncio
    async def test_streaming_state_preservation(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test that sse_state is preserved between streaming chunks."""

        # Mock HTTP streaming response with multiple chunks
        mock_chunks = [
            b'data: {"id":"test-1","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"id":"test-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"id":"test-1","choices":[{"delta":{"content":" world"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        # Mock the HTTP client stream properly
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def aiter_bytes():
            for chunk in mock_chunks:
                yield chunk

        mock_response.aiter_bytes = aiter_bytes

        # Create a proper async context manager mock
        stream_context_manager = AsyncMock()
        stream_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        stream_context_manager.__aexit__ = AsyncMock(return_value=False)

        provider.http_client.stream = Mock(return_value=stream_context_manager)

        # Process the request
        handles = mock_dumper.begin('test')
        results = []
        async for chunk in provider.process_request(mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles):
            results.append(chunk)

        # Verify state was preserved across chunks
        response_transformer = provider.response_transformers[0]
        assert len(response_transformer.chunk_calls) > 1

        # All calls should have sse_state (except possibly the first)
        state_ids = [call['state_id'] for call in response_transformer.chunk_calls if call['state_id']]
        assert len(set(state_ids)) == 1, 'All chunks should share the same state object'

        # Verify chunks were processed
        assert len(results) > 0
        assert any(b'[test_response]' in chunk for chunk in results)

    @pytest.mark.asyncio
    async def test_prevents_duplicate_message_starts(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test that only one message_start event is generated across multiple chunks."""

        # Mock realistic OpenAI streaming chunks that would trigger multiple message_starts
        openai_chunks = [
            b'data: {"id":"test-1","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            b'data: {"id":"test-1","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"test-1","model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def aiter_bytes():
            for chunk in openai_chunks:
                yield chunk

        mock_response.aiter_bytes = aiter_bytes

        # Create a proper async context manager mock
        stream_context_manager = AsyncMock()
        stream_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        stream_context_manager.__aexit__ = AsyncMock(return_value=False)

        provider.http_client.stream = Mock(return_value=stream_context_manager)

        # Use actual OpenAI transformer for realistic testing
        from app.config.log import get_logger
        from app.services.transformers.openai import OpenAIResponseTransformer

        provider.response_transformers = [OpenAIResponseTransformer(get_logger(__name__))]

        handles = mock_dumper.begin('test')
        results = []
        async for chunk in provider.process_request(mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles):
            results.append(chunk.decode())

        # Count message_start events
        message_starts = [chunk for chunk in results if 'event: message_start' in chunk]
        assert len(message_starts) == 1, f'Expected exactly 1 message_start, got {len(message_starts)}'

        # Count content_block_start events for text
        content_starts = [chunk for chunk in results if 'event: content_block_start' in chunk and '"type":"text"' in chunk]
        assert len(content_starts) == 1, f'Expected exactly 1 content_block_start for text, got {len(content_starts)}'

    @pytest.mark.asyncio
    async def test_non_streaming_transformer_chaining(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test response transformer chaining for non-streaming responses."""

        # Mock non-streaming request
        mock_anthropic_request.stream = False

        # Mock HTTP client response with content as a list (Anthropic format)
        mock_response_data = {'id': 'test-response-1', 'content': [{'type': 'text', 'text': 'Hello from provider'}], 'model': 'test-model'}

        mock_http_response = Mock()
        mock_http_response.raise_for_status = Mock()
        mock_http_response.json.return_value = mock_response_data

        provider.http_client.post = AsyncMock(return_value=mock_http_response)

        handles = mock_dumper.begin('test')
        results = []
        async for chunk in provider.process_request(mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles):
            results.append(chunk.decode())

        # Verify transformer was called
        response_transformer = provider.response_transformers[0]
        assert len(response_transformer.response_calls) == 1

        # Verify content was modified by transformer (look in content blocks)
        combined_output = ''.join(results)
        assert '[test_response]' in combined_output

    @pytest.mark.asyncio
    async def test_multiple_transformer_state_preservation(self, provider_config, mock_transformer_loader):
        """Test multiple response transformers with state preservation."""

        # Create multiple transformers
        transformer1 = MockResponseTransformer('first')
        transformer2 = MockResponseTransformer('second')

        mock_transformer_loader.load_transformers.side_effect = lambda names: [transformer1, transformer2] if names == ['test_response'] else []
        provider = Provider(provider_config, mock_transformer_loader)

        # Mock streaming chunks
        mock_chunks = [b'data: {"test": "chunk1"}\n\n', b'data: {"test": "chunk2"}\n\n']

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def aiter_bytes():
            for chunk in mock_chunks:
                yield chunk

        mock_response.aiter_bytes = aiter_bytes

        # Create a proper async context manager mock
        stream_context_manager = AsyncMock()
        stream_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        stream_context_manager.__aexit__ = AsyncMock(return_value=False)

        provider.http_client.stream = Mock(return_value=stream_context_manager)

        # Process request
        request = AnthropicRequest(model='test', messages=[], stream=True)
        fastapi_request = Mock(spec=Request)
        fastapi_request.headers = {}
        dumper = Mock(spec=Dumper)
        handles = DumpHandles(None, None, None, None, None, None, None, None, 'test')

        results = []
        async for chunk in provider.process_request(request, fastapi_request, 'test', dumper, handles):
            results.append(chunk)

        # Verify both transformers processed chunks and preserved state
        assert len(transformer1.chunk_calls) == 2
        assert len(transformer2.chunk_calls) == 2

        # Verify state was maintained within each transformer
        transformer1_states = [call['state_id'] for call in transformer1.chunk_calls if call['state_id']]
        transformer2_states = [call['state_id'] for call in transformer2.chunk_calls if call['state_id']]

        # Each transformer should maintain its own consistent state
        assert len(set(transformer1_states)) == 1, 'First transformer should maintain consistent state'
        assert len(set(transformer2_states)) == 1, 'Second transformer should maintain consistent state'

    @pytest.mark.asyncio
    async def test_error_handling_propagates_transformer_errors(self, provider, mock_anthropic_request, mock_fastapi_request, mock_dumper):
        """Test that transformer errors are properly propagated."""

        # Create a transformer that fails on the second chunk
        class FailingTransformer(ResponseTransformer):
            def __init__(self):
                self.call_count = 0

            async def transform_chunk(self, params):
                self.call_count += 1
                if self.call_count == 2:
                    raise Exception('Simulated transformer failure')
                yield params['chunk']

            async def transform_response(self, params):
                """Required method for ResponseTransformer."""
                return params['response']

        provider.response_transformers = [FailingTransformer()]

        mock_chunks = [b'data: {"test": "chunk1"}\n\n', b'data: {"test": "chunk2"}\n\n']

        # Mock the HTTP client properly
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_bytes():
            for chunk in mock_chunks:
                yield chunk

        mock_response.aiter_bytes = mock_aiter_bytes

        # Create a proper async context manager mock
        stream_context_manager = AsyncMock()
        stream_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        stream_context_manager.__aexit__ = AsyncMock(return_value=False)

        provider.http_client.stream = Mock(return_value=stream_context_manager)

        # Process should propagate transformer errors
        handles = mock_dumper.begin('test')

        # The test should raise the transformer exception
        with pytest.raises(Exception, match='Simulated transformer failure'):
            results = []
            async for chunk in provider.process_request(mock_anthropic_request, mock_fastapi_request, 'test-key', mock_dumper, handles):
                results.append(chunk)


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
