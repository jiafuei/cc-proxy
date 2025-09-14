"""Tests for capability interfaces and implementations."""

import pytest

from app.config.user_models import ProviderConfig
from app.services.capabilities import MessagesCapability, OpenAITokenCountCapability, TokenCountCapability, UnsupportedOperationError


@pytest.fixture
def sample_config():
    """Sample provider configuration for testing."""
    return ProviderConfig(
        name='test-provider', 
        url='https://api.test.com/v1/messages', 
        api_key='test-api-key', 
        transformers={'request': [], 'response': []}, 
        timeout=30,
        capabilities=[
            {'operation': 'messages', 'class_name': 'MessagesCapability'},
            {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
        ]
    )


@pytest.fixture
def sample_request():
    """Sample request data for testing."""
    return {'model': 'claude-3-haiku', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 100}


@pytest.fixture
def sample_context():
    """Sample context data for testing."""
    return {'headers': {'x-api-key': 'test-key'}, 'routing_key': 'default', 'operation': 'messages'}


class TestMessagesCapability:
    """Test MessagesCapability implementation."""

    def test_operation_name(self):
        """Test that MessagesCapability returns correct operation name."""
        capability = MessagesCapability()
        assert capability.get_operation_name() == 'messages'

    def test_supports_provider(self):
        """Test that MessagesCapability supports all providers by default."""
        capability = MessagesCapability()
        assert capability.supports_provider('anthropic')
        assert capability.supports_provider('openai')
        assert capability.supports_provider('any-provider')

    @pytest.mark.asyncio
    async def test_prepare_request_passthrough(self, sample_request, sample_config, sample_context):
        """Test that prepare_request is a passthrough for messages."""
        capability = MessagesCapability()
        result_config = await capability.prepare_request(sample_request, sample_config, sample_context)

        # Should return the same config unchanged
        assert result_config == sample_config
        assert result_config.url == sample_config.url

    @pytest.mark.asyncio
    async def test_process_response_passthrough(self, sample_request, sample_context):
        """Test that process_response is a passthrough for messages."""
        capability = MessagesCapability()
        response_data = {'id': 'msg_123', 'content': 'Hello back!'}

        result = await capability.process_response(response_data, sample_request, sample_context)

        # Should return the same response unchanged
        assert result == response_data


class TestTokenCountCapability:
    """Test TokenCountCapability implementation."""

    def test_operation_name(self):
        """Test that TokenCountCapability returns correct operation name."""
        capability = TokenCountCapability()
        assert capability.get_operation_name() == 'count_tokens'

    def test_supports_provider(self):
        """Test that TokenCountCapability supports all providers by default."""
        capability = TokenCountCapability()
        assert capability.supports_provider('anthropic')
        assert capability.supports_provider('openai')
        assert capability.supports_provider('any-provider')

    @pytest.mark.asyncio
    async def test_prepare_request_modifies_url(self, sample_request, sample_config, sample_context):
        """Test that prepare_request modifies URL to add /count_tokens."""
        capability = TokenCountCapability()
        result_config = await capability.prepare_request(sample_request, sample_config, sample_context)

        # Should modify URL to add count_tokens suffix
        assert result_config.url == 'https://api.test.com/v1/messages/count_tokens'
        # Other config should remain unchanged
        assert result_config.name == sample_config.name
        assert result_config.api_key == sample_config.api_key

    @pytest.mark.asyncio
    async def test_prepare_request_url_already_has_suffix(self, sample_request, sample_context):
        """Test that prepare_request doesn't double-add suffix if already present."""
        config_with_suffix = ProviderConfig(
            name='test-provider', 
            url='https://api.test.com/v1/messages/count_tokens', 
            api_key='test-api-key', 
            transformers={'request': [], 'response': []}, 
            timeout=30,
            capabilities=[
                {'operation': 'messages', 'class_name': 'MessagesCapability'},
                {'operation': 'count_tokens', 'class_name': 'TokenCountCapability'}
            ]
        )

        capability = TokenCountCapability()
        result_config = await capability.prepare_request(sample_request, config_with_suffix, sample_context)

        # Should not double-add the suffix
        assert result_config.url == 'https://api.test.com/v1/messages/count_tokens'

    @pytest.mark.asyncio
    async def test_process_response_validates_fields(self, sample_request, sample_context):
        """Test that process_response validates expected response fields."""
        capability = TokenCountCapability()

        # Test with valid response
        valid_response = {'input_tokens': 10, 'output_tokens': 0, 'total_tokens': 10}
        result = await capability.process_response(valid_response, sample_request, sample_context)
        assert result == valid_response

        # Test with missing fields (should still work but may log warnings)
        incomplete_response = {'total_tokens': 10}
        result = await capability.process_response(incomplete_response, sample_request, sample_context)
        assert result == incomplete_response


class TestOpenAITokenCountCapability:
    """Test OpenAITokenCountCapability implementation."""

    def test_operation_name(self):
        """Test that OpenAITokenCountCapability returns correct operation name."""
        capability = OpenAITokenCountCapability()
        assert capability.get_operation_name() == 'count_tokens'

    def test_supports_provider(self):
        """Test that OpenAITokenCountCapability only supports OpenAI providers."""
        capability = OpenAITokenCountCapability()
        assert capability.supports_provider('openai-provider')
        assert capability.supports_provider('openai-gpt4')
        assert not capability.supports_provider('anthropic')
        assert not capability.supports_provider('gemini')

    @pytest.mark.asyncio
    async def test_prepare_request_still_modifies_url(self, sample_request, sample_config, sample_context):
        """Test that OpenAI capability still tries URL modification (for now)."""
        capability = OpenAITokenCountCapability()
        result_config = await capability.prepare_request(sample_request, sample_config, sample_context)

        # Should still modify URL (this could be changed in future implementations)
        assert result_config.url == 'https://api.test.com/v1/messages/count_tokens'


class TestUnsupportedOperationError:
    """Test UnsupportedOperationError exception."""

    def test_error_creation_with_defaults(self):
        """Test creating error with default message."""
        error = UnsupportedOperationError('embeddings', 'test-provider')

        assert error.operation == 'embeddings'
        assert error.provider_name == 'test-provider'
        assert 'test-provider' in error.message
        assert 'embeddings' in error.message

    def test_error_creation_with_custom_message(self):
        """Test creating error with custom message."""
        custom_message = 'Custom error message'
        error = UnsupportedOperationError('embeddings', 'test-provider', custom_message)

        assert error.operation == 'embeddings'
        assert error.provider_name == 'test-provider'
        assert error.message == custom_message

    def test_error_str_representation(self):
        """Test that error can be converted to string."""
        error = UnsupportedOperationError('embeddings', 'test-provider')
        error_str = str(error)

        assert 'test-provider' in error_str
        assert 'embeddings' in error_str
