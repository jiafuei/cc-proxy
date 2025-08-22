from unittest.mock import Mock

import pytest
from fastapi import Request

from app.common.models import ClaudeRequest
from app.config.models import ConfigModel
from app.services.pipeline.models import ProxyRequest, ProxyResponse, StreamChunk, TransformationContext
from app.services.pipeline.transformers.anthropic import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer


class TestAnthropicRequestTransformer:
    @pytest.fixture
    def config(self):
        return ConfigModel(anthropic_api_url='https://api.anthropic.com/v1/messages', anthropic_api_key='test-key')

    @pytest.fixture
    def transformer(self, config):
        return AnthropicRequestTransformer(config)

    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.headers = {'content-type': 'application/json', 'content-length': '100', 'accept': 'application/json', 'connection': 'keep-alive', 'custom-header': 'custom-value'}
        request.query_params = {'param1': 'value1', 'param2': 'value2'}
        return request

    @pytest.fixture
    def sample_claude_request(self):
        return ClaudeRequest(model='claude-3', messages=[], max_tokens=1000)

    @pytest.fixture
    def proxy_request(self, sample_claude_request, mock_request):
        return ProxyRequest.from_claude_request(sample_claude_request, mock_request, 'test-correlation-id')

    @pytest.mark.asyncio
    async def test_transform_request_with_api_key(self, transformer, proxy_request):
        result = await transformer.transform(proxy_request)

        assert result.url == 'https://api.anthropic.com/v1/messages'
        assert result.claude_request.model == 'claude-3'
        assert result.params == {'param1': 'value1', 'param2': 'value2'}

        headers = result.headers
        assert headers['authorization'] == 'Bearer test-key'
        assert headers['host'] == 'api.anthropic.com'
        assert headers['custom-header'] == 'custom-value'

        assert 'content-length' not in headers
        assert 'accept' not in headers
        assert 'connection' not in headers

    @pytest.mark.asyncio
    async def test_transform_request_without_api_key(self, mock_request, sample_claude_request):
        config = ConfigModel(anthropic_api_url='https://api.anthropic.com/v1/messages', anthropic_api_key=None)
        transformer = AnthropicRequestTransformer(config)
        proxy_request = ProxyRequest.from_claude_request(sample_claude_request, mock_request, 'test-correlation-id')

        result = await transformer.transform(proxy_request)

        headers = result.headers
        assert 'authorization' not in headers
        assert headers['host'] == 'api.anthropic.com'

    @pytest.mark.asyncio
    async def test_transform_request_with_empty_headers(self, transformer, sample_claude_request):
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        proxy_request = ProxyRequest.from_claude_request(sample_claude_request, request, 'test-correlation-id')

        result = await transformer.transform(proxy_request)

        headers = result.headers
        assert headers['authorization'] == 'Bearer test-key'
        assert headers['host'] == 'api.anthropic.com'


class TestAnthropicResponseTransformer:
    @pytest.fixture
    def transformer(self):
        return AnthropicResponseTransformer()

    @pytest.fixture
    def sample_proxy_request(self):
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}
        claude_request = ClaudeRequest(model='claude-3', messages=[], max_tokens=1000)
        return ProxyRequest.from_claude_request(claude_request, mock_request, 'test-correlation-id')

    @pytest.fixture
    def sample_proxy_response(self):
        return ProxyResponse(
            content={'id': 'msg_123', 'type': 'message', 'content': [{'type': 'text', 'text': 'Hello!'}]}, headers={'content-type': 'application/json'}, status_code=200
        )

    @pytest.mark.asyncio
    async def test_transform_response_passthrough(self, transformer, sample_proxy_response, sample_proxy_request):
        result = await transformer.transform(sample_proxy_response, sample_proxy_request)

        assert result == sample_proxy_response
        assert result.content == {'id': 'msg_123', 'type': 'message', 'content': [{'type': 'text', 'text': 'Hello!'}]}


class TestAnthropicStreamTransformer:
    @pytest.fixture
    def transformer(self):
        return AnthropicStreamTransformer()

    @pytest.fixture
    def sample_proxy_request(self):
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}
        claude_request = ClaudeRequest(model='claude-3', messages=[], max_tokens=1000)
        return ProxyRequest.from_claude_request(claude_request, mock_request, 'test-correlation-id')

    @pytest.fixture
    def sample_stream_chunk(self):
        return StreamChunk(data=b'{"type": "message_start", "message": {"id": "msg_123"}}')

    @pytest.mark.asyncio
    async def test_transform_chunk_passthrough(self, transformer, sample_stream_chunk, sample_proxy_request):
        result = await transformer.transform_chunk(sample_stream_chunk, sample_proxy_request)

        assert result == sample_stream_chunk
        assert result.data == b'{"type": "message_start", "message": {"id": "msg_123"}}'
