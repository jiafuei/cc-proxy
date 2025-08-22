from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import Request

from app.common.models import ClaudeRequest
from app.config.models import ConfigModel
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.models import ProxyResponse
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.pipeline.transformers.anthropic import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer


class TestMessagesPipelineService:
    @pytest.fixture
    def config(self):
        return ConfigModel(anthropic_api_url='https://api.anthropic.com/v1/messages', anthropic_api_key='test-key')

    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.headers = {'content-type': 'application/json'}
        request.query_params = {}
        return request

    @pytest.fixture
    def sample_claude_request(self):
        return ClaudeRequest(model='claude-3-sonnet', messages=[], max_tokens=1000, stream=True)

    @pytest.fixture
    def non_streaming_claude_request(self):
        return ClaudeRequest(model='claude-3-sonnet', messages=[], max_tokens=1000, stream=False)

    @pytest.fixture
    def mock_http_client(self):
        return Mock(spec=HttpClientService)

    @pytest.fixture
    def pipeline_service(self, config, mock_http_client):
        request_transformer = AnthropicRequestTransformer(config)
        response_transformer = AnthropicResponseTransformer()
        stream_transformer = AnthropicStreamTransformer()

        request_pipeline = RequestPipeline([request_transformer])
        response_pipeline = ResponsePipeline([response_transformer], [stream_transformer])

        return MessagesPipelineService(request_pipeline, response_pipeline, mock_http_client)

    @pytest.mark.asyncio
    async def test_streaming_request(self, pipeline_service, mock_request, mock_http_client, sample_claude_request):
        mock_chunks = [b'chunk1', b'chunk2', b'chunk3']

        async def mock_stream_generator(prepared_request):
            for chunk in mock_chunks:
                yield chunk

        mock_http_client.stream_request = mock_stream_generator

        chunks = []
        async for chunk in pipeline_service.process_stream(sample_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        assert chunks == mock_chunks

    @pytest.mark.asyncio
    async def test_non_streaming_request(self, pipeline_service, mock_request, mock_http_client, non_streaming_claude_request):
        mock_response = Mock()
        mock_response.content = b'{"response": "Hello there!"}'
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.status_code = 200
        mock_http_client.post_request = AsyncMock(return_value=mock_response)

        result = await pipeline_service.process_request(non_streaming_claude_request, mock_request, 'test-correlation-id')

        assert isinstance(result, ProxyResponse)
        assert result.content == b'{"response": "Hello there!"}'
        assert result.status_code == 200
        assert result.headers == {'content-type': 'application/json'}
        mock_http_client.post_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_with_real_transformers(self, config, mock_request, sample_claude_request):
        mock_httpx_client = Mock(spec=httpx.AsyncClient)
        http_client = HttpClientService(mock_httpx_client)

        request_transformer = AnthropicRequestTransformer(config)
        response_transformer = AnthropicResponseTransformer()
        stream_transformer = AnthropicStreamTransformer()

        request_pipeline = RequestPipeline([request_transformer])
        response_pipeline = ResponsePipeline([response_transformer], [stream_transformer])

        pipeline_service = MessagesPipelineService(request_pipeline, response_pipeline, http_client)

        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock()
        mock_stream_context.__aexit__ = AsyncMock()

        mock_response = Mock()

        async def mock_aiter_bytes():
            for chunk in [b'chunk1', b'chunk2']:
                yield chunk

        mock_response.aiter_bytes = mock_aiter_bytes
        mock_response.raise_for_status = Mock()

        mock_stream_context.__aenter__.return_value = mock_response
        mock_httpx_client.stream = Mock(return_value=mock_stream_context)

        chunks = []
        async for chunk in pipeline_service.process_stream(sample_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        assert chunks == [b'chunk1', b'chunk2']
        mock_httpx_client.stream.assert_called_once()

        # Verify the request was properly transformed
        call_args = mock_httpx_client.stream.call_args
        assert call_args[0] == ('POST', 'https://api.anthropic.com/v1/messages')
        assert 'authorization' in call_args[1]['headers']
        assert call_args[1]['json']['model'] == 'claude-3-sonnet'

    @pytest.mark.asyncio
    async def test_legacy_method_streaming(self, pipeline_service, mock_request, mock_http_client):
        """Test the legacy method for backward compatibility"""
        request_data = {'model': 'claude-3-sonnet', 'messages': [], 'stream': True, 'max_tokens': 1000}

        mock_chunks = [b'chunk1', b'chunk2', b'chunk3']

        async def mock_stream_generator(prepared_request):
            for chunk in mock_chunks:
                yield chunk

        mock_http_client.stream_request = mock_stream_generator

        chunks = []
        async for chunk in pipeline_service.process_request_legacy(request_data, mock_request):
            chunks.append(chunk)

        assert chunks == mock_chunks

    @pytest.mark.asyncio
    async def test_legacy_method_non_streaming(self, pipeline_service, mock_request, mock_http_client):
        """Test the legacy method for backward compatibility"""
        request_data = {'model': 'claude-3-sonnet', 'messages': [], 'stream': False, 'max_tokens': 1000}

        mock_response = Mock()
        mock_response.content = b'{"response": "Hello there!"}'
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.status_code = 200
        mock_http_client.post_request = AsyncMock(return_value=mock_response)

        chunks = []
        async for chunk in pipeline_service.process_request_legacy(request_data, mock_request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == b'{"response": "Hello there!"}'
        mock_http_client.post_request.assert_called_once()
