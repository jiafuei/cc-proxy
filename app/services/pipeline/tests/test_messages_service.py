from unittest.mock import AsyncMock, Mock

import httpx
import orjson
import pytest
from fastapi import Request

from app.common.models import ClaudeRequest
from app.config.models import ConfigModel
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer


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

        # Add SSE formatter to the service
        sse_formatter = AnthropicSseFormatter()

        return MessagesPipelineService(request_pipeline, response_pipeline, mock_http_client, sse_formatter)

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
        async for chunk in pipeline_service.process_unified(sample_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        assert chunks == [b'chunk1', b'chunk2']
        mock_httpx_client.stream.assert_called_once()

        # Verify the request was properly transformed
        call_args = mock_httpx_client.stream.call_args
        assert call_args[0] == ('POST', 'https://api.anthropic.com/v1/messages')
        assert 'authorization' in call_args[1]['headers']
        assert call_args[1]['json']['model'] == 'claude-3-sonnet'

    @pytest.mark.asyncio
    async def test_unified_processing_streaming(self, pipeline_service, mock_request, mock_http_client, sample_claude_request):
        """Test unified processing with streaming upstream"""
        # Mock SSE-formatted chunks from upstream
        mock_chunks = [
            b'event: message_start\ndata: {"type": "message_start"}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n',
            b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]

        async def mock_stream_generator(prepared_request):
            for chunk in mock_chunks:
                yield chunk

        mock_http_client.stream_request = mock_stream_generator

        chunks = []
        async for chunk in pipeline_service.process_unified(sample_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        # Should pass through SSE chunks as-is
        assert chunks == mock_chunks

    @pytest.mark.asyncio
    async def test_unified_processing_non_streaming(self, pipeline_service, mock_request, mock_http_client, non_streaming_claude_request):
        """Test unified processing with non-streaming upstream"""
        # Mock non-streaming JSON response from upstream
        mock_response_data = {
            'id': 'msg_123',
            'type': 'message',
            'role': 'assistant',
            'model': 'claude-3-sonnet',
            'content': [{'type': 'text', 'text': 'Hello there!'}],
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        }

        mock_response = Mock()
        mock_response.content = orjson.dumps(mock_response_data)
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.status_code = 200
        mock_http_client.post_request = AsyncMock(return_value=mock_response)

        chunks = []
        async for chunk in pipeline_service.process_unified(non_streaming_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        # Should convert to SSE format
        assert len(chunks) > 0

        # Verify SSE events are generated
        all_data = b''.join(chunks).decode()
        assert 'event: message_start' in all_data
        assert 'event: content_block_start' in all_data
        assert 'event: content_block_delta' in all_data
        assert 'event: content_block_stop' in all_data
        assert 'event: message_delta' in all_data
        assert 'event: message_stop' in all_data
        assert 'Hello there!' in all_data

    @pytest.mark.asyncio
    async def test_unified_processing_transform_changes_stream_flag(self, config, mock_request, mock_http_client):
        """Test that stream decision is made after transformations"""

        # Create a custom transformer that changes stream flag
        class StreamToggleTransformer:
            async def transform(self, proxy_request):
                # Toggle the stream flag
                proxy_request.claude_request.stream = not proxy_request.claude_request.stream
                return proxy_request

        # Start with stream=True
        claude_request = ClaudeRequest(model='claude-3-sonnet', messages=[], max_tokens=1000, stream=True)

        # Setup pipeline with stream toggle transformer
        request_pipeline = RequestPipeline([StreamToggleTransformer(), AnthropicRequestTransformer(config)])
        response_pipeline = ResponsePipeline([AnthropicResponseTransformer()], [AnthropicStreamTransformer()])
        sse_formatter = AnthropicSseFormatter()
        pipeline_service = MessagesPipelineService(request_pipeline, response_pipeline, mock_http_client, sse_formatter)

        # Mock non-streaming response (because transformer will change stream=True to stream=False)
        mock_response = Mock()
        mock_response.content = b'{"content": [{"type": "text", "text": "Response"}]}'
        mock_response.headers = {}
        mock_response.status_code = 200
        mock_http_client.post_request = AsyncMock(return_value=mock_response)
        mock_http_client.stream_request = AsyncMock()  # Should NOT be called

        chunks = []
        async for chunk in pipeline_service.process_unified(claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        # Verify non-streaming path was taken (post_request called, not stream_request)
        mock_http_client.post_request.assert_called_once()
        mock_http_client.stream_request.assert_not_called()

        # Verify SSE conversion happened
        all_data = b''.join(chunks).decode()
        assert 'event: message_start' in all_data

    @pytest.mark.asyncio
    async def test_unified_processing_error_response(self, pipeline_service, mock_request, mock_http_client, non_streaming_claude_request):
        """Test unified processing with error response"""
        # Mock error response
        mock_response = Mock()
        mock_response.content = b'event: error\ndata: {"type": "error", "error": {"type": "api_error", "message": "Something went wrong"}}\n\n'
        mock_response.headers = {}
        mock_response.status_code = 500
        mock_http_client.post_request = AsyncMock(return_value=mock_response)

        chunks = []
        async for chunk in pipeline_service.process_unified(non_streaming_claude_request, mock_request, 'test-correlation-id'):
            chunks.append(chunk.data)

        # Should pass through error SSE event
        assert len(chunks) == 1
        assert b'event: error' in chunks[0]
        assert b'Something went wrong' in chunks[0]
