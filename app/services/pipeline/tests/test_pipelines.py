from unittest.mock import Mock

import pytest
from fastapi import Request

from app.common.models import ClaudeRequest
from app.services.pipeline.interfaces import RequestTransformer, ResponseTransformer, StreamTransformer
from app.services.pipeline.models import ProxyRequest, ProxyResponse, StreamChunk
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline


class MockRequestTransformer(RequestTransformer):
    async def transform(self, proxy_request: ProxyRequest) -> ProxyRequest:
        # Add a marker to show transformation occurred
        proxy_request.headers['x-transformed'] = 'true'
        return proxy_request


class MockResponseTransformer(ResponseTransformer):
    async def transform(self, proxy_response: ProxyResponse, proxy_request: ProxyRequest) -> ProxyResponse:
        # Add a marker to show transformation occurred
        proxy_response.headers['x-response-transformed'] = 'true'
        return proxy_response


class MockStreamTransformer(StreamTransformer):
    async def transform_chunk(self, chunk: StreamChunk, proxy_request: ProxyRequest) -> StreamChunk:
        # Add metadata to show transformation occurred
        chunk.metadata = {'transformed': True}
        return chunk


class TestRequestPipeline:
    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.headers = {'content-type': 'application/json'}
        request.query_params = {}
        return request

    @pytest.fixture
    def sample_claude_request(self):
        return ClaudeRequest(model='claude-3', messages=[], max_tokens=1000)

    @pytest.fixture
    def proxy_request(self, sample_claude_request, mock_request):
        return ProxyRequest.from_claude_request(sample_claude_request, mock_request, 'test-correlation-id')

    @pytest.mark.asyncio
    async def test_single_transformer(self, proxy_request):
        transformer = MockRequestTransformer()
        pipeline = RequestPipeline([transformer])

        result = await pipeline.execute(proxy_request)

        assert result.headers['x-transformed'] == 'true'
        assert result.claude_request.model == 'claude-3'

    @pytest.mark.asyncio
    async def test_multiple_transformers(self, proxy_request):
        transformer1 = MockRequestTransformer()
        transformer2 = MockRequestTransformer()
        pipeline = RequestPipeline([transformer1, transformer2])

        result = await pipeline.execute(proxy_request)

        assert result.headers['x-transformed'] == 'true'

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, proxy_request):
        pipeline = RequestPipeline([])

        result = await pipeline.execute(proxy_request)

        assert result == proxy_request
        assert 'x-transformed' not in result.headers


class TestResponsePipeline:
    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.headers = {'content-type': 'application/json'}
        request.query_params = {}
        return request

    @pytest.fixture
    def sample_claude_request(self):
        return ClaudeRequest(model='claude-3', messages=[], max_tokens=1000)

    @pytest.fixture
    def proxy_request(self, sample_claude_request, mock_request):
        return ProxyRequest.from_claude_request(sample_claude_request, mock_request, 'test-correlation-id')

    @pytest.fixture
    def proxy_response(self):
        return ProxyResponse(content={'id': 'msg_123', 'content': 'Hello'}, headers={'content-type': 'application/json'}, status_code=200)

    @pytest.fixture
    def stream_chunk(self):
        return StreamChunk(data=b'{"type": "message_start"}')

    @pytest.mark.asyncio
    async def test_single_transformer(self, proxy_response, proxy_request):
        transformer = MockResponseTransformer()
        pipeline = ResponsePipeline([transformer])

        result = await pipeline.execute(proxy_response, proxy_request)

        assert result.headers['x-response-transformed'] == 'true'
        assert result.content['id'] == 'msg_123'

    @pytest.mark.asyncio
    async def test_multiple_transformers(self, proxy_response, proxy_request):
        transformer1 = MockResponseTransformer()
        transformer2 = MockResponseTransformer()
        pipeline = ResponsePipeline([transformer1, transformer2])

        result = await pipeline.execute(proxy_response, proxy_request)

        assert result.headers['x-response-transformed'] == 'true'

    @pytest.mark.asyncio
    async def test_empty_pipeline(self, proxy_response, proxy_request):
        pipeline = ResponsePipeline([])

        result = await pipeline.execute(proxy_response, proxy_request)

        assert result == proxy_response
        assert 'x-response-transformed' not in result.headers

    @pytest.mark.asyncio
    async def test_stream_chunk_transformation(self, stream_chunk, proxy_request):
        stream_transformer = MockStreamTransformer()
        pipeline = ResponsePipeline([], [stream_transformer])

        result = await pipeline.execute_stream_chunk(stream_chunk, proxy_request)

        assert result.metadata == {'transformed': True}
        assert result.data == b'{"type": "message_start"}'

    @pytest.mark.asyncio
    async def test_empty_stream_pipeline(self, stream_chunk, proxy_request):
        pipeline = ResponsePipeline([])

        result = await pipeline.execute_stream_chunk(stream_chunk, proxy_request)

        assert result == stream_chunk
        assert result.metadata is None
