from typing import AsyncIterator

from fastapi import Request

from app.common.models import ClaudeRequest
from app.services.sse_formatter.interfaces import SseFormatter

from .http_client import HttpClientService
from .models import ProxyRequest, ProxyResponse, StreamChunk
from .request_pipeline import RequestPipeline
from .response_pipeline import ResponsePipeline


class MessagesPipelineService:
    def __init__(self, request_pipeline: RequestPipeline, response_pipeline: ResponsePipeline, http_client: HttpClientService, sse_formatter: SseFormatter = None):
        self._request_pipeline = request_pipeline
        self._response_pipeline = response_pipeline
        self._http_client = http_client
        self._sse_formatter = sse_formatter

    async def process_unified(self, claude_request: ClaudeRequest, original_request: Request, correlation_id: str) -> AsyncIterator[StreamChunk]:
        """Unified processing that always returns SSE-formatted stream chunks.

        The stream decision is made AFTER transformations are applied.
        Returns SSE-formatted chunks regardless of upstream streaming mode.
        """
        # Step 1: Create proxy request and apply ALL transformations
        proxy_request = ProxyRequest.from_claude_request(claude_request, original_request, correlation_id)
        prepared_request = await self._request_pipeline.execute(proxy_request)

        # Step 2: Check stream flag AFTER transformations
        should_stream_upstream = prepared_request.claude_request.stream

        # Step 3: Route based on transformed stream flag
        if should_stream_upstream:
            # Stream from upstream - pass through SSE events
            async for chunk in self._process_stream_internal(prepared_request):
                yield chunk
        else:
            # Non-streaming from upstream - convert response to SSE format
            response = await self._process_request_internal(prepared_request)
            if self._sse_formatter:
                async for chunk in self._sse_formatter.format_response(response):
                    yield chunk
            else:
                # Fallback to basic pass-through if no formatter
                yield StreamChunk(data=response.to_bytes())

    async def _process_request_internal(self, prepared_request: ProxyRequest) -> ProxyResponse:
        """Internal non-streaming processing (request already transformed)"""
        # Get raw response from HTTP client
        raw_response = await self._http_client.post_request(prepared_request)

        # Create ProxyResponse from raw response
        proxy_response = ProxyResponse(content=raw_response.content, headers=dict(raw_response.headers), status_code=raw_response.status_code)

        # Apply response transformations
        return await self._response_pipeline.execute(proxy_response, prepared_request)

    async def _process_stream_internal(self, prepared_request: ProxyRequest) -> AsyncIterator[StreamChunk]:
        """Internal streaming processing (request already transformed)"""
        # Stream from HTTP client and convert to StreamChunk objects
        async for raw_chunk in self._http_client.stream_request(prepared_request):
            chunk = StreamChunk(data=raw_chunk)

            # Apply response transformations to each chunk
            transformed_chunk = await self._response_pipeline.execute_stream_chunk(chunk, prepared_request)

            yield transformed_chunk
