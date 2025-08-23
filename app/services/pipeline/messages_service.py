from typing import AsyncIterator

from fastapi import Request

from app.common.models import ClaudeRequest

from .http_client import HttpClientService
from .models import ProxyRequest, ProxyResponse, StreamChunk
from .request_pipeline import RequestPipeline
from .response_pipeline import ResponsePipeline


class MessagesPipelineService:
    def __init__(self, request_pipeline: RequestPipeline, response_pipeline: ResponsePipeline, http_client: HttpClientService):
        self._request_pipeline = request_pipeline
        self._response_pipeline = response_pipeline
        self._http_client = http_client

    async def process_request(self, claude_request: ClaudeRequest, original_request: Request, correlation_id: str) -> ProxyResponse:
        """Process non-streaming request through transformer pipeline and return response"""

        proxy_request = ProxyRequest.from_claude_request(claude_request, original_request, correlation_id)
        prepared_request = await self._request_pipeline.execute(proxy_request)

        # Get raw response from HTTP client
        raw_response = await self._http_client.post_request(prepared_request)

        # Create ProxyResponse from raw response
        proxy_response = ProxyResponse(content=raw_response.content, headers=dict(raw_response.headers), status_code=raw_response.status_code)

        # Apply response transformations
        transformed_response = await self._response_pipeline.execute(proxy_response, prepared_request)

        return transformed_response

    async def process_stream(self, claude_request: ClaudeRequest, original_request: Request, correlation_id: str) -> AsyncIterator[StreamChunk]:
        """Process streaming request through transformer pipeline and return stream chunks"""

        proxy_request = ProxyRequest.from_claude_request(claude_request, original_request, correlation_id)
        prepared_request = await self._request_pipeline.execute(proxy_request)

        # Stream from HTTP client and convert to StreamChunk objects
        async for raw_chunk in self._http_client.stream_request(prepared_request):
            chunk = StreamChunk(data=raw_chunk)

            # Apply response transformations to each chunk
            transformed_chunk = await self._response_pipeline.execute_stream_chunk(chunk, prepared_request)

            yield transformed_chunk

    # Legacy method for backward compatibility - TODO: Remove once router is updated
    async def process_request_legacy(self, request_data: dict, original_request: Request) -> AsyncIterator[bytes]:
        """Legacy method for backward compatibility"""
        # Convert dict to ClaudeRequest for internal processing
        claude_request = ClaudeRequest(**request_data)
        correlation_id = 'legacy'  # This should come from the router

        if request_data.get('stream', True):
            async for chunk in self.process_stream(claude_request, original_request, correlation_id):
                yield chunk.data
        else:
            response = await self.process_request(claude_request, original_request, correlation_id)
            yield response.to_bytes()
