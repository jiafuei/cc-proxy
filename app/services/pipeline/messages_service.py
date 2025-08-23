from typing import AsyncIterator

import orjson
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
            async for chunk in self._convert_response_to_sse_stream(response):
                yield chunk

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

    async def _convert_response_to_sse_stream(self, response: ProxyResponse) -> AsyncIterator[StreamChunk]:
        """Convert a non-streaming response to SSE format following Anthropic's specification."""
        # Parse the response content
        response_bytes = response.to_bytes()

        # If the response is already SSE-formatted (unlikely for non-streaming), pass through
        if response_bytes.startswith(b'event:') or response_bytes.startswith(b'data:'):
            yield StreamChunk(data=response_bytes)
            return

        # Parse JSON response and convert to SSE events 
        # TODO: Make sure required fields are entered, token usage is correct
        # TODO: Make sure fields are processed correctly in order - Thinking - tool - content
        try:
            response_json = orjson.loads(response_bytes)

            # Generate message_start event
            message_start = {
                'type': 'message_start',
                'message': {
                    'id': response_json.get('id'),
                    'type': response_json.get('type'),
                    'role': response_json.get('role'),
                    'model': response_json.get('model'),
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {'input_tokens': 0, 'output_tokens': 0},
                },
            }
            yield StreamChunk(data=f'event: message_start\ndata: {orjson.dumps(message_start).decode()}\n\n'.encode())

            # Process content blocks
            content = response_json.get('content', [])
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]

            for index, block in enumerate(content):
                # content_block_start
                block_start = {'type': 'content_block_start', 'index': index, 'content_block': block}
                yield StreamChunk(data=f'event: content_block_start\ndata: {orjson.dumps(block_start).decode()}\n\n'.encode())

                # content_block_delta (for text blocks, send the full text as delta)
                if block.get('type') == 'text':
                    block_delta = {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': block.get('text', '')}}
                    yield StreamChunk(data=f'event: content_block_delta\ndata: {orjson.dumps(block_delta).decode()}\n\n'.encode())

                # content_block_stop
                block_stop = {'type': 'content_block_stop', 'index': index}
                yield StreamChunk(data=f'event: content_block_stop\ndata: {orjson.dumps(block_stop).decode()}\n\n'.encode())

            # message_delta with usage
            if 'usage' in response_json:
                message_delta = {
                    'type': 'message_delta',
                    'delta': {'stop_reason': response_json.get('stop_reason'), 'stop_sequence': response_json.get('stop_sequence')},
                    'usage': {'output_tokens': response_json['usage'].get('output_tokens', 0)},
                }
                yield StreamChunk(data=f'event: message_delta\ndata: {orjson.dumps(message_delta).decode()}\n\n'.encode())

            # message_stop
            message_stop = {'type': 'message_stop'}
            yield StreamChunk(data=f'event: message_stop\ndata: {orjson.dumps(message_stop).decode()}\n\n'.encode())

        except Exception:
            # If we can't parse as JSON, return as-is (might be an error or other format)
            yield StreamChunk(data=response_bytes)
