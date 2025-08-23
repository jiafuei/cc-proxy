"""Anthropic SSE formatter implementation."""

from typing import AsyncIterator

import orjson

from app.services.pipeline.models import ProxyResponse, StreamChunk

from .interfaces import SseFormatter


class AnthropicSseFormatter(SseFormatter):
    """Formats responses according to Anthropic's SSE specification."""

    async def format_response(self, response: ProxyResponse) -> AsyncIterator[StreamChunk]:
        """Convert a non-streaming response to Anthropic SSE format."""
        response_bytes = response.to_bytes()

        # If already SSE-formatted, pass through
        if response_bytes.startswith(b'event:') or response_bytes.startswith(b'data:'):
            yield StreamChunk(data=response_bytes)
            return

        # Parse JSON response and convert to SSE events
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

                # content_block_delta (for text blocks)
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
            # If can't parse as JSON, return as-is
            yield StreamChunk(data=response_bytes)

    async def format_error(self, error_data: dict, correlation_id: str = None) -> StreamChunk:
        """Format error data as an SSE error event."""
        if correlation_id and 'request_id' not in error_data:
            error_data['request_id'] = correlation_id

        json_data = orjson.dumps(error_data).decode('utf-8')
        sse_event = f'event: error\ndata: {json_data}\n\n'
        return StreamChunk(data=sse_event.encode('utf-8'))
