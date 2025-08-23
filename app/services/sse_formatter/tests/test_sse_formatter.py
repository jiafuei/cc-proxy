"""Tests for SSE formatter service."""

import pytest

from app.services.pipeline.models import ProxyResponse, StreamChunk
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter


@pytest.fixture
def sse_formatter():
    return AnthropicSseFormatter()


@pytest.mark.asyncio
async def test_format_json_response_to_sse(sse_formatter):
    """Test converting JSON response to SSE format."""
    response_data = {
        'id': 'msg_123',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-3',
        'content': [{'type': 'text', 'text': 'Hello, world!'}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 10, 'output_tokens': 5},
    }
    response = ProxyResponse(content=response_data, headers={}, status_code=200)

    chunks = []
    async for chunk in sse_formatter.format_response(response):
        chunks.append(chunk.data.decode('utf-8'))

    # Check that we got SSE events
    assert any('event: message_start' in chunk for chunk in chunks)
    assert any('event: content_block_start' in chunk for chunk in chunks)
    assert any('event: content_block_delta' in chunk for chunk in chunks)
    assert any('event: content_block_stop' in chunk for chunk in chunks)
    assert any('event: message_delta' in chunk for chunk in chunks)
    assert any('event: message_stop' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_format_error_to_sse(sse_formatter):
    """Test formatting error to SSE format."""
    error_data = {'type': 'error', 'error': {'type': 'api_error', 'message': 'Something went wrong'}}

    chunk = await sse_formatter.format_error(error_data, 'test-correlation-id')

    assert isinstance(chunk, StreamChunk)
    sse_text = chunk.data.decode('utf-8')
    assert 'event: error' in sse_text
    assert 'test-correlation-id' in sse_text


@pytest.mark.asyncio
async def test_passthrough_already_sse_formatted(sse_formatter):
    """Test that already SSE-formatted responses pass through."""
    sse_data = b'event: message_start\ndata: {"type": "message_start"}\n\n'
    response = ProxyResponse(content=sse_data, headers={}, status_code=200)

    chunks = []
    async for chunk in sse_formatter.format_response(response):
        chunks.append(chunk.data)

    # Should pass through as-is
    assert len(chunks) == 1
    assert chunks[0] == sse_data
