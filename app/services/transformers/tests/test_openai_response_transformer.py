"""Tests for OpenAI Response Transformer."""

from unittest.mock import Mock

import pytest

from app.services.transformers.openai import OpenAIResponseTransformer


class TestOpenAIResponseTransformer:
    """Test cases for the OpenAI response transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        mock_logger = Mock()
        return OpenAIResponseTransformer(mock_logger)

    def test_streaming_tool_call_start_chunk(self, transformer):
        """Test OpenAI tool call start chunk conversion to Claude format."""
        # OpenAI chunk with tool call metadata (first chunk)
        openai_chunk = {
            'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_rZWtuw75uzo1VSx9oL13zV3g', 'type': 'function', 'function': {'name': 'get_meaning', 'arguments': ''}}]}}]
        }

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        expected = {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'tool_use', 'id': 'call_rZWtuw75uzo1VSx9oL13zV3g', 'name': 'get_meaning', 'input': {}}}
        assert result == expected

    def test_streaming_tool_call_argument_chunk(self, transformer):
        """Test OpenAI tool call argument chunk conversion to Claude format."""
        # OpenAI chunk with argument fragment
        openai_chunk = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"subject"'}}]}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        expected = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': '{"subject"'}}
        assert result == expected

    def test_streaming_tool_call_with_different_index(self, transformer):
        """Test tool call with non-zero index uses correct index in Claude format."""
        # OpenAI chunk with tool call at index 1
        openai_chunk = {'choices': [{'delta': {'tool_calls': [{'index': 1, 'id': 'call_test123', 'type': 'function', 'function': {'name': 'test_function', 'arguments': ''}}]}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result['index'] == 1
        assert result['type'] == 'content_block_start'

    def test_streaming_tool_call_argument_with_different_index(self, transformer):
        """Test tool call argument chunk with non-zero index."""
        # OpenAI chunk with argument fragment at index 2
        openai_chunk = {'choices': [{'delta': {'tool_calls': [{'index': 2, 'function': {'arguments': '":"yellow"'}}]}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result['index'] == 2
        assert result['type'] == 'content_block_delta'
        assert result['delta']['partial_json'] == '":"yellow"'

    def test_streaming_tool_call_finish_reason(self, transformer):
        """Test tool call completion with finish_reason."""
        openai_chunk = {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result == {'type': 'message_stop'}

    def test_streaming_text_content(self, transformer):
        """Test regular text content streaming."""
        openai_chunk = {'choices': [{'delta': {'content': 'Hello world'}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        expected = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello world'}}
        assert result == expected

    def test_empty_chunk_returns_ping(self, transformer):
        """Test empty chunk returns ping."""
        openai_chunk = {'choices': [{'delta': {}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result == {'type': 'ping'}

    def test_no_choices_returns_ping(self, transformer):
        """Test chunk with no choices returns ping."""
        openai_chunk = {'choices': []}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result == {'type': 'ping'}

    def test_tool_call_missing_index_defaults_to_zero(self, transformer):
        """Test tool call without index defaults to 0."""
        openai_chunk = {'choices': [{'delta': {'tool_calls': [{'id': 'call_test', 'function': {'name': 'test_func', 'arguments': ''}}]}}]}

        result = transformer._convert_openai_chunk_to_claude(openai_chunk)

        assert result['index'] == 0

    @pytest.mark.asyncio
    async def test_multiple_data_events_in_single_chunk(self, transformer):
        """Test processing multiple SSE data events in a single network chunk."""
        # Simulate a chunk with multiple data events like in the examples
        multi_event_chunk = (
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
            '\n'
            'data: {"choices": [{"delta": {"content": " world"}}]}\n'
            '\n'
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
            '\n'
            'data: [DONE]\n'
            '\n'
        ).encode('utf-8')

        results = []
        async for result in transformer.transform_chunk({'chunk': multi_event_chunk}):
            results.append(result.decode('utf-8'))

        # Should yield 4 events: 2 text deltas, 1 message_stop, 1 final message_stop
        assert len(results) == 4

        # Check first text event
        assert 'event: content_block_delta' in results[0]
        assert 'data: ' in results[0]
        assert '"text":"Hello"' in results[0]

        # Check second text event
        assert 'event: content_block_delta' in results[1]
        assert 'data: ' in results[1]
        assert '"text":" world"' in results[1]

        # Check finish_reason converted to message_stop
        assert 'event: message_stop' in results[2]
        assert 'data: ' in results[2]
        assert '"type":"message_stop"' in results[2]

        # Check [DONE] converted to message_stop
        assert 'event: message_stop' in results[3]
        assert 'data: ' in results[3]
        assert '"type":"message_stop"' in results[3]

    @pytest.mark.asyncio
    async def test_multiple_tool_call_events_in_single_chunk(self, transformer):
        """Test processing multiple tool call events in a single chunk."""
        tool_call_chunk = (
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "type": "function", "function": {"name": "get_meaning", "arguments": ""}}]}}]}\n'
            '\n'
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"subject"}}]}}]}\n'
            '\n'
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\":\\"yellow sunsets\\"}"}}]}}]}\n'
            '\n'
            'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n'
            '\n'
            'data: [DONE]\n'
            '\n'
        ).encode('utf-8')

        results = []
        async for result in transformer.transform_chunk({'chunk': tool_call_chunk}):
            results.append(result.decode('utf-8'))

        # Should yield 5 events: tool start, 2 argument deltas, message_stop, final message_stop
        assert len(results) == 5

        # Check tool call start
        assert 'event: content_block_start' in results[0]
        assert '"type":"content_block_start"' in results[0]
        assert '"name":"get_meaning"' in results[0]

        # Check argument deltas
        assert 'event: content_block_delta' in results[1]
        assert '"type":"content_block_delta"' in results[1]
        assert '"partial_json":"{\\"subject"' in results[1]

        assert 'event: content_block_delta' in results[2]
        assert '"type":"content_block_delta"' in results[2]
        assert '"partial_json":"\\":\\"yellow sunsets\\"}' in results[2]

    @pytest.mark.asyncio
    async def test_empty_lines_are_skipped(self, transformer):
        """Test that empty lines in chunks are properly skipped."""
        chunk_with_empty_lines = ('\ndata: {"choices": [{"delta": {"content": "test"}}]}\n\n\ndata: [DONE]\n\n\n').encode('utf-8')

        results = []
        async for result in transformer.transform_chunk({'chunk': chunk_with_empty_lines}):
            results.append(result.decode('utf-8'))

        # Should yield 2 events: text delta and final message_stop
        assert len(results) == 2
        assert 'event: content_block_delta' in results[0]
        assert '"text":"test"' in results[0]
        assert 'event: message_stop' in results[1]
        assert '"type":"message_stop"' in results[1]

    @pytest.mark.asyncio
    async def test_malformed_json_continues_processing(self, transformer):
        """Test that malformed JSON doesn't stop processing of other events."""
        chunk_with_malformed_json = (
            'data: {"choices": [{"delta": {"content": "good"}}]}\n\ndata: {invalid json}\n\ndata: {"choices": [{"delta": {"content": "also good"}}]}\n\ndata: [DONE]\n'
        ).encode('utf-8')

        results = []
        async for result in transformer.transform_chunk({'chunk': chunk_with_malformed_json}):
            results.append(result.decode('utf-8'))

        # Should yield 3 events: good text, also good text, final message_stop
        # The malformed JSON should be skipped with warning
        assert len(results) == 3
        assert 'event: content_block_delta' in results[0]
        assert '"text":"good"' in results[0]
        assert 'event: content_block_delta' in results[1]
        assert '"text":"also good"' in results[1]
        assert 'event: message_stop' in results[2]
        assert '"type":"message_stop"' in results[2]

    @pytest.mark.asyncio
    async def test_non_sse_content_passthrough(self, transformer):
        """Test that non-SSE content is passed through unchanged."""
        non_sse_chunk = b'some random content without data: prefix'

        results = []
        async for result in transformer.transform_chunk({'chunk': non_sse_chunk}):
            results.append(result)

        # Should pass through unchanged
        assert len(results) == 1
        assert results[0] == non_sse_chunk
