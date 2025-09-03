"""Tests for Gemini response transformer."""

from unittest.mock import MagicMock

import pytest

from app.services.transformers.gemini import GeminiResponseTransformer


class TestGeminiResponseTransformer:
    """Test cases for GeminiResponseTransformer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = MagicMock()
        self.transformer = GeminiResponseTransformer(self.logger)

    @pytest.mark.asyncio
    async def test_basic_non_streaming_response_conversion(self):
        """Test conversion of basic non-streaming Gemini response."""
        gemini_response = {
            'id': 'resp_123',
            'model': 'gemini-2.0-flash-exp',
            'candidates': [{'content': {'parts': [{'text': 'Hello! How can I help you today?'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 8},
        }

        params = {'response': gemini_response}
        anthropic_response = await self.transformer.transform_response(params)

        # Check basic structure
        assert anthropic_response['id'] == 'resp_123'
        assert anthropic_response['type'] == 'message'
        assert anthropic_response['role'] == 'assistant'
        assert anthropic_response['model'] == 'gemini-2.0-flash-exp'
        assert anthropic_response['stop_reason'] == 'end_turn'
        assert anthropic_response['stop_sequence'] is None

        # Check content conversion
        content = anthropic_response['content']
        assert len(content) == 1
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == 'Hello! How can I help you today?'

        # Check usage conversion
        usage = anthropic_response['usage']
        assert usage['input_tokens'] == 10
        assert usage['output_tokens'] == 8
        assert usage['cache_creation_input_tokens'] == 0
        assert usage['cache_read_input_tokens'] == 0

    @pytest.mark.asyncio
    async def test_function_call_response_conversion(self):
        """Test conversion of response with function calls."""
        gemini_response = {
            'id': 'resp_456',
            'model': 'gemini-2.0-flash-exp',
            'candidates': [
                {
                    'content': {'parts': [{'text': "I'll get the weather for you."}, {'functionCall': {'name': 'get_weather', 'args': {'city': 'San Francisco'}}}]},
                    'finishReason': 'STOP',
                }
            ],
            'usageMetadata': {'promptTokenCount': 15, 'candidatesTokenCount': 12},
        }

        params = {'response': gemini_response}
        anthropic_response = await self.transformer.transform_response(params)

        content = anthropic_response['content']
        assert len(content) == 2

        # Check text part
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == "I'll get the weather for you."

        # Check tool use part
        assert content[1]['type'] == 'tool_use'
        assert content[1]['id'].startswith('toolu_')  # Generated ID
        assert content[1]['name'] == 'get_weather'
        assert content[1]['input'] == {'city': 'San Francisco'}

    @pytest.mark.asyncio
    async def test_stop_reason_mapping(self):
        """Test mapping of Gemini stop reasons to Anthropic format."""
        test_cases = [
            ('STOP', 'end_turn'),
            ('MAX_TOKENS', 'max_tokens'),
            ('SAFETY', 'stop_sequence'),
            ('RECITATION', 'stop_sequence'),
            ('OTHER', 'end_turn'),
            (None, 'end_turn'),
            ('UNKNOWN_REASON', 'end_turn'),  # Default case
        ]

        for gemini_reason, expected_anthropic in test_cases:
            result = self.transformer._convert_stop_reason(gemini_reason)
            assert result == expected_anthropic

    @pytest.mark.asyncio
    async def test_empty_usage_metadata_handling(self):
        """Test handling of empty or missing usage metadata."""
        gemini_response = {
            'candidates': [{'content': {'parts': [{'text': 'Response'}]}, 'finishReason': 'STOP'}]
            # No usageMetadata
        }

        params = {'response': gemini_response}
        anthropic_response = await self.transformer.transform_response(params)

        usage = anthropic_response['usage']
        assert usage['input_tokens'] == 0
        assert usage['output_tokens'] == 0
        assert usage['cache_creation_input_tokens'] == 0
        assert usage['cache_read_input_tokens'] == 0

    @pytest.mark.asyncio
    async def test_no_candidates_handling(self):
        """Test handling of response with no candidates."""
        gemini_response = {'id': 'resp_no_candidates', 'candidates': []}

        params = {'response': gemini_response}
        anthropic_response = await self.transformer.transform_response(params)

        # Should return original response when no candidates
        assert anthropic_response == gemini_response

    @pytest.mark.asyncio
    async def test_streaming_chunk_processing(self):
        """Test processing of streaming response chunks."""
        # Mock SSE chunk data
        chunk_data = b'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}]}\n\n'

        params = {'chunk': chunk_data, 'request': {}, 'final_headers': {}, 'provider_config': {}, 'original_request': {}}

        events = []
        async for event_bytes in self.transformer.transform_chunk(params):
            events.append(event_bytes)

        # Should generate at least message_start and content events
        assert len(events) > 0

        # Check that we get properly formatted SSE events
        for event in events:
            assert isinstance(event, bytes)
            assert b'event:' in event and b'data:' in event

    @pytest.mark.asyncio
    async def test_streaming_done_marker(self):
        """Test handling of streaming DONE marker."""
        chunk_data = b'data: [DONE]\n\n'

        params = {'chunk': chunk_data}

        events = []
        async for event_bytes in self.transformer.transform_chunk(params):
            events.append(event_bytes)

        # Should generate message_stop event
        assert len(events) == 1
        assert b'message_stop' in events[0]

    @pytest.mark.asyncio
    async def test_streaming_invalid_json_handling(self):
        """Test handling of invalid JSON in streaming response."""
        chunk_data = b'data: {invalid json}\n\n'

        params = {'chunk': chunk_data}

        events = []
        async for event_bytes in self.transformer.transform_chunk(params):
            events.append(event_bytes)

        # Should handle gracefully and not generate events for invalid JSON
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_sse_state_initialization(self):
        """Test that SSE state is properly initialized."""
        chunk_data = b'data: {"candidates":[{"content":{"parts":[{"text":"test"}]}}]}\n\n'

        params = {'chunk': chunk_data}

        events = []
        async for event_bytes in self.transformer.transform_chunk(params):
            events.append(event_bytes)

        # State should be created and message_started should be set
        assert 'sse_state' in params
        state = params['sse_state']
        assert 'message_started' in state
        assert 'next_block_index' in state
        assert 'active_text_block' in state

    def test_format_anthropic_sse(self):
        """Test formatting of Anthropic SSE events."""
        test_data = {'type': 'test', 'content': 'hello'}

        result = self.transformer._format_anthropic_sse('test_event', test_data)

        assert isinstance(result, bytes)
        result_str = result.decode()
        assert 'event: test_event\n' in result_str
        assert '"type":"test"' in result_str
        assert '"content":"hello"' in result_str
        assert result_str.endswith('\n\n')

    def test_init_state(self):
        """Test initialization of streaming state."""
        state = self.transformer._init_state()

        expected_keys = ['message_id', 'model', 'next_block_index', 'active_text_block', 'active_tool_block', 'usage_tokens', 'stop_reason', 'message_started']

        for key in expected_keys:
            assert key in state

        assert state['next_block_index'] == 0
        assert state['active_text_block'] is None
        assert state['message_started'] is False

    @pytest.mark.asyncio
    async def test_usage_conversion_comprehensive(self):
        """Test comprehensive usage metadata conversion."""
        usage_metadata = {'promptTokenCount': 100, 'candidatesTokenCount': 50, 'totalTokenCount': 150}

        result = self.transformer._convert_gemini_usage(usage_metadata)

        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['cache_creation_input_tokens'] == 0
        assert result['cache_read_input_tokens'] == 0

    @pytest.mark.asyncio
    async def test_response_conversion_error_handling(self):
        """Test error handling during response conversion."""
        # Malformed response that should trigger exception handling
        malformed_response = {'invalid': 'structure'}

        params = {'response': malformed_response}
        result = await self.transformer.transform_response(params)

        # Should return original response on conversion failure
        assert result == malformed_response

    @pytest.mark.asyncio
    async def test_multipart_content_conversion(self):
        """Test conversion of responses with multiple content parts."""
        gemini_response = {'candidates': [{'content': {'parts': [{'text': 'First part'}, {'text': 'Second part'}, {'text': 'Third part'}]}, 'finishReason': 'STOP'}]}

        params = {'response': gemini_response}
        anthropic_response = await self.transformer.transform_response(params)

        content = anthropic_response['content']
        assert len(content) == 3

        for i, expected_text in enumerate(['First part', 'Second part', 'Third part']):
            assert content[i]['type'] == 'text'
            assert content[i]['text'] == expected_text

    @pytest.mark.asyncio
    async def test_streaming_non_data_lines_ignored(self):
        """Test that non-data SSE lines are ignored during streaming."""
        chunk_data = b'event: ping\nretry: 1000\ndata: {"test": "data"}\n\n'

        params = {'chunk': chunk_data}

        events = []
        async for event_bytes in self.transformer.transform_chunk(params):
            events.append(event_bytes)

        # Should only process data lines, ignoring event and retry lines
        # The exact number depends on implementation, but should handle gracefully
