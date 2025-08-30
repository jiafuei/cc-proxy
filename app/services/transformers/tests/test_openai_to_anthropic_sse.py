"""Tests for OpenAIToAnthropicSSETransformer."""

from unittest.mock import AsyncMock

import orjson
import pytest

from app.config.log import get_logger
from app.services.transformers.openai_to_anthropic_sse import OpenAIToAnthropicSSETransformer

logger = get_logger(__name__)


class TestOpenAIToAnthropicSSETransformer:
    """Test cases for OpenAI to Anthropic SSE transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return OpenAIToAnthropicSSETransformer(logger)

    @pytest.fixture
    def mock_params(self):
        """Mock parameters for transformer methods."""
        return {
            'chunk': b'',
            'request': {'model': 'gpt-4', 'stream': True},
            'final_headers': {'content-type': 'text/event-stream'},
            'provider_config': {},
            'original_request': AsyncMock()
        }

    def test_init_state(self, transformer):
        """Test state initialization."""
        state = transformer._init_state()
        
        assert state['message_id'] == ''
        assert state['model'] == ''
        assert state['next_block_index'] == 0
        assert state['active_text_block'] is None
        assert state['active_tool_block'] is None
        assert state['usage_tokens'] == {}
        assert state['stop_reason'] is None
        assert state['message_started'] is False

    def test_format_anthropic_sse(self, transformer):
        """Test Anthropic SSE formatting."""
        event_type = 'message_start'
        data = {'type': 'message_start', 'message': {'id': 'msg_123'}}
        
        result = transformer._format_anthropic_sse(event_type, data)
        expected = b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_123"}}\n\n'
        
        assert result == expected

    def test_convert_stop_reason(self, transformer):
        """Test stop reason conversion."""
        assert transformer._convert_stop_reason('stop') == 'end_turn'
        assert transformer._convert_stop_reason('length') == 'max_tokens'
        assert transformer._convert_stop_reason('content_filter') == 'stop_sequence'
        assert transformer._convert_stop_reason('tool_calls') == 'tool_use'
        assert transformer._convert_stop_reason('unknown') == 'end_turn'
        assert transformer._convert_stop_reason(None) == 'end_turn'

    def test_convert_openai_usage(self, transformer):
        """Test OpenAI usage conversion."""
        openai_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'prompt_tokens_details': {'cached_tokens': 20},
            'completion_tokens_details': {'reasoning_tokens': 10}
        }
        
        result = transformer._convert_openai_usage(openai_usage)
        
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['cache_read_input_tokens'] == 20
        assert result['cache_creation_input_tokens'] == 0

    def test_convert_openai_usage_empty(self, transformer):
        """Test OpenAI usage conversion with empty input."""
        result = transformer._convert_openai_usage({})
        
        assert result['input_tokens'] == 0
        assert result['output_tokens'] == 0
        assert result['cache_creation_input_tokens'] == 0
        assert result['cache_read_input_tokens'] == 0

    @pytest.mark.asyncio
    async def test_transform_chunk_done_marker(self, transformer, mock_params):
        """Test handling of [DONE] marker."""
        mock_params['chunk'] = b'data: [DONE]\n\n'
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk)
        
        assert len(results) == 1
        assert b'event: message_stop' in results[0]

    @pytest.mark.asyncio
    async def test_transform_chunk_invalid_json(self, transformer, mock_params):
        """Test handling of invalid JSON."""
        mock_params['chunk'] = b'data: {invalid json}\n\n'
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk)
        
        # Should skip invalid JSON without crashing
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_transform_chunk_text_only(self, transformer, mock_params):
        """Test transformation of text-only response."""
        # Role chunk
        role_chunk = {
            'id': 'chatcmpl-123',
            'model': 'gpt-4',
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]
        }
        role_data = b'data: ' + orjson.dumps(role_chunk) + b'\n\n'
        
        # Text chunk
        text_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{'index': 0, 'delta': {'content': 'Hello world'}}]
        }
        text_data = b'data: ' + orjson.dumps(text_chunk) + b'\n\n'
        
        # Finish chunk
        finish_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        finish_data = b'data: ' + orjson.dumps(finish_chunk) + b'\n\n'
        
        mock_params['chunk'] = role_data + text_data + finish_data
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk.decode())
        
        # Should have: message_start, content_block_start, content_block_delta, content_block_stop
        assert len(results) >= 4
        assert 'event: message_start' in results[0]
        assert 'event: content_block_start' in results[1]
        assert 'event: content_block_delta' in results[2]
        assert 'event: content_block_stop' in results[3]
        
        # Check content
        assert '"role":"assistant"' in results[0]
        assert '"type":"text"' in results[1]
        assert '"text":"Hello world"' in results[2]

    @pytest.mark.asyncio
    async def test_transform_chunk_tool_calls(self, transformer, mock_params):
        """Test transformation of tool calls."""
        # Role chunk
        role_chunk = {
            'id': 'chatcmpl-123',
            'model': 'gpt-4',
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]
        }
        role_data = b'data: ' + orjson.dumps(role_chunk) + b'\n\n'
        
        # Tool call start
        tool_start_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0, 
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'id': 'call_123',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': ''}
                    }]
                }
            }]
        }
        tool_start_data = b'data: ' + orjson.dumps(tool_start_chunk) + b'\n\n'
        
        # Tool arguments
        tool_args_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'function': {'arguments': '{"location": "'}
                    }]
                }
            }]
        }
        tool_args_data = b'data: ' + orjson.dumps(tool_args_chunk) + b'\n\n'
        
        # More arguments
        tool_args2_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'function': {'arguments': 'New York"}'}
                    }]
                }
            }]
        }
        tool_args2_data = b'data: ' + orjson.dumps(tool_args2_chunk) + b'\n\n'
        
        # Finish chunk
        finish_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]
        }
        finish_data = b'data: ' + orjson.dumps(finish_chunk) + b'\n\n'
        
        mock_params['chunk'] = role_data + tool_start_data + tool_args_data + tool_args2_data + finish_data
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk.decode())
        
        # Should have: message_start, content_block_start (tool), multiple deltas, content_block_stop
        assert len(results) >= 5
        assert 'event: message_start' in results[0]
        assert 'event: content_block_start' in results[1]
        assert '"type":"tool_use"' in results[1]
        assert '"id":"call_123"' in results[1]
        assert '"name":"get_weather"' in results[1]
        
        # Check argument streaming
        delta_events = [r for r in results if 'content_block_delta' in r]
        assert len(delta_events) >= 2
        assert '"input_json_delta"' in delta_events[0]

    @pytest.mark.asyncio
    async def test_transform_chunk_multiple_tools(self, transformer, mock_params):
        """Test transformation with multiple sequential tool calls."""
        # Setup first tool call
        tool1_start = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'tool1', 'arguments': ''}
                    }]
                }
            }]
        }
        
        tool1_args = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 0,
                        'function': {'arguments': '{"param": "value1"}'}
                    }]
                }
            }]
        }
        
        # Second tool call
        tool2_start = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 1,
                        'id': 'call_2', 
                        'type': 'function',
                        'function': {'name': 'tool2', 'arguments': ''}
                    }]
                }
            }]
        }
        
        tool2_args = {
            'id': 'chatcmpl-123',
            'choices': [{
                'index': 0,
                'delta': {
                    'tool_calls': [{
                        'index': 1,
                        'function': {'arguments': '{"param": "value2"}'}
                    }]
                }
            }]
        }
        
        finish_chunk = {
            'id': 'chatcmpl-123',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]
        }
        
        chunks = [tool1_start, tool1_args, tool2_start, tool2_args, finish_chunk]
        combined_data = b''.join(b'data: ' + orjson.dumps(chunk) + b'\n\n' for chunk in chunks)
        
        mock_params['chunk'] = combined_data
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk.decode())
        
        # Should have multiple content blocks
        start_events = [r for r in results if 'content_block_start' in r]
        stop_events = [r for r in results if 'content_block_stop' in r]
        
        assert len(start_events) == 2  # Two tool blocks
        assert len(stop_events) == 2   # Both should be stopped
        
        # Check that blocks have different indices
        assert '"index":0' in start_events[0]
        assert '"index":1' in start_events[1]

    @pytest.mark.asyncio
    async def test_transform_chunk_usage_only(self, transformer, mock_params):
        """Test handling of usage-only chunks at the end."""
        usage_chunk = {
            'id': 'chatcmpl-123',
            'choices': [],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            }
        }
        usage_data = b'data: ' + orjson.dumps(usage_chunk) + b'\n\n'
        
        # Initialize state with stop reason
        mock_params['sse_state'] = transformer._init_state()
        mock_params['sse_state']['stop_reason'] = 'stop'
        
        mock_params['chunk'] = usage_data
        
        results = []
        async for chunk in transformer.transform_chunk(mock_params):
            results.append(chunk.decode())
        
        assert len(results) == 1
        assert 'event: message_delta' in results[0]
        assert '"stop_reason":"end_turn"' in results[0]
        assert '"input_tokens":100' in results[0]
        assert '"output_tokens":50' in results[0]

    @pytest.mark.asyncio
    async def test_transform_response_non_streaming(self, transformer, mock_params):
        """Test transform_response method (should just return response with warning)."""
        mock_response = {'id': 'test', 'choices': []}
        mock_params['response'] = mock_response
        
        result = await transformer.transform_response(mock_params)
        assert result == mock_response

    def test_data_prefix_constant(self, transformer):
        """Test DATA_PREFIX constant."""
        assert transformer.DATA_PREFIX == b'data: '

    def test_done_marker_constant(self, transformer):
        """Test DONE_MARKER constant."""
        assert transformer.DONE_MARKER == b'[DONE]'

    def test_stop_reason_mapping_constant(self, transformer):
        """Test STOP_REASON_MAPPING constant."""
        expected_mapping = {
            'stop': 'end_turn',
            'length': 'max_tokens',
            'content_filter': 'stop_sequence', 
            'tool_calls': 'tool_use'
        }
        assert transformer.STOP_REASON_MAPPING == expected_mapping