"""Tests for OpenAI Response Transformer."""

from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock

import orjson
import pytest

from app.config.log import get_logger
from app.services.transformers.openai import OpenAIResponseTransformer

logger = get_logger(__name__)


class TestOpenAIResponseTransformer:
    """Test cases for the OpenAI response transformer."""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        mock_logger = Mock()
        return OpenAIResponseTransformer(mock_logger)

    @pytest.fixture
    def mock_params_streaming(self):
        """Mock parameters for streaming transformer methods."""
        return {
            'chunk': b'',
            'request': {'model': 'gpt-4', 'stream': True},
            'final_headers': {'content-type': 'text/event-stream'},
            'provider_config': {},
            'original_request': AsyncMock(),
        }

    def _load_sse_file(self, filename: str) -> List[str]:
        """Load SSE file and return list of data lines."""
        # Get the directory containing this test file
        test_dir = Path(__file__).parent
        # Navigate to examples directory - go up to project root then to examples
        examples_dir = test_dir.parent.parent.parent.parent / 'examples' / 'concrete'
        sse_file_path = examples_dir / filename

        if not sse_file_path.exists():
            raise FileNotFoundError(f'SSE file not found: {sse_file_path}')

        # Read file and extract data lines
        with open(sse_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter to only data lines (skip empty lines)
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and line.startswith('data:'):
                data_lines.append(line)

        return data_lines

    def test_convert_openai_usage_comprehensive(self, transformer):
        """Test comprehensive OpenAI usage conversion with new method name."""
        openai_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'prompt_tokens_details': {'cached_tokens': 20},
            'completion_tokens_details': {'reasoning_tokens': 10},
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

    def test_usage_mapping_with_missing_fields(self, transformer):
        """Test usage mapping handles missing fields gracefully."""
        # Test with minimal usage
        minimal_usage = {'prompt_tokens': 10}
        result = transformer._convert_openai_usage(minimal_usage)
        assert result['input_tokens'] == 10
        assert result['cache_read_input_tokens'] == 0
        assert result['cache_creation_input_tokens'] == 0

        # Test with empty usage
        empty_result = transformer._convert_openai_usage({})
        assert empty_result['input_tokens'] == 0

    def test_init_state(self, transformer):
        """Test SSE state initialization."""
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

    @pytest.mark.asyncio
    async def test_transform_chunk_done_marker(self, transformer, mock_params_streaming):
        """Test handling of [DONE] marker."""
        mock_params_streaming['chunk'] = b'data: [DONE]\n\n'

        results = []
        async for chunk in transformer.transform_chunk(mock_params_streaming):
            results.append(chunk)

        assert len(results) == 1
        assert b'event: message_stop' in results[0]

    @pytest.mark.asyncio
    async def test_transform_chunk_invalid_json(self, transformer, mock_params_streaming):
        """Test handling of invalid JSON."""
        mock_params_streaming['chunk'] = b'data: {invalid json}\n\n'

        results = []
        async for chunk in transformer.transform_chunk(mock_params_streaming):
            results.append(chunk)

        # Should skip invalid JSON without crashing
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_transform_chunk_text_only(self, transformer, mock_params_streaming):
        """Test transformation of text-only streaming response."""
        # Role chunk
        role_chunk = {'id': 'chatcmpl-123', 'model': 'gpt-4', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]}
        role_data = b'data: ' + orjson.dumps(role_chunk) + b'\n\n'

        # Text chunk
        text_chunk = {'id': 'chatcmpl-123', 'choices': [{'index': 0, 'delta': {'content': 'Hello world'}}]}
        text_data = b'data: ' + orjson.dumps(text_chunk) + b'\n\n'

        # Finish chunk
        finish_chunk = {'id': 'chatcmpl-123', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        finish_data = b'data: ' + orjson.dumps(finish_chunk) + b'\n\n'

        mock_params_streaming['chunk'] = role_data + text_data + finish_data

        results = []
        async for chunk in transformer.transform_chunk(mock_params_streaming):
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

    def test_data_prefix_constant(self, transformer):
        """Test DATA_PREFIX constant."""
        assert transformer.DATA_PREFIX == b'data: '

    def test_done_marker_constant(self, transformer):
        """Test DONE_MARKER constant."""
        assert transformer.DONE_MARKER == b'[DONE]'

    def test_stop_reason_mapping_constant(self, transformer):
        """Test STOP_REASON_MAPPING constant."""
        expected_mapping = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'stop_sequence', 'tool_calls': 'tool_use'}
        assert transformer.STOP_REASON_MAPPING == expected_mapping

    @pytest.mark.asyncio
    async def test_transform_chunk_three_lines_at_a_time(self, transformer, mock_params_streaming):
        """Test transform_chunk with SSE data processed 3 lines at a time."""
        # Use text+multitool file for comprehensive testing of batched processing
        sse_lines = self._load_sse_file('1-openai-chat-completion-response-text-multitool.sse')

        # Group lines into batches of 3
        batches = [sse_lines[i : i + 3] for i in range(0, len(sse_lines), 3)]

        # Collect all emitted events
        all_results = []
        collected_arguments = []

        # Process each batch as a single chunk
        for batch in batches:
            # Combine 3 lines into a single chunk with proper SSE formatting
            chunk_data = '\n\n'.join(batch) + '\n\n'
            mock_params_streaming['chunk'] = chunk_data.encode()

            # Collect results from this batch
            async for chunk in transformer.transform_chunk(mock_params_streaming):
                result = chunk.decode()
                all_results.append(result)

                # Track tool argument deltas for validation
                if 'event: content_block_delta' in result and '"type":"input_json_delta"' in result:
                    # Extract partial_json from the delta
                    import re

                    match = re.search(r'"partial_json":"([^"]*)"', result)
                    if match:
                        collected_arguments.append(match.group(1))

        # Validate event sequence and structure
        event_types = []
        for result in all_results:
            if result.startswith('event:'):
                event_type = result.split('\n')[0].split('event: ')[1]
                event_types.append(event_type)

        # Expected event sequence for text + tools response with batched processing
        assert event_types[0] == 'message_start', 'First event should be message_start'
        assert event_types[1] == 'content_block_start', 'Second event should be content_block_start'
        assert event_types[-2] == 'message_delta', 'Second to last event should be message_delta'
        assert event_types[-1] == 'message_stop', 'Last event should be message_stop'

        # Validate message_start contains correct information
        message_start = all_results[0]
        assert '"role":"assistant"' in message_start
        assert '"model":"gpt-5-mini-2025-08-07"' in message_start

        # Check for both text and tool content blocks
        text_block_found = False
        tool_blocks_found = 0

        for result in all_results:
            if 'event: content_block_start' in result:
                if '"type":"text"' in result:
                    text_block_found = True
                elif '"type":"tool_use"' in result:
                    tool_blocks_found += 1

        assert text_block_found, 'Should have found text content block'
        assert tool_blocks_found == 2, 'Should have found exactly 2 tool call blocks'

        # Validate text content exists
        text_deltas = [r for r in all_results if 'content_block_delta' in r and '"type":"text_delta"' in r]
        assert len(text_deltas) > 0, 'Should have text delta events'

        # Validate tool arguments exist
        tool_deltas = [r for r in all_results if 'content_block_delta' in r and '"type":"input_json_delta"' in r]
        assert len(tool_deltas) > 0, 'Should have tool argument delta events'

        # Verify incremental argument building works with batched processing
        full_arguments = ''.join(collected_arguments)
        # Handle basic JSON escaping in the extracted arguments
        full_arguments = full_arguments.replace('\\', '').replace('"', '"')
        assert 'llow sunsets' in full_arguments, f'Should contain expected tool arguments. Got: {full_arguments}'

        # Validate final usage information in message_delta
        message_delta_events = [r for r in all_results if 'event: message_delta' in r]
        assert len(message_delta_events) == 1
        usage_result = message_delta_events[0]
        assert '"stop_reason":"tool_use"' in usage_result

        # Validate proper event counts for batched processing (text block + 2 tool blocks = 3 total)
        event_counts = {}
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        assert event_counts['message_start'] == 1
        assert event_counts['content_block_start'] == 3  # 1 text + 2 tools
        assert event_counts['content_block_stop'] == 3
        assert event_counts['message_delta'] == 1
        assert event_counts['message_stop'] == 1

        # Validate that batching didn't break the streaming behavior
        assert len(batches) > 1, f'Should have processed multiple batches. Got {len(batches)} batches'
        assert len(all_results) > 10, 'Should have many events from batched processing'

    @pytest.mark.asyncio
    async def test_transform_chunk_line_by_line_text_only(self, transformer, mock_params_streaming):
        """Test transform_chunk called line by line with text-only SSE file."""
        # Load SSE lines from example file
        sse_lines = self._load_sse_file('1-openai-chat-completion-response-text.sse')

        # Collect all emitted events
        all_results = []

        # Process each line individually to test line-by-line behavior
        for line in sse_lines:
            mock_params_streaming['chunk'] = (line + '\n\n').encode()

            # Collect results from this line
            async for chunk in transformer.transform_chunk(mock_params_streaming):
                result = chunk.decode()
                all_results.append(result)

        # Validate event sequence and structure
        event_types = []
        for result in all_results:
            if result.startswith('event:'):
                event_type = result.split('\n')[0].split('event: ')[1]
                event_types.append(event_type)

        # Expected event sequence for text-only response
        assert event_types[0] == 'message_start', 'First event should be message_start'
        assert event_types[1] == 'content_block_start', 'Second event should be content_block_start'
        assert event_types[-2] == 'message_delta', 'Second to last event should be message_delta'
        assert event_types[-1] == 'message_stop', 'Last event should be message_stop'

        # Validate message_start contains correct information
        message_start = all_results[0]
        assert '"role":"assistant"' in message_start
        assert '"model":"gpt-5-mini-2025-08-07"' in message_start

        # Validate content_block_start for text
        content_block_start = all_results[1]
        assert '"type":"text"' in content_block_start
        assert '"index":0' in content_block_start

        # Validate that content_block_delta events contain text
        delta_events = [r for r in all_results if 'event: content_block_delta' in r]
        assert len(delta_events) > 5, 'Should have multiple content_block_delta events for text streaming'

        # Validate text deltas contain expected content fragments
        text_deltas = [r for r in all_results if 'event: content_block_delta' in r and '"type":"text_delta"' in r]
        assert len(text_deltas) > 0, 'Should have text delta events'

        # Check that some expected text appears in deltas
        # Extract the actual text content from JSON in the results
        import re

        text_content_parts = []
        for result in all_results:
            if 'text_delta' in result:
                # Extract text from JSON: "text":"content"
                match = re.search(r'"text":"([^"]*)"', result)
                if match:
                    text_content_parts.append(match.group(1))

        all_text_content = ''.join(text_content_parts).lower()
        assert 'let me get that for ya' in all_text_content, f'Should contain expected text fragments. Got: {all_text_content}'

        # Validate final usage information in message_delta
        message_delta_events = [r for r in all_results if 'event: message_delta' in r]
        assert len(message_delta_events) == 1
        usage_result = message_delta_events[0]
        assert '"stop_reason":"end_turn"' in usage_result

        # Validate proper event counts
        event_counts = {}
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        assert event_counts['message_start'] == 1
        assert event_counts['content_block_start'] == 1
        assert event_counts['content_block_stop'] == 1
        assert event_counts['message_delta'] == 1
        assert event_counts['message_stop'] == 1

    @pytest.mark.asyncio
    async def test_transform_chunk_line_by_line_text_and_tools(self, transformer, mock_params_streaming):
        """Test transform_chunk called line by line with text and multitool SSE file."""
        # Load SSE lines from example file
        sse_lines = self._load_sse_file('1-openai-chat-completion-response-text-multitool.sse')

        # Collect all emitted events
        all_results = []
        collected_arguments = []

        # Process each line individually to test line-by-line behavior
        for line in sse_lines:
            mock_params_streaming['chunk'] = (line + '\n\n').encode()

            # Collect results from this line
            async for chunk in transformer.transform_chunk(mock_params_streaming):
                result = chunk.decode()
                all_results.append(result)

                # Track tool argument deltas for validation
                if 'event: content_block_delta' in result and '"type":"input_json_delta"' in result:
                    # Extract partial_json from the delta
                    import re

                    match = re.search(r'"partial_json":"([^"]*)"', result)
                    if match:
                        collected_arguments.append(match.group(1))

        # Validate event sequence and structure
        event_types = []
        for result in all_results:
            if result.startswith('event:'):
                event_type = result.split('\n')[0].split('event: ')[1]
                event_types.append(event_type)

        # Expected event sequence for text + tools response
        assert event_types[0] == 'message_start', 'First event should be message_start'
        assert event_types[1] == 'content_block_start', 'Second event should be content_block_start'
        assert event_types[-2] == 'message_delta', 'Second to last event should be message_delta'
        assert event_types[-1] == 'message_stop', 'Last event should be message_stop'

        # Validate message_start contains correct information
        message_start = all_results[0]
        assert '"role":"assistant"' in message_start
        assert '"model":"gpt-5-mini-2025-08-07"' in message_start

        # Check for both text and tool content blocks
        text_block_found = False
        tool_blocks_found = 0

        for result in all_results:
            if 'event: content_block_start' in result:
                if '"type":"text"' in result:
                    text_block_found = True
                elif '"type":"tool_use"' in result:
                    tool_blocks_found += 1
                    if 'get_meaning' in result:
                        assert '"index":1' in result
                    elif 'get_extra_meaning' in result:
                        assert '"index":2' in result

        assert text_block_found, 'Should have found text content block'
        assert tool_blocks_found == 2, 'Should have found exactly 2 tool call blocks'

        # Validate text content
        text_deltas = [r for r in all_results if 'content_block_delta' in r and '"type":"text_delta"' in r]
        assert len(text_deltas) > 0, 'Should have text delta events'

        # Validate tool arguments
        tool_deltas = [r for r in all_results if 'content_block_delta' in r and '"type":"input_json_delta"' in r]
        assert len(tool_deltas) > 0, 'Should have tool argument delta events'

        # Verify incremental argument building
        full_arguments = ''.join(collected_arguments)
        # Handle basic JSON escaping in the extracted arguments
        full_arguments = full_arguments.replace('\\', '').replace('"', '"')
        assert 'llow sunsets' in full_arguments, f'Should contain expected tool arguments. Got: {full_arguments}'

        # Validate final usage information in message_delta
        message_delta_events = [r for r in all_results if 'event: message_delta' in r]
        assert len(message_delta_events) == 1
        usage_result = message_delta_events[0]
        assert '"stop_reason":"tool_use"' in usage_result

        # Validate proper event counts (text block + 2 tool blocks = 3 total)
        event_counts = {}
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        assert event_counts['message_start'] == 1
        assert event_counts['content_block_start'] == 3  # 1 text + 2 tools
        assert event_counts['content_block_stop'] == 3
        assert event_counts['message_delta'] == 1
        assert event_counts['message_stop'] == 1

    @pytest.mark.asyncio
    async def test_transform_chunk_line_by_line_pure_tools(self, transformer, mock_params_streaming):
        """Test transform_chunk called line by line with pure multitool SSE file."""
        # Load SSE lines from example file
        sse_lines = self._load_sse_file('1-openai-chat-completion-response-multitool.sse')

        # Collect all emitted events
        all_results = []
        collected_arguments = []

        # Process each line individually to test line-by-line behavior
        for line in sse_lines:
            mock_params_streaming['chunk'] = (line + '\n\n').encode()

            # Collect results from this line
            async for chunk in transformer.transform_chunk(mock_params_streaming):
                result = chunk.decode()
                all_results.append(result)

                # Track tool argument deltas for validation
                if 'event: content_block_delta' in result and '"type":"input_json_delta"' in result:
                    # Extract partial_json from the delta
                    import re

                    match = re.search(r'"partial_json":"([^"]*)"', result)
                    if match:
                        collected_arguments.append(match.group(1))

        # Validate event sequence and structure
        event_types = []
        for result in all_results:
            if result.startswith('event:'):
                event_type = result.split('\n')[0].split('event: ')[1]
                event_types.append(event_type)

        # Expected event sequence for pure tools response
        assert event_types[0] == 'message_start', 'First event should be message_start'
        assert event_types[1] == 'content_block_start', 'Second event should be content_block_start'
        assert event_types[-2] == 'message_delta', 'Second to last event should be message_delta'
        assert event_types[-1] == 'message_stop', 'Last event should be message_stop'

        # Validate message_start contains correct information
        message_start = all_results[0]
        assert '"role":"assistant"' in message_start
        assert '"model":"gpt-5-mini-2025-08-07"' in message_start

        # Check for tool content blocks only (no text)
        text_block_found = False
        tool_blocks_found = 0

        for result in all_results:
            if 'event: content_block_start' in result:
                if '"type":"text"' in result:
                    text_block_found = True
                elif '"type":"tool_use"' in result:
                    tool_blocks_found += 1

        assert not text_block_found, 'Should not have found text content block'
        assert tool_blocks_found == 2, 'Should have found exactly 2 tool call blocks'

        # Validate tool arguments
        tool_deltas = [r for r in all_results if 'content_block_delta' in r and '"type":"input_json_delta"' in r]
        assert len(tool_deltas) > 0, 'Should have tool argument delta events'

        # Verify incremental argument building
        full_arguments = ''.join(collected_arguments)
        # Handle basic JSON escaping in the extracted arguments
        full_arguments = full_arguments.replace('\\', '').replace('"', '"')
        assert 'llow sunsets' in full_arguments, f'Should contain expected tool arguments. Got: {full_arguments}'

        # Validate final usage information in message_delta
        message_delta_events = [r for r in all_results if 'event: message_delta' in r]
        assert len(message_delta_events) == 1
        usage_result = message_delta_events[0]
        assert '"stop_reason":"tool_use"' in usage_result

        # Validate proper event counts (2 tool blocks only)
        event_counts = {}
        for event_type in event_types:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        assert event_counts['message_start'] == 1
        assert event_counts['content_block_start'] == 2  # 2 tools only
        assert event_counts['content_block_stop'] == 2
        assert event_counts['message_delta'] == 1
        assert event_counts['message_stop'] == 1
