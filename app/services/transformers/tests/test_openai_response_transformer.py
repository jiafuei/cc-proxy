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
